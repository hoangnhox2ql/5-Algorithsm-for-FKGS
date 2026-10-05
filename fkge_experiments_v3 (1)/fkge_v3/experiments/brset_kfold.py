"""
experiments/brset_kfold.py — Chạy FISA (FKG-Pairs, như FKGS v3.2) và FKG-E (Chương 3) trên cơ sở luật FRB của
BRSET đã chia 5 fold theo ID bệnh nhân (data_real/BRSET_Data/fold_XX/{TrainDataRule,TestDataRule}.csv).

Quy trình cho mỗi fold (tập kiểm tra chỉ dùng để đánh giá cuối cùng):
  1. Nạp FRB của fold: mỗi dòng TrainDataRule là một luật và cũng là một mẫu huấn luyện; nhóm hoá các dòng
     trùng nhau (do cân bằng lớp bằng lặp mẫu) để mọi phép chia nội bộ giữ các bản sao trong cùng một phần.
  2. Tính chéo theo phần (F_c phần theo nhóm): với mỗi phần, xây FISA trên các luật của các phần còn lại và suy
     diễn cho các mẫu của phần đó -> giá trị D ngoài phần, nhãn ŷ_F và căn cứ luật a_F (Định nghĩa 3.18).
  3. Ngưỡng τ_F của FISA chọn trên điểm ngoài phần của toàn bộ tập huấn luyện (tiêu chí độ chính xác cân bằng);
     FISA cuối cùng xây trên toàn bộ luật của fold, dùng τ_F.
  4. FKG-E: tách 20% nhóm làm tập xác thực; huấn luyện với giáo viên ngoài phần, loại luật của chính mẫu khi
     tính mất mát; dừng sớm và chọn τ_E trên tập xác thực.
  5. Đánh giá trên tập kiểm tra: AUC, độ chính xác, độ chính xác cân bằng, độ nhạy, độ đặc hiệu, F1 lớp dương,
     đồng thuận với FISA (R2), độ lệch căn cứ luật Dev (R3), thời gian mỗi truy vấn (FISA bảng tra, FISA tuần tự,
     FKG-E).
Tổng hợp: trung bình, độ lệch chuẩn và khoảng tin cậy 95% (phân phối t, 4 bậc tự do) trên 5 fold.

  python experiments/brset_kfold.py                 # đầy đủ
  python experiments/brset_kfold.py --quick         # T_ep nhỏ, để kiểm tra nhanh
  python experiments/brset_kfold.py --folds fold_01,fold_02
"""
import sys, os, json, time, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import config as C
from data.brset_folds import list_folds, load_fold, subset_rulebase
from models.fisa import FISA, FISASequential, _balanced_acc, _auc
from models.fkge import FKGE

T_CRIT_4DF = 2.776   # t_{0,975; 4}


def group_partition(groups, n_parts, seed):
    """Chia các nhóm (không phải từng dòng) thành n_parts phần, xáo trộn theo seed."""
    ug = np.unique(groups)
    rng = np.random.RandomState(seed); rng.shuffle(ug)
    part_of = {g: i % n_parts for i, g in enumerate(ug)}
    return np.array([part_of[g] for g in groups])


def binary_metrics(y_true, y_pred, score, pos):
    y = np.array([1 if v == pos else 0 for v in y_true]); p = np.array([1 if v == pos else 0 for v in y_pred])
    tp = int(((y == 1) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan"); spec = tn / (tn + fp) if tn + fp else float("nan")
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens > 0 else 0.0
    return {"auc": _auc(y.tolist(), list(score)), "accuracy": (tp + tn) / len(y), "balanced_accuracy": (sens + spec) / 2,
            "sensitivity": sens, "specificity": spec, "precision": prec, "f1_pos": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def oof_fisa(fkg, samples, parts, n_parts):
    """D ngoài phần (log), nhãn argmax và căn cứ luật a_F cho mọi mẫu huấn luyện."""
    nL = len(fkg.class_tokens); n = len(samples); nR = len(fkg.rules)
    pos_of = {r["id"]: k for k, r in enumerate(fkg.rules)}
    Dlog = np.zeros((n, nL)); D = np.zeros((n, nL)); matched = np.zeros(n, dtype=bool)
    models = {}
    for p in range(n_parts):
        idx = np.where(parts == p)[0]
        keep = [samples[i]["rule_id"] for i in np.where(parts != p)[0]]
        f = FISA(subset_rulebase(fkg, keep), decision="argmax").fit()
        models[p] = f
        for i in idx:
            d, _, mt = f.decision_values(samples[i]["membership"])
            D[i] = [d[c] for c in fkg.class_tokens]; matched[i] = mt
            Dlog[i] = np.log(D[i] + 1e-12)
    return D, Dlog, matched, models, pos_of


def run_fold(root, fold, cfg):
    t_fold = time.time()
    fkg, train, test, info = load_fold(root, fold)
    pos = fkg.class_tokens[-1]
    groups = np.array([s["group"] for s in train])
    parts = group_partition(groups, cfg["F_c"], cfg["seed"])
    # ---- FISA: giáo viên ngoài phần + ngưỡng τ_F trên điểm ngoài phần
    D, Dlog, matched, oof_models, pos_of = oof_fisa(fkg, train, parts, cfg["F_c"])
    s_oof = np.where(matched, Dlog[:, 1] - Dlog[:, 0], 0.0)
    y_tr = np.array([1 if s["label"] == pos else 0 for s in train])
    cand = np.unique(np.concatenate([[0.0], np.quantile(s_oof, np.linspace(0.01, 0.99, 197))]))
    bacc = [0.5 * (np.mean((s_oof > t)[y_tr == 1]) + np.mean((s_oof <= t)[y_tr == 0])) for t in cand]
    tau_F = float(cand[int(np.argmax(bacc))])
    fisa = FISA(fkg, decision="calibrated").fit(); fisa.threshold = tau_F
    # căn cứ luật và nhãn của giáo viên (ngoài phần, cùng ngưỡng τ_F)
    nR = len(fkg.rules); eF = np.zeros((len(train), nR)); yF = np.zeros(len(train), dtype=int)
    for i, s in enumerate(train):
        f = oof_models[parts[i]]; f.decision = "calibrated"; f.threshold = tau_F
        d = {c: D[i, k] for k, c in enumerate(fkg.class_tokens)}
        yF[i] = fkg.class_tokens.index(f._decide(d, matched[i]))
        for rid, a in f.attribution(s["membership"]).items():
            eF[i, pos_of[rid]] = a
    # ---- đánh giá FISA trên tập kiểm tra
    t0 = time.time(); res_F = fisa.evaluate(test); tF = res_F["avg_time_per_query_ms"]
    seq = FISASequential(fkg, decision="calibrated").fit(); seq.threshold = tau_F
    test_seq = test[: min(len(test), cfg["n_seq_timing"])]
    tS = seq.evaluate(test_seq)["avg_time_per_query_ms"]
    sF = []
    for s in test:
        d, _, mt = fisa.decision_values(s["membership"]); sF.append(fisa._score(d) if mt else 0.0)
    mF = binary_metrics(res_F["y_true"], res_F["y_pred"], sF, pos)
    mF.update(accuracy_argmax=res_F["accuracy_argmax"], balanced_accuracy_argmax=res_F["balanced_accuracy_argmax"],
              accuracy_ratio9=res_F["accuracy_ratio9"], ms_per_query_table=tF, ms_per_query_sequential=tS,
              threshold=tau_F, n_unseen=res_F["n_unseen"])
    # ---- FKG-E
    vparts = group_partition(groups, int(round(1 / cfg["val_fraction"])), cfg["seed"] + 1)
    vidx = np.where(vparts == 0)[0]; tidx = np.where(vparts != 0)[0]
    beta = np.zeros(len(fkg.class_tokens)); beta[1] = -tau_F
    td = {"Dlog": Dlog[tidx], "eF": eF[tidx], "yF": yF[tidx], "Dlog_val": Dlog[vidx], "beta": beta}
    m = FKGE(fkg, T_ep=cfg["T_ep"], seed=cfg["seed"])
    m.fit(fisa_model=fisa, train_samples=[train[i] for i in tidx], val_samples=[train[i] for i in vidx],
          teacher_data=td)
    res_E = m.evaluate(test, fisa_model=fisa)
    Q = m._query_matrix(test); _, pE = m._infer(Q)
    mE = binary_metrics(res_E["y_true"], res_E["y_pred"], m._logratio(pE).tolist(), pos)
    mE.update(agreement_fisa=res_E["agreement_fisa"], dev_rule=res_E["dev_rule"], log_loss=res_E["log_loss"],
              ms_per_query=res_E["avg_time_per_query_ms"], threshold=m.threshold, epochs=m.epochs_run,
              stop=m.stop_reason, train_time_s=m.train_time_s, final_grad_norm=m.history["grad_norm"][-1],
              teacher_T=m.teacher_T)
    out = {"fold": fold, "info": info, "FISA": mF, "FKGE": mE, "time_s": time.time() - t_fold}
    print(f"[{fold}] n_train={info['n_train']} (nhóm {info['n_groups']}), n_test={info['n_test']} (dương {info['test_pos']})")
    for name, r in [("FISA ", mF), ("FKG-E", mE)]:
        print(f"   {name}: AUC={r['auc']:.4f}  BalAcc={r['balanced_accuracy']:.4f}  Sens={r['sensitivity']:.4f}  "
              f"Spec={r['specificity']:.4f}  F1+={r['f1_pos']:.4f}  Acc={r['accuracy']:.4f}")
    print(f"   FKG-E: đồng thuận FISA={mE['agreement_fisa']:.4f}  Dev={mE['dev_rule']:.4f}  epoch={mE['epochs']} ({mE['stop']})"
          f"  ‖∇L‖={mE['final_grad_norm']:.2e}  | ms/truy vấn: FISA bảng tra {tF:.3f}, FISA tuần tự {tS:.3f}, "
          f"FKG-E {mE['ms_per_query']:.3f}  | {out['time_s']:.0f}s")
    return out


def summarize(results):
    keys = ["auc", "balanced_accuracy", "sensitivity", "specificity", "f1_pos", "accuracy"]
    summ = {}
    for model, extra in [("FISA", ["accuracy_argmax", "balanced_accuracy_argmax", "accuracy_ratio9",
                                   "ms_per_query_table", "ms_per_query_sequential"]),
                         ("FKGE", ["agreement_fisa", "dev_rule", "ms_per_query", "log_loss"])]:
        summ[model] = {}
        for k in keys + extra:
            x = np.array([r[model][k] for r in results], dtype=float)
            mu, sd = float(np.nanmean(x)), float(np.nanstd(x, ddof=1)) if len(x) > 1 else 0.0
            hw = T_CRIT_4DF * sd / math.sqrt(len(x)) if len(x) == 5 else float("nan")
            summ[model][k] = {"mean": mu, "sd": sd, "ci95": [mu - hw, mu + hw], "per_fold": x.tolist()}
    return summ


def write_report(results, summ, path, cfg):
    L = ["# FISA (FKG-Pairs, FKGS v3.2) và FKG-E (Chương 3) trên BRSET — 5 fold theo ID bệnh nhân", "",
         f"Cấu hình: F_c = {cfg['F_c']} phần tính chéo; tập xác thực FKG-E = {int(cfg['val_fraction']*100)}% nhóm; "
         f"T_ep = {cfg['T_ep']}; tham số FKG-E theo config.FKGE.", "",
         "| Độ đo | FISA (trung bình ± SD) | FISA CI95 | FKG-E (trung bình ± SD) | FKG-E CI95 |", "|---|---|---|---|---|"]
    names = {"auc": "AUC-ROC", "balanced_accuracy": "Độ chính xác cân bằng", "sensitivity": "Độ nhạy",
             "specificity": "Độ đặc hiệu", "f1_pos": "F1 lớp dương", "accuracy": "Độ chính xác"}
    for k, nm in names.items():
        a, b = summ["FISA"][k], summ["FKGE"][k]
        L.append(f"| {nm} | {a['mean']:.4f} ± {a['sd']:.4f} | [{a['ci95'][0]:.4f}; {a['ci95'][1]:.4f}] | "
                 f"{b['mean']:.4f} ± {b['sd']:.4f} | [{b['ci95'][0]:.4f}; {b['ci95'][1]:.4f}] |")
    L += ["", "| FISA — quy tắc khác | Giá trị |", "|---|---|",
          f"| Acc / BalAcc theo argmax (công thức (12)) | {summ['FISA']['accuracy_argmax']['mean']:.4f} / {summ['FISA']['balanced_accuracy_argmax']['mean']:.4f} |",
          f"| Acc theo D0 > 9·D1 (notebook gốc) | {summ['FISA']['accuracy_ratio9']['mean']:.4f} |", "",
          "| FKG-E — ràng buộc Bài toán 3.1A | Giá trị |", "|---|---|",
          f"| Đồng thuận nhãn với FISA (R2) | {summ['FKGE']['agreement_fisa']['mean']:.4f} ± {summ['FKGE']['agreement_fisa']['sd']:.4f} |",
          f"| Độ lệch căn cứ luật Dev (R3) | {summ['FKGE']['dev_rule']['mean']:.4f} ± {summ['FKGE']['dev_rule']['sd']:.4f} |",
          f"| ms/truy vấn: FISA bảng tra / FISA tuần tự / FKG-E (R4) | {summ['FISA']['ms_per_query_table']['mean']:.3f} / "
          f"{summ['FISA']['ms_per_query_sequential']['mean']:.3f} / {summ['FKGE']['ms_per_query']['mean']:.3f} |", "",
          "## Theo fold", "", "| Fold | n_train (nhóm) | n_test (dương) | FISA AUC | FISA BalAcc | FKG-E AUC | FKG-E BalAcc | Đồng thuận | Dev | Epoch (dừng) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        i, F, E = r["info"], r["FISA"], r["FKGE"]
        L.append(f"| {r['fold']} | {i['n_train']} ({i['n_groups']}) | {i['n_test']} ({i['test_pos']}) | {F['auc']:.4f} | "
                 f"{F['balanced_accuracy']:.4f} | {E['auc']:.4f} | {E['balanced_accuracy']:.4f} | {E['agreement_fisa']:.4f} | "
                 f"{E['dev_rule']:.4f} | {E['epochs']} ({E['stop']}) |")
    L += ["", "Ghi chú: dữ liệu chỉ có mức ngôn ngữ rời rạc nên suy diễn là trường hợp Crisp. Tập huấn luyện đã được "
          "cân bằng lớp bằng lặp mẫu; các dòng trùng được gom nhóm để mọi phép chia nội bộ giữ chúng trong cùng một phần, "
          "và FKG-E loại luật của chính mẫu (cùng nhóm) khi tính mất mát."]
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(C.PATHS.DATA_DIR, "BRSET_Data"))
    ap.add_argument("--folds", default="")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--T_ep", type=int, default=None)
    a = ap.parse_args()
    cfg = {"F_c": 5, "val_fraction": 0.2, "seed": C.FKGE.seed, "n_seq_timing": 50,
           "T_ep": a.T_ep or (10 if a.quick else C.FKGE.T_ep)}
    folds = a.folds.split(",") if a.folds else list_folds(a.root)
    if not folds:
        raise SystemExit(f"Không tìm thấy fold_* trong {a.root}")
    os.makedirs(C.PATHS.OUTPUT_DIR, exist_ok=True)
    results = [run_fold(a.root, f, cfg) for f in folds]
    summ = summarize(results)
    tag = "_quick" if a.quick else ""
    json.dump({"config": cfg, "folds": results, "summary": summ},
              open(os.path.join(C.PATHS.OUTPUT_DIR, f"brset_kfold{tag}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=float)
    write_report(results, summ, os.path.join(C.PATHS.OUTPUT_DIR, f"brset_kfold{tag}.md"), cfg)
    print("\nTổng hợp 5 fold (trung bình ± SD):")
    for k in ["auc", "balanced_accuracy", "sensitivity", "specificity", "f1_pos"]:
        print(f"  {k:18s} FISA {summ['FISA'][k]['mean']:.4f} ± {summ['FISA'][k]['sd']:.4f}   "
              f"FKG-E {summ['FKGE'][k]['mean']:.4f} ± {summ['FKGE'][k]['sd']:.4f}")
    print(f"Đã lưu {C.PATHS.OUTPUT_DIR}/brset_kfold{tag}.json và .md")


if __name__ == "__main__":
    main()
