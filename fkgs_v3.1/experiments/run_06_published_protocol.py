"""
experiments/run_06_published_protocol.py (bước PUB) — Đối chiếu với giao thức của bài APIN [CT2]:
chia cố định 70% huấn luyện / 30% kiểm tra, FKG đầy đủ, FISA gốc (tổ hợp ba thuộc tính).

Tách hai nguyên nhân làm FISA "sụp về lớp đa số" trong mã v3:
  (1) Mờ hoá: điểm cắt q10/q90 dồn khoảng 80% giá trị vào mức Medium, làm mất
      khả năng phân biệt; mờ hoá theo tam phân vị (q1/3, q2/3) giữ được thông tin.
  (2) Quy tắc quyết định: argmax D_l, hoặc ngưỡng cố định D0 > 9*D1, không thích
      nghi với tỉ lệ lớp; ngưỡng trên tỉ số log(D1/D0) được chọn trên tập xác thực
      tách từ tập huấn luyện (không dùng tập kiểm tra).
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score, confusion_matrix
import config as C
from analysis.diag_fisa import fisa_scores

def build(df_tr):
    from src.fuzzification import FuzzyFitter, rules_from_fuzzy_df
    f = FuzzyFitter().fit(df_tr)
    return f, rules_from_fuzzy_df(f.transform(df_tr))

def orig_scores(R, Q):
    from src.fkg_fast import EncodedRuleBase, caculateA_fast, caculateM_fast, caculateB_fast, caculateC_fast
    enc = EncodedRuleBase(R); A = caculateA_fast(enc); M = caculateM_fast(enc)
    Cm = caculateC_fast(enc, caculateB_fast(enc, A, M))
    D = np.array([fisa_scores(enc, Cm, q[:-1]) for q in Q])
    return D

def pair_scores(R, Q):
    from src.fisa_corrected import FISACorrected
    m = FISACorrected(R, pair_scope="global").fit()
    D = []
    for q in Q:
        d = m.decision_values(q[:-1])
        D.append((d.get(0, 0.0), d.get(1, 0.0)) if d else (0.0, 0.0))
    return np.array(D)

def logratio(D): return np.log(D[:, 1] + 1e-12) - np.log(D[:, 0] + 1e-12)

def best_threshold(s, y, crit):
    cand = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 197)))
    best, bt = -1, 0.0
    for t in cand:
        p = (s > t).astype(int)
        v = accuracy_score(y, p) if crit == "acc" else balanced_accuracy_score(y, p)
        if v > best: best, bt = v, t
    return bt

def evaluate(y, pred, s):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(acc=accuracy_score(y, pred), bacc=balanced_accuracy_score(y, pred),
                f1=f1_score(y, pred, zero_division=0), auc=roc_auc_score(y, s),
                sens=tp / (tp + fn), spec=tn / (tn + fp), pos_rate=float(np.mean(pred)))

def run(dataset, dedup, quant, variant, seed):
    C.set_dataset(dataset); C.FUZZY_QUANTILES = quant
    df = pd.read_csv(C.PATHS.DATA_CSV)[C.FEATURE_COLUMNS + [C.LABEL_COLUMN]]
    if dedup: df = df.drop_duplicates().reset_index(drop=True)
    tr, te = train_test_split(df, test_size=0.3, stratify=df[C.LABEL_COLUMN], random_state=seed)
    tr = tr.reset_index(drop=True); te = te.reset_index(drop=True)
    score_fn = orig_scores if variant == "orig" else pair_scores
    # ngưỡng chọn trên tập xác thực tách 20% từ tập huấn luyện
    itr, iva = train_test_split(tr, test_size=0.2, stratify=tr[C.LABEL_COLUMN], random_state=seed + 1)
    f_in, R_in = build(itr.reset_index(drop=True))
    from src.fuzzification import rules_from_fuzzy_df
    V = rules_from_fuzzy_df(f_in.transform(iva.reset_index(drop=True)))
    sv = logratio(score_fn(R_in, V)); yv = np.array([r[-1] for r in V])
    t_acc = best_threshold(sv, yv, "acc"); t_bacc = best_threshold(sv, yv, "bacc")
    t0 = time.time()
    f, R = build(tr)
    Q = rules_from_fuzzy_df(f.transform(te))
    D = score_fn(R, Q); s = logratio(D); y = np.array([r[-1] for r in Q])
    t = time.time() - t0
    res = {"n_train": len(R), "n_test": len(Q), "time_s": t,
           "argmax": evaluate(y, (D[:, 1] >= D[:, 0]).astype(int), s),
           "ratio9": evaluate(y, (~(D[:, 0] > 9 * D[:, 1])).astype(int), s),
           "calib_acc": evaluate(y, (s > t_acc).astype(int), s),
           "calib_bacc": evaluate(y, (s > t_bacc).astype(int), s),
           "t_acc": float(t_acc), "t_bacc": float(t_bacc)}
    return res

def run_pub():
    """Chạy giao thức công bố (70/30, không loại trùng) cho các cấu hình và lưu PUB_results.json."""
    from experiments.common import save_json
    ds = C.DATASET
    saved_q = list(C.FUZZY_QUANTILES)
    saved_paths = (C.PATHS.DATA_CSV, C.PATHS.RESULTS_DIR, C.PATHS.DATA_CLEAN)
    configs = [("orig", "q1090", [42]), ("orig", "tertile", C.PUB.SEEDS),
               ("pair", "q1090", C.PUB.SEEDS), ("pair", "tertile", C.PUB.SEEDS)]
    out = []
    for variant, qn, seeds in configs:
        q = [1/3, 2/3] if qn == "tertile" else [0.10, 0.90]
        for sd in seeds:
            r = run(ds, False, q, variant, sd)
            C.PATHS.DATA_CSV, C.PATHS.RESULTS_DIR, C.PATHS.DATA_CLEAN = saved_paths
            r.update(variant=variant, quant=qn, seed=sd, dedup=False)
            out.append(r)
            print(f"  PUB {variant} {qn} seed={sd}: argmax acc={r['argmax']['acc']:.4f}, "
                  f"calib acc={r['calib_acc']['acc']:.4f}, AUC={r['argmax']['auc']:.4f}", flush=True)
    C.FUZZY_QUANTILES = saved_q
    C.PATHS.DATA_CSV, C.PATHS.RESULTS_DIR, C.PATHS.DATA_CLEAN = saved_paths
    save_json("PUB_results.json", dict(results=out, test_size=0.3, published_acc=C.PUB.PUBLISHED_ACC.get(ds)))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="diabetes"); ap.add_argument("--dedup", type=int, default=0)
    ap.add_argument("--quant", default="tertile"); ap.add_argument("--variant", default="orig")
    ap.add_argument("--seeds", default="42"); ap.add_argument("--out", default="")
    a = ap.parse_args()
    q = [1/3, 2/3] if a.quant == "tertile" else [0.10, 0.90]
    allr = {}
    for sd in [int(x) for x in a.seeds.split(",")]:
        allr[sd] = run(a.dataset, bool(a.dedup), q, a.variant, sd)
        print(sd, json.dumps({k: ({m: round(v, 4) for m, v in val.items()} if isinstance(val, dict) else val) for k, val in allr[sd].items()}), flush=True)
    if a.out:
        json.dump(dict(args=vars(a), results=allr), open(a.out, "w"), indent=1, default=float)
