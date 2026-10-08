"""
experiments/common.py — Nguồn dữ liệu và vòng huấn luyện/đánh giá dùng chung cho KB1–KB6, Ablation, Baseline.

Thứ tự ưu tiên dữ liệu (config.PATHS):
  1. BRSET thật — BRSET_FOLDS_DIR/fold_XX/{TrainDataRule,TestDataRule}.csv (5 fold theo ID bệnh nhân). Mỗi fold được
     chuẩn bị đúng như experiments/brset_kfold.py (prepare_fold): giáo viên FISA tính chéo theo nhóm, ngưỡng τ_F trên
     điểm ngoài phần, tập xác thực FKG-E 20% nhóm, FKG-E che luật cùng nhóm với mẫu — chống rò rỉ nhãn vì mỗi luật FRB
     sinh từ chính một mẫu huấn luyện.
  2. JSON — BRSET_RULES_FILE + BRSET_TEST_FILE: một lần chia 60/40 các mẫu như bản trước.
  3. Dữ liệu tổng hợp — chỉ để kiểm tra pipeline (in cảnh báo).

Tổng hợp mỗi cấu hình: trung bình trên các seed trong một fold, rồi trung bình ± độ lệch chuẩn giữa các fold (nếu chỉ
có một lần chia: độ lệch chuẩn giữa các seed). Huấn luyện FKG-E chạy song song (C.EVAL.N_JOBS tiến trình, fork); đánh
giá — kể cả đo thời gian suy diễn — chạy tuần tự trong tiến trình chính để mọi phép đo thời gian cùng điều kiện.
FKG-E tất định theo (fold, seed, tham số), nên một cấu hình đã huấn luyện được dùng lại giữa các kịch bản (ví dụ cấu
hình mặc định xuất hiện ở KB1, KB2, KB3 với d = 32, KB4 với λ_I = λ_P = 1, Ablation "Full", Baseline).
"""
import os
import math
import random
import time
import warnings
import multiprocessing as mp

import numpy as np

import config as C
from data.fkg_io import FKGRuleBase, load_test_samples, generate_synthetic_fkg, generate_synthetic_test_samples
from data.brset_folds import list_folds, load_fold
from models.fisa import FISA, FISASequential
from models.fkge import FKGE
from experiments.brset_kfold import prepare_fold, binary_metrics

FKGE_KEYS = ["accuracy", "balanced_accuracy", "auc", "f1_macro", "log_loss", "sensitivity", "specificity", "f1_pos",
             "agreement_fisa", "dev_rule", "avg_time_per_query_ms", "train_time_s"]
FISA_KEYS = ["accuracy", "balanced_accuracy", "auc", "f1_macro", "sensitivity", "specificity", "f1_pos",
             "accuracy_argmax", "accuracy_ratio9", "avg_time_per_query_ms", "seq_avg_time_per_query_ms",
             "fit_time_s", "total_time_s"]


# ============================================================================ nguồn dữ liệu
class Fold:
    """Một lần chia huấn luyện/kiểm tra với FISA (giáo viên, đối chứng) đã dựng sẵn.
    kind: "real" (BRSET 5 fold) | "json" | "synthetic"; fit_kwargs: tham số của FKGE.fit ngoài fisa_model."""

    def __init__(self, key, name, kind, fkg, train, test, fisa, fit_kwargs, info=None):
        self.key, self.name, self.kind = key, name, kind
        self.fkg, self.train, self.test, self.fisa = fkg, train, test, fisa
        self.fit_kwargs = fit_kwargs
        self.info = info or {}
        self._fisa_metrics = None

    @property
    def n_rules(self):
        return len(self.fkg.rules)


_FOLDS = {}    # khoá -> Fold đã chuẩn bị; tiến trình con (fork) kế thừa
_STATES = {}   # (khoá fold, seed, T_ep, tham số) -> trạng thái FKG-E đã huấn luyện


def using_real_data():
    d = C.PATHS.BRSET_FOLDS_DIR
    return os.path.isdir(d) and bool(list_folds(d))


def real_fold_names():
    return list(C.EVAL.REAL_FOLDS or list_folds(C.PATHS.BRSET_FOLDS_DIR))


def describe_source():
    if using_real_data():
        return (f"BRSET THẬT — {len(real_fold_names())} fold ({', '.join(real_fold_names())}) tại "
                f"{C.PATHS.BRSET_FOLDS_DIR}; {C.EVAL.REAL_SEEDS_PER_FOLD} seed FKG-E mỗi fold")
    if os.path.exists(C.PATHS.BRSET_RULES_FILE) and os.path.exists(C.PATHS.BRSET_TEST_FILE):
        return f"BRSET JSON — {C.PATHS.BRSET_RULES_FILE} (một lần chia 60/40)"
    return "DỮ LIỆU TỔNG HỢP (chưa có dữ liệu thật — chỉ để kiểm tra pipeline, không dùng để báo cáo)"


def seeds_for(folds, n_seeds=None):
    """Số seed mỗi fold: BRSET thật dùng C.EVAL.REAL_SEEDS_PER_FOLD; một lần chia dùng n_seeds (mặc định N_SEEDS)."""
    if folds and folds[0].kind == "real":
        return int(C.EVAL.REAL_SEEDS_PER_FOLD)
    return int(n_seeds or C.EVAL.N_SEEDS)


def _real_fold(fold, rule_ratio=None, rule_seed=0):
    sub = rule_ratio is not None and rule_ratio < 1
    key = ("real", fold, rule_ratio if sub else None, rule_seed if sub else None)
    if key in _FOLDS:
        return _FOLDS[key]
    keep, rules = None, "full"
    if sub:   # mô phỏng FKGS: giữ ngẫu nhiên tỉ lệ rule_ratio số luật của fold (mẫu huấn luyện giữ nguyên)
        fkg_full = load_fold(C.PATHS.BRSET_FOLDS_DIR, fold)[0]
        ids = [r["id"] for r in fkg_full.rules]
        keep = random.Random(rule_seed).sample(ids, max(1, int(math.ceil(len(ids) * rule_ratio))))
        rules = f"random_{rule_ratio:g}"
    cfg = {"F_c": C.EVAL.F_C, "val_fraction": C.FKGE.val_fraction, "seed": C.FKGE.seed}
    t0 = time.time()
    P = prepare_fold(C.PATHS.BRSET_FOLDS_DIR, fold, cfg, keep_rule_ids=keep)
    info = dict(P["info"], rules=rules, prepare_s=time.time() - t0)
    f = Fold(key, fold, "real", P["fkg"], P["train"], P["test"], P["fisa"],
             {"train_samples": P["train_fit"], "val_samples": P["val_fit"], "teacher_data": P["teacher_data"]}, info)
    print(f"  [chuẩn bị {fold}{'' if not sub else f', {rule_ratio:.0%} luật'}] |R|={f.n_rules}, "
          f"n_train={info['n_train']}, n_test={info['n_test']} (dương {info['test_pos']}), τ_F={P['tau_F']:.3f} "
          f"({info['prepare_s']:.1f}s)")
    _FOLDS[key] = f
    return f


def _single_split_fold(name, rules_path, test_path, seed_offset, rule_ratio=None, rule_seed=0, fkgs_file=None,
                       allow_synthetic=True):
    """Một lần chia 60/40 (JSON thật hoặc dữ liệu tổng hợp). Trả về None nếu không có JSON và không cho tổng hợp."""
    sub = rule_ratio is not None and rule_ratio < 1
    use_fkgs = bool(fkgs_file) and os.path.exists(fkgs_file)
    key = ("single", name, rules_path, fkgs_file if use_fkgs else None,
           rule_ratio if sub and not use_fkgs else None, rule_seed if sub and not use_fkgs else None)
    if key in _FOLDS:
        return _FOLDS[key]
    if os.path.exists(rules_path) and os.path.exists(test_path):
        fkg, samples, kind = FKGRuleBase.load(rules_path), load_test_samples(test_path), "json"
        print(f"  [{name}] Đã nạp dữ liệu THẬT (JSON): {len(fkg)} luật, {len(samples)} mẫu")
    elif allow_synthetic:
        print(f"  [{name}] !! CHƯA TÌM THẤY dữ liệu thật tại '{rules_path}'. Dùng DỮ LIỆU TỔNG HỢP để kiểm thử pipeline.")
        fkg = generate_synthetic_fkg(n_rules=250, seed=100 + seed_offset, dataset_name=name)
        samples = generate_synthetic_test_samples(fkg, n_samples=200, seed=200 + seed_offset)
        kind = "synthetic"
    else:
        return None
    rules = "full"
    if use_fkgs:
        fkg = FKGRuleBase.load(fkgs_file); rules = "fkgs_file"
        print(f"  [{name}] Đã nạp FKGS THẬT: {len(fkg)} luật ({fkgs_file})")
    elif sub:
        fkg = fkg.sample_subset(rule_ratio, seed=rule_seed); rules = f"random_{rule_ratio:g}"
    n_train = int(0.6 * len(samples))
    train, test = samples[:n_train], samples[n_train:]
    fisa = FISA(fkg).fit(val_samples=train)    # ngưỡng chọn trên tập huấn luyện của FKG-E, không dùng tập kiểm tra
    f = Fold(key, name, kind, fkg, train, test, fisa, {"train_samples": train},
             {"n_train": len(train), "n_test": len(test), "n_rules": len(fkg), "rules": rules})
    _FOLDS[key] = f
    return f


def brset_folds(rule_ratio=None, rule_seed=0, fkgs_file=None):
    """Danh sách Fold của BRSET theo thứ tự ưu tiên: 5 fold thật > JSON > tổng hợp.
    rule_ratio < 1: chỉ giữ ngẫu nhiên tỉ lệ đó số luật (mô phỏng FKGS); fkgs_file: file luật FKGS thật (chỉ JSON)."""
    if using_real_data():
        return [_real_fold(f, rule_ratio, rule_seed) for f in real_fold_names()]
    return [_single_split_fold("BRSET", C.PATHS.BRSET_RULES_FILE, C.PATHS.BRSET_TEST_FILE, 0,
                               rule_ratio, rule_seed, fkgs_file)]


def other_dataset_folds(name, rules_path, test_path, seed_offset):
    """Bộ dữ liệu phụ của KB1 (Diabetes): JSON nếu có; khi đang chạy BRSET thật thì KHÔNG thay bằng dữ liệu tổng hợp."""
    f = _single_split_fold(name, rules_path, test_path, seed_offset, allow_synthetic=not using_real_data())
    return [f] if f is not None else []


# ============================================================================ huấn luyện và đánh giá
def _canon(overrides):
    """Bỏ các tham số trùng mặc định của config.FKGE để cùng một cấu hình được nhận ra ở mọi kịch bản."""
    out = {}
    for k, v in sorted((overrides or {}).items()):
        if getattr(C.FKGE, k, object()) == v:
            continue
        out[k] = v
    return tuple(out.items())


def _train_model(key, seed, T_ep, ov):
    fold = _FOLDS[key]
    m = FKGE(fold.fkg, T_ep=T_ep, seed=seed, **dict(ov))
    m.fit(fisa_model=fold.fisa, **fold.fit_kwargs)
    return m


def _state(m):
    gn = m.history.get("grad_norm") or [float("nan")]
    return {"params": [np.asarray(p) for p in m.params], "threshold": m.threshold, "epochs_run": m.epochs_run,
            "stop_reason": m.stop_reason, "train_time_s": m.train_time_s, "grad_norm": gn[-1],
            "teacher_T": getattr(m, "teacher_T", None)}


def _rebuild(key, seed, T_ep, ov, st):
    fold = _FOLDS[key]
    m = FKGE(fold.fkg, T_ep=T_ep, seed=seed, **dict(ov))
    m.params = [p.copy() for p in st["params"]]
    m.threshold, m.epochs_run, m.stop_reason = st["threshold"], st["epochs_run"], st["stop_reason"]
    m.train_time_s, m.teacher_T, m.teacher = st["train_time_s"], st["teacher_T"], fold.fisa
    m.history["grad_norm"] = [st["grad_norm"]]
    m._finalize()
    return m


def _init_worker():
    try:   # mỗi tiến trình một luồng BLAS, tránh tranh chấp lõi
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass


def _worker(item):
    return item, _state(_train_model(*item))


def fkge_metrics(fold, m):
    r = m.evaluate(fold.test, fisa_model=fold.fisa)
    out = {k: r[k] for k in ("accuracy", "balanced_accuracy", "f1_macro", "log_loss", "avg_time_per_query_ms",
                             "agreement_fisa", "dev_rule") if k in r}
    out["auc"] = r.get("auc", float("nan"))
    if m.nL == 2:
        _, pE = m._infer(m._query_matrix(fold.test))
        b = binary_metrics(r["y_true"], r["y_pred"], m._logratio(pE).tolist(), m.class_tokens[1])
        out.update(sensitivity=b["sensitivity"], specificity=b["specificity"], f1_pos=b["f1_pos"])
    out.update(train_time_s=m.train_time_s, epochs=m.epochs_run, stop=m.stop_reason, threshold=m.threshold)
    return out


def run_fkge(tasks, label=""):
    """tasks: list[(Fold, seed, overrides)] -> list[dict độ đo trên tập kiểm tra], cùng thứ tự.
    Cấu hình đã huấn luyện (trong cùng tiến trình) được dùng lại; phần còn lại huấn luyện song song."""
    T_ep = int(C.FKGE.T_ep)
    items = []
    for fold, seed, ov in tasks:
        _FOLDS[fold.key] = fold
        items.append((fold.key, int(seed), T_ep, _canon(ov)))
    uniq = list(dict.fromkeys(items))
    todo = [it for it in uniq if it not in _STATES]
    metrics, t0 = {}, time.time()
    jobs = max(1, min(int(C.EVAL.N_JOBS), len(todo)))
    if todo:
        print(f"  >> {label}: huấn luyện {len(todo)} mô hình FKG-E (T_ep={T_ep}, {jobs} tiến trình"
              f"{'' if len(todo) == len(uniq) else f', dùng lại {len(uniq) - len(todo)} mô hình đã có'})")

    def progress(n, it, st):
        ov = dict(it[3])
        print(f"     [{n}/{len(todo)}] {_FOLDS[it[0]].name} seed={it[1]}{' ' + str(ov) if ov else ''}: "
              f"{st['epochs_run']} epoch ({st['stop_reason']}), {st['train_time_s']:.0f}s "
              f"| đã chạy {(time.time() - t0) / 60:.1f} phút", flush=True)

    if jobs > 1:
        with mp.get_context("fork").Pool(jobs, initializer=_init_worker) as pool:
            for n, (it, st) in enumerate(pool.imap_unordered(_worker, todo), 1):
                _STATES[it] = st
                progress(n, it, st)
    else:
        for n, it in enumerate(todo, 1):
            m = _train_model(*it)
            _STATES[it] = st = _state(m)
            metrics[it] = fkge_metrics(_FOLDS[it[0]], m)
            progress(n, it, st)
    for it in uniq:
        if it not in metrics:
            metrics[it] = fkge_metrics(_FOLDS[it[0]], _rebuild(*it, _STATES[it]))
    return [metrics[it] for it in items]


def fisa_metrics(fold):
    """Độ đo FISA (bảng tra) trên tập kiểm tra + thời gian FISA tuần tự trên C.EVAL.N_SEQ_TIMING truy vấn đầu."""
    if fold._fisa_metrics is None:
        f = fold.fisa
        r = f.evaluate(fold.test)
        seq = FISASequential(fold.fkg, decision=f.decision).fit(); seq.threshold = f.threshold
        rs = seq.evaluate(fold.test[:max(1, int(C.EVAL.N_SEQ_TIMING))])
        out = {k: r[k] for k in ("accuracy", "balanced_accuracy", "f1_macro", "accuracy_argmax",
                                 "balanced_accuracy_argmax", "accuracy_ratio9", "threshold", "node_size", "n_unseen",
                                 "avg_time_per_query_ms", "total_time_s", "fit_time_s")}
        out["auc"] = r.get("auc", float("nan"))
        out["seq_avg_time_per_query_ms"] = rs["avg_time_per_query_ms"]
        if len(f.class_tokens) == 2:
            sc = []
            for s in fold.test:
                D, _, mt = f.decision_values(s["membership"]); sc.append(f._score(D) if mt else 0.0)
            b = binary_metrics(r["y_true"], r["y_pred"], sc, f.class_tokens[1])
            out.update(sensitivity=b["sensitivity"], specificity=b["specificity"], f1_pos=b["f1_pos"])
        fold._fisa_metrics = out
    return fold._fisa_metrics


# ============================================================================ tổng hợp
def aggregate(runs, keys):
    """runs: list[(khoá fold, dict độ đo)] -> {k_mean, k_std}. Nhiều fold: trung bình theo seed trong fold rồi
    trung bình ± độ lệch chuẩn (ddof = 1) giữa các fold; một fold: trung bình ± độ lệch chuẩn giữa các seed."""
    by_fold = {}
    for fk, m in runs:
        by_fold.setdefault(fk, []).append(m)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in keys:
            if len(by_fold) > 1:
                x = np.array([np.nanmean([float(m.get(k, np.nan)) for m in ms]) for ms in by_fold.values()])
                sd = float(np.nanstd(x, ddof=1))
            else:
                x = np.array([float(m.get(k, np.nan)) for ms in by_fold.values() for m in ms])
                sd = float(np.nanstd(x))
            out[k + "_mean"], out[k + "_std"] = float(np.nanmean(x)), sd
    return out


def sweep(folds, configs, n_seeds=None, label=""):
    """configs: list[(tên, overrides)] -> list[(tên, overrides, độ đo tổng hợp)] — mọi (cấu hình, fold, seed) gửi
    một lượt để huấn luyện song song tối đa."""
    n = seeds_for(folds, n_seeds)
    tasks = [(f, C.FKGE.seed + s, ov) for _, ov in configs for f in folds for s in range(n)]
    runs = run_fkge(tasks, label)
    per, out = len(folds) * n, []
    for i, (name, ov) in enumerate(configs):
        chunk = list(zip(tasks[i * per:(i + 1) * per], runs[i * per:(i + 1) * per]))
        out.append((name, ov, aggregate([(t[0].key, m) for t, m in chunk], FKGE_KEYS)))
    return out


def fisa_summary(folds):
    return aggregate([(f.key, fisa_metrics(f)) for f in folds], FISA_KEYS)


def check_collapse(name, m, margin=0.02):
    """Cảnh báo khi mô hình sụp về một lớp: độ chính xác cân bằng không vượt rõ 0,5 (với tập kiểm tra mất cân bằng
    như BRSET, accuracy thấp hơn tỉ lệ lớp đa số là bình thường khi ngưỡng chọn theo độ chính xác cân bằng)."""
    b = m.get("balanced_accuracy", float("nan"))
    if not b > 0.5 + margin:
        print(f"  !! CẢNH BÁO: {name} có độ chính xác cân bằng={b:.4f} ≈ 0,5 — mô hình gần như luôn đoán một lớp. "
              f"Kiểm tra ngưỡng quyết định và trọng số λ_P/λ_I.")
        return False
    return True
