"""
experiments/common.py — Tiện ích dùng chung (v3): nạp dữ liệu có kiểm soát nguồn
gốc (QA-07), run manifest (QA-03), đánh giá FKGS bằng FISA, ghi JSON kèm run_id.
"""
import sys, os, json, time, hashlib, platform, datetime, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

import config as C
from src.fisa_corrected import FISACorrected
from src.utils_stats import safe_auc


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_fingerprint():
    files = sorted(glob.glob(os.path.join(C.ROOT, "src", "*.py")) +
                   glob.glob(os.path.join(C.ROOT, "experiments", "*.py")) +
                   [os.path.join(C.ROOT, "config.py"), os.path.join(C.ROOT, "run_all.py")])
    per = {os.path.relpath(f, C.ROOT): sha256_file(f) for f in files if os.path.exists(f)}
    agg = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(per.items())).encode()).hexdigest()
    return agg, per


def environment():
    import scipy, sklearn
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                scipy=scipy.__version__, sklearn=sklearn.__version__,
                platform=platform.platform(), processor=platform.processor() or "unknown",
                cpu_count=os.cpu_count())


def load_raw_data():
    """Đọc dữ liệu THẬT theo cấu hình; báo lỗi rõ nếu thiếu (không tự sinh dữ
    liệu giả trừ khi RUN.ALLOW_SYNTHETIC=True). Trả (df, provenance)."""
    path = C.PATHS.DATA_CSV
    if os.path.exists(path):
        df = pd.read_csv(path)
        prov = dict(source="real", path=os.path.relpath(path, C.ROOT), sha256=sha256_file(path))
    elif C.RUN.ALLOW_SYNTHETIC:
        from experiments.synthetic import generate_synthetic
        df = generate_synthetic()
        prov = dict(source="synthetic", path=None, sha256=None)
    else:
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu thật tại {path}. Đặt file đúng vị trí hoặc chạy "
            f"với --allow-synthetic (kết quả sẽ bị đánh dấu synthetic).")
    missing = [c for c in C.FEATURE_COLUMNS + [C.LABEL_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Dữ liệu thiếu cột: {missing}")
    return df, prov


def preprocess(df):
    if len(df) == 0:
        raise ValueError("Dữ liệu rỗng")
    df = df[C.FEATURE_COLUMNS + [C.LABEL_COLUMN]].copy()
    n0 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    X = df[C.FEATURE_COLUMNS].astype(str).agg("|".join, axis=1)
    conflicts = int((df.groupby(X)[C.LABEL_COLUMN].nunique() > 1).sum())
    info = dict(n_raw=n0, n_clean=len(df), n_duplicates_removed=n0 - len(df),
                n_conflicting_X=conflicts,
                label_counts={str(k): int(v) for k, v in df[C.LABEL_COLUMN].value_counts().sort_index().items()},
                zero_or_nan_counts={c: int(((df[c] == 0) | df[c].isna()).sum()) for c in C.ZERO_AS_MISSING_COLUMNS})
    return df, info


def results_path(name):
    os.makedirs(C.PATHS.RESULTS_DIR, exist_ok=True)
    return os.path.join(C.PATHS.RESULTS_DIR, name)


def save_json(name, obj):
    obj = dict(obj)
    obj["run_id"] = C.RUN.RUN_ID
    with open(results_path(name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=_json_default)


def load_json(name):
    p = results_path(name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def load_clean():
    if not os.path.exists(C.PATHS.DATA_CLEAN):
        raise FileNotFoundError("Chưa có Data_clean.csv của lần chạy này -- chạy bước PREPROCESS trước")
    df = pd.read_csv(C.PATHS.DATA_CLEAN)
    return df


def evaluate_fisa(rules_sample, rules_test):
    """Xây FISA trên tập luật mẫu, suy diễn trên test. Trả các độ đo + thời gian."""
    t0 = time.perf_counter()
    model = FISACorrected(rules_sample).fit()
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    ev = model.evaluate(rules_test)
    t_pred = time.perf_counter() - t0
    auc, flag = safe_auc(ev["y"], ev["score"])
    y, p = ev["y"], ev["pred"]
    return dict(auc=auc, auc_flag=flag, acc=float(accuracy_score(y, p)),
                bacc=float(balanced_accuracy_score(y, p)),
                f1=float(f1_score(y, p, zero_division=0)), pred_pos=float(np.mean(p == 1)),
                n_unique_scores=int(len(np.unique(np.round(ev["score"], 12)))),
                n_unseen=int(ev["n_unseen"]), t_fit=t_fit, t_pred=t_pred)
