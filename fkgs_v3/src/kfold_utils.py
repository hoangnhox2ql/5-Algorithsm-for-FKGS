"""
src/kfold_utils.py — Chia fold và dựng FKG theo fold (v3).

SỬA SO VỚI v2
  * k được kiểm tra: 2 <= k <= số mẫu của lớp nhỏ nhất (v2 nhánh dự phòng
    chấp nhận k=1 hoặc k>n).
  * `stratified_subsample` phân bổ theo phần dư lớn nhất nên trả ĐÚNG n_sample
    hàng (v2 làm tròn riêng từng lớp, có thể trả thiếu — ví dụ trả 0 hàng).
  * `build_fkg_fold(..., compute_sim=False)` không dựng ma trận Sim trên toàn
    tập huấn luyện khi E1/E4 chỉ cần mẫu con (tránh nhiều GB bộ nhớ).
  * Trả về chỉ số hàng gốc của tập nền để ghi vào manifest (truy vết).
"""
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from src.fuzzification import FuzzyFitter, rules_from_fuzzy_df
from src.fkg_sim_rep import compute_sim_matrix
from src.fkgs_algorithms import allocate_largest_remainder

try:
    from sklearn.model_selection import StratifiedKFold
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def make_stratified_kfold(df, k=None, seed=None, label_column=None):
    k = C.KFOLD.K if k is None else k
    seed = C.KFOLD.SEED if seed is None else seed
    label_column = label_column or C.LABEL_COLUMN
    y = df[label_column].values
    if len(y) == 0:
        raise ValueError("Không thể chia fold trên dữ liệu rỗng")
    min_class = min(np.bincount(pd.factorize(y)[0]))
    if int(k) != k or k < 2 or k > min_class:
        raise ValueError(f"k={k} không hợp lệ: cần 2 <= k <= {min_class} (cỡ lớp nhỏ nhất)")
    if _HAS_SKLEARN:
        skf = StratifiedKFold(n_splits=int(k), shuffle=True, random_state=seed)
        return [(tr, te) for tr, te in skf.split(np.zeros(len(y)), y)]
    return _manual_stratified_kfold(y, int(k), seed)


def _manual_stratified_kfold(y, k, seed):
    rng = np.random.RandomState(seed)
    fold_of = np.zeros(len(y), dtype=int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        fold_of[idx] = np.arange(len(idx)) % k
    return [(np.where(fold_of != f)[0], np.where(fold_of == f)[0]) for f in range(k)]


def stratified_subsample(n_or_labels, n_sample, seed):
    """Trả về mảng chỉ số (không hoàn lại) có ĐÚNG min(n_sample, n) phần tử,
    giữ tỉ lệ lớp bằng phân bổ phần dư lớn nhất."""
    labels = np.asarray(n_or_labels)
    n = len(labels)
    if n_sample < 0:
        raise ValueError("n_sample âm")
    n_sample = min(int(n_sample), n)
    if n_sample == 0:
        return np.array([], dtype=int)
    classes = sorted(set(labels.tolist()))
    idx_by = [np.where(labels == c)[0] for c in classes]
    alloc = allocate_largest_remainder([len(i) for i in idx_by], n_sample)
    rng = np.random.RandomState(seed)
    chosen = []
    for idx, a in zip(idx_by, alloc):
        chosen.extend(rng.choice(idx, a, replace=False).tolist())
    chosen = np.array(chosen, dtype=int)
    rng.shuffle(chosen)
    return chosen


def build_fkg_fold(df_train, df_test, n_sample=None, seed=0, compute_sim=True):
    fitter = FuzzyFitter().fit(df_train)
    rules_train_full = rules_from_fuzzy_df(fitter.transform(df_train))
    rules_test = rules_from_fuzzy_df(fitter.transform(df_test)) if df_test is not None else []
    labels = [r[-1] for r in rules_train_full]
    if n_sample is not None and n_sample < len(rules_train_full):
        pool_pos = stratified_subsample(labels, n_sample, seed)
    else:
        pool_pos = np.arange(len(rules_train_full))
    rules_pool = [rules_train_full[i] for i in pool_pos]
    return dict(rules_train_full=rules_train_full, rules_train=rules_pool,
                pool_positions=pool_pos, rules_test=rules_test, fitter=fitter,
                sim=compute_sim_matrix(rules_pool) if compute_sim else None)
