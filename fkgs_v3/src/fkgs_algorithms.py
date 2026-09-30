"""
src/fkgs_algorithms.py — Thuật toán lấy mẫu FKGS (v3).

  GreedyFKGS : greedy trên toàn bộ R           -> Rep >= (1-1/e) OPT
  SFKGS      : greedy độc lập trong từng nhãn  -> Rep >= (1-1/e) OPT_pb(Pi_L)
  CFKGS      : greedy trong từng cụm, cụm nằm TRONG nhãn
                                               -> Rep >= (1-1/e) OPT_pb(Pi_C)
  Random, StratRandom : đối chứng không có bảo chứng.

SỬA SO VỚI v2
  * QA-04: ngân sách của SFKGS/CFKGS được phân bổ bằng phương pháp phần dư
    lớn nhất nên tổng ĐÚNG BẰNG m = ceil(theta*n) như GreedyFKGS/Random
    (BUDGET_MODE="exact"); chế độ "ceil" của v2 giữ lại chỉ để đối chứng.
  * QA-10: độ phức tạp một lần gọi greedy là O(k * n * |ứng viên|) = O(k n^2)
    với cài đặt vector hoá hiện tại (không phải O(k n)); bảo chứng của
    SFKGS/CFKGS là so với OPT phân bổ, KHÔNG phải OPT toàn cục. Đã bỏ khẳng
    định "Greedy luôn >= Random" (không phải hệ quả của định lý).
  * CFKGS gom cụm TRONG từng nhãn trên mã one-hot của nhãn ngôn ngữ
    (khoảng cách Euclid^2 = 2 x Hamming, nhất quán với nhân Sim), thay cho
    KMeans trên mã thứ tự chữ cái của v2 (QA mapping #17).
  * Kiểm tra đầu vào: ngân sách âm, theta ngoài [0,1], design sai -> ValueError.
  * Dừng sớm: coverage_mode="mean" (Rep trung bình >= delta, như PDF v4) hoặc
    "all" (mọi luật có cov >= delta, đúng định nghĩa phủ Chương 2) (QA-06).
"""
import math
import random
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


def budget_from_theta(theta, n):
    if not (0.0 <= theta <= 1.0):
        raise ValueError(f"theta phải trong [0,1], nhận {theta}")
    if n < 0:
        raise ValueError("n âm")
    return int(math.ceil(theta * n - 1e-12))


def greedy_submodular_select(sim_matrix, omega_idx, target_size,
                             coverage_threshold=None, coverage_mode="mean"):
    """Greedy trên tập nền omega_idx (chỉ số toàn cục). Trả về list chỉ số toàn
    cục theo thứ tự chọn. Phá hoà theo vị trí nhỏ nhất trong omega_idx."""
    if int(target_size) != target_size or target_size < 0:
        raise ValueError(f"Ngân sách phải là số nguyên không âm, nhận {target_size}")
    if coverage_mode not in ("mean", "all"):
        raise ValueError("coverage_mode phải là 'mean' hoặc 'all'")
    omega_idx = np.asarray(omega_idx, dtype=int)
    n_omega = len(omega_idx)
    if n_omega == 0 or target_size == 0:
        return []
    target_size = min(int(target_size), n_omega)
    sub = np.maximum(sim_matrix[np.ix_(omega_idx, omega_idx)], 0.0)
    cov = np.zeros(n_omega)
    remaining = np.ones(n_omega, dtype=bool)
    chosen = []
    for _ in range(target_size):
        cand = np.where(remaining)[0]
        gains = (np.maximum(cov[:, None], sub[:, cand]) - cov[:, None]).sum(axis=0) / n_omega
        best = cand[int(np.argmax(gains))]
        chosen.append(best)
        cov = np.maximum(cov, sub[:, best])
        remaining[best] = False
        if coverage_threshold is not None:
            stat = cov.mean() if coverage_mode == "mean" else cov.min()
            if stat >= coverage_threshold - 1e-12:
                break
    return omega_idx[chosen].tolist()


def GreedyFKGS(sim_matrix, theta=None, m=None, coverage_threshold=None, coverage_mode="mean"):
    n = sim_matrix.shape[0]
    if m is None:
        m = budget_from_theta(theta, n)
    return greedy_submodular_select(sim_matrix, np.arange(n), m, coverage_threshold, coverage_mode)


def RandomSampling(n, theta=None, m=None, seed=0):
    if m is None:
        m = budget_from_theta(theta, n)
    if m < 0:
        raise ValueError("Ngân sách âm")
    return random.Random(seed).sample(range(n), min(m, n))


def allocate_largest_remainder(sizes, m, weights=None):
    """Phân bổ m suất cho các phần có sức chứa `sizes` theo trọng số (mặc định
    tỉ lệ kích thước), tổng ĐÚNG BẰNG min(m, sum(sizes)), m_k <= sizes[k]."""
    sizes = [int(s) for s in sizes]
    if m < 0:
        raise ValueError("Ngân sách âm")
    total_cap = sum(sizes)
    m = min(int(m), total_cap)
    if m == 0 or total_cap == 0:
        return [0] * len(sizes)
    w = np.asarray(sizes if weights is None else weights, dtype=float)
    if w.sum() <= 0:
        w = np.asarray(sizes, dtype=float)
    q = m * w / w.sum()
    alloc = [min(int(math.floor(qi)), cap) for qi, cap in zip(q, sizes)]
    r = m - sum(alloc)
    while r > 0:
        best, best_key = None, None
        for k in range(len(sizes)):
            if alloc[k] < sizes[k]:
                key = (q[k] - alloc[k], -k)
                if best_key is None or key > best_key:
                    best, best_key = k, key
        alloc[best] += 1
        r -= 1
    return alloc


def _allocate(sizes, theta, n_total, method, budget_mode, strata_sim_div=None):
    if method not in ("proportional", "equal", "neyman"):
        raise ValueError(f"alloc_method không hỗ trợ: {method!r}")
    if budget_mode not in ("exact", "ceil"):
        raise ValueError(f"budget_mode không hỗ trợ: {budget_mode!r}")
    if budget_mode == "ceil":   # hành vi v2 (có thể vượt ngân sách toàn cục)
        if method == "proportional":
            m_list = [int(math.ceil(theta * s - 1e-12)) for s in sizes]
        elif method == "equal":
            m_list = [int(math.ceil(theta * n_total / len(sizes) - 1e-12))] * len(sizes)
        else:
            w = np.asarray(sizes) * np.asarray(strata_sim_div)
            m_list = [int(math.ceil(theta * n_total * wi / w.sum() - 1e-12)) for wi in w]
        return [min(a, s) for a, s in zip(m_list, sizes)]
    m = budget_from_theta(theta, n_total)
    if method == "proportional":
        weights = sizes
    elif method == "equal":
        weights = [1.0] * len(sizes)
    else:
        weights = np.asarray(sizes, float) * np.asarray(strata_sim_div, float)
    return allocate_largest_remainder(sizes, m, weights)


def _intra_diversity(sim_matrix, idx):
    """Độ phân tán nội tầng sigma = 1 - trung bình a(x,y) trong tầng."""
    sub = np.maximum(sim_matrix[np.ix_(idx, idx)], 0.0)
    return float(max(1.0 - sub.mean(), 1e-9))


def SFKGS(sim_matrix, labels, theta, alloc_method=None, budget_mode=None):
    alloc_method = alloc_method or C.SAMPLING.ALLOC_METHOD
    budget_mode = budget_mode or C.SAMPLING.BUDGET_MODE
    n = sim_matrix.shape[0]
    budget_from_theta(theta, n)
    labels = np.asarray(labels)
    strata = [np.where(labels == l)[0] for l in sorted(set(labels.tolist()))]
    sizes = [len(s) for s in strata]
    div = [_intra_diversity(sim_matrix, s) for s in strata] if alloc_method == "neyman" else None
    m_list = _allocate(sizes, theta, n, alloc_method, budget_mode, div)
    S = []
    for idx, mh in zip(strata, m_list):
        S += greedy_submodular_select(sim_matrix, idx, mh)
    return S


def StratRandom(labels, theta, seed=0):
    """Đối chứng: cùng phân bổ nhãn như SFKGS (exact) nhưng chọn ngẫu nhiên trong tầng."""
    labels = np.asarray(labels)
    n = len(labels)
    strata = [np.where(labels == l)[0] for l in sorted(set(labels.tolist()))]
    m_list = allocate_largest_remainder([len(s) for s in strata], budget_from_theta(theta, n))
    rng = random.Random(seed)
    S = []
    for idx, mh in zip(strata, m_list):
        S += rng.sample(list(idx.tolist()), mh)
    return S


# ------------------------------------------------------------------ CFKGS
def _one_hot(rules):
    p = len(rules[0]) - 1
    cols = []
    for j in range(p):
        vals = sorted(set(str(r[j]) for r in rules))
        lut = {v: k for k, v in enumerate(vals)}
        oh = np.zeros((len(rules), len(vals)))
        for i, r in enumerate(rules):
            oh[i, lut[str(r[j])]] = 1.0
        cols.append(oh)
    return np.hstack(cols)


def kmeans_simple(X, n_clusters, seed=0, n_iter=100):
    """KMeans (Lloyd) với khởi tạo k-means++ tất định theo seed. Trả về nhãn cụm
    0..k'-1 (k' <= n_clusters, các cụm rỗng bị loại)."""
    X = np.asarray(X, float)
    n = X.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int)
    k = max(1, min(int(n_clusters), n))
    rng = np.random.RandomState(seed)
    centers = [X[rng.randint(n)]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.asarray(centers)[None]) ** 2).sum(-1), axis=1)
        if d2.sum() <= 1e-12:
            break
        centers.append(X[rng.choice(n, p=d2 / d2.sum())])
    centers = np.asarray(centers)
    lab = np.full(n, -1)
    for _ in range(n_iter):
        new = ((X[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
        if np.array_equal(new, lab):
            break
        lab = new
        for c in range(len(centers)):
            if (lab == c).any():
                centers[c] = X[lab == c].mean(0)
    _, lab = np.unique(lab, return_inverse=True)
    return lab


def form_clusters(rules, n_clusters_per_label=None, seed=0, within_label=None):
    """Trả về list mảng chỉ số (mỗi mảng là một cụm)."""
    n_clusters_per_label = n_clusters_per_label or C.SAMPLING.N_CLUSTERS_PER_LABEL
    within_label = C.SAMPLING.CLUSTER_WITHIN_LABEL if within_label is None else within_label
    X = _one_hot(rules)
    labels = np.array([r[-1] for r in rules])
    groups = ([np.where(labels == l)[0] for l in sorted(set(labels.tolist()))]
              if within_label else [np.arange(len(rules))])
    k_each = n_clusters_per_label if within_label else n_clusters_per_label * len(set(labels.tolist()))
    clusters = []
    for g in groups:
        lab = kmeans_simple(X[g], k_each, seed=seed)
        clusters += [g[lab == c] for c in range(lab.max() + 1)]
    return clusters


def CFKGS(sim_matrix, rules, theta, n_clusters_per_label=None, design="two_stage",
          seed=0, budget_mode=None, within_label=None):
    if design not in ("one_stage", "two_stage"):
        raise ValueError(f"design không hỗ trợ: {design!r}")
    budget_mode = budget_mode or C.SAMPLING.BUDGET_MODE
    n = sim_matrix.shape[0]
    m = budget_from_theta(theta, n)
    if m == 0:
        return []
    clusters = form_clusters(rules, n_clusters_per_label, seed, within_label)
    if design == "one_stage":   # lấy trọn mọi cụm = toàn bộ R (không nén); giữ để đối chứng
        return [int(i) for c in clusters for i in c]
    sizes = [len(c) for c in clusters]
    if budget_mode == "ceil":
        m_list = [min(int(math.ceil(theta * s - 1e-12)), s) for s in sizes]
    else:
        labels = np.array([r[-1] for r in rules])
        if within_label if within_label is not None else C.SAMPLING.CLUSTER_WITHIN_LABEL:
            # phân bổ 2 cấp: nhãn (phần dư lớn nhất) rồi cụm trong nhãn
            cls = sorted(set(labels.tolist()))
            m_lab = allocate_largest_remainder([(labels == l).sum() for l in cls], m)
            m_list = [0] * len(clusters)
            for l, ml in zip(cls, m_lab):
                ids = [i for i, c in enumerate(clusters) if labels[c[0]] == l]
                sub = allocate_largest_remainder([sizes[i] for i in ids], ml)
                for i, v in zip(ids, sub):
                    m_list[i] = v
        else:
            m_list = allocate_largest_remainder(sizes, m)
    S = []
    for c, mc in zip(clusters, m_list):
        S += greedy_submodular_select(sim_matrix, c, mc)
    return S
