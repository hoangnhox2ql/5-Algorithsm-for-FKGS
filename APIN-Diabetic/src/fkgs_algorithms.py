"""
src/fkgs_algorithms.py — 4 thuật toán lấy mẫu duy nhất được giữ lại (theo
yêu cầu), dựa trên MỘT hàm lõi tham lam dùng chung `greedy_submodular_select`
(chính là Bổ đề 1 trong Dieu_chinh_5_thuat_toan_Submodular_v4.pdf):

  1. GreedyFKGS   : greedy trên toàn bộ R                    (Định lý 5)
  2. RandomSampling: chọn ngẫu nhiên đều -- KHÔNG có bảo chứng (baseline E2)
  3. SFKGS         : greedy ĐỘC LẬP trong từng tầng (Outcome) (Algorithm 2, Định lý 2)
  4. CFKGS         : greedy ĐỘC LẬP trong từng cụm (KMeans)    (Algorithm 5, Định lý 4)

3/4 thuật toán (Greedy, SFKGS, CFKGS) đều là tham lam-trên-một-hàm-dưới-mô-đun
(chỉ khác PHẠM VI: toàn cục / theo tầng / theo cụm) nên đều thừa hưởng bảo
chứng (1-1/e), theo đúng Định lý 5/2/4. Random KHÔNG thuộc nhóm này, dùng
làm baseline không có bảo chứng lý thuyết trong E2.
"""
import numpy as np
import random
from itertools import combinations as _combinations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.fkg_sim_rep import compute_sim_matrix, rep_value


def greedy_submodular_select(sim_matrix, omega_idx, target_size, coverage_threshold=None):
    """Hàm lõi DÙNG CHUNG cho GreedyFKGS/SFKGS/CFKGS -- chạy tham lam tất
    định TRÊN tập nền omega_idx (chỉ số toàn cục), duy trì bảng Cov cục bộ
    trong omega (đúng Bổ đề 1), trả về list chỉ số TOÀN CỤC đã chọn.

    `coverage_threshold`: nếu khác None, dừng sớm khi Cov trung bình trong
    omega đạt ngưỡng này (Bước 2(d) của Thuật toán 2.3 sửa đổi / Định nghĩa
    2.12 "điều kiện phủ").
    """
    omega_idx = np.asarray(omega_idx)
    n_omega = len(omega_idx)
    if n_omega == 0 or target_size <= 0:
        return []
    target_size = min(target_size, n_omega)

    sim_sub = sim_matrix[np.ix_(omega_idx, omega_idx)]  # Sim CỤC BỘ trong omega (Ω)
    cov = np.zeros(n_omega)
    remaining_mask = np.ones(n_omega, dtype=bool)
    S_local = []

    for _ in range(target_size):
        candidates_local = np.where(remaining_mask)[0]
        if len(candidates_local) == 0:
            break
        sim_cand = sim_sub[:, candidates_local]              # (n_omega, n_cand)
        new_cov = np.maximum(cov[:, None], sim_cand)
        gains = (new_cov - cov[:, None]).sum(axis=0) / n_omega
        best_pos = candidates_local[int(np.argmax(gains))]

        S_local.append(best_pos)
        cov = np.maximum(cov, sim_sub[:, best_pos])
        remaining_mask[best_pos] = False

        if coverage_threshold is not None and cov.mean() >= coverage_threshold:
            break

    return omega_idx[S_local].tolist()


# ============================================================
# 1) GreedyFKGS -- greedy toàn cục (Định lý 5)
# ============================================================
def GreedyFKGS(sim_matrix, theta=None, m=None, coverage_threshold=None):
    n = sim_matrix.shape[0]
    if m is None:
        m = int(np.ceil(theta * n))
    S = greedy_submodular_select(sim_matrix, np.arange(n), m, coverage_threshold)
    return S


# ============================================================
# 2) RandomSampling -- KHÔNG có bảo chứng, dùng làm baseline E2
# ============================================================
def RandomSampling(n, theta=None, m=None, seed=0):
    if m is None:
        m = int(np.ceil(theta * n))
    rng = random.Random(seed)
    return rng.sample(range(n), min(m, n))


# ============================================================
# 3) SFKGS -- Algorithm 2 (Định lý 2), phân tầng theo nhãn Outcome
# ============================================================
def _allocate_budget(strata_sizes, theta, n_total, method="proportional",
                      strata_values=None):
    """Phân bổ ngân sách m_s cho từng tầng theo alloc_method (dòng
    322-343 Algorithm 2 trong PDF)."""
    S = len(strata_sizes)
    if method == "proportional":
        m_list = [int(np.ceil(theta * ns)) for ns in strata_sizes]
    elif method == "equal":
        m_each = int(np.ceil(theta * n_total / S))
        m_list = [m_each] * S
    elif method == "neyman":
        if strata_values is None:
            raise ValueError("Neyman allocation cần strata_values (dữ liệu mỗi tầng) để ước lượng phương sai.")
        sigmas = [np.std(np.array(v, dtype=float)) + 1e-9 for v in strata_values]
        denom = sum(ns * sig for ns, sig in zip(strata_sizes, sigmas))
        m_list = [int(np.ceil(theta * n_total * ns * sig / denom))
                  for ns, sig in zip(strata_sizes, sigmas)]
    else:
        raise ValueError(f"Không hỗ trợ alloc_method='{method}'")
    m_list = [min(m, ns) for m, ns in zip(m_list, strata_sizes)]
    return m_list


def SFKGS(sim_matrix, labels, theta, alloc_method="proportional", strata_values=None):
    """Algorithm 2 (sửa đổi): phân tầng theo `labels` (Outcome), phân bổ
    ngân sách, greedy độc lập trong từng tầng, gộp kết quả."""
    n = sim_matrix.shape[0]
    labels = np.asarray(labels)
    unique_labels = sorted(set(labels.tolist()))
    strata_idx = [np.where(labels == l)[0] for l in unique_labels]
    strata_sizes = [len(idx) for idx in strata_idx]

    m_list = _allocate_budget(strata_sizes, theta, n, alloc_method, strata_values)

    S = []
    for idx_s, m_s in zip(strata_idx, m_list):
        S += greedy_submodular_select(sim_matrix, idx_s, m_s)
    return S


# ============================================================
# 4) CFKGS -- Algorithm 5 (Định lý 4), gom cụm KMeans
# ============================================================
def _kmeans_simple(X, n_clusters, seed=0, n_iter=50):
    """KMeans đơn giản tự viết (tránh phụ thuộc sklearn nếu không có sẵn) --
    dùng trên vector đặc trưng đã mã hoá số (X: n x d)."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    n_clusters = min(n_clusters, n)
    init_idx = rng.choice(n, n_clusters, replace=False)
    centroids = X[init_idx].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(n_iter):
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = dists.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                centroids[k] = X[mask].mean(axis=0)
    return labels


def _encode_rules_numeric(rules):
    """Mã hoá mỗi thuộc tính phạm trù thành số nguyên, dùng làm đầu vào
    KMeans (không dùng Sim -- theo đúng PDF, gom cụm dựa trên vector đặc
    trưng, KHÔNG dựa trên chính ma trận Sim đang được bảo vệ tính chất)."""
    n = len(rules)
    m = len(rules[0]) - 1
    X = np.zeros((n, m))
    for j in range(m):
        col_vals = [r[j] for r in rules]
        uniq = sorted(set(col_vals))
        lut = {v: k for k, v in enumerate(uniq)}
        for i in range(n):
            X[i, j] = lut[col_vals[i]]
    return X


def CFKGS(sim_matrix, rules, theta, n_clusters, design="two_stage", seed=0):
    """Algorithm 5 (sửa đổi): gom cụm bằng KMeans trên vector mã hoá số của
    luật, greedy độc lập trong từng cụm (one_stage: lấy hết mỗi cụm;
    two_stage: lấy theta*|cluster| mỗi cụm), gộp kết quả."""
    n = sim_matrix.shape[0]
    X = _encode_rules_numeric(rules)
    cluster_labels = _kmeans_simple(X, n_clusters, seed=seed)
    unique_clusters = sorted(set(cluster_labels.tolist()))

    S = []
    for c in unique_clusters:
        idx_c = np.where(cluster_labels == c)[0]
        if design == "one_stage":
            target_c = len(idx_c)
        else:
            target_c = int(np.ceil(theta * len(idx_c)))
        S += greedy_submodular_select(sim_matrix, idx_c, target_c)
    return S


if __name__ == "__main__":
    rng = random.Random(0)
    levels = ["Low", "Medium", "High"]
    rules = [[rng.choice(levels) for _ in range(6)] + [rng.randint(0, 1)] for _ in range(80)]
    sim = compute_sim_matrix(rules)
    labels = [r[-1] for r in rules]

    print("=== Kiểm thử 4 thuật toán lấy mẫu ===")
    theta = 0.3
    n = len(rules)
    m_target = int(np.ceil(theta * n))

    S_greedy = GreedyFKGS(sim, theta=theta)
    print(f"GreedyFKGS: |S|={len(S_greedy)} (target={m_target}), "
          f"Rep(S)={rep_value(sim, S_greedy):.4f}")
    assert len(S_greedy) == m_target

    S_random = RandomSampling(n, theta=theta, seed=1)
    print(f"Random    : |S|={len(S_random)}, Rep(S)={rep_value(sim, S_random):.4f}")

    S_sfkgs = SFKGS(sim, labels, theta=theta, alloc_method="proportional")
    print(f"SFKGS     : |S|={len(S_sfkgs)}, Rep(S)={rep_value(sim, S_sfkgs):.4f}")

    S_cfkgs = CFKGS(sim, rules, theta=theta, n_clusters=5, design="two_stage", seed=2)
    print(f"CFKGS     : |S|={len(S_cfkgs)}, Rep(S)={rep_value(sim, S_cfkgs):.4f}")

    # Kiểm tra quan trọng: Greedy/SFKGS/CFKGS (có bảo chứng) phải cho Rep
    # cao hơn RÕ RỆT so với Random trên đa số trường hợp -- đây là kỳ vọng
    # định tính cần đúng (nếu Random THẮNG áp đảo cả 3, có bug ở đâu đó).
    rep_greedy = rep_value(sim, S_greedy)
    rep_random = rep_value(sim, S_random)
    print(f"\nGreedyFKGS Rep={rep_greedy:.4f} vs Random Rep={rep_random:.4f} "
          f"-> {'GreedyFKGS thắng (kỳ vọng đúng)' if rep_greedy >= rep_random else 'CẢNH BÁO: Random thắng, kiểm tra lại!'}")
    assert rep_greedy >= rep_random - 1e-9, "GreedyFKGS phải Rep >= Random (Greedy tối ưu hoá trực tiếp Rep)!"
    print("\nKiểm thử fkgs_algorithms: OK")
