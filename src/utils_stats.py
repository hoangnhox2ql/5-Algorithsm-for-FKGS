"""
src/utils_stats.py — Kiểm định thống kê dùng cho E1-E4, đúng thiet_ke_thuc_nghiem.md:
Wilcoxon signed-rank (E2), Holm-Bonferroni (E2/E4), Cohen's d ghép cặp (E2),
Friedman + Nemenyi/CD (E4), xấp xỉ Kneedle (E3).
"""
import numpy as np

try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def wilcoxon_one_sided(x, y, alternative="greater"):
    """Wilcoxon signed-rank test MỘT PHÍA (H1: x > y theo mặc định). Trả về
    (statistic, p_value). Cần scipy; nếu không có, báo lỗi rõ ràng."""
    if not _HAS_SCIPY:
        raise ImportError("Cần scipy cho wilcoxon_one_sided -- pip install scipy")
    diffs = np.asarray(x) - np.asarray(y)
    if np.allclose(diffs, 0):
        return 0.0, 1.0
    stat, p = _scipy_stats.wilcoxon(x, y, alternative=alternative)
    return stat, p


def cohens_d_paired(x, y):
    """Cohen's d cho dữ liệu GHÉP CẶP: d = mean(diff) / std(diff)."""
    diffs = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    sd = diffs.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    return diffs.mean() / sd


def holm_bonferroni(p_values, alpha=0.05):
    """Hiệu chỉnh Holm-Bonferroni cho nhiều kiểm định. Trả về
    (p_adjusted [cùng thứ tự đầu vào], reject [bool cùng thứ tự])."""
    p_values = np.asarray(p_values)
    n = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted_sorted = np.zeros(n)
    for i in range(n):
        adjusted_sorted[i] = min((n - i) * sorted_p[i], 1.0)
    # Đảm bảo tính đơn điệu không giảm của Holm (chuẩn hoá bước "step-up")
    for i in range(1, n):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i - 1])

    p_adjusted = np.zeros(n)
    p_adjusted[order] = adjusted_sorted
    reject = p_adjusted < alpha
    return p_adjusted, reject


def friedman_test(data_matrix):
    """Friedman test trên ma trận (n_blocks x n_methods). Trả về (stat, p)."""
    if not _HAS_SCIPY:
        raise ImportError("Cần scipy cho friedman_test -- pip install scipy")
    data_matrix = np.asarray(data_matrix)
    cols = [data_matrix[:, j] for j in range(data_matrix.shape[1])]
    stat, p = _scipy_stats.friedmanchisquare(*cols)
    return stat, p


def nemenyi_critical_difference(n_blocks, n_methods, alpha=0.05):
    """Critical Difference (CD) cho Nemenyi post-hoc test:
    CD = q_alpha * sqrt(k(k+1) / (6N)), q_alpha tra bảng Studentized range
    (xấp xỉ cho alpha=0.05, k nhỏ -- bảng chuẩn Demsar 2006)."""
    q_alpha_table = {  # q_0.05 cho k = số phương pháp (Demšar 2006, Bảng 5(b))
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164,
    }
    q_alpha = q_alpha_table.get(n_methods, 3.164)
    cd = q_alpha * np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_blocks))
    return cd


def average_ranks(data_matrix, higher_is_better=True):
    """Hạng trung bình mỗi phương pháp (cột) qua các block (hàng), dùng cho
    Nemenyi/CD. Hạng 1 = tốt nhất."""
    data_matrix = np.asarray(data_matrix, dtype=float)
    if higher_is_better:
        data_matrix = -data_matrix
    n_blocks, n_methods = data_matrix.shape
    ranks = np.zeros_like(data_matrix)
    for i in range(n_blocks):
        ranks[i] = _rankdata_avg(data_matrix[i])
    return ranks.mean(axis=0)


def _rankdata_avg(arr):
    """Xếp hạng trung bình khi có giá trị trùng nhau (tie -> trung bình
    hạng), tự viết để tránh phụ thuộc scipy.stats.rankdata nếu không cần."""
    order = np.argsort(arr)
    ranks = np.empty(len(arr))
    sorted_arr = arr[order]
    i = 0
    while i < len(arr):
        j = i
        while j < len(arr) - 1 and sorted_arr[j + 1] == sorted_arr[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def kneedle_point(x, y):
    """Xấp xỉ Kneedle đơn giản: chuẩn hoá (x,y) về [0,1]^2, tìm điểm có
    khoảng cách vuông góc lớn nhất tới đường nối điểm đầu-cuối."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-12)

    x1, y1 = x_norm[0], y_norm[0]
    x2, y2 = x_norm[-1], y_norm[-1]
    # Khoảng cách từ mỗi điểm tới đường thẳng (x1,y1)-(x2,y2)
    num = np.abs((y2 - y1) * x_norm - (x2 - x1) * y_norm + x2 * y1 - y2 * x1)
    den = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2) + 1e-12
    dist = num / den
    knee_idx = int(np.argmax(dist))
    return knee_idx, x[knee_idx]


if __name__ == "__main__":
    rng = np.random.RandomState(0)

    # Wilcoxon + Cohen's d
    x = rng.normal(0.8, 0.05, 30)
    y = rng.normal(0.7, 0.05, 30)
    stat, p = wilcoxon_one_sided(x, y)
    d = cohens_d_paired(x, y)
    print(f"Wilcoxon (x>y): stat={stat:.2f}, p={p:.6f}")
    print(f"Cohen's d (ghép cặp): {d:.4f}")
    assert p < 0.05, "Wilcoxon phải phát hiện khác biệt rõ trên dữ liệu tổng hợp này!"

    # Holm-Bonferroni
    p_raw = [0.001, 0.02, 0.04, 0.20]
    p_adj, reject = holm_bonferroni(p_raw, alpha=0.05)
    print(f"\nHolm-Bonferroni: p_raw={p_raw}")
    print(f"  p_adjusted={np.round(p_adj, 4).tolist()}, reject={reject.tolist()}")

    # Friedman + Nemenyi
    data = rng.normal(0.7, 0.05, (10, 4))
    data[:, 0] += 0.15  # phương pháp 0 tốt hơn rõ rệt
    stat, p = friedman_test(data)
    ranks = average_ranks(data, higher_is_better=True)
    cd = nemenyi_critical_difference(n_blocks=10, n_methods=4)
    print(f"\nFriedman: stat={stat:.2f}, p={p:.6f}")
    print(f"Hạng trung bình 4 phương pháp: {np.round(ranks, 2).tolist()}")
    print(f"Critical Difference (Nemenyi, alpha=0.05): {cd:.3f}")
    assert ranks[0] < ranks[1], "Phương pháp 0 (tốt hơn) phải có hạng THẤP HƠN (rank 1 = tốt nhất)!"

    # Kneedle
    theta = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    auc_rel = np.array([0.70, 0.85, 0.93, 0.96, 0.98, 0.99, 1.00])
    knee_idx, knee_theta = kneedle_point(theta, auc_rel)
    print(f"\nKneedle: điểm gối tại theta={knee_theta} (index={knee_idx})")

    print("\nKiểm thử utils_stats: OK")
