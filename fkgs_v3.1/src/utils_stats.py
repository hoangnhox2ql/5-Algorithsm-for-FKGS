"""
src/utils_stats.py — Thống kê (v3).

SỬA SO VỚI v2
  * Holm: bác bỏ khi p_adj <= alpha; giá trị NaN được GIỮ ĐÚNG VỊ TRÍ (không
    lọc rồi gán lệch theta) và không tham gia số phép kiểm định.
  * Cohen d_z: độ lệch chuẩn của chênh lệch = 0 mà trung bình khác 0 -> trả
    +/-inf (không xác định hữu hạn), không trả 0.
  * CD Nemenyi: q_alpha = studentized_range.ppf(1-alpha, k, inf)/sqrt(2) cho
    MỌI k, alpha (v2 bỏ qua alpha và dùng lại hằng số k=10 khi k>10).
  * AUC: một lớp -> NaN kèm cờ, không gán 0.5 im lặng.
  * Pareto + knee: điểm gối chỉ tìm trên mặt Pareto (chi phí thấp, lợi ích
    cao), sắp theo chi phí (QA-08).
  * Wilcoxon: dùng phép tính chính xác khi có thể (n nhỏ, không hoà).
  * bootstrap_ci: khoảng tin cậy phân vị cho trung bình chênh lệch.
"""
import numpy as np
from scipy import stats as _st


def wilcoxon_one_sided(x, y, alternative="greater"):
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan"), float("nan"), "empty"
    if np.allclose(d, 0):
        return 0.0, 1.0, "all_zero"
    nz = d[~np.isclose(d, 0)]
    exact = len(nz) <= 50 and len(np.unique(np.abs(nz))) == len(nz)
    method = "exact" if exact else "approx"
    res = _st.wilcoxon(nz, alternative=alternative, method=method)
    return float(res.statistic), float(res.pvalue), method


def cohens_d_paired(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    if len(d) < 2:
        return float("nan")
    sd = d.std(ddof=1)
    if sd < 1e-12:
        m = d.mean()
        return 0.0 if abs(m) < 1e-12 else float(np.sign(m) * np.inf)
    return float(d.mean() / sd)


def holm_bonferroni(p_values, alpha=0.05):
    p = np.asarray(p_values, dtype=float)
    adj = np.full(len(p), np.nan)
    ok = np.where(np.isfinite(p))[0]
    m = len(ok)
    if m:
        order = ok[np.argsort(p[ok])]
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, min((m - rank) * p[i], 1.0))
            adj[i] = running
    reject = np.where(np.isfinite(adj), adj <= alpha, False)
    return adj, reject


def friedman_test(data_matrix):
    X = np.asarray(data_matrix, float)
    if X.shape[0] < 2 or X.shape[1] < 3:
        raise ValueError("Friedman cần >= 2 block và >= 3 phương pháp")
    if np.allclose(X, X[:, :1]):
        return 0.0, 1.0
    stat, p = _st.friedmanchisquare(*[X[:, j] for j in range(X.shape[1])])
    return float(stat), float(p)


def nemenyi_critical_difference(n_blocks, n_methods, alpha=0.05):
    q = _st.studentized_range.ppf(1 - alpha, n_methods, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_blocks)))


def average_ranks(data_matrix, higher_is_better=True):
    X = np.asarray(data_matrix, float)
    if higher_is_better:
        X = -X
    return np.vstack([_st.rankdata(r) for r in X]).mean(axis=0)


def safe_auc(y_true, score):
    """AUC (Mann-Whitney, hoà tính 0.5). Trả (auc, flag)."""
    y = np.asarray(y_true)
    s = np.asarray(score, float)
    if len(np.unique(y)) < 2:
        return float("nan"), "one_class"
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)), "ok"


def bootstrap_ci(values, level=0.95, n_boot=10000, seed=0):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return (float("nan"), float("nan"))
    if len(v) == 1:
        return (float(v[0]), float(v[0]))
    rng = np.random.RandomState(seed)
    boots = v[rng.randint(0, len(v), (n_boot, len(v)))].mean(1)
    a = (1 - level) / 2
    return (float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a)))


def t_ci(values, level=0.95):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (float("nan"), float("nan"))
    h = _st.t.ppf(0.5 + level / 2, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return (float(v.mean() - h), float(v.mean() + h))


def pareto_front(cost, benefit):
    """Chỉ số các điểm không bị trội (chi phí <=, lợi ích >=, ít nhất một ngặt),
    sắp theo chi phí tăng dần."""
    cost = np.asarray(cost, float)
    ben = np.asarray(benefit, float)
    idx = []
    for i in range(len(cost)):
        dominated = np.any((cost <= cost[i]) & (ben >= ben[i]) &
                           ((cost < cost[i]) | (ben > ben[i])))
        if not dominated:
            idx.append(i)
    return sorted(idx, key=lambda i: (cost[i], -ben[i]))


def kneedle_point(x, y):
    """Điểm có khoảng cách lớn nhất tới dây cung nối đầu-cuối, sau khi SẮP theo x
    và chuẩn hoá về [0,1]^2. Trả (chỉ số trong mảng gốc, x tại đó)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) == 0:
        raise ValueError("kneedle_point: dữ liệu rỗng")
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    if len(xs) <= 2:
        k = int(np.argmax(ys))
        return int(order[k]), float(xs[k])
    xn = (xs - xs.min()) / (np.ptp(xs) + 1e-12)
    yn = (ys - ys.min()) / (np.ptp(ys) + 1e-12)
    x1, y1, x2, y2 = xn[0], yn[0], xn[-1], yn[-1]
    dist = np.abs((y2 - y1) * xn - (x2 - x1) * yn + x2 * y1 - y2 * x1) / (np.hypot(y2 - y1, x2 - x1) + 1e-12)
    k = int(np.argmax(dist))
    return int(order[k]), float(xs[k])


def pareto_knee(cost, benefit):
    """Knee trên mặt Pareto; trả chỉ số trong mảng gốc."""
    front = pareto_front(cost, benefit)
    if len(front) == 1:
        return front[0], front
    k, _ = kneedle_point(np.asarray(cost)[front], np.asarray(benefit)[front])
    return front[k], front


def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "NA"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"
