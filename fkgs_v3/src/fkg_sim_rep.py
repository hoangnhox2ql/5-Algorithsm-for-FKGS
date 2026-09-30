"""
src/fkg_sim_rep.py — Nhân tương đồng, hàm đại diện và độ phủ (v3).

  Sim(Ri,Rj) = (#tiền đề trùng)/p nếu cùng nhãn, -1 nếu khác nhãn.
  a(x,s)     = max(0, Sim(x,s))                     (nhân không âm, bị chặn)
  Rep(S)     = (1/n) * sum_x max_{s in S} a(x,s),   max(rỗng) = 0.

Rep đơn điệu, dưới mô-đun vì a trong [0,1] (Mệnh đề 2.28). v3 bổ sung kiểm tra
đầu vào (rỗng, p=0) và hàm `coverage_stats` phân biệt ĐẠI DIỆN TRUNG BÌNH
(Rep) với PHỦ TỪNG LUẬT (min cov, tỉ lệ luật có cov >= delta) (QA-06).
"""
import numpy as np


def _validate_rules(rules):
    if rules is None or len(rules) == 0:
        raise ValueError("Tập luật rỗng")
    p = len(rules[0]) - 1
    if p <= 0:
        raise ValueError("Luật phải có ít nhất một tiền đề (p >= 1)")
    if any(len(r) != p + 1 for r in rules):
        raise ValueError("Các luật có số tiền đề khác nhau")
    return p


def compute_sim_matrix(rules):
    p = _validate_rules(rules)
    attrs = np.array([r[:p] for r in rules], dtype=object)
    labels = np.array([r[-1] for r in rules])
    n = len(rules)
    match = np.zeros((n, n), dtype=np.int32)
    for j in range(p):
        col = attrs[:, j]
        match += (col[:, None] == col[None, :])
    sim = match / p
    sim = np.where(labels[:, None] == labels[None, :], sim, -1.0)
    np.fill_diagonal(sim, 1.0)
    return sim


def cov_vector(sim_matrix, S_idx):
    n = sim_matrix.shape[0]
    S_idx = list(S_idx)
    if len(S_idx) == 0:
        return np.zeros(n)
    return np.maximum(sim_matrix[:, S_idx].max(axis=1), 0.0)


def rep_value(sim_matrix, S_idx):
    return float(cov_vector(sim_matrix, S_idx).mean())


def marginal_gains(sim_matrix, cov, candidates_idx):
    sim_cand = sim_matrix[:, candidates_idx]
    return (np.maximum(cov[:, None], sim_cand) - cov[:, None]).mean(axis=0)


def coverage_stats(sim_matrix, S_idx, delta):
    """Trả về dict: rep (trung bình), min_cov, frac_covered (tỉ lệ luật có
    cov >= delta), full_cover (min_cov >= delta)."""
    cov = cov_vector(sim_matrix, S_idx)
    return dict(rep=float(cov.mean()), min_cov=float(cov.min()),
                frac_covered=float((cov >= delta - 1e-12).mean()),
                full_cover=bool(cov.min() >= delta - 1e-12))
