"""
src/fkg_sim_rep.py — Sim (Định nghĩa 2.2), Rep, Cov, đúng hàm `diff()` gốc
trong FKG.ipynb (đã phân tích ở lượt trước):

    def diff(Rule1, Rule2):
        if Rule1[-1] != Rule2[-1]: return -1
        count số vị trí thuộc tính khớp
        return count/m

Tức: Sim(Ri,Rj) = (số thuộc tính khớp)/m NẾU cùng nhãn Outcome, ngược lại
= -1 (hai luật khác lớp không thể là đại diện của nhau).

Rep(S) = (1/n) * sum_x max_{s in S} Sim(x,s), với quy ước max(rỗng) = 0 —
quy ước này (Bổ đề 1, Dieu_chinh_..._v4.pdf) tự động "triệt tiêu" ảnh hưởng
của Sim âm: nếu S chưa có luật nào cùng nhãn với x, Cov(x) giữ nguyên 0
(không âm), giữ đúng tính đơn điệu và dưới mô-đun của Rep.
"""
import numpy as np
from itertools import combinations as _combinations


def compute_sim_matrix(rules):
    """Ma trận Sim đầy đủ n x n theo đúng Định nghĩa 2.2 -- vector hoá bằng
    radix-encoding (đếm số vị trí khớp qua so sánh mảng NumPy)."""
    n = len(rules)
    m = len(rules[0]) - 1
    attrs = np.array([[r[i] for i in range(m)] for r in rules], dtype=object)
    labels = np.array([r[-1] for r in rules])

    match_count = np.zeros((n, n), dtype=np.int64)
    for j in range(m):
        col = attrs[:, j]
        match_count += (col[:, None] == col[None, :]).astype(np.int64)
    sim = match_count / m

    same_label = (labels[:, None] == labels[None, :])
    sim = np.where(same_label, sim, -1.0)
    np.fill_diagonal(sim, 1.0)  # Sim(Ri,Ri) = 1.0 (khớp mọi thuộc tính, cùng nhãn)
    return sim


def rep_value(sim_matrix, S_idx, full_n=None):
    """Rep(S) = (1/n) * sum_x max_{s in S} Sim(x,s), quy ước max(rỗng)=0."""
    n = full_n if full_n is not None else sim_matrix.shape[0]
    if len(S_idx) == 0:
        return 0.0
    sub = sim_matrix[:, S_idx]                 # (n, |S|)
    cov = np.maximum(sub.max(axis=1), 0.0)      # max(Sim,...), chặn dưới 0
    return cov.sum() / n


def cov_vector(sim_matrix, S_idx):
    """Cov(x) = max(0, max_{s in S} Sim(x,s)) cho MỌI x -- dùng để tính
    lợi ích biên trong GreedyFKGS/SFKGS/CFKGS mà không cần tính lại Rep từ
    đầu mỗi bước (đúng Bổ đề 1)."""
    n = sim_matrix.shape[0]
    if len(S_idx) == 0:
        return np.zeros(n)
    sub = sim_matrix[:, S_idx]
    return np.maximum(sub.max(axis=1), 0.0)


def marginal_gains(sim_matrix, cov, candidates_idx):
    """Delta(r) cho MỌI r trong candidates_idx cùng lúc, dựa trên Cov hiện
    có (Bổ đề 1): Delta(r) = (1/n) * sum_x [max(Cov(x), Sim(x,r)) - Cov(x)]."""
    n = sim_matrix.shape[0]
    sim_cand = sim_matrix[:, candidates_idx]              # (n, |candidates|)
    new_cov = np.maximum(cov[:, None], sim_cand)           # (n, |candidates|)
    gains = (new_cov - cov[:, None]).sum(axis=0) / n
    return gains  # shape (|candidates|,)


if __name__ == "__main__":
    import random
    rng = random.Random(0)
    levels = ["Low", "Medium", "High"]
    rules = [[rng.choice(levels) for _ in range(4)] + [rng.randint(0, 1)] for _ in range(30)]

    sim = compute_sim_matrix(rules)
    print(f"Ma trận Sim: shape={sim.shape}, range=[{sim.min():.3f}, {sim.max():.3f}]")
    assert sim.shape == (30, 30)
    assert np.allclose(np.diag(sim), 1.0), "Sim(Ri,Ri) phải bằng 1.0"
    assert sim.min() >= -1.0 - 1e-9, "Sim không được nhỏ hơn -1"

    # Kiểm tra tính đơn điệu: Rep(S) phải KHÔNG GIẢM khi thêm phần tử
    S = []
    prev_rep = 0.0
    for step in range(10):
        cov = cov_vector(sim, S)
        candidates = [i for i in range(30) if i not in S]
        gains = marginal_gains(sim, cov, candidates)
        best = candidates[int(np.argmax(gains))]
        S.append(best)
        cur_rep = rep_value(sim, S)
        assert cur_rep >= prev_rep - 1e-9, f"VI PHẠM ĐƠN ĐIỆU tại bước {step}: {cur_rep} < {prev_rep}"
        prev_rep = cur_rep
    print(f"Rep(S) sau 10 bước tham lam: {prev_rep:.4f} -- tăng đơn điệu qua từng bước: OK")
    print("Kiểm thử fkg_sim_rep: OK")
