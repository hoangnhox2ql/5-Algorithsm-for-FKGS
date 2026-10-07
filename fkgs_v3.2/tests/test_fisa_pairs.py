"""
tests/test_fisa_pairs.py — Kiểm định src/fisa_pairs.py theo đúng mô hình FKG-Pairs của bài báo
gốc và notebook FKG.ipynb:
  1. trùng khớp với cài đặt vòng lặp trực tiếp của các công thức (7), (8), (10), (11) với k = 1, 2, 3;
  2. tái lập ví dụ số của bài báo (Bảng 3, Bảng 4) với k = 2;
  3. với k = 3, giá trị D̃ trùng với notebook gốc (src/fkg_original.py) khi xét đủ mọi luật;
  4. các quy tắc quyết định argmax (12), ratio9 (notebook) và dự phòng khi không khớp.
"""
import sys, os, random, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fractions import Fraction
import numpy as np
import pytest
from src.fisa_pairs import FISAPairs
from src import fkg_original as O


def reference_D(base, q, k):
    """Cài đặt vòng lặp trực tiếp (7), (8), (10), (11); trả {nhãn: D̃}."""
    n, m = len(base), len(base[0]) - 1
    k = min(k, m); ka = min(k + 1, m)
    sumA = [sum(Fraction(sum(all(base[s][i] == base[t][i] for i in c) for s in range(n)), n)
                for c in itertools.combinations(range(m), ka)) for t in range(n)]
    M = lambda t, i: Fraction(sum(base[s][i] == base[t][i] and base[s][-1] == base[t][-1] for s in range(n)), n)
    labels = sorted(set(r[-1] for r in base))
    D = {}
    for l in labels:
        Cs = []
        for c in itertools.combinations(range(m), k):
            Cs.append(sum((sumA[t] * min(M(t, i) for i in c) for t in range(n)
                           if base[t][-1] == l and all(base[t][i] == q[i] for i in c)), Fraction(0)))
        D[l] = max(Cs) + min(Cs)
    return D


@pytest.mark.parametrize("k", [1, 2, 3])
def test_matches_reference_formulas(k):
    rng = random.Random(10 + k)
    for _ in range(6):
        base = [[rng.choice("LMH") for _ in range(5)] + [rng.randint(0, 1)] for _ in range(18)]
        model = FISAPairs(base, node_size=k).fit()
        queries = [base[rng.randrange(18)][:-1] for _ in range(4)] + \
                  [[rng.choice("LMHX") for _ in range(5)] for _ in range(4)]
        D, unseen = model.decision_matrix(queries)
        for qi, q in enumerate(queries):
            ref = reference_D(base, q, k)
            assert D[qi, 0] == pytest.approx(float(ref[0])) and D[qi, 1] == pytest.approx(float(ref[1]))
            assert unseen[qi] == (ref[0] == 0 and ref[1] == 0 and
                                  all(any(q[i] not in {r[i] for r in base} for i in c) or
                                      not any(all(r[i] == q[i] for i in c) for r in base)
                                      for c in itertools.combinations(range(5), min(k, 5))))


PAPER_TABLE3 = [["M", "M", "M", "M", "M", 1], ["L", "H", "L", "L", "L", 2], ["M", "H", "M", "M", "M", 1],
                ["M", "M", "M", "L", "M", 1], ["L", "H", "L", "H", "L", 2], ["L", "H", "H", "H", "H", 2]]


def test_paper_example_table4_and_B():
    m = FISAPairs(PAPER_TABLE3, node_size=2).fit()
    # Bảng 4, dòng R1: tổng các A^1_{ijk} = 1/3+1/6+1/3+1/3+1/2+1/3+1/6+1/3+1/6+1/3 = 3
    assert m.sumA[0] == pytest.approx(3.0)
    # B^1_{12,1} = 3 * Min(1/2, 1/3) = 1 (ví dụ tính tay trong bài báo)
    assert m.sumA[0] * min(m.M[0, 0], m.M[0, 1]) == pytest.approx(1.0)
    # C̃_{12,1}(R1) = B^1 + B^4 (R1 và R4 cùng (M1, M2), nhãn 1)
    ref = reference_D(PAPER_TABLE3, PAPER_TABLE3[0][:-1], 2)
    D, _ = m.decision_matrix([PAPER_TABLE3[0][:-1]])
    assert D[0, 0] == pytest.approx(float(ref[1])) and D[0, 1] == pytest.approx(0.0)
    assert m.predict_one(PAPER_TABLE3[0][:-1])[0] == 1


def legacy_D_full(base, Cm, q):
    """D̃ của notebook gốc với vòng lặp duyệt ĐỦ mọi luật (sửa lỗi row-1)."""
    m = len(base[0]) - 1; c3 = O.combination(3, m)
    C0, C1 = [0.0] * c3, [0.0] * c3
    for t, (a, b, c) in enumerate(itertools.combinations(range(m), 3)):
        for r in range(len(base)):
            if base[r][a] == q[a] and base[r][b] == q[b] and base[r][c] == q[c]:
                if base[r][-1] == 0:
                    C0[t] = Cm[r][t]
                else:
                    C1[t] = Cm[r][t + c3]
    return max(C0) + min(C0), max(C1) + min(C1)


def test_k3_equals_notebook_without_off_by_one():
    rng = random.Random(7)
    for _ in range(5):
        base = [[rng.choice("LMH") for _ in range(6)] + [rng.randint(0, 1)] for _ in range(20)]
        A = O.caculateA(base); M = O.caculateM(base); Cm = O.caculateC(base, O.caculateB(base, A, M))
        model = FISAPairs(base, node_size=3).fit()
        qs = [base[rng.randrange(20)][:-1] for _ in range(5)]
        D, _ = model.decision_matrix(qs)
        for qi, q in enumerate(qs):
            d0, d1 = legacy_D_full(base, Cm, q)
            assert D[qi, 0] == pytest.approx(d0) and D[qi, 1] == pytest.approx(d1)


def test_decision_rules_and_fallback():
    base = PAPER_TABLE3 + [["M", "M", "L", "M", "M", 1]]
    base = [r[:-1] + [0 if r[-1] == 1 else 1] for r in base]   # nhãn 0/1
    m = FISAPairs(base, node_size=3).fit()
    ev = m.evaluate(base, decision="argmax")
    D = ev["D"]
    assert np.array_equal(ev["pred_argmax"], np.where(D[:, 1] > D[:, 0], 1, 0))
    assert np.array_equal(ev["pred_ratio9"], np.where(D[:, 0] > 9 * D[:, 1], 0, 1))
    # truy vấn không khớp nút nào -> nhãn có nhiều luật nhất, điểm 0
    ev2 = m.evaluate([["X", "X", "X", "X", "X", 0]], decision="argmax")
    assert ev2["n_unseen"] == 1 and ev2["pred"][0] == m.fallback_label and ev2["score"][0] == 0.0
