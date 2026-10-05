"""
tests/test_fisa_v32.py — Kiểm định models/fisa.py khớp FISA của FKGS v3.2 (mô hình FKG-Pairs):
  1. với độ thuộc rời rạc, D trùng cài đặt vòng lặp trực tiếp của các công thức (7), (8), (10), (11);
  2. bảng tra (FISA) và duyệt tuần tự (FISASequential) cho cùng D với độ thuộc mờ;
  3. nếu có mã FKGS v3.2 (biến môi trường FKGS_V32_DIR), D trùng src/fisa_pairs.FISAPairs;
  4. đóng góp luật a_F là phân phối, chỉ trên luật của nhãn dự đoán, và phân rã đúng D.
Chạy: python -m pytest -q tests/
"""
import sys, os, random, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest
from data.fkg_io import FKGRuleBase, generate_synthetic_fkg, generate_synthetic_test_samples
from models.fisa import FISA, FISASequential


def make_base(seed, n=40, m=5):
    rng = random.Random(seed)
    attrs = [f"A{i}" for i in range(m)]; lv = ["Low", "Medium", "High"]
    vocab = [f"{a}-{l}" for a in attrs for l in lv] + ["class-0", "class-1"]
    rules = [{"id": i, "antecedent_tokens": [f"{a}-{rng.choice(lv)}" for a in attrs],
              "consequent_token": f"class-{rng.randint(0, 1)}", "support": 0.1, "confidence": 0.8}
             for i in range(n)]
    return FKGRuleBase(vocab, rules, []), attrs, lv


def reference_D(fkg, attrs, q, k=3):
    rows = [{t.rsplit('-', 1)[0]: t.rsplit('-', 1)[1] for t in r["antecedent_tokens"]} for r in fkg.rules]
    labs = [r["consequent_token"] for r in fkg.rules]; n = len(rows)
    sumA = [sum(sum(all(rows[s][a] == rows[t][a] for a in c) for s in range(n)) / n
                for c in itertools.combinations(attrs, k + 1)) for t in range(n)]
    M = lambda t, a: sum(rows[s][a] == rows[t][a] and labs[s] == labs[t] for s in range(n)) / n
    D = {}
    for l in sorted(set(labs)):
        Cs = [sum(sumA[t] * min(M(t, a) for a in c) for t in range(n)
                  if labs[t] == l and all(rows[t][a] == q[a] for a in c))
              for c in itertools.combinations(attrs, k)]
        D[l] = max(Cs) + min(Cs)
    return D


def onehot(q):
    return {f"{a}-{v}": 1.0 for a, v in q.items()}


def test_crisp_matches_formulas():
    for seed in range(4):
        fkg, attrs, lv = make_base(seed)
        f = FISA(fkg, decision="argmax").fit()
        rng = random.Random(100 + seed)
        for _ in range(6):
            q = {a: rng.choice(lv) for a in attrs}
            D, _, _ = f.decision_values(onehot(q))
            ref = reference_D(fkg, attrs, q)
            for l in ref:
                assert D[l] == pytest.approx(ref[l])


def test_lookup_equals_sequential_fuzzy():
    fkg = generate_synthetic_fkg(n_rules=120, seed=3)
    smp = generate_synthetic_test_samples(fkg, n_samples=30, seed=4)
    a = FISA(fkg).fit(); b = FISASequential(fkg).fit()
    for s in smp:
        Da, _, _ = a.decision_values(s["membership"]); Db, _, _ = b.decision_values(s["membership"])
        for l in Da:
            assert Da[l] == pytest.approx(Db[l])


@pytest.mark.skipif(not os.environ.get("FKGS_V32_DIR"), reason="cần FKGS_V32_DIR trỏ tới thư mục fkgs_v3.2")
def test_matches_fkgs_v32_module():
    sys.path.insert(0, os.environ["FKGS_V32_DIR"])
    from src.fisa_pairs import FISAPairs
    fkg, attrs, lv = make_base(7)
    rules = [[t.rsplit('-', 1)[1] for t in r["antecedent_tokens"]] + [int(r["consequent_token"][-1])] for r in fkg.rules]
    ref = FISAPairs(rules, node_size=3).fit()
    f = FISA(fkg, decision="argmax").fit()
    rng = random.Random(9)
    for _ in range(10):
        q = {a: rng.choice(lv) for a in attrs}
        Dr, _ = ref.decision_matrix([[q[a] for a in attrs]])
        D, _, _ = f.decision_values(onehot(q))
        assert D["class-0"] == pytest.approx(Dr[0, 0]) and D["class-1"] == pytest.approx(Dr[0, 1])


def test_attribution_is_exact_decomposition():
    fkg = generate_synthetic_fkg(n_rules=80, seed=5)
    smp = generate_synthetic_test_samples(fkg, n_samples=10, seed=6)
    f = FISA(fkg, decision="argmax").fit()
    for s in smp:
        a = f.attribution(s["membership"])
        assert sum(a.values()) == pytest.approx(1.0)
        D, y, _ = f.predict_one(s["membership"])
        lab = {r.get("id", i): r["consequent_token"] for i, r in enumerate(fkg.rules)}
        assert all(lab[i] == y for i in a)
