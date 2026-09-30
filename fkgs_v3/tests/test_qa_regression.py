"""
tests/test_qa_regression.py — Bộ kiểm thử hồi quy theo Báo cáo QA FKGS v2.

Tái hiện các ca trong 02_test_results.md (tính tay, biên, bất biến). Các ca
FAIL ở v2 được giữ với kỳ vọng ĐÃ SỬA ở v3; ca nào thay đổi đặc tả có chú thích.
Chạy: python -m pytest -q tests/
"""
import sys, os, math, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import pytest

import config as C
C.set_dataset("diabetes")

from src.fkg_sim_rep import compute_sim_matrix, rep_value, marginal_gains, cov_vector, coverage_stats
from src.fkgs_algorithms import (GreedyFKGS, RandomSampling, SFKGS, CFKGS, StratRandom, _allocate,
                                 greedy_submodular_select, kmeans_simple, allocate_largest_remainder,
                                 budget_from_theta)
from src.fuzzification import FuzzyFitter, rules_from_fuzzy_df
from src.kfold_utils import make_stratified_kfold, stratified_subsample, build_fkg_fold
from src.fisa_corrected import FISACorrected
from src.utils_stats import (holm_bonferroni, cohens_d_paired, average_ranks, nemenyi_critical_difference,
                             kneedle_point, pareto_front, pareto_knee, safe_auc, wilcoxon_one_sided)
from experiments.common import preprocess
from experiments.synthetic import generate_synthetic
from experiments.run_01_E1_theory_validation import brute_force_optimal
from experiments.run_02_E2_greedy_vs_random import select

R = [["L", "L", 0], ["L", "H", 0], ["H", "H", 1]]
SIM = compute_sim_matrix(R)
FEAT = C.FEATURE_COLUMNS


def _df(rows):
    return pd.DataFrame(rows, columns=FEAT + ["Outcome"])


def _base_row(**kw):
    r = dict(Pregnancies=1, Glucose=120, BloodPressure=70, SkinThickness=20, Insulin=80,
             BMI=30.0, DiabetesPedigreeFunction=0.5, Age=40, Outcome=0)
    r.update(kw)
    return r


# ------------------------------------------------------------------ Sim/Rep
def test_sim_hand():
    assert np.allclose(SIM, [[1, .5, -1], [.5, 1, -1], [-1, -1, 1]])


def test_sim_singleton():
    assert np.allclose(compute_sim_matrix([["L", "L", 0]]), [[1]])


def test_sim_empty_rejected():
    with pytest.raises(ValueError):
        compute_sim_matrix([])


def test_sim_no_antecedent_rejected():
    with pytest.raises(ValueError):
        compute_sim_matrix([[0], [1]])


def test_sim_invariants():
    assert np.allclose(SIM, SIM.T) and np.allclose(np.diag(SIM), 1)


def test_rep_hand_empty_duplicate():
    assert rep_value(SIM, [0]) == pytest.approx(0.5)
    assert rep_value(SIM, []) == 0.0
    assert rep_value(SIM, [0, 0]) == rep_value(SIM, [0])


def test_marginal_hand():
    cov = cov_vector(SIM, [0])
    assert np.allclose(marginal_gains(SIM, cov, [1, 2]), [1 / 6, 1 / 3])


def test_rep_submodular_all_subsets():
    n = 3
    subsets = [[i for i in range(n) if mask >> i & 1] for mask in range(2 ** n)]
    for A in subsets:
        for B in subsets:
            if set(A) <= set(B):
                for x in range(n):
                    if x in B:
                        continue
                    gA = rep_value(SIM, A + [x]) - rep_value(SIM, A)
                    gB = rep_value(SIM, B + [x]) - rep_value(SIM, B)
                    assert gA >= gB - 1e-12


# ------------------------------------------------------------------ Greedy / Random
def test_greedy_hand_zero_oversized():
    S = GreedyFKGS(SIM, m=2)
    assert S == [0, 2] and rep_value(SIM, S) == pytest.approx(5 / 6)
    assert GreedyFKGS(SIM, m=0) == []
    assert sorted(GreedyFKGS(SIM, m=99)) == [0, 1, 2]


def test_greedy_negative_budget_rejected():
    with pytest.raises(ValueError):
        GreedyFKGS(SIM, m=-1)


def test_coverage_all_mode_is_full_coverage():
    rng = random.Random(0)
    T = [[rng.choice("LMH") for _ in range(4)] + [rng.randint(0, 1)] for _ in range(30)]
    s = compute_sim_matrix(T)
    S = greedy_submodular_select(s, np.arange(30), 30, coverage_threshold=0.6, coverage_mode="all")
    assert cov_vector(s, S).min() >= 0.6
    S2 = greedy_submodular_select(s, np.arange(30), 30, coverage_threshold=0.6, coverage_mode="mean")
    st = coverage_stats(s, S2, 0.6)
    assert st["rep"] >= 0.6          # 'mean' chỉ bảo đảm Rep trung bình


def test_random_cases():
    assert RandomSampling(10, m=3, seed=7) == RandomSampling(10, m=3, seed=7)
    assert len(set(RandomSampling(10, m=3, seed=7))) == 3
    assert RandomSampling(0, m=0) == []
    assert sorted(RandomSampling(3, m=5, seed=1)) == [0, 1, 2]


def test_theta_out_of_range_rejected():
    with pytest.raises(ValueError):
        budget_from_theta(1.5, 10)


# ------------------------------------------------------------------ SFKGS / budgets
def test_sfkgs_hand_zero_full():
    labels = [r[-1] for r in R]
    assert SFKGS(SIM, labels, 0.5) == [0, 2]
    assert SFKGS(SIM, labels, 0.0) == []
    assert sorted(SFKGS(SIM, labels, 1.0)) == [0, 1, 2]


def test_sfkgs_matched_global_budget():
    rules = [["L", 0], ["H", 1]]
    s = compute_sim_matrix(rules)
    assert len(SFKGS(s, [0, 1], 0.5)) == 1                       # v3 exact
    assert len(SFKGS(s, [0, 1], 0.5, budget_mode="ceil")) == 2   # hành vi v2


def test_allocate_equal_cap_and_invalid():
    assert _allocate([1, 9], 0.2, 10, "equal", "exact") == [1, 1]
    with pytest.raises(ValueError):
        _allocate([1, 9], 0.2, 10, "bad", "exact")


def test_largest_remainder_properties():
    rng = np.random.RandomState(0)
    for _ in range(300):
        sizes = rng.randint(0, 20, rng.randint(1, 6)).tolist()
        m = rng.randint(0, sum(sizes) + 3)
        a = allocate_largest_remainder(sizes, m)
        assert sum(a) == min(m, sum(sizes))
        assert all(0 <= x <= s for x, s in zip(a, sizes))


def test_all_methods_same_budget_random_cases():
    rng = random.Random(3)
    for _ in range(40):
        n = rng.randint(3, 60)
        rules = [[rng.choice("LMH") for _ in range(4)] + [rng.randint(0, 1)] for _ in range(n)]
        s = compute_sim_matrix(rules)
        lab = [r[-1] for r in rules]
        theta = rng.choice([0.05, 0.1, 0.2, 0.5, 0.9])
        m = budget_from_theta(theta, n)
        for mth in ["GreedyFKGS", "SFKGS", "CFKGS", "Random", "StratRandom"]:
            S = select(mth, s, rules, lab, theta, seed=1)
            assert len(S) == m and len(set(S)) == m, mth


def test_e4_hand_budgets_equal():
    lab = [r[-1] for r in R]
    sizes = {m: len(select(m, SIM, R, lab, 0.5)) for m in ["GreedyFKGS", "Random", "SFKGS", "CFKGS"]}
    assert set(sizes.values()) == {2}


# ------------------------------------------------------------------ CFKGS / KMeans
def test_cfkgs_cases():
    rules = [["L", "L", 0], ["H", "H", 0]]
    s = compute_sim_matrix(rules)
    assert len(CFKGS(s, rules, 0.5, n_clusters_per_label=2)) == 1                       # exact
    assert sorted(CFKGS(s, rules, 0.5, n_clusters_per_label=2, budget_mode="ceil")) == [0, 1]
    assert sorted(CFKGS(SIM, R, 0.1, design="one_stage")) == [0, 1, 2]
    assert CFKGS(SIM, R, 0.0) == []
    with pytest.raises(ValueError):
        CFKGS(SIM, R, 0.5, design="typo")


def test_kmeans_identical_points():
    assert len(set(kmeans_simple(np.zeros((4, 3)), 3).tolist())) == 1


# ------------------------------------------------------------------ subsample
def test_subsample_exact():
    assert len(stratified_subsample([0, 1], 1, seed=0)) == 1
    assert len(stratified_subsample([0, 1, 1], 0, seed=0)) == 0
    assert sorted(stratified_subsample([0, 1, 1], 3, seed=0).tolist()) == [0, 1, 2]
    lab = [0] * 7 + [1] * 3
    for k in range(11):
        assert len(stratified_subsample(lab, k, seed=k)) == k


# ------------------------------------------------------------------ fuzzification / leakage
def test_fuzzify_hand():
    f = FuzzyFitter().fit(_df([_base_row(Glucose=g) for g in [50, 100, 150]]))
    f.bounds_["Glucose"] = (10.0, 100.0)
    out = f.transform_features(_df([_base_row(Glucose=10), _base_row(Glucose=100)]))
    assert out["Glucose"].tolist() == ["Low", "Medium"]


def test_no_label_leakage_in_imputation():
    train = _df([_base_row(Glucose=g, Outcome=o) for g, o in [(80, 0), (90, 0), (150, 1), (160, 1)]])
    f = FuzzyFitter().fit(train)
    a = f.transform_features(_df([_base_row(Glucose=0, Outcome=0)]))
    b = f.transform_features(_df([_base_row(Glucose=0, Outcome=1)]))
    assert a.equals(b)


def test_permuting_test_labels_does_not_change_features():
    df = generate_synthetic(400, 3)
    f = FuzzyFitter().fit(df.iloc[:300])
    te = df.iloc[300:].copy()
    x1 = f.transform_features(te)
    te["Outcome"] = np.random.RandomState(0).permutation(te["Outcome"].values)
    assert x1.equals(f.transform_features(te))


def test_inference_without_label_column():
    train = _df([_base_row(Glucose=g) for g in [80, 120, 160]])
    f = FuzzyFitter().fit(train)
    q = pd.DataFrame([_base_row(Glucose=0)]).drop(columns=["Outcome"])
    out = f.transform_features(q)
    assert out.shape == (1, len(FEAT))


def test_nan_is_imputed_not_high():
    train = _df([_base_row(Glucose=g) for g in [60, 80, 100, 120, 140, 160, 180, 200, 220, 240]])
    f = FuzzyFitter().fit(train)
    out = f.transform_features(_df([_base_row(Glucose=np.nan, BMI=np.nan)]))
    assert out["Glucose"].iloc[0] == "Medium"
    assert out["BMI"].iloc[0] == "Obese" or out["BMI"].iloc[0] in C.CLINICAL_BINS["BMI"][1]


def test_categorical_rule_conversion():
    d = pd.DataFrame({"G": ["Low", "High"], "Outcome": [0, 1]})
    assert rules_from_fuzzy_df(d, ["G"], "Outcome") == [["Low", 0], ["High", 1]]


# ------------------------------------------------------------------ preprocess / folds
def test_preprocess_cases():
    d = _df([_base_row(), _base_row()])
    assert len(preprocess(d)[0]) == 1
    assert len(preprocess(_df([_base_row()]))[0]) == 1
    with pytest.raises(ValueError):
        preprocess(_df([]))


def test_synthetic_repeat():
    assert generate_synthetic(60, 7).equals(generate_synthetic(60, 7))


def test_fold_cases():
    d = pd.DataFrame({"Outcome": [0, 0, 1, 1]})
    folds = make_stratified_kfold(d, k=2, seed=0)
    te = np.concatenate([t for _, t in folds])
    assert sorted(te.tolist()) == [0, 1, 2, 3]
    for tr, t in folds:
        assert not set(tr) & set(t)
    with pytest.raises(ValueError):
        make_stratified_kfold(d, k=1)
    with pytest.raises(ValueError):
        make_stratified_kfold(d, k=6)


def test_build_fold_smoke():
    df = generate_synthetic(80, 1).drop_duplicates().reset_index(drop=True)
    b = build_fkg_fold(df.iloc[:40], df.iloc[40:60], n_sample=12, seed=0)
    assert b["sim"].shape == (12, 12) and len(b["rules_test"]) == 20
    assert build_fkg_fold(df.iloc[:40], None, compute_sim=False)["sim"] is None


# ------------------------------------------------------------------ FISA
def test_fisa_upper_matches_v2_hand():
    m = FISACorrected(R, pair_scope="upper").fit()
    D = m.decision_values(["L", "L"])
    assert D[0] == pytest.approx(4 / 9) and D[1] == pytest.approx(0.0)


def test_fisa_ch1_global_oracle_and_scopes():
    rule = [["a", "b", "c", 0]]
    assert FISACorrected(rule, "global").fit().decision_values(["a", "b", "c"])[0] == pytest.approx(6)
    assert FISACorrected(rule, "adjacent").fit().decision_values(["a", "b", "c"])[0] == pytest.approx(4)
    assert FISACorrected(rule, "upper").fit().decision_values(["a", "b", "c"])[0] == pytest.approx(2)


def test_fisa_global_is_column_order_invariant():
    rng = random.Random(5)
    rules = [[rng.choice("LMH") for _ in range(4)] + [rng.randint(0, 1)] for _ in range(40)]
    perm = [2, 0, 3, 1]
    rules_p = [[r[j] for j in perm] + [r[-1]] for r in rules]
    a = FISACorrected(rules, "global").fit()
    b = FISACorrected(rules_p, "global").fit()
    for r in rules[:10]:
        da = a.decision_values(r[:-1])
        db = b.decision_values([r[j] for j in perm])
        assert da == pytest.approx(db)


def test_fisa_edge_cases():
    assert FISACorrected([["L", 0]]).fit().decision_values(["L"])[0] == 0.0
    assert FISACorrected(R).fit().predict_one(["Z", "Z"]) == (None, {})
    p, _ = FISACorrected(R).fit().predict_proba_one(["L", "L"])
    assert sum(p.values()) == pytest.approx(1) and min(p.values()) >= 0
    with pytest.raises(ValueError):
        FISACorrected([])


# ------------------------------------------------------------------ E1 helpers
def test_e1_bruteforce():
    assert brute_force_optimal(SIM, 2)[1] == pytest.approx(5 / 6)
    assert brute_force_optimal(SIM, 0) == ([], 0.0)
    with pytest.raises(ValueError):
        brute_force_optimal(SIM, 4)


# ------------------------------------------------------------------ statistics
def test_auc_cases():
    assert safe_auc([0, 1], [.5, .5])[0] == 0.5
    assert safe_auc([0, 1], [0, 1])[0] == 1.0
    assert safe_auc([0, 1], [1, 0])[0] == 0.0
    v, flag = safe_auc([0, 0], [0.1, 0.2])
    assert math.isnan(v) and flag == "one_class"


def test_holm_cases():
    adj, rej = holm_bonferroni([.001, .02, .04, .2])
    assert np.allclose(adj, [.004, .06, .08, .2])
    assert len(holm_bonferroni([])[0]) == 0
    assert holm_bonferroni([0.05], 0.05)[1].tolist() == [True]
    adj, rej = holm_bonferroni([0.01, np.nan, 0.04])
    assert math.isnan(adj[1]) and adj[0] == pytest.approx(0.02) and adj[2] == pytest.approx(0.04)


def test_cohen_cases():
    assert cohens_d_paired([1, 2, 3], [0, 0, 0]) == pytest.approx(2.0)
    assert cohens_d_paired([1, 1, 1], [0, 0, 0]) == np.inf
    assert math.isnan(cohens_d_paired([1], [0]))


def test_wilcoxon_exact_five_positive():
    _, p, meth = wilcoxon_one_sided([1, 2, 3, 4, 5], [0, 0, 0, 0, 0])
    assert meth == "exact" and p == pytest.approx(1 / 32)


def test_rank_cases():
    assert average_ranks([[2, 2, 1]], higher_is_better=False).tolist() == [2.5, 2.5, 1.0]
    assert average_ranks(np.ones((2, 4))).tolist() == [2.5] * 4
    assert average_ranks([[5]]).tolist() == [1.0]


def test_cd_cases():
    assert nemenyi_critical_difference(50, 4) == pytest.approx(2.569 * math.sqrt(20 / 300), abs=1e-3)
    assert nemenyi_critical_difference(50, 4, 0.01) > nemenyi_critical_difference(50, 4, 0.05)
    q11 = nemenyi_critical_difference(50, 11) / math.sqrt(11 * 12 / 300)
    assert q11 > 3.164 + 0.02    # không dùng lại hằng số của k = 10


def test_knee_and_pareto():
    assert kneedle_point([0, .5, 1], [0, .9, 1])[0] == 1
    assert kneedle_point([1], [1])[0] == 0
    with pytest.raises(ValueError):
        kneedle_point([], [])
    cost = [1, 2, 3, 4]
    ben = [0.5, 0.9, 0.8, 0.95]           # điểm 2 bị điểm 1 trội
    assert pareto_front(cost, ben) == [0, 1, 3]
    k, front = pareto_knee(cost, ben)
    assert k in front


# ------------------------------------------------------------------ legacy
def test_legacy_matrices_and_unknown_token():
    from src.fkg_original import caculateA, caculateM, caculateB, caculateC, FISA
    from src.fkg_fast import EncodedRuleBase, caculateA_fast, caculateM_fast, caculateB_fast, caculateC_fast, FISA_fast
    base = [["H", "L", "L", "L", 0], ["L", "H", "H", "H", 1], ["H", "H", "H", "H", 1]]
    enc = EncodedRuleBase(base)
    A, M = caculateA(base), caculateM(base)
    B, Cm = caculateB(base, A, M), caculateC(base, caculateB(base, A, M))
    Af, Mf = caculateA_fast(enc), caculateM_fast(enc)
    Bf = caculateB_fast(enc, Af, Mf)
    Cf = caculateC_fast(enc, Bf)
    assert np.allclose(A, Af) and np.allclose(M, Mf) and np.allclose(B, Bf) and np.allclose(Cm, Cf)
    for q in (["L", "UNKNOWN", "L", "L", 0], ["UNKNOWN", "H", "H", "H", 1], ["X", "Y", "Z", "W", 0]):
        assert FISA(base, Cm, q) == FISA_fast(enc, Cf, q)
    with pytest.raises(ValueError):
        EncodedRuleBase([])
    with pytest.raises(ValueError):
        caculateA([])


def test_legacy_existing_suite():
    import tests.test_validate_fast_vs_original as tv
    fn = getattr(tv, "run_one_case")
    for seed, (n, p, lv) in enumerate([(20, 4, 3), (25, 5, 3), (18, 8, 3)]):
        assert fn(n, p, lv, seed)[0]
