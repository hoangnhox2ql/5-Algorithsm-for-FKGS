"""
tests/test_fkge_ch3_v2.py — Kiểm định models/fkge.py theo Chương 3 bản hiện hành (Mục 3.2.4, 3.3.2):
  định nghĩa (3.5) a_ii', (3.6) b_il, P_n, nhúng luật (3.pool) với u_out, nhúng truy vấn (3.query), p_E, a_E;
  Định lý nghiệm bảo toàn cạnh (PMI của A); Mệnh đề cận dưới L_node và trường hợp đạt cận;
  Mệnh đề cấu trúc và tồn tại nghiệm (tính bức, phép chiếu); Mệnh đề bất biến quay;
  Thuật toán 3.10: ước lượng theo lô không chệch, gradient, giảm hàm mục tiêu, che luật cùng nhóm, giáo viên;
  Thuật toán 3.11: a_E là phân phối trên luật của nhãn dự đoán; Dev ≥ tỉ lệ bất đồng.
Chạy: python -m pytest -q tests/
"""
import sys, os, math, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest
from autograd import grad
from data.fkg_io import FKGRuleBase, generate_synthetic_fkg, generate_synthetic_test_samples
from models.fisa import FISA
from models.fkge import FKGE


@pytest.fixture(scope="module")
def setup():
    fkg = generate_synthetic_fkg(n_attrs=4, n_rules=40, seed=3)
    train = generate_synthetic_test_samples(fkg, n_samples=60, seed=4)
    test = generate_synthetic_test_samples(fkg, n_samples=40, seed=5)
    fisa = FISA(fkg).fit(val_samples=train)
    return fkg, train, test, fisa


def model(fkg, **kw):
    kw.setdefault("d_e", 6); kw.setdefault("seed", 0)
    return FKGE(fkg, **kw)


def data_of(m, fisa, samples, val):
    pF, eF, yF = m._teacher(fisa, samples, val)
    Q = m._query_matrix(samples); y = np.array([m.class_tokens.index(s["label"]) for s in samples])
    return Q, y, pF, eF, yF


# ------------------------------------------------------------------ định nghĩa (3.5), (3.6)
def test_edge_weights_are_FKG_A(setup):
    fkg = setup[0]; m = model(fkg)
    rows = [dict(t.rsplit("-", 1) for t in R["antecedent_tokens"]) for R in fkg.rules]
    n = len(rows)
    for i, j in [(0, 4), (1, 7), (2, 11)]:
        (ai, vi), (aj, vj) = m.nodes[i].rsplit("-", 1), m.nodes[j].rsplit("-", 1)
        expect = 0.0 if ai == aj else sum(r[ai] == vi and r[aj] == vj for r in rows) / n
        assert m.a[i, j] == pytest.approx(expect)
    assert np.allclose(m.a, m.a.T) and np.all(np.diag(m.a) == 0)
    # với mọi luật t chứa cả hai giá trị, a_ii' = A^t_jk (tỉ lệ luật trùng t trên cặp thuộc tính)
    R0 = rows[0]; a1, a2 = list(R0)[:2]
    i, j = m.node_idx[f"{a1}-{R0[a1]}"], m.node_idx[f"{a2}-{R0[a2]}"]
    assert m.a[i, j] == pytest.approx(sum(r[a1] == R0[a1] and r[a2] == R0[a2] for r in rows) / n)


def test_negative_distribution_and_label_distribution(setup):
    m = model(setup[0])
    assert m.Pn.sum() == pytest.approx(1.0) and np.allclose(m.Pn, m.a_i / m.a_tot)
    assert np.allclose(m.b_dist.sum(1), 1.0) and np.allclose(m.btil.sum(1), 1.0)
    assert np.all(m.btil >= m.eps_b / m.nL - 1e-12)
    # b_il khớp đếm trực tiếp
    fkg = setup[0]; i = m.V_R[0]; tok = m.nodes[i]
    rules = [R for R in fkg.rules if tok in R["antecedent_tokens"]]
    for l, c in enumerate(m.class_tokens):
        assert m.b_dist[0, l] == pytest.approx(sum(R["consequent_token"] == c for R in rules) / len(rules))


def test_rule_and_query_embeddings(setup):
    fkg, train = setup[0], setup[1]; m = model(fkg, alpha=0.6)
    Z, _, U = m.params
    R0 = fkg.rules[0]; ant = [m.node_idx[t] for t in R0["antecedent_tokens"]]
    l0 = m.class_tokens.index(R0["consequent_token"])
    r0 = 0.6 / len(ant) * Z[ant].sum(0) + 0.4 * U[l0]                     # (3.pool) với u_out
    assert np.allclose(np.asarray(m._rule_emb(m.params))[0], r0)
    s = train[0]; q = sum(d * Z[m.node_idx[t]] for t, d in s["membership"].items()) / m.r
    assert np.allclose(m._query_matrix([s]) @ Z, q)                        # (3.query)


def test_prediction_distribution(setup):
    fkg, train = setup[0], setup[1]; m = model(fkg, tau=0.2, tau_c=0.05, eps=0.1)
    S, pE = m._scores(m.params, m._query_matrix(train[:10]))
    S, pE = np.asarray(S), np.asarray(pE)
    assert np.allclose(pE.sum(1), 1) and np.all(pE >= m.eps / m.nL - 1e-12) and np.all(np.abs(S) < 1)
    for l in range(m.nL):
        idx = m.R_l[l]; mx = S[:, idx].max(1)
        sbar = m.tau_c * np.log(np.exp(S[:, idx] / m.tau_c).sum(1))
        assert np.all(sbar >= mx - 1e-12) and np.all(sbar <= mx + m.tau_c * math.log(len(idx)) + 1e-12)


# ------------------------------------------------------------------ Định lý nghiệm bảo toàn cạnh
def test_edge_solution_is_shifted_PMI(setup):
    """ℓ_ii'(x) = −a_ii' log σ(x) − K a_i a_i'/|a| log σ(−x) đạt cực tiểu tại x* = log(a_ii'|a|/(K a_i a_i'))."""
    m = model(setup[0], K=3)
    from scipy.optimize import minimize_scalar
    ls = lambda x: -np.logaddexp(0, -x)
    for e in range(0, len(m.E_I), max(1, len(m.E_I) // 5)):
        i, j = m.E_I[e], m.E_J[e]; a = m.a[i, j]; neg = m.K * m.a_i[i] * m.a_i[j] / m.a_tot
        f = lambda x: -a * ls(x) - neg * ls(-x)
        x_num = minimize_scalar(f, bounds=(-30, 30), method="bounded", options={"xatol": 1e-9}).x
        assert x_num == pytest.approx(math.log(a * m.a_tot / (m.K * m.a_i[i] * m.a_i[j])), abs=1e-5)


def test_edge_loss_decomposes_into_pair_terms(setup):
    """L_edge = (1/|a|) Σ_{i,i'} ℓ_ii'(z_iᵀc_i') (khai triển kỳ vọng của mẫu âm, chứng minh Định lý)."""
    m = model(setup[0], K=2); m.lam = {k: 0.0 for k in m.lam}; m.lam.update(E=1.0, C=1e-3)
    Z, Cc, _ = m.params; X = Z @ Cc.T; ls = lambda x: -np.logaddexp(0, -x)
    tot = 0.0
    for i in range(m.nV):
        for j in range(m.nV):
            tot += -m.a[i, j] * ls(X[i, j]) - m.K * m.a_i[i] * m.a_i[j] / m.a_tot * ls(-X[i, j])
    assert float(m._structural(m.params)) == pytest.approx(tot / m.a_tot, rel=1e-9)


# ------------------------------------------------------------------ Mệnh đề cận dưới L_node
def test_node_loss_lower_bound_and_attainment(setup):
    m = model(setup[0], d_e=max(2, len(setup[0].class_tokens)))
    m.lam = {k: 0.0 for k in m.lam}; m.lam.update(N=1.0, C=1e-12)
    H = -np.mean(np.sum(m.btil * np.log(m.btil), axis=1))
    assert float(m._structural(m.params)) >= H - 1e-12
    Z, Cc, U = [p.copy() for p in m.params]
    U[:] = 0; U[np.arange(m.nL), np.arange(m.nL)] = 1.0
    Z[m.V_R] = 0; Z[m.V_R, :m.nL] = np.log(m.btil)
    assert float(m._structural([Z, Cc, U])) == pytest.approx(H, abs=1e-10)


# ------------------------------------------------------------------ Mệnh đề cấu trúc, bất biến quay
def test_components_nonnegative_and_rotation_invariance(setup):
    fkg, train, _, fisa = setup; m = model(fkg)
    d = data_of(m, fisa, train, train[:10])
    c = m.components(data=d)
    assert all(v >= -1e-12 for v in c.values())
    assert c["inf"] <= math.log(m.nL / m.eps) + 1e-9
    Qr, _ = np.linalg.qr(np.random.RandomState(1).normal(size=(m.d_e, m.d_e)))
    rot = [p @ Qr.T for p in m.params]
    assert float(m.loss(m.params, *d)) == pytest.approx(float(m.loss(rot, *d)), rel=1e-9)


def test_coercivity_and_projection(setup):
    fkg, train, _, fisa = setup
    m = model(fkg, T_ep=3, lam_C=0.05).fit(fisa_model=fisa, train_samples=train)
    L = float(m.loss(m.params, *m._train_data))
    assert L >= m.lam["C"] * sum(float(np.sum(p * p)) for p in m.params) - 1e-12
    assert math.sqrt(sum(float(np.sum(p * p)) for p in m.params)) <= m.B0 + 1e-9
    with pytest.raises(ValueError):
        FKGE(fkg, lam_C=0.0)


# ------------------------------------------------------------------ Thuật toán 3.10
def test_minibatch_estimator_unbiased(setup):
    m = model(setup[0]); m.lam["N"] = 0.0
    full = float(m._structural(m.params))
    nE = len(m.E_I)
    assert float(m._structural(m.params, np.arange(nE))) == pytest.approx(full)
    est = np.mean([float(m._structural(m.params, np.array([e]))) for e in range(nE)])
    assert est == pytest.approx(full)


def test_gradient_matches_finite_differences(setup):
    fkg, train, _, fisa = setup; m = model(fkg, d_e=3)
    d = data_of(m, fisa, train[:8], train[:5])
    f = lambda p: m.loss(p, *d)
    g = grad(f)(m.params); rng = np.random.RandomState(0)
    for k in range(3):
        for _ in range(3):
            idx = tuple(rng.randint(s) for s in m.params[k].shape); h = 1e-6
            P1 = [p.copy() for p in m.params]; P2 = [p.copy() for p in m.params]
            P1[k][idx] += h; P2[k][idx] -= h
            assert g[k][idx] == pytest.approx((float(f(P1)) - float(f(P2))) / (2 * h), rel=1e-4, abs=1e-7)


def test_teacher_consistent_with_fisa(setup):
    fkg, train, _, fisa = setup; m = model(fkg)
    pF, eF, yF = m._teacher(fisa, train, train[:10])
    assert np.allclose(pF.sum(1), 1) and np.allclose(eF.sum(1), 1)
    for i, s in enumerate(train):
        _, yhat, _ = fisa.predict_one(s["membership"])
        assert m.class_tokens[int(np.argmax(pF[i]))] == yhat == m.class_tokens[yF[i]]
        assert np.all(m.rule_lab[eF[i] > 0] == yF[i])


def test_teacher_from_arrays_matches_beta_and_temperature(setup):
    fkg, train, _, fisa = setup; m = model(fkg)
    n = len(train); Dlog = np.log(np.random.RandomState(2).rand(n, m.nL) + 1e-3)
    td = {"Dlog": Dlog, "eF": np.full((n, m.nR), 1.0 / m.nR), "yF": np.zeros(n, dtype=int),
          "Dlog_val": Dlog[:10], "beta": np.array([0.0, -0.3])}
    pF, eF, yF = m._teacher_from_arrays(td, [s["label"] for s in train[:10]])
    z = (Dlog + td["beta"]) / m.teacher_T
    assert np.allclose(pF, np.exp(z) / np.exp(z).sum(1, keepdims=True))
    assert m.teacher_T in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def test_training_and_inference(setup):
    fkg, train, test, fisa = setup
    m = model(fkg, d_e=8, T_ep=25, patience=100).fit(fisa_model=fisa, train_samples=train)
    assert m.history["loss"][-1] < m.history["loss"][0]
    r = m.evaluate(test)
    assert 0 <= r["accuracy"] <= 1 and 0 <= r["dev_rule"] <= 1
    assert r["dev_rule"] >= 1 - r["agreement_fisa"] - 1e-9                 # Mệnh đề quan hệ ràng buộc
    aE, top = m.explain_one(test[0]["membership"], K_act=3)
    _, yhat, _ = m.predict_one(test[0]["membership"])
    lab = {R.get("id", k): R["consequent_token"] for k, R in enumerate(fkg.rules)}
    assert sum(aE.values()) == pytest.approx(1.0) and all(lab[k] == yhat for k in aE) and len(top) == 3


def test_group_mask_removes_own_rule(setup):
    fkg = setup[0]
    rules = [dict(R, group=k // 2) for k, R in enumerate(fkg.rules)]       # mỗi nhóm hai luật
    fkg2 = FKGRuleBase(fkg.vocab, rules, [])
    m = model(fkg2)
    smp = [{"membership": {t: 1.0 for t in rules[k]["antecedent_tokens"]}, "label": rules[k]["consequent_token"],
            "group": rules[k]["group"]} for k in range(6)]
    G = m._mask(smp)
    for i in range(6):
        assert set(np.where(G[i])[0]) == {2 * (i // 2), 2 * (i // 2) + 1}
    S, _ = m._scores(m.params, m._query_matrix(smp), G)
    assert np.all(np.asarray(S)[G] < -1e5)                                 # luật bị che không tham gia


def test_legacy_parameter_names(setup):
    with pytest.warns(DeprecationWarning):
        m = FKGE(setup[0], d=7, K_neg=3, lam_node=0.5, beta_rule=0.2, gamma_inf=2.0, delta_pred=3.0, lr=0.02,
                 epochs=4, w=2)
    assert (m.d_e, m.K, m.lam["N"], m.lam["E"], m.lam["I"], m.lam["P"], m.eta, m.T_ep) == (7, 3, 0.5, 0.2, 2.0, 3.0, 0.02, 4)


def test_full_gradient_projected_descent_monotone(setup):
    """Thuật toán 3.10 (gd_full): mỗi bước thoả L(Θ⁺) ≤ L(Θ) − (η/2)‖∇L‖², nên hàm mục tiêu giảm đơn điệu,
    và mọi bước lặp nằm trong hình cầu bán kính B_0 (Định lý hội tụ, Mệnh đề tồn tại nghiệm)."""
    fkg, train, _, fisa = setup
    m = model(fkg, d_e=6, T_ep=15, patience=1000, optimizer="gd_full").fit(fisa_model=fisa, train_samples=train)
    h = m.history["loss"]
    assert all(h[k + 1] <= h[k] + 1e-12 for k in range(len(h) - 1))
    assert all(e > 0 for e in m.history["eta"])
    assert math.sqrt(sum(float(np.sum(p * p)) for p in m.params)) <= m.B0 + 1e-9
