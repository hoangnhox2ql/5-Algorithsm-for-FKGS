"""
tests/test_fkge_ch3.py — Kiểm định models/fkge.py theo Chương 3 (Mục 3.2.5, 3.3.2):
định nghĩa (đồng xuất hiện, P_n, ω, ā, b̃, nhúng luật/truy vấn, p_E, a_E), các mệnh đề về hàm mục tiêu
(miền giá trị, cận dưới L_B, bất biến quay, tính bức, cận c_τ), tính không chệch của ước lượng theo lô
trong Thuật toán 3.10, gradient, và hành vi huấn luyện / suy diễn (Thuật toán 3.10–3.11).
Chạy: python -m pytest -q tests/
"""
import sys, os, math, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest
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


def small_model(fkg, **kw):
    kw.setdefault("d_e", 6); kw.setdefault("seed", 0)
    return FKGE(fkg, **kw)


# ----------------------------------------------------------------- định nghĩa
def test_cooccurrence_full_rule_and_proposition(setup):
    fkg = setup[0]; m = small_model(fkg)
    Cm = m.Cm
    assert np.allclose(Cm, Cm.T) and np.all(np.diag(Cm) == 0)
    # (3.cooc): c_ij = số luật chứa cả hai token
    toks = [set(R["antecedent_tokens"]) | {R["consequent_token"]} for R in fkg.rules]
    for i, j in [(0, 1), (0, 5), (2, 9)]:
        ti, tj = m.vocab[i], m.vocab[j]
        assert Cm[i, j] == sum(1 for t in toks if ti in t and tj in t)
    # Mệnh đề đồng xuất hiện ↔ A, B: c_ij = |R|·ā_ij trên cặp tiền đề; b_il = c_il / Σ_l c_il
    for p in range(len(m.PA_I)):
        assert Cm[m.PA_I[p], m.PA_J[p]] == pytest.approx(m.nR * m.abar[p])
    for a, i in enumerate(m.V_ant):
        cl = Cm[i, m.label_tok]
        b = (m.btil[a] - m.eps_b / m.nL) / (1 - m.eps_b)
        assert np.allclose(b, cl / cl.sum())


def test_negative_distribution_and_omega(setup):
    m = small_model(setup[0])
    assert m.Pn.sum() == pytest.approx(1.0) and np.allclose(m.Pn, m.c_i / m.C_tot)
    assert np.all((m.omega > 0) & (m.omega <= 1))
    # ω đối xứng (Mệnh đề đối xứng hoá)
    om = {(i, j): w for i, j, w in zip(m.E_I, m.E_J, m.omega)}
    assert all(om[(i, j)] == pytest.approx(om[(j, i)]) for (i, j) in om)


def test_rule_and_query_embeddings(setup):
    fkg, train = setup[0], setup[1]; m = small_model(fkg, alpha=0.6)
    Zs = m.params[0]
    R0 = fkg.rules[0]; ant = [m.tok[t] for t in R0["antecedent_tokens"]]; out = m.tok[R0["consequent_token"]]
    r0 = 0.6 / len(ant) * Zs[ant].sum(0) + 0.4 * Zs[out]                  # (3.pool)
    assert np.allclose((m.Pm @ Zs)[0], r0)
    s = train[0]; q = sum(deg * Zs[m.tok[t]] for t, deg in s["membership"].items()) / m.r   # (3.hj)-(3.query)
    assert np.allclose(m._query_matrix([s]) @ Zs, q)
    mu = FKGE(fkg, d_e=6, pooling="mean")                                 # gộp đều: α = r/(r+1)
    assert np.allclose(mu.Pm.sum(1), 1.0) and np.allclose(mu.Pm[0, ant[0]], 1 / (len(ant) + 1))


def test_prediction_distribution_and_lse_bounds(setup):
    fkg, train = setup[0], setup[1]; m = small_model(fkg, tau=0.2, tau_c=0.05, eps=0.1)
    Q = m._query_matrix(train[:10]); S, pE = m._scores(m.params, Q)
    S, pE = np.asarray(S), np.asarray(pE)
    assert np.allclose(pE.sum(1), 1) and np.all(pE >= m.eps / m.nL - 1e-12)
    assert np.all(np.abs(S) < 1)                                           # cos_ε0 ∈ (−1, 1)
    for l in range(m.nL):
        idx = m.R_l[l]; mx = S[:, idx].max(1)
        sbar = m.tau_c * np.log(np.exp(S[:, idx] / m.tau_c).sum(1))
        assert np.all(sbar >= mx - 1e-12) and np.all(sbar <= mx + m.tau_c * math.log(len(idx)) + 1e-12)


# ----------------------------------------------------------------- hàm mục tiêu (Mục 3.2.5.x)
def test_component_ranges_and_lower_bounds(setup):
    fkg, train, _, fisa = setup; m = small_model(fkg)
    pF, eF, yF = m._teacher(fisa, train, train[:10])
    Q = m._query_matrix(train); y = np.array([m.class_tokens.index(s["label"]) for s in train])
    c = m.components(data=(Q, y, pF, eF, yF))
    assert c["SGNS"] >= 0 and 0 <= c["edge"] <= 1 and 0 <= c["A"] <= 1
    H = -np.mean(np.sum(m.btil * np.log(m.btil), axis=1))
    assert c["B"] >= H - 1e-9                                              # Mệnh đề cận dưới L_B
    assert 0 <= c["inf"] <= math.log(m.nL / m.eps) + 1e-9
    assert c["rule"] >= 0 and c["rule"] <= math.log(max(len(i) for i in m.R_l)) + 2 / m.tau_c + 1e-9
    Delta = 2 + m.tau_c * math.log(max(len(i) for i in m.R_l))             # Mệnh đề cận c_τ
    c_tau = -math.log((1 - m.eps) / (1 + (m.nL - 1) * math.exp(-Delta / m.tau)) + m.eps / m.nL)
    assert c["pred"] >= c_tau - 1e-9


def test_lower_bound_LB_attained_with_smoothing(setup):
    m = small_model(setup[0], d_e=max(2, len(setup[0].class_tokens)))
    Zs, Zc, U = [p.copy() for p in m.params]
    U[:] = 0; U[np.arange(m.nL), np.arange(m.nL)] = 1.0                  # Mệnh đề 3.coer(iii)
    Zs[m.V_ant] = 0; Zs[m.V_ant, :m.nL] = np.log(m.btil)
    m.lam = {k: 0.0 for k in m.lam}; m.lam.update(N=1.0, B=1.0, C=1e-12)
    H = -np.mean(np.sum(m.btil * np.log(m.btil), axis=1))
    assert float(m._structural([Zs, Zc, U])) == pytest.approx(H, abs=1e-9)


def test_rotation_invariance(setup):
    fkg, train, _, fisa = setup; m = small_model(fkg)
    pF, eF, yF = m._teacher(fisa, train, train[:10])
    Q = m._query_matrix(train); y = np.array([m.class_tokens.index(s["label"]) for s in train])
    Qr, _ = np.linalg.qr(np.random.RandomState(1).normal(size=(m.d_e, m.d_e)))
    rot = [p @ Qr.T for p in m.params]
    assert float(m.loss(m.params, Q, y, pF, eF, yF)) == pytest.approx(float(m.loss(rot, Q, y, pF, eF, yF)), rel=1e-9)


def test_coercivity_and_projection(setup):
    fkg, train, _, fisa = setup
    m = small_model(fkg, T_ep=3, lam_C=0.05).fit(fisa_model=fisa, train_samples=train)
    L = float(m.loss(m.params, *m._train_data))
    assert L >= m.lam["C"] * sum(float(np.sum(p * p)) for p in m.params) - 1e-12
    assert math.sqrt(sum(float(np.sum(p * p)) for p in m.params)) <= m.B0 + 1e-9
    with pytest.raises(ValueError):
        FKGE(fkg, lam_C=0.0)


def test_minibatch_estimator_is_unbiased(setup):
    """Thuật toán 3.10: lấy toàn bộ cặp làm một lô (hệ số = 1) phải trùng hàm đầy đủ; và trung bình của ước lượng
    trên mọi lô một cặp bằng giá trị đầy đủ (kỳ vọng dưới lấy mẫu đều)."""
    m = small_model(setup[0]); full = float(m._structural(m.params))
    nE, nA = len(m.E_I), len(m.PA_I)
    assert float(m._structural(m.params, np.arange(nE), np.arange(nA))) == pytest.approx(full)
    m.lam["N"] = 0.0
    full_noN = float(m._structural(m.params))
    est = np.mean([float(m._structural(m.params, np.array([e]), None)) for e in range(nE)])
    assert est == pytest.approx(full_noN)


def test_gradient_matches_finite_differences(setup):
    fkg, train, _, fisa = setup; m = small_model(fkg, d_e=3)
    pF, eF, yF = m._teacher(fisa, train[:8], train[:5])
    Q = m._query_matrix(train[:8]); y = np.array([m.class_tokens.index(s["label"]) for s in train[:8]])
    from autograd import grad
    f = lambda p: m.loss(p, Q, y, pF, eF, yF)
    g = grad(f)(m.params)
    rng = np.random.RandomState(0)
    for k in range(3):
        for _ in range(3):
            idx = tuple(rng.randint(s) for s in m.params[k].shape); h = 1e-6
            P1 = [p.copy() for p in m.params]; P2 = [p.copy() for p in m.params]
            P1[k][idx] += h; P2[k][idx] -= h
            fd = (float(f(P1)) - float(f(P2))) / (2 * h)
            assert g[k][idx] == pytest.approx(fd, rel=1e-4, abs=1e-7)


# ----------------------------------------------------------------- giáo viên, huấn luyện, suy diễn
def test_teacher_matches_fisa(setup):
    fkg, train, _, fisa = setup; m = small_model(fkg)
    pF, eF, yF = m._teacher(fisa, train, train[:10])
    assert np.allclose(pF.sum(1), 1) and np.allclose(eF.sum(1), 1)
    for i, s in enumerate(train):
        _, yhat, _ = fisa.predict_one(s["membership"])
        assert m.class_tokens[int(np.argmax(pF[i]))] == yhat == m.class_tokens[yF[i]]
        assert np.all(m.rule_lab[eF[i] > 0] == yF[i])                    # e^F chỉ trên luật của ŷ_F


def test_training_decreases_objective_and_inference(setup):
    fkg, train, test, fisa = setup
    m = small_model(fkg, d_e=8, T_ep=25, patience=100).fit(fisa_model=fisa, train_samples=train)
    h = m.history["loss"]
    assert h[-1] < h[0]
    assert m.history["grad_norm"][-1] < m.history["grad_norm"][0]
    r = m.evaluate(test)
    assert 0 <= r["accuracy"] <= 1 and 0 <= r["dev_rule"] <= 1 and 0 <= r["agreement_fisa"] <= 1
    aE, top = m.explain_one(test[0]["membership"], K_act=3)
    proba, yhat, _ = m.predict_one(test[0]["membership"])
    assert sum(aE.values()) == pytest.approx(1.0) and len(top) == 3
    lab = {R.get("id", k): R["consequent_token"] for k, R in enumerate(fkg.rules)}
    assert all(lab[k] == yhat for k in aE)                                 # a_E chỉ trên luật của nhãn dự đoán
    # chặn trên của Mệnh đề quan hệ ràng buộc: Dev ≥ tỉ lệ bất đồng
    assert r["dev_rule"] >= 1 - r["agreement_fisa"] - 1e-9


def test_legacy_parameter_names(setup):
    m = FKGE(setup[0], d=7, K_neg=3, lam_node=0.5, beta_rule=0.2, gamma_inf=2.0, delta_pred=3.0, lr=0.02, epochs=4)
    assert (m.d_e, m.K, m.lam["N"], m.lam["S"], m.lam["I"], m.lam["P"], m.eta, m.T_ep) == (7, 3, 0.5, 0.2, 2.0, 3.0, 0.02, 4)
