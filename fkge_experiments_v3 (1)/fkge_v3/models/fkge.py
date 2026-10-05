"""
models/fkge.py — Mô hình FKG-E cài đặt đúng Chương 3 luận án (Mục 3.2.5 và Mục 3.3.2).

Ký hiệu và công thức (số hiệu theo bản LaTeX hiện hành):
  Token: V_tok = V_ant ∪ L; luật R = [t_1..t_r, t_out]; tok(R) = ant(R) ∪ {out(R)}.
  Đồng xuất hiện toàn luật (3.cooc): c_ij = |{R : i, j ∈ tok(R)}|, c_ii = 0;  c_i = Σ_j c_ij;  |C| = Σ c_ij.
  Phân phối mẫu âm: P_n(j) = c_j / |C|.
  L_SGNS (3.sgns) = −(1/|C|) Σ_{c_ij>0} c_ij [log σ(z_i^s·z_j^c) + K E_{n~P_n} log σ(−z_i^s·z_n^c)].
  cos_ε0(u,v) = <u,v> / (√(‖u‖²+ε0) √(‖v‖²+ε0))  (3.cos).
  ω_ij = ½(c_ij/c_i + c_ij/c_j), ω̂_ij = ½(1 + cos_ε0(z_i^s, z_j^s))  (3.omega);  L_edge = mean_E (ω − ω̂)².
  ā_ij = |{R : i,j ∈ ant(R)}|/|R|;  b_il = |{R : i ∈ ant(R), out(R)=l}| / |{R : i ∈ ant(R)}|  (3.ab).
  L_A = mean_{P_A} (σ(z_i^s·z_j^s) − ā_ij)²;  L_B^comp = −mean_{V_ant} Σ_l b̃_il log softmax_l(z_i^s·u_l),
        b̃ = (1−ε_b) b + ε_b/n_L;  L_node = λ_A L_A + λ_B L_B^comp.
  Nhúng luật (3.pool): r_k = (α/r) Σ_a z^s_{t_a} + (1−α) z^s_{t_out}.
  Nhúng truy vấn (3.hj, 3.query): h_j = Σ_l μ_l(x_j) z^s_{(j,l)},  q = (1/r) Σ_j h_j.
  s_k = cos_ε0(q, r_k);  s̄_l = τ_c log Σ_{k∈R_l} exp(s_k/τ_c);  p_E = (1−ε) softmax(s̄/τ) + ε/n_L  (3.pE).
  Đóng góp luật (3.attrE): w_{l,k} = softmax_{k∈R_l}(s_k/τ_c);  a_E = w_{ŷ,·}.
  Giáo viên (Định nghĩa 3.18): ŷ_F, e^F = a_F (Định nghĩa 3.attrF), p_F^cal = softmax((log(D+ε0)+β)/T).
  L_rule = mean_i KL(e^F_i ‖ w_{ŷ_F(i),·});  L_inf = mean_i KL(p_F^cal ‖ p_E);  L_pred = −mean_i log p_E(y_i).
  Hàm mục tiêu (3.28): L = λ_S L_SGNS + λ_E L_edge + λ_N L_node + λ_R L_rule + λ_I L_inf + λ_P L_pred + λ_C ‖Θ‖².
  Θ = {z^(s), z^(c)} trên V_tok ∪ {u_l} trên L.

Thuật toán 3.10 (huấn luyện): ước lượng gradient không chệch (lô cặp E⁺ với hệ số |E⁺|/b_p, lô cặp P_A với
hệ số |P_A|/b_p, L_B^comp đầy đủ, thành phần theo mẫu lấy trung bình trên lô), bước gradient có chiếu lên hình cầu
bán kính B_0 = √(L(Θ_0)/λ_C) (Mệnh đề 3.coer, Định lý 3.sgd), tuỳ chọn AMSGrad rồi chiếu, dừng khi ‖∇L‖ ≤ ε_g
(Hệ quả 3.stop) hoặc dừng sớm theo L_pred trên tập xác thực; cuối cùng chọn ngưỡng τ_E trên tập xác thực.
Thuật toán 3.11 (suy diễn): predict_one / explain_one.

Gradient tính bằng thư viện autograd (vi phân tự động trên NumPy).
"""
import time
import math
import warnings
from collections import defaultdict

import numpy as np
import autograd.numpy as anp
from autograd import grad
from autograd.scipy.special import logsumexp

try:
    import config as _C
    _CFG = getattr(_C, "FKGE", None)
except Exception:  # pragma: no cover
    _CFG = None


def _cfg(name, default):
    return getattr(_CFG, name, default) if _CFG is not None else default


# Tên tham số cũ (bản v2) -> tên theo Chương 3, để các kịch bản cũ chạy được.
_LEGACY = {"d": "d_e", "K_neg": "K", "lam_node": "lam_N", "beta_rule": "lam_S", "gamma_inf": "lam_I",
           "delta_pred": "lam_P", "lr": "eta", "epochs": "T_ep", "alpha_pool": "alpha",
           "tau_softmax": "tau", "batch_size": "b"}


def _log_sigmoid(x):
    return -anp.logaddexp(0.0, -x)


def _sigmoid(x):
    return 0.5 * (anp.tanh(0.5 * x) + 1.0)


def _cos(Q, R, eps0):
    num = anp.dot(Q, R.T)
    nq = anp.sqrt(anp.sum(Q * Q, axis=1) + eps0)
    nr = anp.sqrt(anp.sum(R * R, axis=1) + eps0)
    return num / (nq[:, None] * nr[None, :])


def _cos_rows(A, B, eps0):
    return anp.sum(A * B, axis=1) / (anp.sqrt(anp.sum(A * A, axis=1) + eps0) * anp.sqrt(anp.sum(B * B, axis=1) + eps0))


class FKGE:
    def __init__(self, fkg, d_e=None, K=None, alpha=None, tau=None, tau_c=None, eps=None, eps0=None, eps_b=None,
                 lam_S=None, lam_E=None, lam_N=None, lam_A=None, lam_B=None, lam_R=None, lam_I=None, lam_P=None,
                 lam_C=None, eta=None, optimizer=None, eps_g=None, b=None, b_p=None, T_ep=None, patience=None,
                 sigma=None, val_fraction=None, cooc=None, w=None, pooling=None, seed=None, verbose=False,
                 **legacy):
        for k, v in legacy.items():
            if k in _LEGACY:
                setattr(self, "_legacy_" + _LEGACY[k], v)   # ánh xạ tên cũ -> tên Chương 3
            elif k == "weight_decay":
                pass
            else:
                raise TypeError(f"FKGE: tham số không hỗ trợ '{k}'")
        g = lambda name, val, default: (getattr(self, "_legacy_" + name) if hasattr(self, "_legacy_" + name)
                                        else (val if val is not None else _cfg(name, default)))
        self.fkg = fkg
        self.d_e = int(g("d_e", d_e, 32)); self.K = int(g("K", K, 5))
        self.tau = float(g("tau", tau, 0.1)); self.tau_c = float(g("tau_c", tau_c, 0.1))
        self.eps = float(g("eps", eps, 0.05)); self.eps0 = float(g("eps0", eps0, 1e-6))
        self.eps_b = float(g("eps_b", eps_b, 0.05))
        self.lam = {n: float(g("lam_" + n, v, dv)) for n, v, dv in
                    [("S", lam_S, 1.0), ("E", lam_E, 1.0), ("N", lam_N, 1.0), ("A", lam_A, 1.0), ("B", lam_B, 1.0),
                     ("R", lam_R, 1.0), ("I", lam_I, 1.0), ("P", lam_P, 1.0), ("C", lam_C, 1e-3)]}
        if self.lam["C"] <= 0:
            raise ValueError("λ_C phải > 0 (Mệnh đề tính bức; Nhận xét 3.coer)")
        if not (0 < self.eps_b < 1):
            raise ValueError("ε_b phải thuộc (0, 1) (Mệnh đề 3.coer(iii))")
        self.eta = float(g("eta", eta, 0.01)); self.optimizer = g("optimizer", optimizer, "amsgrad")
        self.eps_g = float(g("eps_g", eps_g, 1e-4)); self.b = int(g("b", b, 64)); self.b_p = int(g("b_p", b_p, 256))
        self.T_ep = int(g("T_ep", T_ep, 100)); self.patience = int(g("patience", patience, 10))
        self.sigma = float(g("sigma", sigma, 0.1)); self.val_fraction = float(g("val_fraction", val_fraction, 0.2))
        self.cooc_mode = g("cooc", cooc, "full"); self.window = w if w is not None else _cfg("w", 2)
        if pooling == "mean":
            self.alpha = None   # gộp trung bình đều: α = r/(r+1) (Nhận xét 3.space)
        else:
            self.alpha = float(g("alpha", alpha, 0.7))
        self.seed = int(seed if seed is not None else _cfg("seed", 42))
        self.verbose = verbose
        self.class_tokens = list(fkg.class_tokens)
        self.nL = len(self.class_tokens)
        self.history = defaultdict(list)
        self._prepare()
        rng = np.random.RandomState(self.seed)
        V, d = self.V, self.d_e
        self.params = [rng.normal(0, self.sigma, (V, d)), rng.normal(0, self.sigma, (V, d)),
                       rng.normal(0, self.sigma, (self.nL, d))]
        self.threshold = None
        self.teacher = None
        self.train_time_s = 0.0

    # ======================================================================== chuẩn bị (hằng số)
    def _prepare(self):
        fkg = self.fkg
        self.vocab = list(fkg.vocab); self.tok = {t: i for i, t in enumerate(self.vocab)}; self.V = len(self.vocab)
        self.attrs = sorted({t.rsplit("-", 1)[0] for t in self.vocab if not t.startswith("class-")})
        self.r = len(self.attrs)
        self.label_tok = [self.tok[c] for c in self.class_tokens]
        rules_ant = [[self.tok[t] for t in R["antecedent_tokens"]] for R in fkg.rules]
        rules_out = [self.class_tokens.index(R["consequent_token"]) for R in fkg.rules]
        self.rule_ids = [R.get("id", k) for k, R in enumerate(fkg.rules)]
        self.rule_pos = {rid: k for k, rid in enumerate(self.rule_ids)}
        nR = len(rules_ant); self.nR = nR
        self.rule_lab = np.array(rules_out)
        # nhóm của luật (các luật trùng nhau có cùng nhóm); luật không khai báo nhóm có nhóm riêng âm
        self.rule_group = np.array([R.get("group", -1 - k) for k, R in enumerate(fkg.rules)])
        self.R_l = [np.where(self.rule_lab == l)[0] for l in range(self.nL)]
        # (3.cooc) đồng xuất hiện toàn luật (hoặc theo cửa sổ vị trí, chỉ để đối chứng ở KB5)
        Cm = np.zeros((self.V, self.V))
        for ant, out in zip(rules_ant, rules_out):
            seq = ant + [self.label_tok[out]]
            if self.cooc_mode == "full":
                S = sorted(set(seq))
                for i in S:
                    for j in S:
                        if i != j:
                            Cm[i, j] += 1
            else:
                for p, i in enumerate(seq):
                    for q in range(max(0, p - self.window), min(len(seq), p + self.window + 1)):
                        if q != p and seq[q] != i:
                            Cm[i, seq[q]] += 1
        self.Cm = Cm
        self.c_i = Cm.sum(axis=1); self.C_tot = Cm.sum()
        self.Pn = self.c_i / self.C_tot
        I, J = np.nonzero(Cm)
        self.E_I, self.E_J, self.E_c = I, J, Cm[I, J]
        self.omega = 0.5 * (Cm[I, J] / self.c_i[I] + Cm[I, J] / self.c_i[J])
        # (3.ab) ā và b trên V_ant (token xuất hiện trong tiền đề)
        ant_count = np.zeros(self.V); co_ant = np.zeros((self.V, self.V)); bc = np.zeros((self.V, self.nL))
        for ant, out in zip(rules_ant, rules_out):
            S = sorted(set(ant))
            for i in S:
                ant_count[i] += 1; bc[i, out] += 1
                for j in S:
                    if i != j:
                        co_ant[i, j] += 1
        self.V_ant = np.where(ant_count > 0)[0]
        PA = [(i, j) for i in self.V_ant for j in self.V_ant if i != j]
        self.PA_I = np.array([p[0] for p in PA], dtype=int); self.PA_J = np.array([p[1] for p in PA], dtype=int)
        self.abar = co_ant[self.PA_I, self.PA_J] / nR if len(PA) else np.zeros(0)
        b = bc[self.V_ant] / ant_count[self.V_ant][:, None]
        self.btil = (1 - self.eps_b) * b + self.eps_b / self.nL
        # (3.pool) ma trận gộp luật |R| x V
        Pm = np.zeros((nR, self.V))
        for k, (ant, out) in enumerate(zip(rules_ant, rules_out)):
            a = self.alpha if self.alpha is not None else len(ant) / (len(ant) + 1)
            for i in ant:
                Pm[k, i] += a / len(ant)
            Pm[k, self.label_tok[out]] += 1 - a
        self.Pm = Pm

    def _query_matrix(self, samples):
        """(3.hj)-(3.query): hàng i là hệ số μ_l(x_j)/r của token (j, l)."""
        Q = np.zeros((len(samples), self.V))
        for n, s in enumerate(samples):
            for t, deg in s["membership"].items():
                if deg > 0 and t in self.tok and not t.startswith("class-"):
                    Q[n, self.tok[t]] += deg / self.r
        return Q

    # ======================================================================== các thành phần mất mát
    def _mask(self, samples):
        """G[i, k] = True nếu luật k thuộc cùng nhóm với mẫu i (luật do chính mẫu sinh ra hoặc bản sao của nó).
        Khi cơ sở luật được sinh từ chính các mẫu huấn luyện, các luật này bị loại khỏi suy diễn cho mẫu đó để
        tránh rò rỉ nhãn (tương tự tính chéo theo phần của Định nghĩa 3.18)."""
        g = np.array([s.get("group", -10**9) for s in samples])
        G = g[:, None] == self.rule_group[None, :]
        return G if G.any() else None

    def _scores(self, params, Q, G=None):
        Zs = params[0]
        q = anp.dot(Q, Zs); Rm = anp.dot(self.Pm, Zs)
        S = _cos(q, Rm, self.eps0)
        if G is not None:
            S = S - 1e6 * G
        sbar = []
        for l in range(self.nL):
            idx = self.R_l[l]
            if len(idx) == 0:
                sbar.append(-anp.ones(Q.shape[0]))
            else:
                sbar.append(self.tau_c * logsumexp(S[:, idx] / self.tau_c, axis=1))
        sbar = anp.stack(sbar, axis=1)
        logits = sbar / self.tau
        pE = (1 - self.eps) * anp.exp(logits - logsumexp(logits, axis=1, keepdims=True)) + self.eps / self.nL
        return S, pE

    def _structural(self, params, eI=None, aI=None):
        """λ_S L_SGNS + λ_E L_edge + λ_N(λ_A L_A + λ_B L_B^comp), ước lượng không chệch trên lô cặp
        (eI: chỉ số trong E⁺, aI: chỉ số trong P_A); None nghĩa là dùng toàn bộ."""
        Zs, Zc, U = params
        lam = self.lam; tot = 0.0
        nE = len(self.E_I)
        if eI is None:
            eI = np.arange(nE); se = 1.0
        else:
            se = nE / len(eI)
        I, J, c = self.E_I[eI], self.E_J[eI], self.E_c[eI]
        if lam["S"] > 0:
            x = anp.sum(Zs[I] * Zc[J], axis=1)
            neg = anp.dot(_log_sigmoid(-anp.dot(Zs[I], Zc.T)), self.Pn)
            tot = tot + lam["S"] * se * (-anp.sum(c * (_log_sigmoid(x) + self.K * neg)) / self.C_tot)
        if lam["E"] > 0:
            om_hat = 0.5 * (1 + _cos_rows(Zs[I], Zs[J], self.eps0))
            tot = tot + lam["E"] * se * anp.sum((self.omega[eI] - om_hat) ** 2) / nE
        if lam["N"] > 0:
            nA = len(self.PA_I)
            if lam["A"] > 0 and nA:
                if aI is None:
                    aI = np.arange(nA); sa = 1.0
                else:
                    sa = nA / len(aI)
                pa = _sigmoid(anp.sum(Zs[self.PA_I[aI]] * Zs[self.PA_J[aI]], axis=1))
                tot = tot + lam["N"] * lam["A"] * sa * anp.sum((pa - self.abar[aI]) ** 2) / nA
            if lam["B"] > 0:
                lg = anp.dot(Zs[self.V_ant], U.T)
                logb = lg - logsumexp(lg, axis=1, keepdims=True)
                tot = tot + lam["N"] * lam["B"] * (-anp.sum(self.btil * logb) / len(self.V_ant))
        return tot

    def _instance(self, params, Q, y, pF, eF, yF, G=None):
        """λ_R L_rule + λ_I L_inf + λ_P L_pred, trung bình trên các mẫu của Q."""
        S, pE = self._scores(params, Q, G)
        lam = self.lam; tot = 0.0; n = Q.shape[0]
        logpE = anp.log(pE)
        if lam["P"] > 0:
            tot = tot + lam["P"] * (-anp.sum(logpE[np.arange(n), y]) / n)
        if lam["I"] > 0 and pF is not None:
            tot = tot + lam["I"] * anp.sum(pF * (np.log(pF + 1e-300) - logpE)) / n
        if lam["R"] > 0 and eF is not None:
            acc = 0.0
            for l in range(self.nL):
                rows = np.where(yF == l)[0]; idx = self.R_l[l]
                if len(rows) == 0 or len(idx) == 0:
                    continue
                Sl = S[rows][:, idx] / self.tau_c
                logw = Sl - logsumexp(Sl, axis=1, keepdims=True)
                e = eF[rows][:, idx]
                acc = acc + anp.sum(e * (np.log(np.where(e > 0, e, 1.0)) - logw))
            tot = tot + lam["R"] * acc / n
        return tot

    def _reg(self, params):
        return self.lam["C"] * sum(anp.sum(p * p) for p in params)

    def loss(self, params=None, Q=None, y=None, pF=None, eF=None, yF=None, eI=None, aI=None, G=None):
        """Hàm mục tiêu (3.28) — toàn bộ (mặc định) hoặc ước lượng trên lô."""
        params = self.params if params is None else params
        tot = self._structural(params, eI, aI) + self._reg(params)
        if Q is not None and Q.shape[0] > 0:
            tot = tot + self._instance(params, Q, y, pF, eF, yF, G)
        return tot

    def components(self, params=None, data=None):
        """Giá trị từng thành phần (không nhân trọng số) — để kiểm thử và báo cáo."""
        params = self.params if params is None else params
        saved = dict(self.lam); out = {}
        try:
            for name, keys in [("SGNS", {"S": 1}), ("edge", {"E": 1}), ("A", {"N": 1, "A": 1}), ("B", {"N": 1, "B": 1})]:
                self.lam = {k: 0.0 for k in saved}; self.lam["C"] = saved["C"]; self.lam.update(keys)
                out[name] = float(self._structural(params))
            if data is not None:
                Q, y, pF, eF, yF = data
                for name, keys in [("pred", {"P": 1}), ("inf", {"I": 1}), ("rule", {"R": 1})]:
                    self.lam = {k: 0.0 for k in saved}; self.lam["C"] = saved["C"]; self.lam.update(keys)
                    out[name] = float(self._instance(params, Q, y, pF, eF, yF))
        finally:
            self.lam = saved
        return out

    # ======================================================================== giáo viên (Định nghĩa 3.18)
    def _teacher(self, fisa, samples, val):
        n = len(samples); yF = np.zeros(n, dtype=int); eF = np.zeros((n, self.nR)); Dlog = np.zeros((n, self.nL))
        for i, s in enumerate(samples):
            D, _, matched = fisa.decision_values(s["membership"])
            yF[i] = self.class_tokens.index(fisa._decide(D, matched))
            for rid, a in fisa.attribution(s["membership"]).items():
                eF[i, self.rule_pos[rid]] = a
            Dlog[i] = [math.log(D[c] + 1e-12) for c in self.class_tokens]
        beta = np.zeros(self.nL)
        if self.nL == 2 and fisa.threshold is not None and fisa.decision == "calibrated":
            beta[1] = -fisa.threshold
        # chọn nhiệt độ T cực tiểu log-mất mát trên tập xác thực
        Dv = np.array([[math.log(fisa.decision_values(s["membership"])[0][c] + 1e-12) for c in self.class_tokens]
                       for s in val]) if val else None
        yv = np.array([self.class_tokens.index(s["label"]) for s in val]) if val else None
        best_T, best = 1.0, np.inf
        for T in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
            if Dv is None:
                break
            z = (Dv + beta) / T; p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
            nll = -np.mean(np.log(p[np.arange(len(yv)), yv] + 1e-12))
            if nll < best:
                best, best_T = nll, T
        z = (Dlog + beta) / best_T; pF = np.exp(z - z.max(1, keepdims=True)); pF /= pF.sum(1, keepdims=True)
        self.teacher_T, self.teacher_beta = best_T, beta
        return pF, eF, yF

    # ======================================================================== Thuật toán 3.10
    def _teacher_from_arrays(self, td, val_labels):
        """Giáo viên tính sẵn bên ngoài (ví dụ tính chéo theo phần): td = {Dlog, eF, yF, Dlog_val, beta}."""
        beta = np.asarray(td.get("beta", np.zeros(self.nL)), dtype=float)
        best_T, best = 1.0, np.inf
        Dv = td.get("Dlog_val")
        if Dv is not None and len(Dv):
            yv = np.array([self.class_tokens.index(l) for l in val_labels])
            for T in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
                z = (Dv + beta) / T; p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
                nll = -np.mean(np.log(p[np.arange(len(yv)), yv] + 1e-12))
                if nll < best:
                    best, best_T = nll, T
        z = (np.asarray(td["Dlog"]) + beta) / best_T
        pF = np.exp(z - z.max(1, keepdims=True)); pF /= pF.sum(1, keepdims=True)
        self.teacher_T, self.teacher_beta = best_T, beta
        return pF, np.asarray(td["eF"]), np.asarray(td["yF"], dtype=int)

    def fit(self, fisa_model=None, train_samples=None, val_samples=None, batch_size=None, teacher_data=None):
        t0 = time.time()
        rng = np.random.RandomState(self.seed)
        if batch_size:
            self.b = int(batch_size)
        train = list(train_samples or [])
        if val_samples is None and train:
            perm = rng.permutation(len(train)); nv = max(1, int(round(self.val_fraction * len(train))))
            val = [train[i] for i in perm[:nv]]; train = [train[i] for i in perm[nv:]]
        else:
            val = list(val_samples or [])
        self.teacher = fisa_model
        if fisa_model is None and teacher_data is None and (self.lam["I"] > 0 or self.lam["R"] > 0):
            warnings.warn("FKG-E: không có FISA giáo viên — bỏ L_inf và L_rule (λ_I = λ_R = 0).")
            self.lam["I"] = self.lam["R"] = 0.0
        Q = self._query_matrix(train); y = np.array([self.class_tokens.index(s["label"]) for s in train], dtype=int)
        pF = eF = yF = None
        if teacher_data is not None:
            pF, eF, yF = self._teacher_from_arrays(teacher_data, [s["label"] for s in val])
        elif fisa_model is not None and train:
            pF, eF, yF = self._teacher(fisa_model, train, val)
        G = self._mask(train) if train else None
        Gv = self._mask(val) if val else None
        Qv = self._query_matrix(val) if val else None
        yv = np.array([self.class_tokens.index(s["label"]) for s in val], dtype=int) if val else None
        self._train_data = (Q, y, pF, eF, yF)
        self._train_mask = G

        full = lambda p: self.loss(p, Q, y, pF, eF, yF, G=G)
        L0 = float(full(self.params))
        self.B0 = math.sqrt(L0 / self.lam["C"])                    # Mệnh đề 3.coer
        g_batch = grad(lambda p, Qb, yb, pFb, eFb, yFb, eI, aI, Gb: self.loss(p, Qb, yb, pFb, eFb, yFb, eI, aI, Gb))
        g_full = grad(full)
        N = len(train); B_ep = max(1, int(math.ceil(N / self.b))) if N else 1
        m = [np.zeros_like(p) for p in self.params]; v = [np.zeros_like(p) for p in self.params]
        vhat = [np.zeros_like(p) for p in self.params]; b1, b2 = 0.9, 0.999; step = 0
        best_val, best_params, wait = np.inf, [p.copy() for p in self.params], 0
        nE, nA = len(self.E_I), len(self.PA_I)
        self.stop_reason = "T_ep"
        for ep in range(self.T_ep):
            perm = rng.permutation(N) if N else np.array([], dtype=int)
            for t in range(B_ep):
                bi = perm[t * self.b:(t + 1) * self.b]
                eI = rng.choice(nE, size=min(self.b_p, nE), replace=False) if nE else None
                aI = rng.choice(nA, size=min(self.b_p, nA), replace=False) if nA else None
                sub = lambda a: None if a is None else a[bi]
                gr = g_batch(self.params, Q[bi], y[bi], sub(pF), sub(eF), sub(yF), eI, aI, sub(G))
                step += 1
                if self.optimizer == "amsgrad":
                    new = []
                    for k in range(3):
                        m[k] = b1 * m[k] + (1 - b1) * gr[k]; v[k] = b2 * v[k] + (1 - b2) * gr[k] ** 2
                        vhat[k] = np.maximum(vhat[k], v[k])
                        new.append(self.params[k] - self.eta * m[k] / (np.sqrt(vhat[k]) + 1e-8))
                else:
                    new = [self.params[k] - self.eta * gr[k] for k in range(3)]
                nrm = math.sqrt(sum(float(np.sum(p * p)) for p in new))   # chiếu lên hình cầu bán kính B0
                if nrm > self.B0:
                    new = [p * (self.B0 / nrm) for p in new]
                self.params = new
            gf = g_full(self.params)
            gnorm = math.sqrt(sum(float(np.sum(gg * gg)) for gg in gf))
            self.history["grad_norm"].append(gnorm); self.history["loss"].append(float(full(self.params)))
            if Qv is not None and len(val):
                _, pEv = self._scores(self.params, Qv, Gv)
                vl = float(-np.mean(np.log(pEv[np.arange(len(yv)), yv])))
                self.history["val_pred"].append(vl)
                if vl < best_val - 1e-9:
                    best_val, best_params, wait = vl, [p.copy() for p in self.params], 0
                else:
                    wait += 1
            if gnorm <= self.eps_g:                                   # Hệ quả 3.stop
                self.stop_reason = "grad"; break
            if Qv is not None and wait >= self.patience:
                self.stop_reason = "early"; break
        if Qv is not None and len(val) and best_params is not None:
            self.params = best_params
        self.epochs_run = ep + 1
        self._finalize()
        if self.nL == 2 and Qv is not None and len(val):
            self._calibrate(Qv, yv, Gv)
        self.train_time_s = time.time() - t0
        return self

    def _finalize(self):
        Zs = self.params[0]
        self.rule_emb = self.Pm @ Zs

    def _logratio(self, pE):
        return np.log(pE[:, 1]) - np.log(pE[:, 0])

    def _calibrate(self, Qv, yv, Gv=None):
        """τ_E: ngưỡng trên log p_E(1) − log p_E(0) cực đại độ chính xác cân bằng trên tập xác thực."""
        _, pE = self._infer(Qv, Gv); s = self._logratio(pE)
        if len(np.unique(yv)) < 2:
            self.threshold = 0.0; return
        cand = np.unique(np.concatenate([[0.0], np.quantile(s, np.linspace(0.01, 0.99, 197))]))
        best, bt = -1, 0.0
        for t in cand:
            p = (s > t).astype(int); v = 0.5 * (np.mean(p[yv == 1] == 1) + np.mean(p[yv == 0] == 0))
            if v > best + 1e-12:
                best, bt = v, float(t)
        self.threshold = bt

    # ======================================================================== Thuật toán 3.11
    def _infer(self, Q, G=None):
        """Thuật toán 3.11: q -> s_k (một phép nhân |R| x d_e với r_k tính sẵn) -> s̄_l -> p_E."""
        q = Q @ self.params[0]
        S = np.asarray(_cos(q, self.rule_emb, self.eps0))
        if G is not None:
            S = S - 1e6 * G
        sbar = np.full((Q.shape[0], self.nL), -1.0)
        for l in range(self.nL):
            idx = self.R_l[l]
            if len(idx):
                z = S[:, idx] / self.tau_c; mx = z.max(axis=1, keepdims=True)
                sbar[:, l] = self.tau_c * (mx[:, 0] + np.log(np.exp(z - mx).sum(axis=1)))
        z = sbar / self.tau; z = z - z.max(axis=1, keepdims=True)
        pE = (1 - self.eps) * np.exp(z) / np.exp(z).sum(axis=1, keepdims=True) + self.eps / self.nL
        return S, pE

    def _decide(self, pE):
        if self.nL == 2 and self.threshold is not None:
            return np.where(self._logratio(pE) > self.threshold, 1, 0)
        return np.argmax(pE, axis=1)

    def predict_one(self, membership):
        t0 = time.time()
        Q = self._query_matrix([{"membership": membership}])
        _, pE = self._infer(Q)
        yh = self.class_tokens[int(self._decide(pE)[0])]
        return {c: float(pE[0, i]) for i, c in enumerate(self.class_tokens)}, yh, time.time() - t0

    def explain_one(self, membership, K_act=5):
        """a_E (Định nghĩa 3.attrE) và K_act luật có đóng góp lớn nhất."""
        Q = self._query_matrix([{"membership": membership}])
        S, pE = self._infer(Q)
        l = int(self._decide(pE)[0]); idx = self.R_l[l]
        z = S[0, idx] / self.tau_c; w = np.exp(z - z.max()); w /= w.sum()
        aE = {self.rule_ids[k]: float(wk) for k, wk in zip(idx, w)}
        top = sorted(aE.items(), key=lambda kv: -kv[1])[:K_act]
        return aE, top

    def evaluate(self, samples, fisa_model=None):
        fisa = fisa_model or self.teacher
        t0 = time.time()
        Q = self._query_matrix(samples)
        y = np.array([self.class_tokens.index(s["label"]) for s in samples])
        times = []
        for n in range(len(samples)):                                # đo thời gian theo từng truy vấn
            t1 = time.time(); self._infer(Q[n:n + 1]); times.append(time.time() - t1)
        S, pE = self._infer(Q); yh = self._decide(pE)
        from models.fisa import _macro_f1, _balanced_acc, _auc
        yt = [self.class_tokens[i] for i in y]; yp = [self.class_tokens[i] for i in yh]
        out = {"accuracy": float(np.mean(yh == y)), "f1_macro": _macro_f1(yt, yp, self.class_tokens),
               "balanced_accuracy": _balanced_acc(yt, yp, self.class_tokens),
               "log_loss": float(-np.mean(np.log(pE[np.arange(len(y)), y]))),
               "avg_time_per_query_ms": float(np.mean(times) * 1000), "total_time_s": time.time() - t0,
               "threshold": self.threshold, "y_true": yt, "y_pred": yp}
        if self.nL == 2:
            out["auc"] = _auc(y.tolist(), self._logratio(pE).tolist())
        if fisa is not None:                                         # (R2), (R3)
            agree, dev = [], []
            for n, s in enumerate(samples):
                D, _, matched = fisa.decision_values(s["membership"])
                yF = fisa._decide(D, matched); agree.append(yF == yp[n])
                aF = fisa.attribution(s["membership"])
                l = int(yh[n]); idx = self.R_l[l]; z = S[n, idx] / self.tau_c
                w = np.exp(z - z.max()); w /= w.sum()
                aE = {self.rule_ids[k]: wk for k, wk in zip(idx, w)}
                keys = set(aE) | set(aF)
                dev.append(0.5 * sum(abs(aE.get(k, 0.0) - aF.get(k, 0.0)) for k in keys))
            out["agreement_fisa"] = float(np.mean(agree)); out["dev_rule"] = float(np.mean(dev))
        return out


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.fkg_io import generate_synthetic_fkg, generate_synthetic_test_samples
    from models.fisa import FISA
    fkg = generate_synthetic_fkg(n_rules=200, seed=1)
    train = generate_synthetic_test_samples(fkg, n_samples=200, seed=10)
    test = generate_synthetic_test_samples(fkg, n_samples=150, seed=11)
    fisa = FISA(fkg).fit(val_samples=train)
    print("FISA :", {k: round(v, 4) for k, v in fisa.evaluate(test).items() if k in ("accuracy", "balanced_accuracy", "auc")})
    m = FKGE(fkg, T_ep=30).fit(fisa_model=fisa, train_samples=train)
    r = m.evaluate(test)
    print("FKG-E:", {k: round(v, 4) for k, v in r.items() if isinstance(v, float)},
          "dừng:", m.stop_reason, "epoch:", m.epochs_run)
