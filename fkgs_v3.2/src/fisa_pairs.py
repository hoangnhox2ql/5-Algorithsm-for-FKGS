"""
src/fisa_pairs.py — Suy diễn FISA theo đúng mô hình FKG-Pairs của bài báo gốc
(Multimedia Tools and Applications 81:26505–26534, 2022) và notebook FKG.ipynb.

Ký hiệu: tập luật R gồm n luật, m thuộc tính; k = kích thước nút (số thuộc tính trong
một nút, "FKG-Pairs k"; notebook gốc dùng k = 3).

  (7)  A^t_{c}  = |{s ∈ R : s khớp t trên mọi thuộc tính của c}| / |R|,   c là bộ k+1 thuộc tính
  (8)  B^t_{c'} = (Σ_c A^t_c) · min_{i ∈ c'} M^t_i,                         c' là nút gồm k thuộc tính
       với M^t_i = |{s ∈ R : s_i = t_i và nhãn(s) = nhãn(t)}| / |R|
  (10) C̃_{c',l}(x) = Σ_{t ∈ R : nhãn(t) = l, t khớp x trên c'} B^t_{c'}
  (11) D̃_l(x) = max_{c'} C̃_{c',l}(x) + min_{c'} C̃_{c',l}(x)     (nút không khớp có C̃ = 0)
  (12) nhãn = argmax_l D̃_l(x)

So với notebook gốc, mô-đun này:
  * giữ nguyên công thức (7), (8), (10), (11) và cách gộp CÓ ĐIỀU KIỆN THEO NHÃN ở (10);
  * sửa lỗi bỏ sót luật cuối (`range(row-1)`) — mọi luật đều tham gia;
  * quy tắc quyết định mặc định là (12) argmax như bài báo; quy tắc D0 > 9·D1 của notebook
    được giữ dưới tên "ratio9" chỉ để đối chiếu; quy tắc "calibrated" (luận án, v3.1) chọn
    ngưỡng trên s = log D1 − log D0 bằng tập xác thực;
  * quy tắc dự phòng khi truy vấn không khớp nút nào (D̃_l = 0 với mọi l): dự đoán nhãn có
    nhiều luật nhất, điểm s = 0, và được đếm trong n_unseen;
  * hỗ trợ k bất kỳ (k = 1: FKG-Pairs1, dạng một cặp; k = 3: notebook gốc) và nhiều nhãn.

Tính toán được vector hoá bằng mã cơ số (radix) + np.unique; kết quả trùng khớp với cài đặt
vòng lặp trực tiếp của các công thức trên (tests/test_fisa_pairs.py).
"""
from itertools import combinations
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

DECISIONS = ("argmax", "ratio9", "calibrated")


class FISAPairs:
    EPS = 1e-12

    def __init__(self, rules, node_size=None):
        if rules is None or len(rules) == 0:
            raise ValueError("FISA: tập luật rỗng")
        self.rules = rules
        self.n_rules = len(rules)
        self.n_attr = len(rules[0]) - 1
        if self.n_attr <= 0:
            raise ValueError("FISA: luật không có tiền đề")
        k = int(node_size if node_size is not None else getattr(C.FISA, "NODE_SIZE", 3))
        if k < 1:
            raise ValueError("FISA: kích thước nút k phải >= 1")
        # Khi số thuộc tính ít hơn k (hoặc k+1), dùng toàn bộ thuộc tính.
        self.k = min(k, self.n_attr)
        self.k_a = min(self.k + 1, self.n_attr)
        self.classes = sorted(set(r[-1] for r in rules))
        self.threshold = None
        self._fitted = False

    # ------------------------------------------------------------------ mã hoá
    def _encode(self):
        n, m = self.n_rules, self.n_attr
        self.luts = []
        codes = np.zeros((n, m), dtype=np.int64)
        levels = np.zeros(m, dtype=np.int64)
        for j in range(m):
            vals = [str(r[j]) for r in self.rules]
            uniq = sorted(set(vals))
            lut = {v: i for i, v in enumerate(uniq)}
            self.luts.append(lut)
            codes[:, j] = [lut[v] for v in vals]
            levels[j] = len(uniq)
        self.codes, self.levels = codes, levels
        self.class_index = {c: i for i, c in enumerate(self.classes)}
        self.lab_idx = np.array([self.class_index[r[-1]] for r in self.rules], dtype=np.int64)

    def _radix(self, codes, idx):
        out = np.zeros(codes.shape[0], dtype=np.int64)
        for j in idx:
            out = out * self.levels[j] + codes[:, j]
        return out

    def encode_queries(self, queries):
        """Mã hoá truy vấn theo bảng mã học từ tập luật; giá trị chưa thấy -> -1."""
        Q = np.full((len(queries), self.n_attr), -1, dtype=np.int64)
        for qi, q in enumerate(queries):
            for j in range(self.n_attr):
                Q[qi, j] = self.luts[j].get(str(q[j]), -1)
        return Q

    # ------------------------------------------------------------------ xây đồ thị
    def fit(self):
        self._encode()
        n, m = self.n_rules, self.n_attr
        # (7) tổng các trọng số A^t trên mọi bộ k+1 thuộc tính
        sumA = np.zeros(n)
        for c in combinations(range(m), self.k_a):
            _, inv, cnt = np.unique(self._radix(self.codes, c), return_inverse=True, return_counts=True)
            sumA += cnt[inv] / n
        # M^t_i: cùng giá trị thuộc tính i và cùng nhãn với luật t
        nL = len(self.classes)
        M = np.zeros((n, m))
        for i in range(m):
            key = self.codes[:, i] * nL + self.lab_idx
            _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
            M[:, i] = cnt[inv] / n
        self.sumA, self.M = sumA, M
        # (8) B^t_{c'} và (10) bảng C̃ theo nút, giá trị nút và nhãn
        self.nodes = list(combinations(range(m), self.k))
        self.tables = []
        for c in self.nodes:
            B = sumA * M[:, list(c)].min(axis=1)
            key = self._radix(self.codes, c)
            uniq, inv = np.unique(key, return_inverse=True)
            T = np.zeros((len(uniq), nL))
            np.add.at(T, (inv, self.lab_idx), B)
            self.tables.append((uniq, T))
        counts = np.bincount(self.lab_idx, minlength=nL)
        self.fallback_label = self.classes[int(np.argmax(counts))]
        self._fitted = True
        return self

    # ------------------------------------------------------------------ suy diễn
    def decision_matrix(self, queries):
        """Ma trận D̃ (số truy vấn x số nhãn) theo (10)-(11), và cờ 'không khớp nút nào'."""
        if not self._fitted:
            raise RuntimeError("FISA: chưa gọi fit()")
        Q = self.encode_queries(queries)
        nq, nL = len(queries), len(self.classes)
        mx = np.full((nq, nL), -np.inf)
        mn = np.full((nq, nL), np.inf)
        any_match = np.zeros(nq, dtype=bool)
        for c, (uniq, T) in zip(self.nodes, self.tables):
            valid = np.all(Q[:, list(c)] >= 0, axis=1)
            vals = np.zeros((nq, nL))
            if valid.any():
                key = self._radix(np.where(Q < 0, 0, Q), c)
                pos = np.searchsorted(uniq, key)
                pos_c = np.minimum(pos, len(uniq) - 1)
                hit = valid & (uniq[pos_c] == key)
                vals[hit] = T[pos_c[hit]]
                any_match |= hit
            mx = np.maximum(mx, vals)
            mn = np.minimum(mn, vals)
        D = mx + mn
        return D, ~any_match

    def decision_values(self, query):
        D, unseen = self.decision_matrix([query])
        if unseen[0]:
            return None
        return {c: float(D[0, i]) for i, c in enumerate(self.classes)}

    def _argmax(self, D):
        # (12); khi hoà chọn nhãn nhỏ nhất
        return np.array([self.classes[int(np.argmax(row))] for row in D])

    def _logratio(self, D):
        if len(self.classes) != 2:
            raise ValueError("Điểm log-tỉ số chỉ xác định cho bài toán hai lớp")
        return np.log(D[:, 1] + self.EPS) - np.log(D[:, 0] + self.EPS)

    def predict_one(self, query):
        D = self.decision_values(query)
        if D is None:
            return None, {}
        return max(self.classes, key=lambda l: (D[l], -self.class_index[l])), D

    def logratio_one(self, query):
        D, unseen = self.decision_matrix([query])
        if unseen[0] or len(self.classes) < 2:
            return 0.0
        return float(self._logratio(D)[0])

    def score_one(self, query):
        D = self.decision_values(query)
        if D is None or len(self.classes) < 2:
            return 0.0
        return float(D[self.classes[1]] - D[self.classes[0]])

    def predict_proba_one(self, query, temperature=1.0):
        y_hat, D = self.predict_one(query)
        if not D:
            return {c: 1.0 / len(self.classes) for c in self.classes}, y_hat
        v = np.array([D[c] for c in self.classes]) / max(temperature, 1e-6)
        e = np.exp(v - v.max())
        return dict(zip(self.classes, e / e.sum())), y_hat

    def calibrate(self, rules_val, criterion="bacc"):
        """Chọn ngưỡng tau trên s = log D1 − log D0 bằng tập xác thực (v3.1)."""
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        D, unseen = self.decision_matrix([r[:-1] for r in rules_val])
        s = np.where(unseen, 0.0, self._logratio(D))
        y = np.array([r[-1] for r in rules_val])
        if len(np.unique(y)) < 2:
            self.threshold = 0.0
            return self.threshold
        cand = np.unique(np.concatenate([[0.0], np.quantile(s, np.linspace(0.01, 0.99, 197))]))
        best, bt = -1.0, 0.0
        for t in cand:
            p = (s > t).astype(int)
            v = accuracy_score(y, p) if criterion == "acc" else balanced_accuracy_score(y, p)
            if v > best + 1e-12:
                best, bt = v, float(t)
        self.threshold = bt
        return bt

    def evaluate(self, rules_test, decision=None):
        """Trả dict: y, pred (theo quy tắc quyết định cấu hình), pred_argmax (12), pred_ratio9
        (quy tắc notebook), score (s = log D1 − log D0), D, n_unseen, threshold."""
        decision = decision or getattr(C.FISA, "DECISION", "argmax")
        if decision not in DECISIONS:
            raise ValueError(f"decision phải thuộc {DECISIONS}")
        y = np.array([r[-1] for r in rules_test])
        D, unseen = self.decision_matrix([r[:-1] for r in rules_test])
        pa = self._argmax(D)
        pa[unseen] = self.fallback_label
        out = dict(y=y, pred_argmax=pa, D=D, n_unseen=int(unseen.sum()), threshold=self.threshold)
        if len(self.classes) == 2:
            s = np.where(unseen, 0.0, self._logratio(D))
            r9 = np.where(D[:, 0] > 9 * D[:, 1], self.classes[0], self.classes[1])
            r9[unseen] = self.fallback_label
            out.update(score=s, pred_ratio9=r9)
            if decision == "calibrated" and self.threshold is not None:
                pred = np.where(s > self.threshold, self.classes[1], self.classes[0])
            elif decision == "ratio9":
                pred = r9
            else:
                pred = pa
        else:
            out.update(score=D.max(axis=1), pred_ratio9=pa)
            pred = pa
        out["pred"] = pred
        return out
