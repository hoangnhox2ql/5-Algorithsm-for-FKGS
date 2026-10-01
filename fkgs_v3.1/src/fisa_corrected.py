"""
src/fisa_corrected.py — Suy diễn FISA (v3) dùng cho độ đo downstream.

  A^t_{jk} = |{r: r_j = t_j và r_k = t_k}| / |R|             (đồng thuộc cặp)
  S^t_i    = tổng A^t_{jk} trên miền cặp theo PAIR_SCOPE:
               "global"   : mọi cặp j<k            (công thức (1.17), ví dụ Ch.1)
               "adjacent" : mọi cặp chứa i (j != i)
               "upper"    : j > i                   (cách cài v2)
  W_{i,v,l} = (tổng_{t: t_i = v} S^t_i) * |{t: t_i = v, nhãn l}| / |R|
  C_{il}(x) = W_{i, x_i, l};  D_l(x) = min_i C_{il}(x) + max_i C_{il}(x)
  dự đoán  = argmax_l D_l(x);  điểm cho AUC = D_1 - D_0 (lề), không bão hoà.

SỬA SO VỚI v2 (QA-02): v2 chỉ cộng j>i nên thuộc tính cuối luôn có trọng số 0
và kết quả phụ thuộc thứ tự cột. v3 mặc định "global" theo đúng cách đọc
(1.17) và ví dụ tính tay ở Chương 1 (một luật 3 thuộc tính, mọi A=1 cho
W=3, D=6). Hai quy ước còn lại giữ để phân tích độ nhạy. CẦN người chịu trách
nhiệm khoa học xác nhận lại quy ước chuẩn (xem CHANGELOG_v3.md).

Chính sách truy vấn: thuộc tính có giá trị chưa từng thấy trong tập luật bị
bỏ khỏi min/max; nếu mọi thuộc tính đều chưa thấy, không dự đoán được
(trả None, {}); điểm lề khi đó = 0 và được đếm trong `n_unseen`.
"""
import numpy as np
from itertools import combinations
from collections import defaultdict

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C

PAIR_SCOPES = ("global", "adjacent", "upper")


class FISACorrected:
    def __init__(self, rules, pair_scope=None):
        if rules is None or len(rules) == 0:
            raise ValueError("FISA: tập luật rỗng")
        self.pair_scope = pair_scope or C.FISA.PAIR_SCOPE
        if self.pair_scope not in PAIR_SCOPES:
            raise ValueError(f"pair_scope phải thuộc {PAIR_SCOPES}")
        self.rules = rules
        self.n_rules = len(rules)
        self.n_attr = len(rules[0]) - 1
        if self.n_attr <= 0:
            raise ValueError("FISA: luật không có tiền đề")
        self.classes = sorted(set(r[-1] for r in rules))
        self.W = None

    def _codes(self):
        n, m = self.n_rules, self.n_attr
        codes = np.zeros((n, m), dtype=np.int64)
        levels = np.zeros(m, dtype=np.int64)
        for j in range(m):
            _, inv = np.unique(np.array([str(r[j]) for r in self.rules]), return_inverse=True)
            codes[:, j] = inv
            levels[j] = inv.max() + 1
        return codes, levels

    def _pair_sums(self):
        """Ma trận S (n x m): S[t, i] theo PAIR_SCOPE."""
        n, m = self.n_rules, self.n_attr
        codes, levels = self._codes()
        S = np.zeros((n, m))
        total = np.zeros(n)
        for (i, j) in combinations(range(m), 2):
            key = codes[:, i] * levels[j] + codes[:, j]
            _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
            a = cnt[inv] / n
            total += a
            if self.pair_scope == "upper":
                S[:, i] += a
            elif self.pair_scope == "adjacent":
                S[:, i] += a
                S[:, j] += a
        if self.pair_scope == "global":
            S[:] = total[:, None]
        return S

    def fit(self):
        n, m = self.n_rules, self.n_attr
        S = self._pair_sums()
        labels = np.array([r[-1] for r in self.rules])
        W = defaultdict(dict)
        for i in range(m):
            vals = np.array([str(r[i]) for r in self.rules])
            for v in np.unique(vals):
                mask = vals == v
                s = S[mask, i].sum()
                W[i][v] = {l: s * np.sum(mask & (labels == l)) / n for l in self.classes}
        self.W = W
        return self

    def decision_values(self, query):
        D_parts = [self.W[i][str(query[i])] for i in range(self.n_attr)
                   if str(query[i]) in self.W[i]]
        if not D_parts:
            return None
        return {l: min(p[l] for p in D_parts) + max(p[l] for p in D_parts) for l in self.classes}

    def predict_one(self, query):
        D = self.decision_values(query)
        if D is None:
            return None, {}
        return max(self.classes, key=lambda l: (D[l], -l)), D

    EPS = 1e-12

    def logratio_one(self, query):
        """Điểm s = log D1 - log D0. Quy tắc D0 > k*D1 của FISA gốc tương đương
        s < -log k; argmax tương ứng k = 1 (ngưỡng 0)."""
        D = self.decision_values(query)
        if D is None or len(self.classes) < 2:
            return 0.0
        return float(np.log(D.get(1, 0.0) + self.EPS) - np.log(D.get(0, 0.0) + self.EPS))

    def calibrate(self, rules_val, criterion="bacc"):
        """v3.1: chọn ngưỡng tau trên s bằng tập xác thực. D_l chứa tần suất
        |Val(P_i)->l|/|R| nên thiên về lớp đông; tau = log k bù lại tỉ lệ tiên nghiệm."""
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        s = np.array([self.logratio_one(r[:-1]) for r in rules_val])
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

    def score_one(self, query):
        """Lề D1 - D0 (bài toán hai lớp); 0 nếu không suy diễn được."""
        D = self.decision_values(query)
        if D is None or len(self.classes) < 2:
            return 0.0
        return float(D.get(1, 0.0) - D.get(0, 0.0))

    def predict_proba_one(self, query, temperature=1.0):
        y_hat, D = self.predict_one(query)
        if not D:
            return {c: 1.0 / len(self.classes) for c in self.classes}, y_hat
        v = np.array([D[c] for c in self.classes]) / max(temperature, 1e-6)
        e = np.exp(v - v.max())
        return dict(zip(self.classes, e / e.sum())), y_hat

    def evaluate(self, rules_test):
        """Trả dict: y, pred, score, n_unseen."""
        y, pred, pred_am, score, unseen = [], [], [], [], 0
        tau = getattr(self, "threshold", None)
        for r in rules_test:
            q = r[:-1]
            D = self.decision_values(q)
            y.append(r[-1])
            if D is None:
                unseen += 1
                s = 0.0
                am = self.classes[0]
            else:
                s = float(np.log(D.get(1, 0.0) + self.EPS) - np.log(D.get(0, 0.0) + self.EPS))
                am = max(self.classes, key=lambda l: (D[l], -l))
            score.append(s)
            pred_am.append(am)
            pred.append(int(s > tau) if tau is not None else am)
        return dict(y=np.array(y), pred=np.array(pred), pred_argmax=np.array(pred_am),
                    score=np.array(score), n_unseen=unseen, threshold=tau)
