"""
src/fisa_corrected.py — FISA ĐÃ SỬA đúng công thức (1.17)-(1.20) đã hiệu
chỉnh và xác nhận ở Chương 3 (khác với src/fkg_original.py — vốn cố tình
GIỮ NGUYÊN cấu trúc tổ hợp 4/3 thuộc tính và 3 lỗi của FKG.ipynb chỉ để đối
chiếu số học với bản vector hoá).

    A^t_ij = |{đồng thuộc(i,j) trong luật t}| / |R|                (3.56)
    B^t_il = (sum_{i<j} A^t_ij) * |Val(P_i)->l| / |R|               (3.9 = 1.17)
    W_{i,v,l} = sum_{t: Val_t(P_i)=v} B^t_il                        (3.67, bảng 3 chiều)
    C_il(x) = W_{i, x_i, l}                                         (3.68, TRA CỨU)
    D_l(x) = min_i C_il(x) + max_i C_il(x)                          (3.69 = 1.19)
    y_hat = argmax_l D_l(x)                                          (3.70 = 1.20)

File này dùng CHO THẬT trong AUC downstream của E2/E3/E4 — không phải để
đối chiếu số học (đó là vai trò của fkg_original.py/fkg_fast.py).

⚠️ HẠN CHẾ ĐÃ BIẾT (quyết định GIỮ NGUYÊN công thức gốc, không sửa —
xem thảo luận và lựa chọn "Hướng A"): số hạng |Val(P_i)->l| trong (1.17)
đếm số luật TUYỆT ĐỐI có giá trị v tại thuộc tính i và nhãn l, CHIA CHO
|R| TỔNG THỂ (không chia riêng theo |R_l| của từng lớp). Trên dữ liệu MẤT
CÂN BẰNG LỚP mạnh (ví dụ tỉ lệ 3:1), số hạng này cho lớp đa số lớn hơn một
cách HỆ THỐNG tại MỌI giá trị thuộc tính, khiến D_l của lớp đa số luôn
thắng bất kể nội dung truy vấn — mô hình "sụp" về việc luôn đoán lớp đa số
(AUC downstream tiến về 0.5). Đây là ĐẶC ĐIỂM CỦA CHÍNH CÔNG THỨC (1.17)
trên dữ liệu lệch lớp, KHÔNG PHẢI lỗi cài đặt. `warn_if_fisa_collapsed()`
dưới đây tự động phát hiện và cảnh báo hiện tượng này mỗi khi xảy ra, để
không vô tình báo cáo AUC gây hiểu lầm mà không có chú thích đi kèm.
"""
import numpy as np
from itertools import combinations as _combinations
from collections import defaultdict, Counter


class FISACorrected:
    def __init__(self, rules):
        """`rules`: list các luật dạng list[str/int], cột cuối là nhãn
        (0/1), giống định dạng base trong fkg_original.py."""
        self.rules = rules
        self.n_rules = len(rules)
        self.n_attr = len(rules[0]) - 1
        self.classes = sorted(set(r[-1] for r in rules))
        self.W = None   # W[i][v][l] -> float

    # ---------------- Xây dựng A, B, W ----------------
    def _encode_columns(self):
        """Mã hoá mỗi cột thuộc tính thành số nguyên 0..K-1, dùng cho
        radix-encoding (tăng tốc đếm đồng thuộc bằng np.unique thay vòng
        lặp O(n) lồng nhau)."""
        n, m = self.n_rules, self.n_attr
        codes = np.zeros((n, m), dtype=np.int64)
        n_levels = np.zeros(m, dtype=np.int64)
        for j in range(m):
            col_vals = [r[j] for r in self.rules]
            uniq = sorted(set(col_vals))
            lut = {v: k for k, v in enumerate(uniq)}
            n_levels[j] = len(uniq)
            for t in range(n):
                codes[t, j] = lut[col_vals[t]]
        return codes, n_levels

    def _calculate_A_pairwise(self):
        """A^t_ij (Công thức 3.56): với mỗi cặp (i,j), i<j, và mỗi luật t,
        tỉ lệ số luật đồng thuộc cả (i,j) với luật t -- vector hoá bằng
        radix-encoding + np.unique (tránh vòng lặp O(n) lồng nhau)."""
        n, m = self.n_rules, self.n_attr
        codes, n_levels = self._encode_columns()
        pairs = list(_combinations(range(m), 2))
        A = np.zeros((n, len(pairs)))
        for p_idx, (i, j) in enumerate(pairs):
            radix = codes[:, i] * n_levels[j] + codes[:, j]
            _, inverse, counts = np.unique(radix, return_inverse=True, return_counts=True)
            A[:, p_idx] = counts[inverse] / n
        self._pairs = pairs
        return A

    def fit(self):
        n, m = self.n_rules, self.n_attr
        A = self._calculate_A_pairwise()
        pairs = self._pairs
        labels = [r[-1] for r in self.rules]

        # sum_{i<j} A^t_ij THEO TỪNG THUỘC TÍNH i cố định (chỉ cộng các cặp
        # có i là chỉ số NHỎ trong pair, đúng \sum_{i<j} A^t_ij của (1.17))
        sumA_per_attr = np.zeros((n, m))  # sumA_per_attr[t, i] = sum_{j>i} A^t_{ij}
        for p_idx, (i, j) in enumerate(pairs):
            sumA_per_attr[:, i] += A[:, p_idx]

        # W[i][v][l] = sum_{t: Val_t(P_i)=v} B^t_il, với
        # B^t_il = sumA_per_attr[t,i] * |{t': Val_t'(P_i)=v, nhãn=l}| / n
        # -- QUAN TRỌNG: với MỌI t có Val_t(P_i)=v, số hạng thứ hai
        # |Val(P_i)->l|/n LÀ HẰNG SỐ (không phụ thuộc t, chỉ phụ thuộc v,l),
        # nên W[i][v][l] = (|{t: Val_t(P_i)=v, nhãn=l}|/n) * sum_{t: Val_t(P_i)=v} sumA_per_attr[t,i]
        # -- vector hoá bằng nhóm theo (i,v) một lần, KHÔNG cần vòng lặp t2 lồng.
        W = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        for i in range(m):
            col_i = [r[i] for r in self.rules]
            # Nhóm chỉ số luật theo giá trị v tại thuộc tính i
            idx_by_value = defaultdict(list)
            for t in range(n):
                idx_by_value[col_i[t]].append(t)

            for v, idx_list in idx_by_value.items():
                sum_sumA = sum(sumA_per_attr[t, i] for t in idx_list)  # sum_{t: Val_t(P_i)=v} sumA_per_attr[t,i]
                # |{t: Val_t(P_i)=v, nhãn=l}| cho từng l
                label_counts = Counter(labels[t] for t in idx_list)
                for l in self.classes:
                    freq_val_to_l = label_counts.get(l, 0) / n
                    W[i][v][l] = sum_sumA * freq_val_to_l

        self.W = W
        self._values_per_attr = {i: sorted(set(r[i] for r in self.rules)) for i in range(m)}
        return self

    # ---------------- Suy diễn ----------------
    def _C(self, query):
        """C_il(x) = W[i][x_i][l] -- TRA CỨU trực tiếp (3.68), query đã mờ
        hoá thành list[str] cùng định dạng self.rules[t][:m]."""
        C = {}
        for i in range(self.n_attr):
            v = query[i] if i < len(query) else None
            if v is None or v not in self.W.get(i, {}):
                continue
            C[i] = dict(self.W[i][v])
        return C

    def predict_one(self, query):
        C = self._C(query)
        if not C:
            return None, {}
        D = {}
        for l in self.classes:
            vals = [C[i].get(l, 0.0) for i in C]
            D[l] = (min(vals) + max(vals)) if vals else 0.0
        y_hat = max(D, key=D.get)
        return y_hat, D

    def predict_proba_one(self, query, temperature=1.0):
        y_hat, D = self.predict_one(query)
        if not D:
            n = len(self.classes)
            return {c: 1.0 / n for c in self.classes}, y_hat
        vals = [D.get(c, 0.0) for c in self.classes]
        m = max(vals)
        exps = [np.exp((v - m) / max(temperature, 1e-6)) for v in vals]
        s = sum(exps) + 1e-12
        proba = {c: e / s for c, e in zip(self.classes, exps)}
        return proba, y_hat

    def evaluate(self, queries, labels):
        preds = []
        for q in queries:
            y_hat, _ = self.predict_one(q)
            preds.append(y_hat if y_hat is not None else self.classes[0])
        acc = sum(1 for p, y in zip(preds, labels) if p == y) / len(labels)
        return {"accuracy": acc, "y_pred": preds}


def warn_if_fisa_collapsed(model: FISACorrected, sample_queries, tolerance=0.02):
    """Kiểm tra tự động: lấy một mẫu truy vấn, nếu xác suất dự đoán lớp
    thiểu số gần như KHÔNG BAO GIỜ vượt 0.5 (độ lệch chuẩn của proba(lớp
    thiểu số) qua các truy vấn nhỏ hơn `tolerance`), cảnh báo hiện tượng
    "sụp về lớp đa số" đã biết của công thức (1.17) trên dữ liệu lệch lớp
    (xem docstring module). KHÔNG dừng chương trình -- chỉ cảnh báo, vì đây
    là hạn chế đã CHỦ ĐỘNG chấp nhận (Hướng A), không phải lỗi cần sửa."""
    if not sample_queries:
        return True
    minority_class = min(model.classes, key=lambda c:
                          sum(1 for r in model.rules if r[-1] == c))
    probas = []
    for q in sample_queries:
        proba, _ = model.predict_proba_one(q)
        probas.append(proba.get(minority_class, 0.0))
    std_proba = float(np.std(probas))
    if std_proba < tolerance:
        print(f"  !! CẢNH BÁO (đã biết, Hướng A -- xem fisa_corrected.py): FISA có dấu "
              f"hiệu SỤP VỀ LỚP ĐA SỐ trên fold này (std của proba lớp thiểu số qua "
              f"{len(sample_queries)} truy vấn = {std_proba:.4f} < {tolerance}). "
              f"AUC downstream có thể ~0.5 và KHÔNG phản ánh đúng chất lượng lấy mẫu "
              f"-- diễn giải Rep là chỉ số chính, AUC chỉ mang tính tham khảo cho fold này.")
        return False
    return True


if __name__ == "__main__":
    rules = [
        ["Low", "Medium", "High", "Low", 0],
        ["Low", "Medium", "Low", "Low", 0],
        ["High", "High", "High", "High", 1],
        ["High", "Medium", "High", "Low", 1],
        ["Low", "Low", "Low", "Low", 0],
        ["High", "High", "Medium", "High", 1],
        ["Medium", "Medium", "Medium", "Medium", 0],
        ["High", "Low", "High", "High", 1],
    ]
    model = FISACorrected(rules).fit()
    res = model.evaluate([r[:-1] for r in rules], [r[-1] for r in rules])
    print(f"FISA đã sửa -- accuracy trên chính tập huấn luyện (sanity check): "
          f"{res['accuracy']:.4f}")
    assert res["accuracy"] >= 0.5, "FISA đã sửa cho accuracy quá thấp ngay trên train -- có bug!"
    print("Kiểm thử fisa_corrected: OK")
