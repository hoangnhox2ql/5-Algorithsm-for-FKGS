"""
models/fisa.py — Suy luận FISA (Fuzzy Inference by Similarity Aggregation),
đúng công thức "Nhắc lại ký hiệu suy luận FISA" trong Chương 3 luận án:

    W_il = sum_t B^t_il                                  (3.67)
    C_il(x) = sum_{v in V_i} mu_v(x_i) * W_vl             (3.68)
    D_l(x) = min_i C_il(x) + max_i C_il(x)                (3.69)
    y_hat = argmax_l D_l(x)                                (3.70)

Vì luật đã cho ở dạng (antecedent_tokens -> consequent_token) thay vì ma
trận B^t tường minh, FISA ở đây suy ra W_vl trực tiếp từ tập luật: mỗi
luật "bỏ phiếu" cho token tiền đề của nó với trọng số = support*confidence
của luật, cộng dồn vào W[v][l]. Đây là một cách hiện thực hoá tương đương
công thức (3.67) khi B^t được định nghĩa qua support/confidence của luật t.
"""
import time
import math
from collections import defaultdict, Counter


class FISA:
    def __init__(self, fkg):
        self.fkg = fkg
        self.class_tokens = fkg.class_tokens
        self.W = None          # W[v][l] -> float
        self._attr_of = {}      # token -> tên thuộc tính (phần trước dấu '-')
        for t in fkg.vocab:
            if not t.startswith("class-"):
                self._attr_of[t] = t.rsplit("-", 1)[0]

    def fit(self):
        """Xây ma trận W theo (3.67), tổng hợp từ toàn bộ luật."""
        t0 = time.time()
        W = defaultdict(lambda: defaultdict(float))
        for r in self.fkg.rules:
            w = r["support"] * r["confidence"]
            l = r["consequent_token"]
            for v in r["antecedent_tokens"]:
                W[v][l] += w
        self.W = W
        self.fit_time = time.time() - t0
        return self

    def _C(self, membership):
        """C_il(x) theo (3.68): tổng hợp theo TỪNG THUỘC TÍNH i (không phải
        từng token), để đúng ngữ nghĩa 'thuộc tính quan sát' trong công thức
        gốc — với mỗi thuộc tính, cộng dồn qua các token/nhãn ngôn ngữ của
        chính thuộc tính đó."""
        C = defaultdict(lambda: defaultdict(float))  # C[attr][class] 
        for v, deg in membership.items():
            if deg <= 0 or v not in self.W:
                continue
            attr = self._attr_of.get(v, v)
            for l, w in self.W[v].items():
                C[attr][l] += deg * w
        return C

    def predict_one(self, membership):
        """Trả về (D_l dict, nhãn dự đoán, thời gian suy diễn)."""
        t0 = time.time()
        C = self._C(membership)
        D = defaultdict(float)
        if len(C) == 0:
            # Không có thuộc tính nào khớp bất kỳ token nào trong W -> không đủ dữ liệu
            elapsed = time.time() - t0
            return {}, None, elapsed
        for l in self.class_tokens:
            vals = [C[attr].get(l, 0.0) for attr in C]
            if not vals:
                D[l] = 0.0
            else:
                D[l] = min(vals) + max(vals)
        y_hat = max(D, key=D.get)
        elapsed = time.time() - t0
        return dict(D), y_hat, elapsed

    def predict_proba_one(self, membership, temperature=1.0):
        """Chuyển D_l thành phân phối xác suất bằng softmax (dùng làm
        p^F_cal trong Mục 3.4.5 để tính KL-divergence với FKG-E)."""
        D, y_hat, elapsed = self.predict_one(membership)
        if not D:
            n = len(self.class_tokens)
            return {c: 1.0 / n for c in self.class_tokens}, y_hat, elapsed
        vals = [D.get(c, 0.0) for c in self.class_tokens]
        m = max(vals)
        exps = [math.exp((v - m) / max(temperature, 1e-6)) for v in vals]
        s = sum(exps) + 1e-12
        proba = {c: e / s for c, e in zip(self.class_tokens, exps)}
        return proba, y_hat, elapsed

    def evaluate(self, samples):
        """Chạy suy diễn trên toàn bộ tập mẫu, trả về accuracy, F1 (macro),
        tổng thời gian và thời gian suy diễn trung bình/mẫu."""
        y_true, y_pred = [], []
        total_time = 0.0
        per_query_times = []
        for s in samples:
            D, y_hat, elapsed = self.predict_one(s["membership"])
            total_time += elapsed
            per_query_times.append(elapsed)
            y_true.append(s["label"])
            y_pred.append(y_hat if y_hat is not None else "NONE")
        acc = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)
        f1 = _macro_f1(y_true, y_pred, self.class_tokens)
        return {
            "accuracy": acc, "f1_macro": f1,
            "total_time_s": total_time,
            "avg_time_per_query_ms": (total_time / len(samples)) * 1000,
            "fit_time_s": getattr(self, "fit_time", 0.0),
            "y_true": y_true, "y_pred": y_pred,
        }


def majority_class_baseline(samples):
    """Accuracy của bộ dự đoán ngây thơ 'luôn đoán lớp xuất hiện nhiều nhất
    trong tập test'. MỌI mô hình phải được so sánh với con số này -- nếu một
    mô hình không vượt qua rõ rệt baseline này, đó là dấu hiệu mô hình đang
    "sụp" về việc đoán theo lớp đa số (majority-class collapse), một lỗi rất
    dễ xảy ra khi tín hiệu học biểu diễn (không giám sát) áp đảo tín hiệu
    phân loại (có giám sát) trong hàm mất mát tổng hợp."""
    labels = [s["label"] for s in samples]
    counts = Counter(labels)
    top_label, top_count = counts.most_common(1)[0]
    return top_count / len(samples), top_label


def warn_if_not_beating_majority(result, samples, model_name="Mô hình", margin=0.02):
    """In cảnh báo rõ ràng nếu accuracy không vượt majority baseline + margin."""
    base_acc, base_label = majority_class_baseline(samples)
    acc = result["accuracy"]
    if acc <= base_acc + margin:
        print(f"  !! CẢNH BÁO: {model_name} có accuracy={acc:.4f}, KHÔNG vượt rõ rệt "
              f"baseline lớp đa số ({base_acc:.4f}, luôn đoán '{base_label}'). "
              f"Có thể mô hình đang sụp về đoán theo lớp đa số -- kiểm tra lại "
              f"siêu tham số (đặc biệt delta_pred/gamma_inf) trước khi báo cáo "
              f"kết quả này.")
        return False
    return True


def _macro_f1(y_true, y_pred, classes):
    f1s = []
    for c in classes:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == c and b != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return sum(f1s) / len(f1s) if f1s else 0.0


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.fkg_io import generate_synthetic_fkg, generate_synthetic_test_samples

    fkg = generate_synthetic_fkg(n_rules=200, seed=1)
    samples = generate_synthetic_test_samples(fkg, n_samples=200, seed=2)
    fisa = FISA(fkg).fit()
    res = fisa.evaluate(samples)
    print(f"FISA trên dữ liệu tổng hợp: Accuracy={res['accuracy']:.4f}, "
          f"F1={res['f1_macro']:.4f}, TG suy diễn TB={res['avg_time_per_query_ms']:.4f} ms")
    assert res["accuracy"] > 0.4, "FISA cho accuracy quá thấp trên dữ liệu có cấu trúc -- kiểm tra lại logic!"
    print("Kiểm thử FISA: OK (accuracy vượt ngưỡng ngẫu nhiên có ý nghĩa)")
