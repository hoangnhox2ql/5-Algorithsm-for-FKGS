"""
src/fkg_fast.py — Vector hoá A/M/B/C/FISA bằng radix-encoding + np.unique,
PHẢI cho kết quả khớp TUYỆT ĐỐI với src/fkg_original.py (kể cả giữ nguyên 3
lỗi đã biết) — đây là điều kiện bắt buộc, kiểm định bằng
test_validate_fast_vs_original.py trước khi tin dùng bản này cho mọi thực
nghiệm E1-E4 (vốn cần chạy hàng trăm/nghìn lần, không thể dùng bản Python
thuần O(n^2 . . . n^4) của fkg_original.py).

Ý TƯỞNG RADIX-ENCODING: với K mức categorical mỗi cột (K nhỏ, ví dụ 3-4),
mã hoá tổ hợp giá trị tại một tập chỉ số cột {a,b,c,...} của MỘT luật thành
một số nguyên duy nhất bằng hệ cơ số K (giống đổi cơ số thập phân->cơ số K).
Hai luật có CÙNG mã số radix tại đúng tập chỉ số đó ⟺ hai luật khớp giá trị
tại TẤT CẢ các chỉ số trong tập đó — biến phép so khớp lồng nhau O(row^2)
thành một lần gọi np.unique (dùng thuật toán sort nội bộ, O(row log row)).
"""
import numpy as np
import math
from itertools import combinations as _combinations


def combination(k, n):
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


class EncodedRuleBase:
    """Mã hoá 1 tập luật (list[list[str/int]]) thành mảng số nguyên NumPy,
    dùng chung cho toàn bộ các hàm vector hoá bên dưới."""

    def __init__(self, base):
        self.base = base
        self.row = len(base)
        self.colum = len(base[0])
        n_attr = self.colum - 1
        self.n_attr = n_attr

        # Mã hoá từng cột thuộc tính thành số nguyên 0..K_j-1
        self.codes = np.zeros((self.row, n_attr), dtype=np.int64)
        self.n_levels = np.zeros(n_attr, dtype=np.int64)
        for j in range(n_attr):
            col_vals = [base[i][j] for i in range(self.row)]
            uniq = sorted(set(col_vals))
            lut = {v: k for k, v in enumerate(uniq)}
            self.n_levels[j] = len(uniq)
            for i in range(self.row):
                self.codes[i, j] = lut[col_vals[i]]

        self.labels = np.array([base[i][-1] for i in range(self.row)], dtype=np.int64)

    def radix_code(self, idx_tuple):
        """Mã radix (mảng shape (row,)) đại diện tổ hợp giá trị tại các cột
        trong idx_tuple, cho MỌI luật cùng lúc."""
        code = np.zeros(self.row, dtype=np.int64)
        for j in idx_tuple:
            code = code * self.n_levels[j] + self.codes[:, j]
        return code

    def radix_code_query(self, idx_tuple, query_codes):
        """Mã radix của MỘT truy vấn (query đã mã hoá bằng cùng LUT) tại các
        cột trong idx_tuple -- số nguyên duy nhất (không phải mảng)."""
        code = 0
        for j in idx_tuple:
            code = code * self.n_levels[j] + query_codes[j]
        return code

    def encode_query(self, query):
        """Mã hoá 1 luật truy vấn (chưa từng thấy hoặc đã có trong base)
        theo đúng LUT đã học từ base. Giá trị lạ (không có trong LUT của
        cột đó) được gán mã -1 (không khớp với bất kỳ luật nào)."""
        codes_q = np.zeros(self.n_attr, dtype=np.int64)
        for j in range(self.n_attr):
            col_vals = [self.base[i][j] for i in range(self.row)]
            uniq = sorted(set(col_vals))
            lut = {v: k for k, v in enumerate(uniq)}
            codes_q[j] = lut.get(query[j], -1)
        return codes_q


def caculateA_fast(enc: EncodedRuleBase):
    """Vector hoá caculateA: với mỗi tổ hợp 4 cột (a,b,c,d), mã hoá radix
    toàn bộ luật, dùng np.unique(..., return_inverse, return_counts) để đếm
    số luật cùng mã -- chính là số luật khớp cả 4 giá trị."""
    row = enc.row
    cols = combination(4, enc.n_attr)
    A = np.zeros((row, cols))
    temp = 0
    for idx in _combinations(range(enc.n_attr), 4):
        code = enc.radix_code(idx)
        _, inverse, counts = np.unique(code, return_inverse=True, return_counts=True)
        A[:, temp] = counts[inverse] / row
        temp += 1
    return A


def caculateM_fast(enc: EncodedRuleBase):
    """Vector hoá caculateM: với mỗi cột đơn i, ghép thêm cột nhãn vào mã
    radix (2 chiều: giá trị thuộc tính + nhãn) để đếm đúng điều kiện 'cùng
    giá trị VÀ cùng nhãn'."""
    row = enc.row
    n_attr = enc.n_attr
    M = np.zeros((row, n_attr))
    n_labels = int(enc.labels.max()) + 1
    for i in range(n_attr):
        combined_code = enc.codes[:, i] * n_labels + enc.labels
        _, inverse, counts = np.unique(combined_code, return_inverse=True, return_counts=True)
        M[:, i] = counts[inverse] / row
    return M


def caculateB_fast(enc: EncodedRuleBase, A, M):
    """Vector hoá caculateB: sum(A, axis=1) rồi min qua 3 cột M tương ứng
    mỗi tổ hợp (a,b,c) -- cả hai đều vector hoá trực tiếp bằng NumPy, không
    cần radix-encoding (không phải phép đếm tần suất)."""
    row = enc.row
    cols = combination(3, enc.n_attr)
    B = np.zeros((row, cols))
    sumA = A.sum(axis=1)  # (row,)
    temp = 0
    for (a, b, c) in _combinations(range(enc.n_attr), 3):
        B[:, temp] = sumA * np.minimum(np.minimum(M[:, a], M[:, b]), M[:, c])
        temp += 1
    return B


def caculateC_fast(enc: EncodedRuleBase, B):
    """Vector hoá caculateC: với mỗi tổ hợp (a,b,c), gom nhóm luật theo mã
    radix (a,b,c) bằng np.unique(..., return_inverse), rồi CỘNG DỒN B trong
    từng nhóm theo TỪNG nhãn riêng biệt (dùng np.add.at, tương đương vòng
    lặp r1,r2 lồng nhau cộng dồn B[r2] vào C[r1] khi khớp (a,b,c))."""
    row = enc.row
    cols_3 = combination(3, enc.n_attr)
    C = np.zeros((row, 2 * cols_3))
    temp = 0
    for idx in _combinations(range(enc.n_attr), 3):
        code = enc.radix_code(idx)               # (row,) mã nhóm theo (a,b,c)
        uniq_codes, inverse = np.unique(code, return_inverse=True)
        n_groups = len(uniq_codes)

        for lbl_val, offset in [(0, 0), (1, cols_3)]:
            mask = (enc.labels == lbl_val)
            group_sum = np.zeros(n_groups)
            # Cộng dồn B[r2][temp] vào đúng nhóm mà luật r2 (nhãn=lbl_val) thuộc về
            np.add.at(group_sum, inverse[mask], B[mask, temp])
            # Mọi luật r1 CÙNG NHÓM (bất kể nhãn của r1) nhận đúng tổng đó
            C[:, temp + offset] = group_sum[inverse]
        temp += 1
    return C


def FISA_fast(enc: EncodedRuleBase, C, query):
    """Vector hoá FISA GỐC — GIỮ NGUYÊN 3 lỗi đã biết:
      1. Chỉ xét luật index 0..row-2 (bỏ sót luật cuối row-1);
      2. GÁN (không cộng dồn): với các luật khớp, giữ giá trị của luật có
         INDEX LỚN NHẤT trong số các luật khớp (vì vòng lặp gốc duyệt tăng
         dần index và GÁN đè liên tục — luật cuối cùng duyệt qua "thắng");
      3. Ngưỡng quyết định bất đối xứng D0 > 9*D1.
    """
    cols_3 = combination(3, enc.n_attr)
    query_codes = enc.encode_query(query)

    C0 = np.zeros(cols_3)
    C1 = np.zeros(cols_3)
    temp = 0
    considered_idx = np.arange(enc.row - 1)   # LỖI GIỮ NGUYÊN: bỏ sót row-1

    for idx in _combinations(range(enc.n_attr), 3):
        q_code = enc.radix_code_query(idx, query_codes)
        code_all = enc.radix_code(idx)
        code_considered = code_all[considered_idx]
        match_mask = (code_considered == q_code)
        matched_positions = considered_idx[match_mask]  # index gốc (0..row-2) các luật khớp

        if len(matched_positions) > 0:
            # LỖI GIỮ NGUYÊN (gán, không cộng dồn): luật có INDEX LỚN NHẤT
            # trong số các luật khớp là luật "thắng" cuối cùng trong vòng lặp gốc.
            last_idx = matched_positions.max()
            if enc.labels[last_idx] == 0:
                C0[temp] = C[last_idx, temp]
            # Cần xét riêng luật khớp có nhãn=1 lớn nhất (có thể KHÁC last_idx
            # nếu luật index lớn nhất trong nhóm khớp lại có nhãn=0) -- vòng lặp
            # gốc cập nhật C0/C1 ĐỘC LẬP theo 2 điều kiện `if` riêng biệt, không
            # phải if/elif, nên cần tìm max index riêng cho từng nhãn:
            matched_label0 = matched_positions[enc.labels[matched_positions] == 0]
            matched_label1 = matched_positions[enc.labels[matched_positions] == 1]
            if len(matched_label0) > 0:
                r0 = matched_label0.max()
                C0[temp] = C[r0, temp]
            if len(matched_label1) > 0:
                r1 = matched_label1.max()
                C1[temp] = C[r1, temp + cols_3]
        temp += 1

    D0 = C0.max() + C0.min()
    D1 = C1.max() + C1.min()
    return 0 if D0 > 9 * D1 else 1


if __name__ == "__main__":
    base = [
        ["Low", "Medium", "High", "Low", 0],
        ["Low", "Medium", "Low", "Low", 0],
        ["High", "High", "High", "High", 1],
        ["High", "Medium", "High", "Low", 1],
        ["Low", "Low", "Low", "Low", 0],
    ]
    enc = EncodedRuleBase(base)
    A = caculateA_fast(enc)
    M = caculateM_fast(enc)
    B = caculateB_fast(enc, A, M)
    Cm = caculateC_fast(enc, B)
    pred = FISA_fast(enc, Cm, base[0])
    print(f"A: {A.shape}, M: {M.shape}, B: {B.shape}, C: {Cm.shape}")
    print(f"Dự đoán FISA_fast cho query=base[0] (nhãn thật={base[0][-1]}): {pred}")
    print("Kiểm thử fkg_fast (chạy được, không lỗi runtime): OK")
    print(">>> Chạy test_validate_fast_vs_original.py để kiểm định số học đầy đủ.")
