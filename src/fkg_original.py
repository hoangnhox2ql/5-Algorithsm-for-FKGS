"""
src/fkg_original.py — Sao chép TRUNG THÀNH logic gốc của FKG.ipynb (đã đọc
và phân tích kỹ ở các lượt trước): ma trận A (tổ hợp 4 thuộc tính), M (tương
đồng đơn-thuộc-tính có điều kiện nhãn), B (= sum(A_r) * min(M_a,M_b,M_c) trên
tổ hợp 3 thuộc tính), C (tổng hợp theo nhãn), và FISA (so khớp rời rạc trên
tổ hợp 3 thuộc tính, ngưỡng quyết định bất đối xứng 9:1).

⚠️ CHỦ ĐÍCH GIỮ NGUYÊN BA LỖI ĐÃ PHÁT HIỆN Ở CÁC LƯỢT TRƯỚC — file này KHÔNG
dùng để suy diễn "thật" cho báo cáo khoa học; nó CHỈ tồn tại để đối chiếu số
học (test_validate_fast_vs_original.py) với bản vector hoá fkg_fast.py, xác
nhận việc tăng tốc bằng NumPy không làm thay đổi kết quả toán học so với cách
hiểu trực tiếp từ mã nguồn gốc:
  1. `for r in range(row-1)` trong FISA — bỏ sót luật cuối cùng (off-by-one).
  2. `C0[t] = C[r][...]` — GÁN (không cộng dồn) khi nhiều luật cùng khớp.
  3. `D0 > 9*D1` — ngưỡng quyết định bất đối xứng, hệ số 9 không có căn cứ
     lý thuyết tường minh trong mã nguồn.

Suy diễn "đã sửa" (dùng thật cho AUC downstream ở E2-E4) nằm ở
src/fisa_corrected.py, theo đúng công thức (1.18)-(1.20) đã hiệu chỉnh và
xác nhận trong Chương 3.

CHẬM CÓ CHỦ ĐÍCH: mọi vòng lặp đều bằng Python thuần (không NumPy vector
hoá) để phản ánh đúng, dễ đọc-đối-chiếu với cấu trúc vòng lặp lồng nhau của
notebook gốc. Chỉ dùng cho bộ dữ liệu rất nhỏ (n <= vài trăm) khi kiểm định.
"""
from itertools import combinations as _itertools_combinations
import math


def combination(k, n):
    """C(n, k) — số tổ hợp chập k của n, đúng hàm `combination` trong notebook."""
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def caculateA(base):
    """Ma trận A: với mỗi tổ hợp 4 chỉ số thuộc tính (a,b,c,d) và mỗi luật r1,
    A[r1][temp] = (số luật r2 có CÙNG giá trị tại cả 4 vị trí a,b,c,d) / row."""
    row = len(base)
    colum = len(base[0])
    cols = combination(4, colum - 1)
    A = [[0.0] * cols for _ in range(row)]
    temp = 0
    for (a, b, c, d) in _itertools_combinations(range(colum - 1), 4):
        for r1 in range(row):
            k = 0
            for r2 in range(row):
                if (base[r1][a] == base[r2][a] and base[r1][b] == base[r2][b]
                        and base[r1][c] == base[r2][c] and base[r1][d] == base[r2][d]):
                    k += 1
            A[r1][temp] = k / row
        temp += 1
    return A


def caculateM(base):
    """Ma trận M: với mỗi thuộc tính đơn i, M[t1][i] = (số luật t2 có CÙNG
    giá trị thuộc tính i VÀ CÙNG nhãn với t1) / row."""
    row = len(base)
    colum = len(base[0])
    M = [[0.0] * (colum - 1) for _ in range(row)]
    for i in range(colum - 1):
        for t1 in range(row):
            k = 0
            for t2 in range(row):
                if base[t1][i] == base[t2][i] and base[t1][-1] == base[t2][-1]:
                    k += 1
            M[t1][i] = k / row
    return M


def caculateB(base, A, M):
    """Ma trận B: với mỗi tổ hợp 3 chỉ số thuộc tính (a,b,c) và mỗi luật r,
    B[r][temp] = sum(A[r]) * min(M[r][a], M[r][b], M[r][c])."""
    row = len(base)
    colum = len(base[0])
    cols = combination(3, colum - 1)
    B = [[0.0] * cols for _ in range(row)]
    for r in range(row):
        sumA_r = sum(A[r])
        temp = 0
        for (a, b, c) in _itertools_combinations(range(colum - 1), 3):
            B[r][temp] = sumA_r * min(M[r][a], M[r][b], M[r][c])
            temp += 1
    return B


def caculateC(base, B):
    """Ma trận C: với mỗi tổ hợp 3 chỉ số thuộc tính (a,b,c), mỗi luật r1,
    cộng dồn B[r2][temp] của các luật r2 khớp CẢ 3 giá trị thuộc tính với r1,
    tách riêng theo nhãn 0/1 (2 khối cols_3 liền nhau)."""
    row = len(base)
    colum = len(base[0])
    cols_3 = combination(3, colum - 1)
    C = [[0.0] * (2 * cols_3) for _ in range(row)]
    temp = 0
    for (a, b, c) in _itertools_combinations(range(colum - 1), 3):
        for r1 in range(row):
            for r2 in range(row):
                if (base[r2][a] == base[r1][a] and base[r2][b] == base[r1][b]
                        and base[r2][c] == base[r1][c]):
                    if base[r2][-1] == 0:
                        C[r1][temp] += B[r2][temp]
                    elif base[r2][-1] == 1:
                        C[r1][temp + cols_3] += B[r2][temp]
        temp += 1
    return C


def FISA(base, C, query):
    """Suy diễn FISA GỐC (giữ nguyên 3 lỗi đã biết — xem docstring module).
    `query` là 1 luật dạng list[str/int], cùng định dạng base[i]."""
    row = len(base)
    colum = len(base[0])
    cols_3 = combination(3, colum - 1)
    C0 = [0.0] * cols_3
    C1 = [0.0] * cols_3
    temp = 0
    for (a, b, c) in _itertools_combinations(range(colum - 1), 3):
        for r in range(row - 1):          # LỖI GIỮ NGUYÊN: bỏ sót luật cuối (row-1)
            if (base[r][a] == query[a] and base[r][b] == query[b] and base[r][c] == query[c]):
                if base[r][-1] == 0:
                    C0[temp] = C[r][temp]              # LỖI GIỮ NGUYÊN: gán, không +=
                elif base[r][-1] == 1:
                    C1[temp] = C[r][temp + cols_3]      # LỖI GIỮ NGUYÊN: gán, không +=
        temp += 1
    D0 = max(C0) + min(C0)
    D1 = max(C1) + min(C1)
    if D0 > 9 * D1:                          # LỖI GIỮ NGUYÊN: ngưỡng bất đối xứng 9:1
        return 0
    else:
        return 1


if __name__ == "__main__":
    # Kiểm thử cực nhỏ: chạy toàn bộ chuỗi A->M->B->C->FISA trên vài luật
    # đồ chơi, chỉ để xác nhận không lỗi runtime (kiểm định số học đầy đủ
    # nằm ở test_validate_fast_vs_original.py, đối chiếu với fkg_fast.py).
    base = [
        ["Low", "Medium", "High", "Low", 0],
        ["Low", "Medium", "Low", "Low", 0],
        ["High", "High", "High", "High", 1],
        ["High", "Medium", "High", "Low", 1],
        ["Low", "Low", "Low", "Low", 0],
    ]
    A = caculateA(base)
    M = caculateM(base)
    B = caculateB(base, A, M)
    Cm = caculateC(base, B)
    pred = FISA(base, Cm, base[0])
    print(f"A: {len(A)}x{len(A[0])}, M: {len(M)}x{len(M[0])}, "
          f"B: {len(B)}x{len(B[0])}, C: {len(Cm)}x{len(Cm[0])}")
    print(f"Dự đoán FISA cho query=base[0] (nhãn thật={base[0][-1]}): {pred}")
    print("Kiểm thử fkg_original (chạy được, không lỗi runtime): OK")
