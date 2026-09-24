"""
test_validate_fast_vs_original.py — Kiểm định số học BẮT BUỘC: đối chiếu
TỪNG PHẦN TỬ của A, M, B, C và TOÀN BỘ dự đoán FISA giữa src/fkg_original.py
(chậm, Python thuần, chuẩn đối chiếu) và src/fkg_fast.py (nhanh, vector hoá
NumPy) trên nhiều bộ dữ liệu ngẫu nhiên có cấu trúc khác nhau.

Phải chạy và PASS trước khi tin dùng fkg_fast.py cho bất kỳ thực nghiệm nào
(E1-E4) — nếu vector hoá sai một ô dù nhỏ, toàn bộ kết quả downstream (Sim,
Rep, GreedyFKGS, AUC) đều sai theo mà không có dấu hiệu lỗi runtime nào.
"""
import sys, os, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from src.fkg_original import caculateA, caculateM, caculateB, caculateC, FISA
from src.fkg_fast import (EncodedRuleBase, caculateA_fast, caculateM_fast,
                           caculateB_fast, caculateC_fast, FISA_fast)


def make_random_base(n_rows, n_attrs, n_levels=3, seed=0):
    rng = random.Random(seed)
    levels = ["Low", "Medium", "High", "VeryHigh"][:n_levels]
    base = []
    for _ in range(n_rows):
        row = [rng.choice(levels) for _ in range(n_attrs)]
        row.append(rng.randint(0, 1))
        base.append(row)
    return base


def compare_matrix(name, M_orig, M_fast, atol=1e-9):
    M_orig_np = np.array(M_orig)
    max_diff = np.abs(M_orig_np - M_fast).max()
    ok = max_diff < atol
    status = "OK" if ok else "SAI"
    print(f"  {name}: shape_orig={M_orig_np.shape} shape_fast={M_fast.shape} "
          f"max|diff|={max_diff:.2e}  -> {status}")
    return ok


def run_one_case(n_rows, n_attrs, n_levels, seed, n_queries=8):
    print(f"\n=== Trường hợp: n_rows={n_rows}, n_attrs={n_attrs}, "
          f"n_levels={n_levels}, seed={seed} ===")
    base = make_random_base(n_rows, n_attrs, n_levels, seed)

    t0 = time.time()
    A_o = caculateA(base); M_o = caculateM(base)
    B_o = caculateB(base, A_o, M_o); C_o = caculateC(base, B_o)
    t_orig = time.time() - t0

    t0 = time.time()
    enc = EncodedRuleBase(base)
    A_f = caculateA_fast(enc); M_f = caculateM_fast(enc)
    B_f = caculateB_fast(enc, A_f, M_f); C_f = caculateC_fast(enc, B_f)
    t_fast = time.time() - t0

    all_ok = True
    all_ok &= compare_matrix("A", A_o, A_f)
    all_ok &= compare_matrix("M", M_o, M_f)
    all_ok &= compare_matrix("B", B_o, B_f)
    all_ok &= compare_matrix("C", C_o, C_f)

    speedup = t_orig / max(t_fast, 1e-9)
    print(f"  Thời gian: gốc={t_orig:.4f}s  vector-hoá={t_fast:.4f}s  "
          f"(speedup {speedup:.1f}x)")

    # Đối chiếu FISA trên nhiều truy vấn ngẫu nhiên (bao gồm cả luật có sẵn
    # trong base lẫn truy vấn hoàn toàn mới)
    rng = random.Random(seed + 999)
    n_fisa_ok = 0
    for qi in range(n_queries):
        if qi < n_queries // 2:
            query = base[rng.randrange(n_rows)]
        else:
            levels = ["Low", "Medium", "High", "VeryHigh"][:n_levels]
            query = [rng.choice(levels) for _ in range(n_attrs)] + [rng.randint(0, 1)]
        pred_o = FISA(base, C_o, query)
        pred_f = FISA_fast(enc, C_f, query)
        match = (pred_o == pred_f)
        n_fisa_ok += int(match)
        if not match:
            print(f"  !! FISA LỆCH tại truy vấn {qi}: gốc={pred_o} vector-hoá={pred_f} "
                  f"query={query}")
    print(f"  FISA: {n_fisa_ok}/{n_queries} truy vấn khớp -> "
          f"{'OK' if n_fisa_ok == n_queries else 'SAI'}")
    all_ok &= (n_fisa_ok == n_queries)

    return all_ok, speedup


def main():
    print("=" * 70)
    print("KIỂM ĐỊNH SỐ HỌC: fkg_original.py (chuẩn) vs fkg_fast.py (vector hoá)")
    print("=" * 70)

    cases = [
        dict(n_rows=15, n_attrs=4, n_levels=3, seed=1),
        dict(n_rows=25, n_attrs=5, n_levels=3, seed=2),
        dict(n_rows=20, n_attrs=4, n_levels=4, seed=3),   # BMI 4 mức
        dict(n_rows=30, n_attrs=6, n_levels=3, seed=4),
        dict(n_rows=18, n_attrs=8, n_levels=3, seed=5),    # đúng 8 đặc trưng thật
    ]

    all_pass = True
    speedups = []
    for case in cases:
        ok, sp = run_one_case(**case)
        all_pass &= ok
        speedups.append(sp)

    print("\n" + "=" * 70)
    if all_pass:
        print(f"KẾT QUẢ: TẤT CẢ {len(cases)} trường hợp KHỚP TUYỆT ĐỐI.")
        print(f"Speedup trung bình: {np.mean(speedups):.1f}x "
              f"(nhỏ nhất {min(speedups):.1f}x, lớn nhất {max(speedups):.1f}x)")
        print("=> fkg_fast.py ĐÁNG TIN CẬY để dùng cho các thực nghiệm E1-E4.")
    else:
        print("KẾT QUẢ: CÓ TRƯỜNG HỢP LỆCH — KHÔNG được dùng fkg_fast.py cho tới "
              "khi tìm ra và sửa nguyên nhân lệch ở trên.")
    print("=" * 70)

    assert all_pass, "Kiểm định số học THẤT BẠI -- xem chi tiết ở trên."
    return all_pass


if __name__ == "__main__":
    main()
