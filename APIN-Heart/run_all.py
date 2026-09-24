"""
run_all.py — Chạy toàn bộ pipeline: preprocess -> E1 -> E2 -> E3 -> E4 -> report.
Tự dừng nếu một bước lỗi.

Cách dùng:
  python run_all.py                  # chạy đầy đủ
  python run_all.py --quick           # rút gọn quy mô để kiểm tra pipeline
  python run_all.py --only E1,E4      # chỉ chạy các bước liệt kê
"""
import argparse, time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# QUAN TRỌNG: import config TRƯỚC, sửa đổi (nếu --quick) NGAY SAU KHI PARSE
# ARGS, rồi mới import các module thực nghiệm -- vì mỗi module con đọc giá
# trị config.py TẠI THỜI ĐIỂM IMPORT (module-level constant), không đọc lại
# sau đó. Chạy trong CÙNG process (không dùng subprocess) để việc sửa config
# ở đây thực sự có tác dụng lan toả xuống mọi bước -- bản đầu tiên dùng
# subprocess khiến --quick hoàn toàn không có tác dụng (mỗi subprocess tự
# import lại config.py mặc định, không thấy thay đổi của process cha).
import config as C


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--only", type=str, default="")
    args = parser.parse_args()

    if args.quick:
        print(">>> CHẾ ĐỘ NHANH: giảm quy mô để kiểm tra pipeline không lỗi.\n")
        C.SCALE.N_EXP = 150
        C.KFOLD.K = 2
        C.E1.N_VALUES = [10, 12]
        C.E1.N_REPEATS = 3
        C.E2.THETA_VALUES = [0.1, 0.3]
        C.E2.N_SEEDS_RANDOM = 5
        C.E3.THETA_GRID = [0.1, 0.3, 0.5, 1.0]
        C.E4.N_BLOCKS = 3
        C.E4.BLOCK_SIZE = 100

    only = set(x.strip().upper() for x in args.only.split(",") if x.strip()) or None
    t_start = time.time()
    os.makedirs(C.PATHS.RESULTS_DIR, exist_ok=True)

    def run_step(name, fn):
        if only is not None and name not in only:
            return
        print(f"\n{'#'*70}\n# BƯỚC: {name}\n{'#'*70}")
        fn()

    from experiments import run_00_preprocess
    run_step("PREPROCESS", run_00_preprocess.main)

    from experiments.run_01_E1_theory_validation import run_e1
    def _e1():
        results, bound, n_violations = run_e1()
        import json
        with open(os.path.join(C.PATHS.RESULTS_DIR, "E1_results.json"), "w", encoding="utf-8") as f:
            json.dump({"results": results, "theoretical_bound": bound,
                       "n_violations": n_violations}, f, ensure_ascii=False, indent=2)
    run_step("E1", _e1)

    from experiments.run_02_E2_greedy_vs_random import run_e2
    def _e2():
        rows = run_e2()
        import json
        with open(os.path.join(C.PATHS.RESULTS_DIR, "E2_results.json"), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    run_step("E2", _e2)

    from experiments.run_03_E3_theta_curve import run_e3
    def _e3():
        summary, theta_threshold, knee_theta = run_e3()
        import json
        with open(os.path.join(C.PATHS.RESULTS_DIR, "E3_results.json"), "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "theta_at_coverage_threshold": theta_threshold,
                       "knee_theta": knee_theta}, f, ensure_ascii=False, indent=2)
    run_step("E3", _e3)

    from experiments.run_04_E4_four_strategies import run_e4
    def _e4():
        out = run_e4()
        import json
        with open(os.path.join(C.PATHS.RESULTS_DIR, "E4_results.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    run_step("E4", _e4)

    from experiments import run_05_generate_report
    run_step("REPORT", run_05_generate_report.main)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}\nHOÀN TẤT trong {elapsed/60:.1f} phút.\n{'='*70}")


if __name__ == "__main__":
    main()
