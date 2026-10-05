"""
run_all.py — Chạy TOÀN BỘ pipeline thực nghiệm theo đúng thứ tự:
  KB1 -> KB2 -> KB3 -> KB4 -> KB5 -> KB6 -> Ablation -> Baseline -> Báo cáo

Cách dùng:
  python3 run_all.py                 # chạy đầy đủ (có thể mất 30-90 phút
                                        tuỳ quy mô dữ liệu thật, xem README)
  python3 run_all.py --quick          # chạy nhanh (epochs/n_seeds giảm) để
                                        kiểm tra toàn bộ pipeline không lỗi
                                        trước khi chạy đầy đủ trên dữ liệu thật
  python3 run_all.py --only kb1,kb6   # chỉ chạy các KB được liệt kê
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Chạy nhanh với epochs/n_seeds giảm để kiểm tra pipeline")
    parser.add_argument("--only", type=str, default="",
                        help="Danh sách KB cách nhau bởi dấu phẩy, ví dụ: kb1,kb3,kb6")
    args = parser.parse_args()

    if args.quick:
        print(">>> CHẾ ĐỘ NHANH: giảm epochs=15, n_seeds=2, thu nhỏ lưới quét để "
              "kiểm tra pipeline không lỗi (không dùng để báo cáo chính thức).\n")
        C.FKGE.epochs = 15
        C.EVAL.N_SEEDS = 2
        # Thu nhỏ mạnh các lưới quét KB3/KB4/KB5/KB6 -- mục đích --quick chỉ là
        # xác nhận KHÔNG LỖI RUNTIME, không phải chạy đủ độ phân giải thật.
        C.KB3_DIMS = [8, 32]
        C.KB4_LAMBDA_GRID = [0.3, 1.0]
        C.KB4_BETA_GRID = [0.3, 1.0]
        C.KB5_W_GRID = [None, 2]
        C.KB5_K_GRID = [2, 5]
        C.KB6_SAMPLE_RATIOS = [0.4, 1.0]

    only = set(x.strip().lower() for x in args.only.split(",") if x.strip()) or None

    os.makedirs(C.PATHS.OUTPUT_DIR, exist_ok=True)
    t_start = time.time()

    if only is None or "kb1" in only or "kb2" in only:
        from experiments.kb1_kb2 import run_kb1, run_kb2
        import json
        if only is None or "kb1" in only:
            r1 = run_kb1()
            json.dump(r1, open(os.path.join(C.PATHS.OUTPUT_DIR, "kb1_results.json"), "w",
                                encoding="utf-8"), ensure_ascii=False, indent=2)
        if only is None or "kb2" in only:
            r2, verdict = run_kb2()
            json.dump({"rows": r2, "verdict": verdict},
                      open(os.path.join(C.PATHS.OUTPUT_DIR, "kb2_results.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=2)

    if only is None or "kb3" in only:
        from experiments.kb3_kb5 import run_kb3
        import json
        r3 = run_kb3()
        json.dump(r3, open(os.path.join(C.PATHS.OUTPUT_DIR, "kb3_results.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)

    if only is None or "kb5" in only:
        from experiments.kb3_kb5 import run_kb5
        import json
        n_seeds_kb5 = 2 if args.quick else 3
        r5 = run_kb5(n_seeds=n_seeds_kb5)
        json.dump(r5, open(os.path.join(C.PATHS.OUTPUT_DIR, "kb5_results.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)

    if only is None or "kb4" in only:
        from experiments.kb4_kb6 import run_kb4
        import json
        n_seeds_kb4 = 2 if args.quick else 3
        r4 = run_kb4(n_seeds=n_seeds_kb4)
        json.dump(r4, open(os.path.join(C.PATHS.OUTPUT_DIR, "kb4_results.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)

    if only is None or "kb6" in only:
        from experiments.kb4_kb6 import run_kb6
        import json
        n_seeds_kb6 = 2 if args.quick else 3
        r6 = run_kb6(n_seeds=n_seeds_kb6)
        json.dump(r6, open(os.path.join(C.PATHS.OUTPUT_DIR, "kb6_results.json"), "w",
                            encoding="utf-8"), ensure_ascii=False, indent=2)

    if only is None or "ablation" in only:
        from experiments.ablation import run_ablation
        import json
        rows = run_ablation()
        json.dump(rows, open(os.path.join(C.PATHS.OUTPUT_DIR, "ablation_results.json"), "w",
                              encoding="utf-8"), ensure_ascii=False, indent=2)

    if only is None or "baseline" in only:
        from experiments.baseline_comparison import run_baseline_comparison
        import json
        rows = run_baseline_comparison()
        json.dump(rows, open(os.path.join(C.PATHS.OUTPUT_DIR, "baseline_comparison.json"), "w",
                              encoding="utf-8"), ensure_ascii=False, indent=2)

    elapsed = time.time() - t_start
    print(f"\n{'='*70}\nHOÀN TẤT TOÀN BỘ THỰC NGHIỆM trong {elapsed/60:.1f} phút.\n{'='*70}")

    print("\nĐang sinh báo cáo tổng hợp (bảng + biểu đồ)...")
    from report.generate_report import main as gen_report
    gen_report()


if __name__ == "__main__":
    main()
