"""
run_all.py — Chạy TOÀN BỘ pipeline thực nghiệm theo đúng thứ tự:
  KB1 -> KB2 -> KB3 -> KB4 -> KB5 -> KB6 -> Ablation -> Baseline -> Báo cáo

Dữ liệu: nếu có data_real/BRSET_Data/fold_XX (BRSET thật, 5 fold theo ID bệnh nhân) thì MỌI kịch bản chạy trên đó
(xem experiments/common.py); ngược lại dùng JSON trong config.PATHS, cuối cùng là dữ liệu tổng hợp (có cảnh báo).

Cách dùng:
  python3 run_all.py                    # chạy đầy đủ
  python3 run_all.py --quick            # chạy nhanh (T_ep = 15, lưới nhỏ; BRSET thật: 2 fold đầu) để kiểm tra
                                          toàn bộ pipeline không lỗi trước khi chạy đầy đủ
  python3 run_all.py --only kb1,kb6     # chỉ chạy các KB được liệt kê
  python3 run_all.py --jobs 12          # số tiến trình huấn luyện FKG-E song song (mặc định config.EVAL.N_JOBS)
  python3 run_all.py --folds fold_01,fold_03 --seeds 3   # BRSET thật: chọn fold, số seed FKG-E mỗi fold
  python3 run_all.py --out outputs_thu  # ghi kết quả vào thư mục khác (mặc định outputs/)
"""
import sys, os, argparse, time, json
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")   # ma trận nhỏ: một luồng BLAS nhanh hơn; song song hoá bằng tiến trình (--jobs)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


def _dump(obj, name):
    with open(os.path.join(C.PATHS.OUTPUT_DIR, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Chạy nhanh với T_ep/n_seeds giảm để kiểm tra pipeline")
    parser.add_argument("--only", type=str, default="",
                        help="Danh sách KB cách nhau bởi dấu phẩy, ví dụ: kb1,kb3,kb6,ablation,baseline")
    parser.add_argument("--jobs", type=int, default=None, help="Số tiến trình huấn luyện FKG-E song song")
    parser.add_argument("--folds", type=str, default="", help="BRSET thật: danh sách fold, ví dụ fold_01,fold_02")
    parser.add_argument("--seeds", type=int, default=None, help="BRSET thật: số seed FKG-E mỗi fold")
    parser.add_argument("--out", type=str, default="", help="Thư mục kết quả (mặc định config.PATHS.OUTPUT_DIR)")
    args = parser.parse_args()

    if args.out:
        C.PATHS.OUTPUT_DIR = os.path.abspath(args.out)
    if args.jobs:
        C.EVAL.N_JOBS = args.jobs
    if args.folds:
        C.EVAL.REAL_FOLDS = [x.strip() for x in args.folds.split(",") if x.strip()]
    if args.seeds:
        C.EVAL.REAL_SEEDS_PER_FOLD = args.seeds

    from experiments.common import using_real_data, real_fold_names, describe_source
    if args.quick:
        print(">>> CHẾ ĐỘ NHANH: T_ep=15, n_seeds=2, thu nhỏ lưới quét để kiểm tra pipeline không lỗi "
              "(không dùng để báo cáo chính thức).\n")
        C.FKGE.T_ep = C.FKGE.epochs = 15
        C.EVAL.N_SEEDS = 2
        # Thu nhỏ mạnh các lưới quét KB3/KB4/KB5/KB6 -- mục đích --quick chỉ là
        # xác nhận KHÔNG LỖI RUNTIME, không phải chạy đủ độ phân giải thật.
        C.KB3_DIMS = [8, 32]
        C.KB4_LAMBDA_GRID = [0.3, 1.0]
        C.KB4_BETA_GRID = [0.3, 1.0]
        C.KB5_W_GRID = [None]
        C.KB5_K_GRID = [2, 5]
        C.KB6_SAMPLE_RATIOS = [0.4, 1.0]
        if using_real_data() and not args.folds:
            C.EVAL.REAL_FOLDS = real_fold_names()[:2]

    only = set(x.strip().lower() for x in args.only.split(",") if x.strip()) or None

    os.makedirs(C.PATHS.OUTPUT_DIR, exist_ok=True)
    print(f"Nguồn dữ liệu: {describe_source()}")
    print(f"T_ep={C.FKGE.T_ep}, {C.EVAL.N_JOBS} tiến trình huấn luyện, kết quả -> {C.PATHS.OUTPUT_DIR}")
    t_start = time.time()
    meta = {"source": describe_source(), "real_data": using_real_data(),
            "folds": real_fold_names() if using_real_data() else None,
            "seeds_per_fold": C.EVAL.REAL_SEEDS_PER_FOLD if using_real_data() else C.EVAL.N_SEEDS,
            "T_ep": C.FKGE.T_ep, "n_jobs": C.EVAL.N_JOBS, "main_metric": C.EVAL.MAIN_METRIC, "quick": args.quick,
            "only": sorted(only) if only else None, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    _dump(meta, "run_meta.json")

    if only is None or "kb1" in only or "kb2" in only:
        from experiments.kb1_kb2 import run_kb1, run_kb2
        if only is None or "kb1" in only:
            _dump(run_kb1(), "kb1_results.json")
        if only is None or "kb2" in only:
            r2, verdict = run_kb2()
            _dump({"rows": r2, "verdict": verdict}, "kb2_results.json")

    if only is None or "kb3" in only:
        from experiments.kb3_kb5 import run_kb3
        _dump(run_kb3(), "kb3_results.json")

    if only is None or "kb5" in only:
        from experiments.kb3_kb5 import run_kb5
        _dump(run_kb5(n_seeds=2 if args.quick else 3), "kb5_results.json")   # n_seeds: chỉ khi một lần chia

    if only is None or "kb4" in only:
        from experiments.kb4_kb6 import run_kb4
        _dump(run_kb4(n_seeds=2 if args.quick else 3), "kb4_results.json")

    if only is None or "kb6" in only:
        from experiments.kb4_kb6 import run_kb6
        _dump(run_kb6(n_seeds=2 if args.quick else 3), "kb6_results.json")

    if only is None or "ablation" in only:
        from experiments.ablation import run_ablation
        _dump(run_ablation(), "ablation_results.json")

    if only is None or "baseline" in only:
        from experiments.baseline_comparison import run_baseline_comparison
        _dump(run_baseline_comparison(), "baseline_comparison.json")

    elapsed = time.time() - t_start
    meta["elapsed_min"] = elapsed / 60
    _dump(meta, "run_meta.json")
    print(f"\n{'='*70}\nHOÀN TẤT TOÀN BỘ THỰC NGHIỆM trong {elapsed/60:.1f} phút.\n{'='*70}")

    print("\nĐang sinh báo cáo tổng hợp (bảng + biểu đồ)...")
    from report.generate_report import main as gen_report
    gen_report(C.PATHS.OUTPUT_DIR)


if __name__ == "__main__":
    main()
