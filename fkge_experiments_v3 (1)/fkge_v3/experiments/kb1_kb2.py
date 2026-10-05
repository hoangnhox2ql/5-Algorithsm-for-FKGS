"""
experiments/kb1_kb2.py — KB1: FKG-E vs FISA trên FKG gốc (3 bộ dữ liệu).
                         KB2: FKG-E vs FISA trên FKGS đã lấy mẫu (Chương 2).

Cách dùng dữ liệu thật: xem README.md mục "Chuẩn bị dữ liệu đầu vào".
Nếu chưa có dữ liệu thật, script tự động dùng dữ liệu tổng hợp (synthetic)
để bạn kiểm tra toàn bộ pipeline chạy đúng trước khi có BRSET/Diabetes.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fkg_io import (FKGRuleBase, load_test_samples,
                          generate_synthetic_fkg, generate_synthetic_test_samples)
from models.fisa import FISA, FISASequential, warn_if_not_beating_majority
from models.fkge import FKGE
import config as C


def _load_or_synthesize(rules_path, test_path, dataset_name, seed_offset=0):
    """Nạp dữ liệu thật nếu tồn tại, ngược lại sinh dữ liệu tổng hợp và in
    cảnh báo rõ ràng — KHÔNG được âm thầm dùng synthetic mà không báo."""
    if os.path.exists(rules_path) and os.path.exists(test_path):
        fkg = FKGRuleBase.load(rules_path)
        samples = load_test_samples(test_path)
        print(f"  [{dataset_name}] Đã nạp dữ liệu THẬT: {len(fkg)} luật, {len(samples)} mẫu test")
        return fkg, samples, False
    else:
        print(f"  [{dataset_name}] !! CHƯA TÌM THẤY dữ liệu thật tại "
              f"'{rules_path}'. Dùng DỮ LIỆU TỔNG HỢP để kiểm thử pipeline.")
        fkg = generate_synthetic_fkg(n_rules=250, seed=100 + seed_offset,
                                      dataset_name=dataset_name)
        samples = generate_synthetic_test_samples(fkg, n_samples=200, seed=200 + seed_offset)
        return fkg, samples, True


def run_one_dataset(fkg, test_samples, train_samples, dataset_name, tag,
                     fkge_kwargs=None, n_seeds=C.EVAL.N_SEEDS):
    """Chạy FISA (1 lần, tất định) và FKG-E (n_seeds lần, lấy mean/std) trên
    một bộ luật, trả về dict kết quả tổng hợp cho một dòng bảng KB1/KB2."""
    fkge_kwargs = fkge_kwargs or {}
    # FISA theo FKGS v3.2 (FKG-Pairs, k = 3); ngưỡng quyết định chọn trên tập huấn luyện của FKG-E
    # (không dùng tập kiểm tra). FISASequential: cùng hàm quyết định, duyệt tuần tự (FISA gốc).
    fisa = FISA(fkg).fit(val_samples=train_samples)
    res_fisa = fisa.evaluate(test_samples)
    res_seq = FISASequential(fkg).fit(val_samples=train_samples).evaluate(test_samples)

    accs, f1s, times, extra = [], [], [], []
    for s in range(n_seeds):
        kwargs = dict(T_ep=C.FKGE.epochs, seed=C.FKGE.seed + s)   # tham số Chương 3 lấy từ config.FKGE
        kwargs.update(fkge_kwargs)
        model = FKGE(fkg, **kwargs)
        model.fit(fisa_model=fisa, train_samples=train_samples)
        res = model.evaluate(test_samples)
        extra.append(res)
        warn_if_not_beating_majority(res, test_samples, model_name=f"FKG-E ({dataset_name}, seed={s})")
        accs.append(res["accuracy"]); f1s.append(res["f1_macro"]); times.append(res["avg_time_per_query_ms"])

    import numpy as np
    return {
        "dataset": dataset_name, "kb": tag,
        "fisa_accuracy": res_fisa["accuracy"], "fisa_f1": res_fisa["f1_macro"],
        "fisa_bacc": res_fisa["balanced_accuracy"], "fisa_auc": res_fisa.get("auc"),
        "fisa_acc_argmax": res_fisa["accuracy_argmax"], "fisa_acc_ratio9": res_fisa["accuracy_ratio9"],
        "fisa_threshold": res_fisa["threshold"], "fisa_node_size": res_fisa["node_size"],
        "fisa_avg_ms": res_fisa["avg_time_per_query_ms"],
        "fisa_seq_avg_ms": res_seq["avg_time_per_query_ms"],
        "fisa_total_time_s": res_fisa["total_time_s"],
        "fkge_accuracy_mean": float(np.mean(accs)), "fkge_accuracy_std": float(np.std(accs)),
        "fkge_f1_mean": float(np.mean(f1s)), "fkge_f1_std": float(np.std(f1s)),
        "fkge_avg_ms_mean": float(np.mean(times)), "fkge_avg_ms_std": float(np.std(times)),
        "fkge_bacc_mean": float(np.mean([r["balanced_accuracy"] for r in extra])),
        "fkge_auc_mean": float(np.mean([r.get("auc", float("nan")) for r in extra])),
        "fkge_agreement_mean": float(np.mean([r["agreement_fisa"] for r in extra])),
        "fkge_dev_mean": float(np.mean([r["dev_rule"] for r in extra])),
        "speedup": (res_fisa["avg_time_per_query_ms"] / max(np.mean(times), 1e-9)),
        "speedup_vs_seq": (res_seq["avg_time_per_query_ms"] / max(np.mean(times), 1e-9)),
        "n_rules": len(fkg), "is_synthetic": getattr(fkg, "_is_synthetic", False),
    }


def run_kb1():
    print("\n" + "=" * 70)
    print("KB1: FKG-E vs FISA trên FKG gốc (Diabetes-Kaggle, Healthcare-Diabetes, BRSET)")
    print("=" * 70)
    datasets = [
        ("Diabetes-Kaggle", C.PATHS.DIABETES_KAGGLE_RULES_FILE, C.PATHS.DIABETES_KAGGLE_TEST_FILE, 1),
        ("Healthcare-Diabetes-Kaggle", C.PATHS.HEALTHCARE_DIABETES_RULES_FILE, C.PATHS.HEALTHCARE_DIABETES_TEST_FILE, 2),
        ("BRSET (FKG-MM)", C.PATHS.BRSET_RULES_FILE, C.PATHS.BRSET_TEST_FILE, 3),
    ]
    results = []
    for name, rules_path, test_path, off in datasets:
        print(f"\n-- {name} --")
        fkg, samples, is_synth = _load_or_synthesize(rules_path, test_path, name, off)
        fkg._is_synthetic = is_synth
        n_train = int(0.6 * len(samples))
        train_samples, test_samples = samples[:n_train], samples[n_train:]
        row = run_one_dataset(fkg, test_samples, train_samples, name, "KB1")
        print(f"  FISA (FKG-Pairs k={row['fisa_node_size']}): acc={row['fisa_accuracy']:.4f}  "
              f"bacc={row['fisa_bacc']:.4f}  f1={row['fisa_f1']:.4f}  AUC={row['fisa_auc']}  "
              f"bảng tra {row['fisa_avg_ms']:.4f} ms/query, tuần tự {row['fisa_seq_avg_ms']:.4f} ms/query")
        print(f"  FKG-E: acc={row['fkge_accuracy_mean']:.4f}±{row['fkge_accuracy_std']:.4f}  "
              f"f1={row['fkge_f1_mean']:.4f}±{row['fkge_f1_std']:.4f}  bacc={row['fkge_bacc_mean']:.4f}  "
              f"đồng thuận={row['fkge_agreement_mean']:.4f}  Dev={row['fkge_dev_mean']:.4f}  "
              f"{row['fkge_avg_ms_mean']:.4f} ms/query  (so với FISA bảng tra {row['speedup']:.2f}x, "
              f"so với FISA tuần tự {row['speedup_vs_seq']:.2f}x)")
        results.append(row)
    return results


def run_kb2():
    print("\n" + "=" * 70)
    print("KB2: FKG-E vs FISA trên FKGS đã lấy mẫu (Chương 2)")
    print("=" * 70)
    fkg_full, samples, is_synth = _load_or_synthesize(
        C.PATHS.BRSET_RULES_FILE, C.PATHS.BRSET_TEST_FILE, "BRSET (FKG-MM đầy đủ)", 3)

    if os.path.exists(C.PATHS.BRSET_FKGS_RULES_FILE):
        fkg_sampled = FKGRuleBase.load(C.PATHS.BRSET_FKGS_RULES_FILE)
        print(f"  Đã nạp FKGS THẬT: {len(fkg_sampled)} luật (từ {len(fkg_full)} luật gốc)")
    else:
        print(f"  !! CHƯA TÌM THẤY FKGS thật tại '{C.PATHS.BRSET_FKGS_RULES_FILE}'. "
              f"Mô phỏng bằng cách lấy mẫu ngẫu nhiên 30% số luật gốc.")
        fkg_sampled = fkg_full.sample_subset(ratio=0.3, seed=99)

    n_train = int(0.6 * len(samples))
    train_samples, test_samples = samples[:n_train], samples[n_train:]

    row_full = run_one_dataset(fkg_full, test_samples, train_samples, "BRSET (FKG-MM đầy đủ)", "KB2-full")
    row_sampled = run_one_dataset(fkg_sampled, test_samples, train_samples, "BRSET (FKGS đã nén)", "KB2-sampled")

    for row, tag in [(row_full, "FKG-MM đầy đủ"), (row_sampled, "FKGS đã nén")]:
        print(f"\n-- {tag} ({row['n_rules']} luật) --")
        print(f"  FISA:  acc={row['fisa_accuracy']:.4f}  {row['fisa_avg_ms']:.4f} ms/query")
        print(f"  FKG-E: acc={row['fkge_accuracy_mean']:.4f}±{row['fkge_accuracy_std']:.4f}  "
              f"{row['fkge_avg_ms_mean']:.4f} ms/query  (speedup={row['speedup']:.2f}x)")

    # Câu hỏi cốt lõi KB2: lợi ích tốc độ của FKG-E có CỘNG HƯỞNG hay bị
    # TRIỆT TIÊU khi luật đã được FKGS nén sẵn?
    synergy = row_sampled["speedup"] - row_full["speedup"]
    verdict = ("CỘNG HƯỞNG (speedup tăng thêm khi kết hợp FKGS+FKG-E)" if synergy > 0.1
               else "TRUNG TÍNH/TRIỆT TIÊU (FKGS không làm tăng thêm lợi ích tốc độ của FKG-E)")
    print(f"\n>>> Speedup trên FKG-MM đầy đủ: {row_full['speedup']:.2f}x")
    print(f">>> Speedup trên FKGS đã nén:   {row_sampled['speedup']:.2f}x")
    print(f">>> Kết luận KB2: {verdict}")

    return [row_full, row_sampled], verdict


if __name__ == "__main__":
    os.makedirs(C.PATHS.OUTPUT_DIR, exist_ok=True)
    r1 = run_kb1()
    r2, verdict = run_kb2()
    with open(os.path.join(C.PATHS.OUTPUT_DIR, "kb1_results.json"), "w", encoding="utf-8") as f:
        json.dump(r1, f, ensure_ascii=False, indent=2)
    with open(os.path.join(C.PATHS.OUTPUT_DIR, "kb2_results.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": r2, "verdict": verdict}, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu kết quả vào {C.PATHS.OUTPUT_DIR}/kb1_results.json và kb2_results.json")
