"""
experiments/kb1_kb2.py — KB1: FKG-E vs FISA trên FKG gốc (BRSET + các bộ Diabetes nếu có).
                         KB2: FKG-E vs FISA trên FKGS đã lấy mẫu (Chương 2).

Dữ liệu: experiments/common.py — BRSET thật 5 fold (data_real/BRSET_Data) nếu có, ngược lại JSON, ngược lại dữ liệu
tổng hợp (có cảnh báo). Các bộ Diabetes chỉ chạy khi có file JSON tương ứng trong config.PATHS.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import config as C
from experiments.common import (brset_folds, other_dataset_folds, using_real_data, describe_source, seeds_for,
                                run_fkge, fisa_metrics, aggregate, check_collapse, FKGE_KEYS, FISA_KEYS)


def run_one_dataset(folds, dataset_name, tag, fkge_kwargs=None, n_seeds=None):
    """FISA (tất định, một lần mỗi fold) và FKG-E (n_seeds seed mỗi fold) -> một dòng bảng KB1/KB2."""
    n = seeds_for(folds, n_seeds)
    tasks = [(f, C.FKGE.seed + s, fkge_kwargs) for f in folds for s in range(n)]
    runs = run_fkge(tasks, f"{tag} {dataset_name}")
    for (f, s, _), m in zip(tasks, runs):
        check_collapse(f"FKG-E ({dataset_name}, {f.name}, seed={s})", m)
    E = aggregate([(t[0].key, m) for t, m in zip(tasks, runs)], FKGE_KEYS)
    F = aggregate([(f.key, fisa_metrics(f)) for f in folds], FISA_KEYS)
    per_fold = []
    for f in folds:
        ms = [m for t, m in zip(tasks, runs) if t[0] is f]
        fm = fisa_metrics(f)
        per_fold.append({"fold": f.name, "n_rules": f.n_rules, "fisa_threshold": fm["threshold"],
                         "fisa": {k: fm.get(k) for k in FISA_KEYS},
                         "fkge": {k: float(np.mean([m.get(k, np.nan) for m in ms])) for k in FKGE_KEYS}})
    return {
        "dataset": dataset_name, "kb": tag, "source": folds[0].kind, "n_folds": len(folds), "n_seeds_per_fold": n,
        "n_rules": int(round(np.mean([f.n_rules for f in folds]))),
        "fisa_accuracy": F["accuracy_mean"], "fisa_accuracy_std": F["accuracy_std"], "fisa_f1": F["f1_macro_mean"],
        "fisa_bacc": F["balanced_accuracy_mean"], "fisa_bacc_std": F["balanced_accuracy_std"],
        "fisa_auc": F["auc_mean"], "fisa_auc_std": F["auc_std"],
        "fisa_sensitivity": F["sensitivity_mean"], "fisa_specificity": F["specificity_mean"],
        "fisa_acc_argmax": F["accuracy_argmax_mean"], "fisa_acc_ratio9": F["accuracy_ratio9_mean"],
        "fisa_threshold": [fisa_metrics(f)["threshold"] for f in folds],
        "fisa_node_size": fisa_metrics(folds[0])["node_size"],
        "fisa_avg_ms": F["avg_time_per_query_ms_mean"], "fisa_seq_avg_ms": F["seq_avg_time_per_query_ms_mean"],
        "fisa_total_time_s": F["total_time_s_mean"],
        "fkge_accuracy_mean": E["accuracy_mean"], "fkge_accuracy_std": E["accuracy_std"],
        "fkge_f1_mean": E["f1_macro_mean"], "fkge_f1_std": E["f1_macro_std"],
        "fkge_avg_ms_mean": E["avg_time_per_query_ms_mean"], "fkge_avg_ms_std": E["avg_time_per_query_ms_std"],
        "fkge_bacc_mean": E["balanced_accuracy_mean"], "fkge_bacc_std": E["balanced_accuracy_std"],
        "fkge_auc_mean": E["auc_mean"], "fkge_auc_std": E["auc_std"],
        "fkge_sensitivity_mean": E["sensitivity_mean"], "fkge_specificity_mean": E["specificity_mean"],
        "fkge_agreement_mean": E["agreement_fisa_mean"], "fkge_dev_mean": E["dev_rule_mean"],
        "fkge_train_time_s_mean": E["train_time_s_mean"],
        "speedup": F["avg_time_per_query_ms_mean"] / max(E["avg_time_per_query_ms_mean"], 1e-9),
        "speedup_vs_seq": F["seq_avg_time_per_query_ms_mean"] / max(E["avg_time_per_query_ms_mean"], 1e-9),
        "is_synthetic": folds[0].kind == "synthetic", "per_fold": per_fold,
    }


def _print_row(row):
    print(f"  FISA (FKG-Pairs k={row['fisa_node_size']}): acc={row['fisa_accuracy']:.4f}  "
          f"bacc={row['fisa_bacc']:.4f}±{row['fisa_bacc_std']:.4f}  AUC={row['fisa_auc']:.4f}±{row['fisa_auc_std']:.4f}  "
          f"f1={row['fisa_f1']:.4f}  bảng tra {row['fisa_avg_ms']:.4f} ms/query, tuần tự {row['fisa_seq_avg_ms']:.4f} ms/query")
    print(f"  FKG-E: acc={row['fkge_accuracy_mean']:.4f}±{row['fkge_accuracy_std']:.4f}  "
          f"bacc={row['fkge_bacc_mean']:.4f}±{row['fkge_bacc_std']:.4f}  AUC={row['fkge_auc_mean']:.4f}±{row['fkge_auc_std']:.4f}  "
          f"f1={row['fkge_f1_mean']:.4f}  đồng thuận={row['fkge_agreement_mean']:.4f}  Dev={row['fkge_dev_mean']:.4f}  "
          f"{row['fkge_avg_ms_mean']:.4f} ms/query  (so với FISA bảng tra {row['speedup']:.2f}x, "
          f"so với FISA tuần tự {row['speedup_vs_seq']:.2f}x)")
    if row["n_folds"] > 1:
        print(f"  (trung bình ± độ lệch chuẩn giữa {row['n_folds']} fold, {row['n_seeds_per_fold']} seed mỗi fold)")


def run_kb1(n_seeds=None):
    print("\n" + "=" * 70)
    print("KB1: FKG-E vs FISA trên FKG gốc (Diabetes-Kaggle, Healthcare-Diabetes, BRSET)")
    print("=" * 70)
    print(f"Nguồn dữ liệu BRSET: {describe_source()}")
    results = []
    for name, rules_path, test_path, off in [
            ("Diabetes-Kaggle", C.PATHS.DIABETES_KAGGLE_RULES_FILE, C.PATHS.DIABETES_KAGGLE_TEST_FILE, 1),
            ("Healthcare-Diabetes-Kaggle", C.PATHS.HEALTHCARE_DIABETES_RULES_FILE, C.PATHS.HEALTHCARE_DIABETES_TEST_FILE, 2)]:
        print(f"\n-- {name} --")
        folds = other_dataset_folds(name, rules_path, test_path, off)
        if not folds:
            print(f"  [Bỏ qua] Chưa có dữ liệu thật '{rules_path}' — đang chạy BRSET thật nên không thay bằng dữ liệu tổng hợp.")
            continue
        row = run_one_dataset(folds, name, "KB1", n_seeds=n_seeds)
        _print_row(row); results.append(row)
    print("\n-- BRSET (FKG-MM) --")
    row = run_one_dataset(brset_folds(), "BRSET (FKG-MM)", "KB1", n_seeds=n_seeds)
    _print_row(row); results.append(row)
    return results


def run_kb2(n_seeds=None, ratio=0.3):
    print("\n" + "=" * 70)
    print("KB2: FKG-E vs FISA trên FKGS đã lấy mẫu (Chương 2)")
    print("=" * 70)
    full = brset_folds()
    sampled = brset_folds(rule_ratio=ratio, rule_seed=99, fkgs_file=C.PATHS.BRSET_FKGS_RULES_FILE)
    simulated = sampled[0].info.get("rules") != "fkgs_file"
    if simulated:
        print(f"  !! Chưa có luật FKGS thật cho {'các fold BRSET' if using_real_data() else 'BRSET'} — MÔ PHỎNG bằng "
              f"cách giữ ngẫu nhiên {ratio:.0%} số luật{' mỗi fold' if using_real_data() else ''}.")
    name_s = f"BRSET (FKGS mô phỏng {ratio:.0%})" if simulated else "BRSET (FKGS đã nén)"
    row_full = run_one_dataset(full, "BRSET (FKG-MM đầy đủ)", "KB2-full", n_seeds=n_seeds)
    row_sampled = run_one_dataset(sampled, name_s, "KB2-sampled", n_seeds=n_seeds)
    row_sampled["fkgs_simulated"] = simulated

    for row in (row_full, row_sampled):
        print(f"\n-- {row['dataset']} ({row['n_rules']} luật) --")
        print(f"  FISA:  acc={row['fisa_accuracy']:.4f}  bacc={row['fisa_bacc']:.4f}  AUC={row['fisa_auc']:.4f}  "
              f"{row['fisa_avg_ms']:.4f} ms/query")
        print(f"  FKG-E: acc={row['fkge_accuracy_mean']:.4f}±{row['fkge_accuracy_std']:.4f}  bacc={row['fkge_bacc_mean']:.4f}  "
              f"AUC={row['fkge_auc_mean']:.4f}  {row['fkge_avg_ms_mean']:.4f} ms/query  (speedup={row['speedup']:.2f}x)")

    # Câu hỏi cốt lõi KB2: lợi ích tốc độ của FKG-E có CỘNG HƯỞNG hay bị
    # TRIỆT TIÊU khi luật đã được FKGS nén sẵn?
    synergy = row_sampled["speedup"] - row_full["speedup"]
    verdict = ("CỘNG HƯỞNG (speedup tăng thêm khi kết hợp FKGS+FKG-E)" if synergy > 0.1
               else "TRUNG TÍNH/TRIỆT TIÊU (FKGS không làm tăng thêm lợi ích tốc độ của FKG-E)")
    if simulated:
        verdict += f" — FKGS MÔ PHỎNG bằng lấy mẫu ngẫu nhiên {ratio:.0%} luật, chưa phải FKGS thật của Chương 2"
    print(f"\n>>> Speedup trên FKG-MM đầy đủ: {row_full['speedup']:.2f}x")
    print(f">>> Speedup trên FKGS:          {row_sampled['speedup']:.2f}x")
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
