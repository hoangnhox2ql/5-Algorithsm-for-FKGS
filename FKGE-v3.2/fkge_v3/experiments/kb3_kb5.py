"""
experiments/kb3_kb5.py — KB3: quét chiều nhúng d.
                         KB5: quét số mẫu âm K (mô hình theo Chương 3 không dùng cửa sổ w).
Dữ liệu: experiments/common.py (BRSET thật 5 fold nếu có).
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from experiments.common import brset_folds, describe_source, fisa_summary, sweep


def run_kb3(dims=None, n_seeds=None):
    dims = list(dims or C.KB3_DIMS)
    print("\n" + "=" * 70)
    print(f"KB3: Độ nhạy theo chiều nhúng d ∈ {dims}")
    print("=" * 70)
    print(f"Nguồn dữ liệu: {describe_source()}")
    folds = brset_folds()
    F = fisa_summary(folds)
    rows = []
    for name, ov, E in sweep(folds, [(d, {"d_e": d}) for d in dims], n_seeds, "KB3"):
        row = {"d": name, "accuracy_mean": E["accuracy_mean"], "accuracy_std": E["accuracy_std"],
               "balanced_accuracy_mean": E["balanced_accuracy_mean"], "balanced_accuracy_std": E["balanced_accuracy_std"],
               "auc_mean": E["auc_mean"], "auc_std": E["auc_std"],
               "avg_ms_mean": E["avg_time_per_query_ms_mean"], "train_time_s_mean": E["train_time_s_mean"],
               "fisa_accuracy": F["accuracy_mean"], "fisa_balanced_accuracy": F["balanced_accuracy_mean"],
               "fisa_auc": F["auc_mean"], "n_folds": len(folds)}
        print(f"  d={name:4d}: accuracy={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f}  "
              f"bacc={row['balanced_accuracy_mean']:.4f}±{row['balanced_accuracy_std']:.4f}  "
              f"AUC={row['auc_mean']:.4f}±{row['auc_std']:.4f}  "
              f"TG suy diễn={row['avg_ms_mean']:.4f}ms  TG huấn luyện={row['train_time_s_mean']:.2f}s")
        rows.append(row)
    return rows


def run_kb5(w_grid=None, k_grid=None, n_seeds=3):
    w_grid = list(w_grid or C.KB5_W_GRID); k_grid = list(k_grid or C.KB5_K_GRID)
    print("\n" + "=" * 70)
    print(f"KB5: Độ nhạy theo số mẫu âm K ∈ {k_grid} (đồng xuất hiện toàn luật)")
    print("=" * 70)
    print(f"Nguồn dữ liệu: {describe_source()}")
    folds = brset_folds()
    configs = [((w, k), {"K": k}) for w in w_grid for k in k_grid]   # w không còn là tham số của mô hình
    rows = []
    for (w, k), ov, E in sweep(folds, configs, n_seeds, "KB5"):
        row = {"w": ("toàn luật" if w is None else w), "K": k,
               "accuracy_mean": E["accuracy_mean"], "accuracy_std": E["accuracy_std"],
               "balanced_accuracy_mean": E["balanced_accuracy_mean"], "balanced_accuracy_std": E["balanced_accuracy_std"],
               "auc_mean": E["auc_mean"], "auc_std": E["auc_std"], "n_folds": len(folds)}
        print(f"  đồng xuất hiện={'toàn luật' if w is None else 'cửa sổ w=' + str(w)} K={k:2d}: "
              f"accuracy={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f}  "
              f"bacc={row['balanced_accuracy_mean']:.4f}±{row['balanced_accuracy_std']:.4f}  "
              f"AUC={row['auc_mean']:.4f}±{row['auc_std']:.4f}")
        rows.append(row)
    return rows


if __name__ == "__main__":
    os.makedirs(C.PATHS.OUTPUT_DIR, exist_ok=True)
    r3 = run_kb3()
    with open(os.path.join(C.PATHS.OUTPUT_DIR, "kb3_results.json"), "w", encoding="utf-8") as f:
        json.dump(r3, f, ensure_ascii=False, indent=2)
    r5 = run_kb5()
    with open(os.path.join(C.PATHS.OUTPUT_DIR, "kb5_results.json"), "w", encoding="utf-8") as f:
        json.dump(r5, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu kb3_results.json và kb5_results.json vào {C.PATHS.OUTPUT_DIR}")
