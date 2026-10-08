"""
experiments/ablation.py — Ablation Study cho FKG-E (Mục 3.5.4 / bảng
Ablation trong thiết kế KB): tách rời từng thành phần kiến trúc để chứng
minh (hoặc bác bỏ) đóng góp riêng của L_node, L_rule (SGNS) và cơ chế
weighted pooling.

4 biến thể:
  1. Rule-embedding-only (lambda=0)      : chỉ L_rule (SGNS) + L_inf/L_pred
  2. Node-embedding-only (beta=0)        : chỉ L_node + L_inf/L_pred
  3. Uniform mean pooling (không alpha)  : pooling="mean" thay vì "weighted"
  4. Full FKG-E                          : đầy đủ như cấu hình mặc định
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from experiments.common import brset_folds, describe_source, sweep


VARIANTS = {   # cắt bỏ thành phần theo hàm mục tiêu (3.28) của Chương 3
    "Full FKG-E":                         {},
    "Bỏ L_pred (λ_P = 0)":                dict(lam_P=0.0),
    "Bỏ L_inf (λ_I = 0)":                 dict(lam_I=0.0),
    "Bỏ L_rule (λ_R = 0)":                dict(lam_R=0.0),
    "Bỏ bảo toàn cạnh (λ_E = 0)":         dict(lam_E=0.0),
    "Bỏ bảo toàn nút (λ_N = 0)":          dict(lam_N=0.0),
    "Bỏ cấu trúc (λ_E = λ_N = 0)":        dict(lam_E=0.0, lam_N=0.0),
    "Gộp trung bình đều (α = r/(r+1))":   dict(pooling="mean"),
}


def run_ablation(n_seeds=None):
    print("\n" + "=" * 70)
    print("Ablation Study cho FKG-E (theo hàm mục tiêu (3.28))")
    print("=" * 70)
    print(f"Nguồn dữ liệu: {describe_source()}")
    folds = brset_folds()
    rows = []
    for name, ov, E in sweep(folds, list(VARIANTS.items()), n_seeds, "Ablation"):
        row = {"variant": name, "n_folds": len(folds)}
        for k in ["accuracy", "balanced_accuracy", "f1_macro", "log_loss", "auc", "agreement_fisa", "dev_rule"]:
            row[k + "_mean"], row[k + "_std"] = E[k + "_mean"], E[k + "_std"]
        row["f1_mean"] = row["f1_macro_mean"]
        print(f"  {name:38s}: acc={row['accuracy_mean']:.4f}  bacc={row['balanced_accuracy_mean']:.4f}  "
              f"AUC={row['auc_mean']:.4f}  log_loss={row['log_loss_mean']:.4f}  "
              f"đồng thuận FISA={row['agreement_fisa_mean']:.4f}  Dev={row['dev_rule_mean']:.4f}")
        rows.append(row)
    return rows


if __name__ == "__main__":
    os.makedirs(C.PATHS.OUTPUT_DIR, exist_ok=True)
    rows = run_ablation()
    with open(os.path.join(C.PATHS.OUTPUT_DIR, "ablation_results.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu ablation_results.json vào {C.PATHS.OUTPUT_DIR}")
