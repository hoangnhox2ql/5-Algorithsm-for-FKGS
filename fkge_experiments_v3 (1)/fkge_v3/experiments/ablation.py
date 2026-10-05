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
import numpy as np

from data.fkg_io import FKGRuleBase, load_test_samples, generate_synthetic_fkg, generate_synthetic_test_samples
from models.fisa import FISA
from models.fkge import FKGE
import config as C


def _get_brset_or_synthetic():
    if os.path.exists(C.PATHS.BRSET_RULES_FILE) and os.path.exists(C.PATHS.BRSET_TEST_FILE):
        fkg = FKGRuleBase.load(C.PATHS.BRSET_RULES_FILE)
        samples = load_test_samples(C.PATHS.BRSET_TEST_FILE)
        print("Đã nạp dữ liệu BRSET THẬT.")
    else:
        print("!! Chưa có BRSET thật -- dùng dữ liệu tổng hợp cho Ablation.")
        fkg = generate_synthetic_fkg(n_rules=250, seed=100, dataset_name="BRSET")
        samples = generate_synthetic_test_samples(fkg, n_samples=200, seed=200)
    n_train = int(0.6 * len(samples))
    return fkg, samples[:n_train], samples[n_train:]


VARIANTS = {   # cắt bỏ thành phần theo hàm mục tiêu (3.28) của Chương 3
    "Full FKG-E":                         {},
    "Bỏ L_pred (λ_P = 0)":                dict(lam_P=0.0),
    "Bỏ L_inf (λ_I = 0)":                 dict(lam_I=0.0),
    "Bỏ L_rule (λ_R = 0)":                dict(lam_R=0.0),
    "Bỏ cấu trúc (λ_S = λ_E = λ_N = 0)":  dict(lam_S=0.0, lam_E=0.0, lam_N=0.0),
    "Gộp trung bình đều (α = r/(r+1))":   dict(pooling="mean"),
    "Đồng xuất hiện theo cửa sổ (w = 2)": dict(cooc="window", w=2),
}


def run_ablation(n_seeds=C.EVAL.N_SEEDS):
    print("\n" + "=" * 70)
    print("Ablation Study cho FKG-E (theo hàm mục tiêu (3.28))")
    print("=" * 70)
    fkg, train, test = _get_brset_or_synthetic()
    fisa = FISA(fkg).fit(val_samples=train)

    rows = []
    for name, overrides in VARIANTS.items():
        res = []
        for s in range(n_seeds):
            kwargs = dict(T_ep=C.FKGE.epochs, seed=C.FKGE.seed + s)
            kwargs.update(overrides)
            m = FKGE(fkg, **kwargs)
            m.fit(fisa_model=fisa, train_samples=train)
            res.append(m.evaluate(test))
        agg = lambda k: (float(np.mean([r[k] for r in res])), float(np.std([r[k] for r in res])))
        row = {"variant": name}
        for k in ["accuracy", "balanced_accuracy", "f1_macro", "log_loss", "auc", "agreement_fisa", "dev_rule"]:
            row[k + "_mean"], row[k + "_std"] = agg(k)
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
