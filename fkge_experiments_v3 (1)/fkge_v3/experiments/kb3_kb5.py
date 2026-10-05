"""
experiments/kb3_kb5.py — KB3: quét chiều nhúng d.
                         KB5: quét (cửa sổ ngữ cảnh w, số mẫu âm K).
"""
import sys, os, json, time
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
        print("!! Chưa có BRSET thật -- dùng dữ liệu tổng hợp cho KB3/KB5.")
        fkg = generate_synthetic_fkg(n_rules=250, seed=100, dataset_name="BRSET")
        samples = generate_synthetic_test_samples(fkg, n_samples=200, seed=200)
    n_train = int(0.6 * len(samples))
    return fkg, samples[:n_train], samples[n_train:]


def run_kb3(dims=C.KB3_DIMS, n_seeds=C.EVAL.N_SEEDS):
    print("\n" + "=" * 70)
    print(f"KB3: Độ nhạy theo chiều nhúng d ∈ {dims}")
    print("=" * 70)
    fkg, train, test = _get_brset_or_synthetic()
    fisa = FISA(fkg).fit(val_samples=train)
    res_fisa = fisa.evaluate(test)
    rows = []
    for d in dims:
        accs, times, train_times = [], [], []
        for s in range(n_seeds):
            m = FKGE(fkg, d=d, w=C.FKGE.w, K_neg=C.FKGE.K_neg,
                     lam_node=C.FKGE.lam_node, beta_rule=C.FKGE.beta_rule,
                     gamma_inf=C.FKGE.gamma_inf, delta_pred=C.FKGE.delta_pred,
                     lr=C.FKGE.lr, epochs=C.FKGE.epochs, seed=C.FKGE.seed + s)
            m.fit(fisa_model=fisa, train_samples=train)
            r = m.evaluate(test)
            accs.append(r["accuracy"]); times.append(r["avg_time_per_query_ms"])
            train_times.append(m.train_time_s)
        row = {"d": d, "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
               "avg_ms_mean": float(np.mean(times)), "train_time_s_mean": float(np.mean(train_times)),
               "fisa_accuracy": res_fisa["accuracy"]}
        print(f"  d={d:4d}: accuracy={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f}  "
              f"TG suy diễn={row['avg_ms_mean']:.4f}ms  TG huấn luyện={row['train_time_s_mean']:.2f}s")
        rows.append(row)
    return rows


def run_kb5(w_grid=C.KB5_W_GRID, k_grid=C.KB5_K_GRID, n_seeds=3):
    print("\n" + "=" * 70)
    print(f"KB5: Độ nhạy theo (w, K) skip-gram — w∈{w_grid}, K∈{k_grid}")
    print("=" * 70)
    fkg, train, test = _get_brset_or_synthetic()
    fisa = FISA(fkg).fit(val_samples=train)
    rows = []
    for w in w_grid:
        for k in k_grid:
            accs = []
            for s in range(n_seeds):
                m = FKGE(fkg, d=C.FKGE.d, cooc=("full" if w is None else "window"), w=w, K_neg=k,
                         lam_node=C.FKGE.lam_node, beta_rule=C.FKGE.beta_rule,
                         gamma_inf=C.FKGE.gamma_inf, delta_pred=C.FKGE.delta_pred,
                         lr=C.FKGE.lr, epochs=C.FKGE.epochs, seed=C.FKGE.seed + s)
                m.fit(fisa_model=fisa, train_samples=train)
                r = m.evaluate(test)
                accs.append(r["accuracy"])
            row = {"w": ("toàn luật" if w is None else w), "K": k, "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs))}
            print(f"  đồng xuất hiện={'toàn luật' if w is None else 'cửa sổ w=' + str(w)} K={k:2d}: accuracy={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f}")
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
