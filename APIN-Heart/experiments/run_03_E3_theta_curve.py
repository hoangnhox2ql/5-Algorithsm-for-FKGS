"""
experiments/run_03_E3_theta_curve.py — E3: đường cong đánh đổi theo theta
(Mục 5, thiet_ke_thuc_nghiem.md), lưới 12 điểm, dày ở vùng theta nhỏ.
Baseline 100% = AUC huấn luyện trên TOÀN BỘ N_EXP (theta=1.0, không lấy mẫu).

K-FOLD: lặp toàn bộ lưới theta trên 5 fold, báo cáo mean±std qua fold.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

import config as C
from src.kfold_utils import make_stratified_kfold, build_fkg_fold
from src.fkg_sim_rep import rep_value
from src.fkgs_algorithms import GreedyFKGS
from src.fisa_corrected import FISACorrected
from src.utils_stats import kneedle_point
from experiments.run_02_E2_greedy_vs_random import _auc_from_fisa
import time


def run_e3():
    print("=" * 70)
    print("E3: Đường cong đánh đổi theo theta")
    print("=" * 70)

    df = pd.read_csv(os.path.join(C.PATHS.DATA_DIR, "Data_clean.csv"))
    folds = make_stratified_kfold(df, k=C.KFOLD.K, seed=C.KFOLD.SEED)

    results_per_theta = {theta: {"rep": [], "auc_rel": [], "time_s": []}
                          for theta in C.E3.THETA_GRID}

    for fold_id, (train_idx, test_idx) in enumerate(folds):
        print(f"\n--- Fold {fold_id} ---")
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)
        built = build_fkg_fold(df_train, df_test, n_sample=C.SCALE.N_EXP, seed=fold_id)
        rules_train, rules_test, sim = built["rules_train"], built["rules_test"], built["sim"]

        # Mốc tham chiếu (theta=1.0): AUC trên TOÀN BỘ N_EXP, không lấy mẫu
        auc_full = _auc_from_fisa(rules_train, rules_test)
        print(f"  AUC(theta=1.0, đầy đủ N_EXP={len(rules_train)}) = {auc_full:.4f}")

        for theta in C.E3.THETA_GRID:
            n = sim.shape[0]
            m = max(1, int(np.ceil(theta * n)))
            t0 = time.time()
            if theta >= 0.999:
                S = list(range(n))
            else:
                S = GreedyFKGS(sim, m=m)
            t_sample = time.time() - t0

            rep_s = rep_value(sim, S)
            auc_s = _auc_from_fisa([rules_train[i] for i in S], rules_test)
            auc_rel = auc_s / auc_full if auc_full > 1e-9 else 1.0

            results_per_theta[theta]["rep"].append(rep_s)
            results_per_theta[theta]["auc_rel"].append(auc_rel)
            results_per_theta[theta]["time_s"].append(t_sample)

        print(f"  Đã quét {len(C.E3.THETA_GRID)} giá trị theta.")

    # Tổng hợp mean qua fold
    summary = []
    for theta in C.E3.THETA_GRID:
        r = results_per_theta[theta]
        summary.append({
            "theta": theta,
            "rep_mean": float(np.mean(r["rep"])), "rep_std": float(np.std(r["rep"])),
            "auc_rel_mean": float(np.mean(r["auc_rel"])), "auc_rel_std": float(np.std(r["auc_rel"])),
            "time_s_mean": float(np.mean(r["time_s"])),
        })
        print(f"theta={theta:.2f}: Rep={summary[-1]['rep_mean']:.4f}±{summary[-1]['rep_std']:.4f}  "
              f"AUC_rel={summary[-1]['auc_rel_mean']:.4f}±{summary[-1]['auc_rel_std']:.4f}  "
              f"time={summary[-1]['time_s_mean']*1000:.2f}ms")

    # Theta tối thiểu đạt ngưỡng phủ delta=0.85 (Định nghĩa 2.12)
    theta_at_threshold = None
    for row in summary:
        if row["rep_mean"] >= C.E3.COVERAGE_THRESHOLD:
            theta_at_threshold = row["theta"]
            break
    print(f"\nTheta tối thiểu đạt Rep >= {C.E3.COVERAGE_THRESHOLD}: "
          f"{theta_at_threshold if theta_at_threshold else 'KHÔNG đạt trong lưới đã quét'}")

    # Điểm gối Kneedle trên đường Pareto (time vs auc_rel)
    times = [row["time_s_mean"] for row in summary]
    aucs = [row["auc_rel_mean"] for row in summary]
    knee_idx, knee_theta = kneedle_point(times, aucs)
    print(f"Điểm gối (Kneedle) trên đồ thị Pareto (thời gian, AUC tương đối): "
          f"theta={C.E3.THETA_GRID[knee_idx]}")

    return summary, theta_at_threshold, C.E3.THETA_GRID[knee_idx]


if __name__ == "__main__":
    os.makedirs(C.PATHS.RESULTS_DIR, exist_ok=True)
    summary, theta_threshold, knee_theta = run_e3()
    out = {"summary": summary, "theta_at_coverage_threshold": theta_threshold,
           "knee_theta": knee_theta}
    with open(os.path.join(C.PATHS.RESULTS_DIR, "E3_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu kết quả vào {C.PATHS.RESULTS_DIR}/E3_results.json")
