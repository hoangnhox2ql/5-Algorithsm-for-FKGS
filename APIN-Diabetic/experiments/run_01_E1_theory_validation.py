"""
experiments/run_01_E1_theory_validation.py — E1: xác nhận Rep(S_G)/Rep(S*)
>= (1-1/e) trên MỌI lần chạy (Mục 3, thiet_ke_thuc_nghiem.md).

QUAN TRỌNG: xây FKG (mờ hoá + khai phá luật) THEO TỪNG FOLD của k-fold
(src/kfold_utils.py) trước khi lấy mẫu con brute-force-able (n<=18) từ
CHÍNH luật của fold đó -- không dùng một FKG cố định chung cho mọi fold.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from itertools import combinations

import config as C
from src.kfold_utils import make_stratified_kfold, build_fkg_fold
from src.fkg_sim_rep import compute_sim_matrix, rep_value
from src.fkgs_algorithms import GreedyFKGS


def brute_force_optimal(sim_matrix, m):
    """Tìm S* tối ưu bằng vét cạn TẤT CẢ tổ hợp C(n,m) -- chỉ khả thi n<=18."""
    n = sim_matrix.shape[0]
    best_rep = -1.0
    best_S = None
    for S in combinations(range(n), m):
        r = rep_value(sim_matrix, list(S))
        if r > best_rep:
            best_rep, best_S = r, S
    return list(best_S), best_rep


def run_e1():
    print("=" * 70)
    print("E1: Xác nhận lý thuyết Rep(S_G)/Rep(S*) >= (1-1/e)")
    print("=" * 70)

    df = pd.read_csv(os.path.join(C.PATHS.DATA_DIR, "Data_clean.csv"))
    folds = make_stratified_kfold(df, k=C.KFOLD.K, seed=C.KFOLD.SEED)

    all_results = []
    bound = 1 - 1 / np.e
    n_violations = 0
    total_runs = 0

    # E1 chỉ cần MỘT fold đại diện (bài toán brute-force rất tốn kém, không
    # cần lặp qua cả 5 fold) -- dùng fold 0, ghi chú rõ trong báo cáo.
    train_idx, test_idx = folds[0]
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    built = build_fkg_fold(df_train, df_test, n_sample=None, seed=0)
    rules_full = built["rules_train"]
    print(f"Đã xây FKG từ fold 0: {len(rules_full)} luật train.")

    for n in C.E1.N_VALUES:
        for ratio in C.E1.RATIO_VALUES:
            m = max(1, int(round(ratio * n)))
            print(f"\n-- n={n}, m/n={ratio} (m={m}) --")
            ratios_this_config = []

            for rep_idx in range(C.E1.N_REPEATS):
                rng = np.random.RandomState(1000 * n + 10 * rep_idx + int(ratio * 100))
                sub_idx = rng.choice(len(rules_full), n, replace=False)
                sub_rules = [rules_full[i] for i in sub_idx]
                sim_sub = compute_sim_matrix(sub_rules)

                S_greedy = GreedyFKGS(sim_sub, m=m)
                rep_greedy = rep_value(sim_sub, S_greedy)

                S_opt, rep_opt = brute_force_optimal(sim_sub, m)
                ratio_val = rep_greedy / rep_opt if rep_opt > 1e-12 else 1.0
                ratios_this_config.append(ratio_val)

                total_runs += 1
                violated = ratio_val < bound - 1e-9
                if violated:
                    n_violations += 1
                    print(f"  !! VI PHẠM tại rep={rep_idx}: rho={ratio_val:.4f} < {bound:.4f}")

                all_results.append({
                    "n": n, "ratio_budget": ratio, "m": m, "repeat": rep_idx,
                    "rep_greedy": rep_greedy, "rep_opt": rep_opt, "rho": ratio_val,
                    "violated": bool(violated),
                })

            arr = np.array(ratios_this_config)
            print(f"  rho trung bình={arr.mean():.4f}, min={arr.min():.4f}, "
                  f"max={arr.max():.4f}  (cận lý thuyết={bound:.4f})")

    print(f"\n{'='*70}")
    print(f"TỔNG KẾT E1: {total_runs} lần chạy, {n_violations} vi phạm cận (1-1/e)")
    if n_violations == 0:
        print(f"=> KHÔNG có vi phạm nào -- đúng lý thuyết Định lý Nemhauser (Định lý 5).")
    else:
        print(f"=> CÓ {n_violations} VI PHẠM -- cần rà lại cài đặt greedy_submodular_select, "
              f"KHÔNG phải phản bác định lý.")
    print("=" * 70)

    return all_results, bound, n_violations


if __name__ == "__main__":
    os.makedirs(C.PATHS.RESULTS_DIR, exist_ok=True)
    results, bound, n_violations = run_e1()
    out = {"results": results, "theoretical_bound": bound, "n_violations": n_violations}
    with open(os.path.join(C.PATHS.RESULTS_DIR, "E1_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu kết quả vào {C.PATHS.RESULTS_DIR}/E1_results.json")
