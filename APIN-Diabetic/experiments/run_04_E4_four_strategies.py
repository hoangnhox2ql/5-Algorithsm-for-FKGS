"""
experiments/run_04_E4_four_strategies.py — E4: so sánh 4 chiến lược
(ĐÃ SỬA từ 5 xuống 4, theo yêu cầu và đúng cơ sở lý thuyết trong
Dieu_chinh_5_thuat_toan_Submodular_v4.pdf — xem config.py để biết lý do):
GreedyFKGS, Random, SFKGS, CFKGS.

10 block độc lập (Mục 6, thiet_ke_thuc_nghiem.md) = 10 lần lấy mẫu con kích
thước BLOCK_SIZE từ pool dữ liệu — giữ nguyên hạn chế đã nêu rõ trong tài
liệu gốc: đo được độ ổn định qua các mẫu con CÙNG một phân phối, KHÔNG đo
được tổng quát hoá đa lĩnh vực. Kết luận chỉ phát biểu "trên bộ dữ liệu
tiểu đường đã dùng".

K-FOLD: toàn bộ 10-block-Friedman lặp lại trên MỖI fold của k-fold ngoài,
kết quả cuối gộp qua (fold x block) = 5*10 = 50 block tổng cộng cho Friedman
-- vừa giữ đúng thiết kế 10-block gốc, vừa tích hợp yêu cầu k-fold.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

import config as C
from src.kfold_utils import make_stratified_kfold, build_fkg_fold, _stratified_subsample
from src.fkg_sim_rep import compute_sim_matrix, rep_value
from src.fkgs_algorithms import GreedyFKGS, RandomSampling, SFKGS, CFKGS
from src.utils_stats import friedman_test, average_ranks, nemenyi_critical_difference
import time


def run_one_block(rules_block, theta):
    sim = compute_sim_matrix(rules_block)
    labels = [r[-1] for r in rules_block]
    n = len(rules_block)
    m = max(1, int(np.ceil(theta * n)))

    results = {}
    for method in C.E4.METHODS:
        t0 = time.time()
        if method == "GreedyFKGS":
            S = GreedyFKGS(sim, m=m)
        elif method == "Random":
            S = RandomSampling(n, m=m, seed=0)
        elif method == "SFKGS":
            S = SFKGS(sim, labels, theta=theta, alloc_method=C.E4.ALLOC_METHOD)
        elif method == "CFKGS":
            S = CFKGS(sim, rules_block, theta=theta, n_clusters=C.E4.N_CLUSTERS,
                      design=C.E4.CLUSTER_DESIGN, seed=0)
        else:
            raise ValueError(f"Không hỗ trợ method={method}")
        elapsed = time.time() - t0
        results[method] = {"rep": rep_value(sim, S), "time_s": elapsed, "size": len(S)}
    return results


def run_e4():
    print("=" * 70)
    print(f"E4: So sánh {len(C.E4.METHODS)} chiến lược lấy mẫu -- {C.E4.METHODS}")
    print("=" * 70)
    print("!! LƯU Ý GIỚI HẠN (nêu theo đúng thiet_ke_thuc_nghiem.md Mục 6): "
          "10 block là 10 mẫu con độc lập từ CÙNG MỘT nguồn dữ liệu tiểu đường, "
          "KHÔNG phải 10 bộ dữ liệu độc lập đa lĩnh vực. Kết luận chỉ phát biểu "
          "'trên bộ dữ liệu tiểu đường đã dùng'.\n")

    df = pd.read_csv(os.path.join(C.PATHS.DATA_DIR, "Data_clean.csv"))
    folds = make_stratified_kfold(df, k=C.KFOLD.K, seed=C.KFOLD.SEED)

    rep_matrix_rows = []   # mỗi dòng: 1 block, cột: 1 phương pháp -- Rep
    time_matrix_rows = []  # tương tự cho thời gian
    block_meta = []

    for fold_id, (train_idx, test_idx) in enumerate(folds):
        print(f"\n--- Fold {fold_id} ---")
        df_train = df.iloc[train_idx].reset_index(drop=True)
        built = build_fkg_fold(df_train, df_train.iloc[:1], n_sample=None, seed=fold_id)
        rules_pool = built["rules_train"]  # toàn bộ luật train của fold (chưa lấy mẫu N_EXP)

        for block_id in range(C.E4.N_BLOCKS):
            seed = fold_id * 1000 + block_id
            rules_block = _stratified_subsample(rules_pool, C.E4.BLOCK_SIZE, seed)
            res = run_one_block(rules_block, C.E4.THETA)

            rep_row = [res[m]["rep"] for m in C.E4.METHODS]
            time_row = [res[m]["time_s"] for m in C.E4.METHODS]
            rep_matrix_rows.append(rep_row)
            time_matrix_rows.append(time_row)
            block_meta.append({"fold": fold_id, "block": block_id})

            print(f"  Block (fold={fold_id}, block={block_id}): "
                  + "  ".join(f"{m}={res[m]['rep']:.4f}" for m in C.E4.METHODS))

    rep_matrix = np.array(rep_matrix_rows)     # (n_blocks_total, n_methods)
    time_matrix = np.array(time_matrix_rows)
    n_blocks_total = rep_matrix.shape[0]

    print(f"\n{'='*70}")
    print(f"KIỂM ĐỊNH FRIEDMAN trên {n_blocks_total} block x {len(C.E4.METHODS)} phương pháp")
    print("=" * 70)

    stat, p = friedman_test(rep_matrix)
    print(f"Friedman: stat={stat:.4f}, p={p:.6f}")

    ranks_rep = average_ranks(rep_matrix, higher_is_better=True)
    ranks_time = average_ranks(time_matrix, higher_is_better=False)  # thời gian: thấp hơn tốt hơn
    print("\nHạng trung bình theo Rep (1=tốt nhất):")
    for m, r in zip(C.E4.METHODS, ranks_rep):
        print(f"  {m:12s}: hạng={r:.2f}")
    print("\nHạng trung bình theo Thời gian (1=nhanh nhất):")
    for m, r in zip(C.E4.METHODS, ranks_time):
        print(f"  {m:12s}: hạng={r:.2f}")

    nemenyi_results = None
    if p < 0.05:
        cd = nemenyi_critical_difference(n_blocks_total, len(C.E4.METHODS))
        print(f"\np < 0.05 -> chạy Nemenyi post-hoc. Critical Difference (CD) = {cd:.4f}")
        print("Các cặp phương pháp có |chênh lệch hạng| > CD được xem là khác biệt có ý nghĩa:")
        pairs_significant = []
        for i in range(len(C.E4.METHODS)):
            for j in range(i + 1, len(C.E4.METHODS)):
                diff = abs(ranks_rep[i] - ranks_rep[j])
                sig = diff > cd
                if sig:
                    pairs_significant.append((C.E4.METHODS[i], C.E4.METHODS[j], float(diff)))
                    print(f"  {C.E4.METHODS[i]} vs {C.E4.METHODS[j]}: "
                          f"|diff hạng|={diff:.3f} > CD={cd:.3f} -> KHÁC BIỆT có ý nghĩa")
        nemenyi_results = {"cd": cd, "significant_pairs": pairs_significant}
    else:
        print(f"\np >= 0.05 -> KHÔNG đủ bằng chứng bác bỏ H0 (4 phương pháp có Rep "
              f"như nhau) -- không chạy Nemenyi post-hoc, báo cáo trung thực kết quả này.")

    return {
        "n_blocks_total": n_blocks_total, "methods": C.E4.METHODS,
        "rep_matrix": rep_matrix.tolist(), "time_matrix": time_matrix.tolist(),
        "block_meta": block_meta,
        "friedman_stat": float(stat), "friedman_p": float(p),
        "ranks_rep": ranks_rep.tolist(), "ranks_time": ranks_time.tolist(),
        "nemenyi": nemenyi_results,
    }


if __name__ == "__main__":
    os.makedirs(C.PATHS.RESULTS_DIR, exist_ok=True)
    out = run_e4()
    with open(os.path.join(C.PATHS.RESULTS_DIR, "E4_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu kết quả vào {C.PATHS.RESULTS_DIR}/E4_results.json")
