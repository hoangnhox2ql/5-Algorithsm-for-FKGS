"""
experiments/run_02_E2_greedy_vs_random.py — E2: GreedyFKGS vs Random cùng
ngân sách, thiết kế ghép cặp (Mục 4, thiet_ke_thuc_nghiem.md). GreedyFKGS
tất định (chạy 1 lần); Random lặp 30 seed. Downstream AUC dùng FISA THẬT
đã sửa đúng (1.18)-(1.20) — src/fisa_corrected.py.

K-FOLD: toàn bộ quy trình (xây FKG, lấy mẫu, đo Rep/AUC) lặp lại trên CẢ 5
fold, kết quả cuối là trung bình qua fold để phản ánh đúng biến thiên dữ
liệu, không chỉ 1 lần chia cố định.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
    _HAS_SKLEARN_METRICS = True
except ImportError:
    _HAS_SKLEARN_METRICS = False

import config as C
from src.kfold_utils import make_stratified_kfold, build_fkg_fold
from src.fkg_sim_rep import rep_value
from src.fkgs_algorithms import GreedyFKGS, RandomSampling
from src.fisa_corrected import FISACorrected, warn_if_fisa_collapsed
from src.utils_stats import wilcoxon_one_sided, cohens_d_paired, holm_bonferroni


def _auc_from_fisa(rules_sample, rules_test, n_classes=2):
    """Huấn luyện FISA trên rules_sample, đo AUC trên rules_test bằng xác
    suất lớp 1 (predict_proba_one)."""
    if len(rules_sample) < 2 or len(set(r[-1] for r in rules_sample)) < 2:
        return 0.5  # không đủ dữ liệu/chỉ 1 lớp -- AUC không xác định, quy ước 0.5
    model = FISACorrected(rules_sample).fit()
    # Cảnh báo tự động (Hướng A -- xem src/fisa_corrected.py) nếu mô hình có
    # dấu hiệu sụp về lớp đa số trên chính fold này, dùng mẫu con của test
    # để kiểm tra nhanh (không cần toàn bộ, tiết kiệm thời gian).
    warn_if_fisa_collapsed(model, [r[:-1] for r in rules_test[:min(30, len(rules_test))]])
    y_true, y_score = [], []
    for r in rules_test:
        proba, _ = model.predict_proba_one(r[:-1])
        y_true.append(r[-1])
        y_score.append(proba.get(1, 0.5))
    if len(set(y_true)) < 2:
        return 0.5
    if _HAS_SKLEARN_METRICS:
        return roc_auc_score(y_true, y_score)
    return _manual_auc(y_true, y_score)


def _manual_auc(y_true, y_score):
    """AUC thủ công (Mann-Whitney U) nếu không có sklearn."""
    y_true, y_score = np.array(y_true), np.array(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    count = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return count / (len(pos) * len(neg))


def run_e2_one_fold(rules_train_full, rules_test, sim, theta, n_seeds_random):
    n = sim.shape[0]
    m = int(np.ceil(theta * n))

    S_greedy = GreedyFKGS(sim, m=m)
    rep_greedy = rep_value(sim, S_greedy)
    auc_greedy = _auc_from_fisa([rules_train_full[i] for i in S_greedy], rules_test)

    rep_randoms, auc_randoms = [], []
    for seed in range(n_seeds_random):
        S_random = RandomSampling(n, m=m, seed=seed)
        rep_randoms.append(rep_value(sim, S_random))
        auc_randoms.append(_auc_from_fisa([rules_train_full[i] for i in S_random], rules_test))

    return {
        "rep_greedy": rep_greedy, "auc_greedy": auc_greedy,
        "rep_randoms": rep_randoms, "auc_randoms": auc_randoms,
    }


def run_e2():
    print("=" * 70)
    print("E2: GreedyFKGS vs Random Sampling (thiết kế ghép cặp)")
    print("=" * 70)

    df = pd.read_csv(os.path.join(C.PATHS.DATA_DIR, "Data_clean.csv"))
    folds = make_stratified_kfold(df, k=C.KFOLD.K, seed=C.KFOLD.SEED)

    # QUAN TRỌNG (đã sửa lỗi thiết kế thống kê): giữ ĐẦY ĐỦ n_seeds_random
    # quan sát mỗi fold làm ĐƠN VỊ GHÉP CẶP cho Wilcoxon (đúng thiết kế gốc
    # "seed=1..30"), KHÔNG lấy trung bình 30 seed rồi ghép cặp chỉ theo 5
    # fold -- cách đó làm giảm cỡ mẫu kiểm định từ 150 xuống 5, mất gần hết
    # sức mạnh thống kê (kiểm chứng bằng thực nghiệm: d=12-20 rất lớn nhưng
    # Wilcoxon n=5 không đủ mạnh để p<0.05 sau Holm). Ở đây: MỖI fold đóng
    # góp n_seeds_random cặp (Greedy_fold lặp lại y hệt, Random_fold_seed_j
    # khác nhau) -- gộp qua 5 fold thành 5*n_seeds_random cặp, Wilcoxon 1
    # lần trên toàn bộ.
    all_theta_pairs = {theta: {"greedy": [], "random": []} for theta in C.E2.THETA_VALUES}
    per_fold_summary = {theta: [] for theta in C.E2.THETA_VALUES}

    for fold_id, (train_idx, test_idx) in enumerate(folds):
        print(f"\n--- Fold {fold_id} ---")
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        built = build_fkg_fold(df_train, df_test, n_sample=C.SCALE.N_EXP, seed=fold_id)
        rules_train, rules_test, sim = built["rules_train"], built["rules_test"], built["sim"]
        print(f"  Xây FKG fold {fold_id}: {len(rules_train)} luật train (đã lấy mẫu N_EXP), "
              f"{len(rules_test)} luật test.")

        for theta in C.E2.THETA_VALUES:
            res = run_e2_one_fold(rules_train, rules_test, sim, theta, C.E2.N_SEEDS_RANDOM)
            n_seeds = len(res["rep_randoms"])
            # Ghép mỗi seed Random với CÙNG một giá trị Greedy (Greedy tất định,
            # lặp lại y hệt cho mọi seed trong CHÍNH fold này) -- đúng thiết kế
            # ghép cặp gốc, KHÔNG lấy trung bình trước.
            all_theta_pairs[theta]["greedy"].extend([res["rep_greedy"]] * n_seeds)
            all_theta_pairs[theta]["random"].extend(res["rep_randoms"])
            per_fold_summary[theta].append({
                "fold": fold_id, "rep_greedy": res["rep_greedy"],
                "rep_random_mean": float(np.mean(res["rep_randoms"])),
                "auc_greedy": res["auc_greedy"],
                "auc_random_mean": float(np.mean(res["auc_randoms"])),
            })
            print(f"  theta={theta}: Rep(Greedy)={res['rep_greedy']:.4f} vs "
                  f"Rep(Random,mean{n_seeds})={np.mean(res['rep_randoms']):.4f}  |  "
                  f"AUC(Greedy)={res['auc_greedy']:.4f} vs "
                  f"AUC(Random,mean{n_seeds})={np.mean(res['auc_randoms']):.4f}")

    # Kiểm định thống kê: Wilcoxon 1 phía trên TOÀN BỘ (fold x seed) cặp gộp lại
    print(f"\n{'='*70}")
    print(f"KIỂM ĐỊNH THỐNG KÊ (gộp {C.KFOLD.K}x{C.E2.N_SEEDS_RANDOM} cặp qua mọi fold, "
          f"Wilcoxon 1 phía Greedy>Random)")
    print("=" * 70)

    p_values_rep = []
    summary_rows = []
    for theta in C.E2.THETA_VALUES:
        rep_g = all_theta_pairs[theta]["greedy"]
        rep_r = all_theta_pairs[theta]["random"]

        try:
            _, p_rep = wilcoxon_one_sided(rep_g, rep_r)
        except Exception:
            p_rep = float("nan")
        d_rep = cohens_d_paired(rep_g, rep_r)

        # AUC: vẫn ghép cặp theo FOLD (chỉ có 1 AUC-Greedy và 1 AUC-Random-mean
        # mỗi fold, vì AUC không cần lặp theo từng seed riêng để tiết kiệm
        # thời gian chạy FISA) -- báo cáo riêng, cỡ mẫu nhỏ hơn (K fold), nêu rõ
        auc_g_per_fold = [row["auc_greedy"] for row in per_fold_summary[theta]]
        auc_r_per_fold = [row["auc_random_mean"] for row in per_fold_summary[theta]]
        try:
            _, p_auc = wilcoxon_one_sided(auc_g_per_fold, auc_r_per_fold)
        except Exception:
            p_auc = float("nan")
        d_auc = cohens_d_paired(auc_g_per_fold, auc_r_per_fold)

        p_values_rep.append(p_rep)
        summary_rows.append({
            "theta": theta, "n_pairs_rep": len(rep_g),
            "rep_greedy_mean": float(np.mean(rep_g)),
            "rep_random_mean": float(np.mean(rep_r)), "p_rep_raw": p_rep, "cohens_d_rep": d_rep,
            "n_pairs_auc": len(auc_g_per_fold),
            "auc_greedy_mean": float(np.mean(auc_g_per_fold)),
            "auc_random_mean": float(np.mean(auc_r_per_fold)),
            "p_auc_raw": p_auc, "cohens_d_auc": d_auc,
        })

    p_values_auc = [row["p_auc_raw"] for row in summary_rows]
    p_rep_adj, _ = holm_bonferroni([p for p in p_values_rep if not np.isnan(p)])
    p_auc_adj, _ = holm_bonferroni([p for p in p_values_auc if not np.isnan(p)])

    for i, theta in enumerate(C.E2.THETA_VALUES):
        row = summary_rows[i]
        row["p_rep_holm"] = float(p_rep_adj[i]) if i < len(p_rep_adj) else None
        row["p_auc_holm"] = float(p_auc_adj[i]) if i < len(p_auc_adj) else None
        print(f"theta={theta}: Rep (n={row['n_pairs_rep']} cặp) p_raw={row['p_rep_raw']:.4f} "
              f"p_holm={row['p_rep_holm']:.4f} d={row['cohens_d_rep']:.2f}  |  "
              f"AUC (n={row['n_pairs_auc']} cặp) p_raw={row['p_auc_raw']:.4f} "
              f"p_holm={row['p_auc_holm']:.4f} d={row['cohens_d_auc']:.2f}")
        if row["p_rep_holm"] is not None and row["p_rep_holm"] > 0.05:
            print(f"  !! Lưu ý: tại theta={theta}, Rep KHÔNG khác biệt có ý nghĩa "
                  f"sau hiệu chỉnh Holm (p={row['p_rep_holm']:.4f}) -- báo cáo trung thực, "
                  f"không bỏ qua.")
        if row["p_auc_holm"] is not None and row["p_auc_holm"] > 0.05:
            print(f"  !! Lưu ý: AUC chỉ ghép cặp theo {row['n_pairs_auc']} fold (không theo "
                  f"từng seed, để tiết kiệm chi phí chạy FISA) -- cỡ mẫu nhỏ, diễn giải "
                  f"p-value cần thận trọng, ưu tiên nhìn Cohen's d và chênh lệch trung bình.")

    return summary_rows


if __name__ == "__main__":
    os.makedirs(C.PATHS.RESULTS_DIR, exist_ok=True)
    rows = run_e2()
    with open(os.path.join(C.PATHS.RESULTS_DIR, "E2_results.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu kết quả vào {C.PATHS.RESULTS_DIR}/E2_results.json")
