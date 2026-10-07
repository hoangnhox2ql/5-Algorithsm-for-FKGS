"""
E4 (RQ4) — So sánh các chiến lược tại theta cố định, CÙNG NGÂN SÁCH (v3).

Sửa QA-04/05: mọi phương pháp chọn đúng m = ceil(theta*n) luật (kiểm tra
bằng assert, |S| được lưu); Friedman/Nemenyi được báo ở hai mức:
  * PRIMARY: 5 block = trung bình theo fold (đơn vị gần độc lập nhất có được);
  * mô tả: 50 block (fold x mẫu con) -- có điều kiện trên tập huấn luyện của fold,
    KHÔNG coi là 50 bộ dữ liệu độc lập.
Chỉ dựng Sim cho từng block (không dựng trên toàn tập huấn luyện).
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import config as C
from experiments.common import load_clean, save_json
from experiments.run_02_E2_greedy_vs_random import select
from src.kfold_utils import make_stratified_kfold, build_fkg_fold, stratified_subsample
from src.fkg_sim_rep import compute_sim_matrix, coverage_stats
from src.fkgs_algorithms import budget_from_theta
from src.utils_stats import friedman_test, average_ranks, nemenyi_critical_difference


def _ranking(M, methods, higher):
    stat, p = friedman_test(M)
    ranks = average_ranks(M, higher_is_better=higher)
    cd = nemenyi_critical_difference(M.shape[0], M.shape[1], C.STATS.ALPHA)
    pairs = [(methods[i], methods[j], float(abs(ranks[i] - ranks[j])))
             for i in range(len(methods)) for j in range(i + 1, len(methods))
             if abs(ranks[i] - ranks[j]) > cd]
    return dict(friedman_stat=stat, friedman_p=p, ranks=ranks.tolist(), cd=cd,
                significant_pairs=pairs, n_blocks=int(M.shape[0]))


def run_e4():
    df = load_clean()
    folds = make_stratified_kfold(df)
    methods = C.E4.METHODS
    rows = []
    for f, (tr, _) in enumerate(folds):
        b = build_fkg_fold(df.iloc[tr].reset_index(drop=True), None, compute_sim=False)
        R_all = b["rules_train_full"]
        labels_all = [r[-1] for r in R_all]
        for blk in range(C.E4.N_BLOCKS):
            pos = stratified_subsample(labels_all, C.E4.BLOCK_SIZE, seed=1000 * f + blk)
            R = [R_all[i] for i in pos]
            sim = compute_sim_matrix(R)
            lab = [r[-1] for r in R]
            m = budget_from_theta(C.E4.THETA, len(R))
            for mth in methods:
                t0 = time.perf_counter()
                S = select(mth, sim, R, lab, C.E4.THETA, seed=0)
                t = time.perf_counter() - t0
                if len(S) != m:
                    raise AssertionError(f"E4 {mth}: |S|={len(S)} != {m}")
                cs = coverage_stats(sim, S, C.E3.COVER_DELTA)
                rows.append(dict(fold=f, block=blk, method=mth, size=len(S), rep=cs["rep"],
                                 frac_covered=cs["frac_covered"], time_s=t,
                                 pos_share=float(np.mean([lab[i] for i in S]))))
        print(f"  E4 fold {f} xong", flush=True)
    def matrix(key, level):
        if level == "block":
            keys = sorted(set((r["fold"], r["block"]) for r in rows))
            return np.array([[next(r[key] for r in rows if (r["fold"], r["block"]) == k and r["method"] == m)
                              for m in methods] for k in keys])
        return np.array([[np.mean([r[key] for r in rows if r["fold"] == f and r["method"] == m])
                          for m in methods] for f in sorted(set(r["fold"] for r in rows))])
    res = dict(methods=methods, theta=C.E4.THETA, block_size=C.E4.BLOCK_SIZE, raw=rows,
               rep_fold=_ranking(matrix("rep", "fold"), methods, True),
               rep_block=_ranking(matrix("rep", "block"), methods, True),
               time_fold=_ranking(matrix("time_s", "fold"), methods, False),
               time_block=_ranking(matrix("time_s", "block"), methods, False))
    res["summary"] = [dict(method=m, size=float(np.mean([r["size"] for r in rows if r["method"] == m])),
                           rep_mean=float(np.mean([r["rep"] for r in rows if r["method"] == m])),
                           rep_sd=float(np.std([r["rep"] for r in rows if r["method"] == m], ddof=1)),
                           frac_covered=float(np.mean([r["frac_covered"] for r in rows if r["method"] == m])),
                           pos_share=float(np.mean([r["pos_share"] for r in rows if r["method"] == m])),
                           time_ms=1000 * float(np.mean([r["time_s"] for r in rows if r["method"] == m])))
                      for m in methods]
    save_json("E4_results.json", res)
    return res


if __name__ == "__main__":
    run_e4()
