"""
E1 (RQ1) — Kiểm chứng cận lý thuyết bằng vét cạn, v3.

Cho MỖI mẫu con nhỏ (n <= 18) lấy từ luật huấn luyện của fold 0:
  * GreedyFKGS : rho = Rep/OPT                 (cận 1-1/e, Định lý 2.22)
  * SFKGS      : Rep/OPT_pb(Pi_L), tỉ lệ cục bộ, sai số phân rã g (=0 theo Hq. 2.33)
  * CFKGS      : Rep/OPT_pb(Pi_C), g >= 0, chuỗi OPT_pb(Pi_C) <= OPT_pb(Pi_L) <= OPT
  * Độ lệch tỉ lệ lớp < 1/m (Mệnh đề 2.40) với ngân sách exact.
In cả m thực và m/n thực (QA Stage 4: v2 in tỉ lệ yêu cầu).
Không dựng Sim trên toàn tập huấn luyện (QA: bộ nhớ).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from itertools import combinations
import numpy as np

import config as C
from experiments.common import load_clean, save_json
from src.kfold_utils import make_stratified_kfold, build_fkg_fold
from src.fkg_sim_rep import compute_sim_matrix, rep_value
from src.fkgs_algorithms import (GreedyFKGS, greedy_submodular_select, allocate_largest_remainder,
                                 form_clusters)

BOUND = 1 - 1 / np.e


def rep_local(sim, omega, S):
    omega = list(omega)
    if len(S) == 0:
        return 0.0
    return float(np.maximum(sim[np.ix_(omega, list(S))].max(axis=1), 0).mean())


def opt_local(sim, omega, k):
    omega = list(omega)
    k = min(k, len(omega))
    if k == 0:
        return 0.0
    return max(rep_local(sim, omega, c) for c in combinations(omega, k))


def brute_force_optimal(sim, m):
    n = sim.shape[0]
    if m > n:
        raise ValueError("m > n")
    if m == 0:
        return [], 0.0
    best, best_S = -1.0, None
    for S in combinations(range(n), m):
        r = rep_value(sim, S)
        if r > best:
            best, best_S = r, list(S)
    return best_S, best


def partition_run(sim, parts, budgets):
    """Greedy trong từng phần với ngân sách cho trước; trả các đại lượng E1."""
    n = sim.shape[0]
    S, locs, wsum, optpb = [], [], 0.0, 0.0
    for P, mb in zip(parts, budgets):
        Si = greedy_submodular_select(sim, P, mb)
        S += Si
        rl, ol = rep_local(sim, P, Si), opt_local(sim, P, mb)
        wsum += len(P) / n * rl
        optpb += len(P) / n * ol
        if ol > 1e-12:
            locs.append(rl / ol)
    rep = rep_value(sim, S)
    return dict(S=S, rep=rep, opt_pb=optpb, ratio_pb=rep / optpb if optpb > 0 else 1.0,
                loc_min=min(locs) if locs else 1.0, g=rep - wsum)


def run_e1():
    df = load_clean()
    tr, te = make_stratified_kfold(df)[0]
    built = build_fkg_fold(df.iloc[tr].reset_index(drop=True), None, compute_sim=False)
    R = built["rules_train_full"]
    rows = []
    for n in C.E1.N_VALUES:
        for ratio in C.E1.RATIO_VALUES:
            m = max(1, int(round(ratio * n)))
            for rep_i in range(C.E1.N_REPEATS):
                rng = np.random.RandomState(1000 * n + 10 * rep_i + int(ratio * 100))
                idx = rng.choice(len(R), n, replace=False)
                Rs = [R[i] for i in idx]
                sim = compute_sim_matrix(Rs)
                lab = np.array([r[-1] for r in Rs])
                _, OPT = brute_force_optimal(sim, m)
                Sg = GreedyFKGS(sim, m=m)
                rep_g = rep_value(sim, Sg)
                # SFKGS: phân tầng theo nhãn, ngân sách exact
                strata = [np.where(lab == c)[0] for c in sorted(set(lab.tolist()))]
                bs = allocate_largest_remainder([len(s) for s in strata], m)
                sf = partition_run(sim, strata, bs)
                # CFKGS: 2 cụm trong mỗi nhãn, phân bổ 2 cấp
                clusters = form_clusters(Rs, n_clusters_per_label=2, seed=0, within_label=True)
                bc = []
                for s, b in zip(strata, bs):
                    cl = [c for c in clusters if lab[c[0]] == lab[s[0]]]
                    bc_l = allocate_largest_remainder([len(c) for c in cl], b)
                    bc.append((cl, bc_l))
                parts_c = [c for cl, _ in bc for c in cl]
                budgets_c = [x for _, b in bc for x in b]
                cf = partition_run(sim, parts_c, budgets_c)
                p_true = lab.mean()
                rows.append(dict(
                    n=n, ratio_requested=ratio, m=m, ratio_actual=m / n, repeat=rep_i,
                    sample_positions=idx.tolist(), opt=OPT,
                    greedy_rep=rep_g, greedy_rho=rep_g / OPT if OPT > 0 else 1.0,
                    greedy_le_opt=bool(rep_g <= OPT + 1e-12),
                    sfkgs_rep=sf["rep"], sfkgs_opt_pb=sf["opt_pb"], sfkgs_ratio_pb=sf["ratio_pb"],
                    sfkgs_loc_min=sf["loc_min"], sfkgs_g=sf["g"],
                    sfkgs_classdev=abs(lab[sf["S"]].mean() - p_true), sfkgs_size=len(sf["S"]),
                    cfkgs_rep=cf["rep"], cfkgs_opt_pb=cf["opt_pb"], cfkgs_ratio_pb=cf["ratio_pb"],
                    cfkgs_loc_min=cf["loc_min"], cfkgs_g=cf["g"],
                    cfkgs_classdev=abs(lab[cf["S"]].mean() - p_true), cfkgs_size=len(cf["S"]),
                    chain_ok=bool(cf["opt_pb"] <= sf["opt_pb"] + 1e-12 and sf["opt_pb"] <= OPT + 1e-12),
                ))
        print(f"  E1 n={n} xong", flush=True)
    viol = dict(
        greedy=sum(r["greedy_rho"] < BOUND - 1e-9 for r in rows),
        sfkgs=sum(r["sfkgs_ratio_pb"] < BOUND - 1e-9 or r["sfkgs_loc_min"] < BOUND - 1e-9 for r in rows),
        cfkgs=sum(r["cfkgs_ratio_pb"] < BOUND - 1e-9 or r["cfkgs_loc_min"] < BOUND - 1e-9 for r in rows),
        sfkgs_decomposition=sum(abs(r["sfkgs_g"]) > 1e-9 for r in rows),
        cfkgs_negative_g=sum(r["cfkgs_g"] < -1e-9 for r in rows),
        chain=sum(not r["chain_ok"] for r in rows),
        classdev_sfkgs=sum(r["sfkgs_classdev"] >= 1 / r["m"] for r in rows),
        classdev_cfkgs=sum(r["cfkgs_classdev"] >= 1 / r["m"] for r in rows),
        greedy_above_opt=sum(not r["greedy_le_opt"] for r in rows),
        size_mismatch=sum(r["sfkgs_size"] != r["m"] or r["cfkgs_size"] != r["m"] for r in rows),
    )
    save_json("E1_results.json", dict(results=rows, theoretical_bound=BOUND, violations=viol,
                                      fold=0, n_rules_fold0_train=len(R)))
    print("  Vi phạm:", viol)
    return rows, viol


if __name__ == "__main__":
    run_e1()
