"""
E2 (RQ2) — So sánh tại cùng ngân sách |S| = ceil(theta*n), v3.

Mỗi fold: FKG đầy đủ (trên tập nền và trên toàn tập huấn luyện) làm mốc;
GreedyFKGS, SFKGS, CFKGS (tất định) và Random, StratRandom (N_SEEDS hạt giống).
Ghi DỮ LIỆU THÔ theo fold/phương pháp/theta/seed: |S|, Rep, độ phủ, tỉ lệ lớp,
AUC/Acc/BalAcc/F1, thời gian tách Sim/chọn luật/xây FISA/suy diễn, và chỉ số
hàng (trong Data_clean) của các luật được chọn (phương pháp tất định).

Suy luận (QA-05): ĐƠN VỊ là fold (n = K). Với Random/StratRandom, giá trị của
fold là trung bình trên các hạt giống; biến thiên theo hạt giống chỉ được báo
cáo mô tả (tỉ lệ hạt giống bị vượt). Các fold dùng tập huấn luyện chồng lấp nên
p-value chỉ mang tính mô tả cho bộ dữ liệu này; kèm chênh lệch và CI.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import config as C
from experiments.common import load_clean, save_json, evaluate_fisa
from src.kfold_utils import make_stratified_kfold, build_fkg_fold
from src.fkg_sim_rep import coverage_stats
from src.fkgs_algorithms import GreedyFKGS, SFKGS, CFKGS, RandomSampling, StratRandom, budget_from_theta
from src.utils_stats import wilcoxon_one_sided, holm_bonferroni, cohens_d_paired, t_ci

DET = ["GreedyFKGS", "SFKGS", "CFKGS"]
RND = ["Random", "StratRandom"]


def select(method, sim, rules, labels, theta, seed=0):
    n = len(rules)
    if method == "GreedyFKGS":
        return GreedyFKGS(sim, theta=theta)
    if method == "SFKGS":
        return SFKGS(sim, labels, theta)
    if method == "CFKGS":
        return CFKGS(sim, rules, theta, seed=0)
    if method == "Random":
        return RandomSampling(n, theta=theta, seed=seed)
    if method == "StratRandom":
        return StratRandom(labels, theta, seed=seed)
    raise ValueError(method)


def run_e2():
    df = load_clean()
    folds = make_stratified_kfold(df)
    raw = []
    for f, (tr, te) in enumerate(folds):
        t0 = time.perf_counter()
        b = build_fkg_fold(df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True),
                           n_sample=C.SCALE.N_EXP, seed=f)
        t_build = time.perf_counter() - t0
        R, RT, sim = b["rules_train"], b["rules_test"], b["sim"]
        row_ids = np.asarray(tr)[b["pool_positions"]]           # chỉ số trong Data_clean
        labels = [r[-1] for r in R]
        n = len(R)
        base = dict(fold=f, n_pool=n, n_train=len(b["rules_train_full"]), n_test=len(RT),
                    pos_share_pool=float(np.mean(labels)))
        for ref, rules_ref in [("FKG_full_pool", R), ("FKG_full_train", b["rules_train_full"])]:
            ev = evaluate_fisa(rules_ref, RT)
            raw.append({**base, "method": ref, "theta": 1.0, "seed": None, "size": len(rules_ref),
                        "rep": 1.0 if ref == "FKG_full_pool" else None, "t_select": 0.0,
                        "t_build_fold": t_build, **ev})
        for theta in C.E2.THETA_VALUES:
            m = budget_from_theta(theta, n)
            jobs = [(mm, 0) for mm in DET] + [(mm, s) for mm in RND for s in range(C.E2.N_SEEDS_RANDOM)]
            for method, seed in jobs:
                t0 = time.perf_counter()
                S = select(method, sim, R, labels, theta, seed)
                t_sel = time.perf_counter() - t0
                if len(S) != m:
                    raise AssertionError(f"{method}: |S|={len(S)} != m={m} (vi phạm ngân sách)")
                cs = coverage_stats(sim, S, C.E3.COVER_DELTA)
                ev = evaluate_fisa([R[i] for i in S], RT)
                rec = {**base, "method": method, "theta": theta, "seed": seed if method in RND else None,
                       "size": len(S), "rep": cs["rep"], "min_cov": cs["min_cov"],
                       "frac_covered": cs["frac_covered"],
                       "pos_share": float(np.mean([labels[i] for i in S])) if S else float("nan"),
                       "t_build_fold": t_build, "t_select": t_sel, **ev}
                if method in DET:
                    rec["selected_row_ids"] = row_ids[S].tolist()
                raw.append(rec)
            print(f"  E2 fold {f} theta={theta} xong", flush=True)
    summary = summarize_e2(raw)
    save_json("E2_results.json", dict(raw=raw, summary=summary,
                                      fold_test_ids=[np.asarray(te).tolist() for _, te in folds]))
    return summary


def _fold_values(raw, method, theta, key):
    vals = []
    for f in sorted(set(r["fold"] for r in raw)):
        v = [r[key] for r in raw if r["fold"] == f and r["method"] == method and r["theta"] == theta
             and r.get(key) is not None]
        vals.append(np.mean(v) if v else np.nan)
    return np.array(vals, float)


def summarize_e2(raw):
    alpha = C.STATS.ALPHA
    out = {"per_method": [], "tests": []}
    methods = DET + RND
    for theta in C.E2.THETA_VALUES:
        for mth in methods:
            rec = dict(theta=theta, method=mth)
            for key in ["size", "rep", "frac_covered", "min_cov", "auc", "acc", "bacc", "pos_share",
                        "t_select", "t_fit", "t_pred"]:
                v = _fold_values(raw, mth, theta, key)
                rec[key + "_mean"] = float(np.nanmean(v))
                rec[key + "_sd"] = float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else float("nan")
            out["per_method"].append(rec)
    for ref in ["FKG_full_pool", "FKG_full_train"]:
        rec = dict(theta=1.0, method=ref)
        for key in ["auc", "acc", "bacc", "pos_share_pool", "t_fit", "t_pred", "size"]:
            v = _fold_values(raw, ref, 1.0, key)
            rec[key + "_mean"] = float(np.nanmean(v))
            rec[key + "_sd"] = float(np.nanstd(v, ddof=1))
        out["per_method"].append(rec)

    # Họ kiểm định: (metric, baseline). Đơn vị = fold. Holm trong mỗi họ.
    families = [("rep", "Random"), ("rep", "StratRandom"), ("auc", "Random"),
                ("auc", "StratRandom"), ("auc", "FKG_full_pool")]
    for metric, baseline in families:
        tests = []
        for theta in C.E2.THETA_VALUES:
            for mth in DET:
                x = _fold_values(raw, mth, theta, metric)
                y = _fold_values(raw, baseline, 1.0 if baseline.startswith("FKG") else theta, metric)
                d = x - y
                stat, p, meth = wilcoxon_one_sided(x, y, "greater")
                seed_frac = None
                if baseline in RND:
                    wins, tot = 0, 0
                    for f in sorted(set(r["fold"] for r in raw)):
                        xv = [r[metric] for r in raw if r["fold"] == f and r["method"] == mth and r["theta"] == theta][0]
                        ys = [r[metric] for r in raw if r["fold"] == f and r["method"] == baseline and r["theta"] == theta]
                        wins += sum(xv > yv for yv in ys)
                        tot += len(ys)
                    seed_frac = wins / tot
                tests.append(dict(metric=metric, baseline=baseline, theta=theta, method=mth,
                                  mean_diff=float(np.nanmean(d)), ci95=t_ci(d),
                                  folds_better=int(np.sum(d > 0)), n_folds=int(np.isfinite(d).sum()),
                                  cohens_dz=cohens_d_paired(x, y), p_raw=p, p_method=meth,
                                  frac_seeds_beaten=seed_frac))
        adj, rej = holm_bonferroni([t["p_raw"] for t in tests], alpha)
        for t, a, r in zip(tests, adj, rej):
            t["p_holm"] = float(a) if np.isfinite(a) else None
            t["reject"] = bool(r)
        out["tests"] += tests
    return out


if __name__ == "__main__":
    run_e2()
