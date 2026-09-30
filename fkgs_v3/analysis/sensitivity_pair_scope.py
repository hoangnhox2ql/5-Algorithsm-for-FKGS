"""
experiments/sensitivity_pair_scope.py — Độ nhạy của AUC theo quy ước miền cặp
của FISA (QA-02): global | adjacent | upper (v2). Chạy trên cùng fold/tập nền
như E2, cho FKG đầy đủ và GreedyFKGS/SFKGS/CFKGS tại theta = 0.3.

  python analysis/sensitivity_pair_scope.py --dataset heart
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--theta", type=float, default=0.3)
    a = ap.parse_args()
    C.set_dataset(a.dataset)
    from experiments.common import load_raw_data, preprocess, evaluate_fisa
    from experiments.run_02_E2_greedy_vs_random import select
    from src.kfold_utils import make_stratified_kfold, build_fkg_fold
    df, prov = load_raw_data()
    df, _ = preprocess(df)
    rows = []
    for f, (tr, te) in enumerate(make_stratified_kfold(df)):
        b = build_fkg_fold(df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True),
                           n_sample=C.SCALE.N_EXP, seed=f)
        R, RT, sim = b["rules_train"], b["rules_test"], b["sim"]
        lab = [r[-1] for r in R]
        sets = {"FKG_full": list(range(len(R)))}
        for m in ["GreedyFKGS", "SFKGS", "CFKGS"]:
            sets[m] = select(m, sim, R, lab, a.theta)
        for scope in ["global", "adjacent", "upper"]:
            C.FISA.PAIR_SCOPE = scope
            for m, S in sets.items():
                ev = evaluate_fisa([R[i] for i in S], RT)
                rows.append(dict(fold=f, scope=scope, method=m, auc=ev["auc"], bacc=ev["bacc"]))
        print(f"  fold {f} xong", flush=True)
    d = pd.DataFrame(rows)
    tab = d.groupby(["scope", "method"])[["auc", "bacc"]].agg(["mean", "std"]).round(4)
    print(tab)
    out = os.path.join(C.PATHS.RESULTS_DIR, "sensitivity_pair_scope.json")
    os.makedirs(C.PATHS.RESULTS_DIR, exist_ok=True)
    json.dump(dict(theta=a.theta, data_sha256=prov["sha256"], rows=rows), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
