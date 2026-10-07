"""
analysis/legacy_build_cost.py — Chi phí xây FKG theo cài đặt gốc (fkg_original,
A/M/B/C) và bản vector hoá (fkg_fast) theo số luật n, trên luật fold 0 (v3).
Dùng cho E5 (RQ5).  python analysis/legacy_build_cost.py --dataset diabetes
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import config as C


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True); a = ap.parse_args()
    C.set_dataset(a.dataset)
    from experiments.common import load_raw_data, preprocess
    from src.kfold_utils import make_stratified_kfold, build_fkg_fold, stratified_subsample
    from src.fkg_original import caculateA, caculateM, caculateB, caculateC
    from src.fkg_fast import EncodedRuleBase, caculateA_fast, caculateM_fast, caculateB_fast, caculateC_fast
    df, prov = load_raw_data(); df, _ = preprocess(df)
    tr, _ = make_stratified_kfold(df)[0]
    R = build_fkg_fold(df.iloc[tr].reset_index(drop=True), None, compute_sim=False)["rules_train_full"]
    lab = [r[-1] for r in R]
    ns = [n for n in [37, 75, 150, 300, 450, 750, 1500] if n <= len(R)]
    if len(R) < 1500:
        ns.append(len(R))
    out = []
    for n in ns:
        Rn = [R[i] for i in stratified_subsample(lab, n, 0)]
        t0 = time.perf_counter(); enc = EncodedRuleBase(Rn); A = caculateA_fast(enc); M = caculateM_fast(enc)
        caculateC_fast(enc, caculateB_fast(enc, A, M)); tf = time.perf_counter() - t0
        t0 = time.perf_counter(); A = caculateA(Rn); M = caculateM(Rn); caculateC(Rn, caculateB(Rn, A, M))
        to = time.perf_counter() - t0
        out.append(dict(n=n, t_original=to, t_fast=tf)); print(out[-1], flush=True)
    x = np.log([o["n"] for o in out]); y = np.log([o["t_original"] for o in out])
    slope = float(np.polyfit(x, y, 1)[0])
    res = dict(dataset=a.dataset, data_sha256=prov["sha256"], points=out, slope_original=slope,
               n_train_full=len(R), t_original_extrapolated_full=float(np.exp(np.polyval(np.polyfit(x, y, 1), np.log(len(R))))))
    os.makedirs(C.PATHS.RESULTS_DIR, exist_ok=True)
    json.dump(res, open(os.path.join(C.PATHS.RESULTS_DIR, "legacy_build_cost.json"), "w"), indent=1)
    print("slope", slope)


if __name__ == "__main__":
    main()
