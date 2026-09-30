"""
experiments/ablation_v2_to_v3.py — Tách ảnh hưởng của từng sửa đổi v2 -> v3 lên AUC.

Ba yếu tố (2 x 2 x 2):
  impute : "class" (v2, đọc nhãn của hàng — rò rỉ QA-01) | "global" (v3)
  scope  : "upper" (v2, QA-02) | "global" (v3)
  score  : "softmax" (v2, xác suất softmax của D) | "margin" (v3, D1 - D0)
Đánh giá FKG đầy đủ trên tập nền và GreedyFKGS tại theta = 0.3, cùng fold/tập nền.

  python analysis/ablation_v2_to_v3.py --dataset diabetes
"""
import sys, os, json, argparse, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

import config as C


class ClassMedianFitter:
    """Tái hiện đúng hành vi v2: điền giá trị thiếu bằng trung vị THEO LỚP, trong
    đó lớp của mỗi hàng (kể cả hàng test) được đọc từ cột nhãn."""
    def __init__(self, base):
        self.b = base

    def fit(self, df):
        self.b.fit(df)
        self.cm = {}
        for col in self.b.numeric:
            if col not in self.b.zero_missing:
                continue
            self.cm[col] = {}
            for cls in df[C.LABEL_COLUMN].unique():
                s = pd.to_numeric(df.loc[df[C.LABEL_COLUMN] == cls, col], errors="coerce")
                s = s[(s != 0) & s.notna()]
                self.cm[col][cls] = float(s.median())
        return self

    def transform(self, df):
        d = df.copy()
        for col, med in self.cm.items():
            d[col] = d[col].astype(float)
            for cls, mv in med.items():
                d.loc[((d[col] == 0) | d[col].isna()) & (d[C.LABEL_COLUMN] == cls), col] = mv
        return self.b.transform(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--theta", type=float, default=0.3)
    a = ap.parse_args()
    C.set_dataset(a.dataset)
    from experiments.common import load_raw_data, preprocess
    from src.kfold_utils import make_stratified_kfold, stratified_subsample
    from src.fuzzification import FuzzyFitter, rules_from_fuzzy_df
    from src.fkg_sim_rep import compute_sim_matrix
    from src.fkgs_algorithms import GreedyFKGS
    from src.fisa_corrected import FISACorrected
    from src.utils_stats import safe_auc

    df, prov = load_raw_data()
    df, _ = preprocess(df)
    rows = []
    for f, (tr, te) in enumerate(make_stratified_kfold(df)):
        dtr, dte = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
        for impute in ["class", "global"]:
            fit = FuzzyFitter()
            fit = ClassMedianFitter(fit).fit(dtr) if impute == "class" else fit.fit(dtr)
            RF = rules_from_fuzzy_df(fit.transform(dtr))
            RT = rules_from_fuzzy_df(fit.transform(dte))
            pos = stratified_subsample([r[-1] for r in RF], C.SCALE.N_EXP, f) \
                if C.SCALE.N_EXP < len(RF) else np.arange(len(RF))
            R = [RF[i] for i in pos]
            S = GreedyFKGS(compute_sim_matrix(R), theta=a.theta)
            for scope in ["upper", "global"]:
                for name, rules in [("FKG_full_pool", R), ("GreedyFKGS", [R[i] for i in S])]:
                    m = FISACorrected(rules, pair_scope=scope).fit()
                    y = [r[-1] for r in RT]
                    sm = [m.predict_proba_one(r[:-1])[0].get(1, 0.5) for r in RT]
                    mg = [m.score_one(r[:-1]) for r in RT]
                    for score, s in [("softmax", sm), ("margin", mg)]:
                        rows.append(dict(fold=f, impute=impute, scope=scope, score=score, method=name,
                                         auc=safe_auc(y, s)[0],
                                         n_unique=len(np.unique(np.round(s, 12)))))
        print(f"  fold {f} xong", flush=True)
    d = pd.DataFrame(rows)
    tab = d.groupby(["method", "impute", "scope", "score"])[["auc", "n_unique"]].mean().round(4)
    print(tab.to_string())
    os.makedirs(C.PATHS.RESULTS_DIR, exist_ok=True)
    json.dump(dict(theta=a.theta, data_sha256=prov["sha256"], rows=rows),
              open(os.path.join(C.PATHS.RESULTS_DIR, "ablation_v2_to_v3.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
