"""
analysis/diag_fisa.py — Chẩn đoán hiện tượng FISA "sụp về lớp đa số" và đối chiếu với
giao thức của bài APIN (chia cố định 70/30, FISA gốc với quy tắc D0 > 9*D1).
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from itertools import combinations as _comb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, f1_score
import config as C

def fisa_scores(enc, Cm, query):
    """FISA gốc (giữ nguyên 3 đặc điểm của cài đặt gốc) nhưng trả về (D0, D1)."""
    from src.fkg_fast import combination
    cols_3 = combination(3, enc.n_attr)
    qc = enc.encode_query(query)
    C0 = np.zeros(cols_3); C1 = np.zeros(cols_3); temp = 0
    considered = np.arange(enc.row - 1)
    for idx in _comb(range(enc.n_attr), 3):
        if np.any(qc[list(idx)] < 0):
            temp += 1; continue
        q = enc.radix_code_query(idx, qc)
        code = enc.radix_code(idx)[considered]
        mp = considered[code == q]
        if len(mp):
            m0 = mp[enc.labels[mp] == 0]; m1 = mp[enc.labels[mp] == 1]
            if len(m0): C0[temp] = Cm[m0.max(), temp]
            if len(m1): C1[temp] = Cm[m1.max(), temp + cols_3]
        temp += 1
    return C0.max() + C0.min(), C1.max() + C1.min()

def metrics(y, pred, score=None):
    d = dict(acc=accuracy_score(y, pred), bacc=balanced_accuracy_score(y, pred),
             f1=f1_score(y, pred, zero_division=0), pos_rate=float(np.mean(pred)))
    if score is not None and len(np.unique(y)) == 2:
        d["auc"] = roc_auc_score(y, score)
    return d

def build_rules(df_tr, df_te, fitter_kind):
    from src.fuzzification import FuzzyFitter, rules_from_fuzzy_df
    if fitter_kind == "v3":
        f = FuzzyFitter().fit(df_tr)
        return rules_from_fuzzy_df(f.transform(df_tr)), rules_from_fuzzy_df(f.transform(df_te))
    # v2: điền thiếu theo trung vị của lớp, dùng nhãn của chính hàng (kể cả hàng kiểm tra)
    from analysis.ablation_v2_to_v3 import ClassMedianFitter
    f = ClassMedianFitter(FuzzyFitter()).fit(df_tr)
    return rules_from_fuzzy_df(f.transform(df_tr)), rules_from_fuzzy_df(f.transform(df_te))

def run(dataset, dedup, fitter_kind, seed, test_size=0.3, max_test=None):
    from src.fkg_fast import EncodedRuleBase, caculateA_fast, caculateM_fast, caculateB_fast, caculateC_fast
    from src.fisa_corrected import FISACorrected
    C.set_dataset(dataset)
    df = pd.read_csv(C.PATHS.DATA_CSV)[C.FEATURE_COLUMNS + [C.LABEL_COLUMN]]
    if dedup:
        df = df.drop_duplicates().reset_index(drop=True)
    tr, te = train_test_split(df, test_size=test_size, stratify=df[C.LABEL_COLUMN], random_state=seed)
    R, RT = build_rules(tr.reset_index(drop=True), te.reset_index(drop=True), fitter_kind)
    if max_test: RT = RT[:max_test]
    y = np.array([r[-1] for r in RT])
    t0 = time.time()
    enc = EncodedRuleBase(R); A = caculateA_fast(enc); M = caculateM_fast(enc)
    Cm = caculateC_fast(enc, caculateB_fast(enc, A, M))
    D = np.array([fisa_scores(enc, Cm, r[:-1]) for r in RT])
    t_orig = time.time() - t0
    out = {"n_train": len(R), "n_test": len(RT), "prior_pos": float(np.mean([r[-1] for r in R]))}
    lr = np.log(D[:, 1] + 1e-12) - np.log(D[:, 0] + 1e-12)
    out["FISA_goc_k9"] = metrics(y, (~(D[:, 0] > 9 * D[:, 1])).astype(int), lr)
    out["FISA_goc_argmax"] = metrics(y, (D[:, 1] >= D[:, 0]).astype(int), lr)
    # quy tắc tỉ số với kappa theo tỉ lệ tiên nghiệm n0/n1 (hiệu chỉnh tiên nghiệm)
    n1 = sum(r[-1] for r in R); n0 = len(R) - n1
    out["FISA_goc_prior"] = metrics(y, (~(D[:, 0] > (n0 / n1) ** 2 * D[:, 1])).astype(int), lr)
    fp = FISACorrected(R, pair_scope="global").fit()
    ev = fp.evaluate(RT)
    out["FISA_P_argmax"] = metrics(ev["y"], ev["pred"], ev["score"])
    out["t_fisa_goc_s"] = t_orig
    # phân tích nguyên nhân: tỉ số trung vị D0/D1 theo nhãn thật
    ratio = D[:, 0] / np.maximum(D[:, 1], 1e-12)
    out["median_D0_over_D1_y0"] = float(np.median(ratio[y == 0])); out["median_D0_over_D1_y1"] = float(np.median(ratio[y == 1]))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="diabetes")
    ap.add_argument("--dedup", type=int, default=0); ap.add_argument("--fitter", default="v3")
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--max_test", type=int, default=0)
    a = ap.parse_args()
    r = run(a.dataset, bool(a.dedup), a.fitter, a.seed, max_test=a.max_test or None)
    print(json.dumps(r, indent=1, default=float))
