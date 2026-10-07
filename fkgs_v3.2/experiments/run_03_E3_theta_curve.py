"""
E3 (RQ3) — Đường cong theo theta, chọn theta trên VALIDATION, đánh giá trên TEST (v3).

Sửa QA-06/08/12 so với v2:
  * Trong mỗi fold ngoài, tập huấn luyện được chia tiếp (phân tầng) thành
    train-trong / validation. Đường cong theta được dựng trên VALIDATION; theta*
    chọn bằng điểm gối trên MẶT PARETO (chi phí = thời gian chọn luật + xây FISA
    + suy diễn; lợi ích = AUC validation). Sau đó dựng lại trên toàn tập huấn
    luyện của fold và báo AUC TEST tại theta* -- test không tham gia lựa chọn.
  * Báo hai ngưỡng khác nhau: theta nhỏ nhất đạt Rep trung bình >= 0.85 (ngưỡng
    ĐẠI DIỆN TRUNG BÌNH) và theta nhỏ nhất đạt PHỦ TOÀN PHẦN (mọi luật có
    cov >= delta).
  * Thời gian tại theta = 1 gồm cả xây FISA + suy diễn (v2 gần 0 vì chỉ tạo list).
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.model_selection import train_test_split

import config as C
from experiments.common import load_clean, save_json, evaluate_fisa
from experiments.run_02_E2_greedy_vs_random import select
from src.kfold_utils import make_stratified_kfold, build_fkg_fold
from src.fkg_sim_rep import coverage_stats
from src.utils_stats import pareto_knee


def curve(R, RV, sim, method):
    labels = [r[-1] for r in R]
    pts = []
    for theta in C.E3.THETA_GRID:
        t0 = time.perf_counter()
        S = list(range(len(R))) if theta >= 1.0 else select(method, sim, R, labels, theta)
        t_sel = time.perf_counter() - t0
        cs = coverage_stats(sim, S, C.E3.COVER_DELTA)
        ev = evaluate_fisa([R[i] for i in S], RV)
        pts.append(dict(theta=theta, size=len(S), rep=cs["rep"], min_cov=cs["min_cov"],
                        frac_covered=cs["frac_covered"], full_cover=cs["full_cover"],
                        auc=ev["auc"], bacc=ev["bacc"], t_select=t_sel, t_fit=ev["t_fit"],
                        t_pred=ev["t_pred"], t_total=t_sel + ev["t_fit"] + ev["t_pred"]))
    return pts


def run_e3():
    df = load_clean()
    folds = make_stratified_kfold(df)
    out = []
    for f, (tr, te) in enumerate(folds):
        d_tr = df.iloc[tr].reset_index(drop=True)
        d_te = df.iloc[te].reset_index(drop=True)
        i_tr, i_va = train_test_split(np.arange(len(d_tr)), test_size=C.E3.INNER_VAL_FRACTION,
                                      stratify=d_tr[C.LABEL_COLUMN], random_state=C.E3.INNER_SEED + f)
        inner = build_fkg_fold(d_tr.iloc[i_tr].reset_index(drop=True), d_tr.iloc[i_va].reset_index(drop=True),
                               n_sample=C.SCALE.N_EXP, seed=f)
        outer = build_fkg_fold(d_tr, d_te, n_sample=C.SCALE.N_EXP, seed=f, val_fraction=C.FISA.VAL_FRACTION)
        for method in C.E3.METHODS:
            pts = curve(inner["rules_train"], inner["rules_test"], inner["sim"], method)
            k, front = pareto_knee([p["t_total"] for p in pts], [p["auc"] for p in pts])
            theta_star = pts[k]["theta"]
            th_rep = next((p["theta"] for p in pts if p["rep"] >= C.E3.REP_MEAN_THRESHOLD), None)
            th_cov = next((p["theta"] for p in pts if p["full_cover"]), None)
            R, RT, sim = outer["rules_train"], outer["rules_test"], outer["sim"]
            labels = [r[-1] for r in R]
            S = list(range(len(R))) if theta_star >= 1.0 else select(method, sim, R, labels, theta_star)
            ev = evaluate_fisa([R[i] for i in S], RT, outer["rules_val"])
            ev_full = evaluate_fisa(R, RT, outer["rules_val"])
            out.append(dict(fold=f, method=method, val_curve=pts,
                            pareto_thetas=[pts[i]["theta"] for i in front],
                            theta_star=theta_star, theta_rep_mean=th_rep, theta_full_cover=th_cov,
                            test_auc_at_star=ev["auc"], test_bacc_at_star=ev["bacc"], test_acc_at_star=ev["acc"],
                            test_acc_full=ev_full["acc"], test_bacc_full=ev_full["bacc"],
                            test_auc_full=ev_full["auc"], test_size_at_star=len(S), n_pool=len(R)))
            print(f"  E3 fold {f} {method}: theta*={theta_star}, test AUC={ev['auc']:.4f} "
                  f"(FKG đầy đủ {ev_full['auc']:.4f})", flush=True)
    save_json("E3_results.json", dict(results=out, rep_mean_threshold=C.E3.REP_MEAN_THRESHOLD,
                                      cover_delta=C.E3.COVER_DELTA))
    return out


if __name__ == "__main__":
    run_e3()
