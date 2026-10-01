"""
experiments/run_05_generate_report.py — Báo cáo tổng hợp v3.

Từ chối gộp kết quả khác run_id (QA Stage 3). Mọi bảng ghi kích thước tập mẫu,
độ lệch chuẩn/CI, p-value dạng khoa học (không in 0.0000), và nêu rõ đơn vị
suy luận.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from experiments.common import load_json
from src.utils_stats import fmt_p

MCOL = {"GreedyFKGS": "#1f5aa8", "SFKGS": "#2e8b57", "CFKGS": "#c0392b",
        "Random": "#7f7f7f", "StratRandom": "#b8860b"}


def f4(x, k=4):
    return "NA" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{k}f}"


def load_all():
    man = json.load(open(os.path.join(C.PATHS.RESULTS_DIR, "manifest.json"), encoding="utf-8"))
    data = {}
    for k in ["E1", "E2", "E3", "E4"]:
        d = load_json(f"{k}_results.json")
        if d is None:
            raise SystemExit(f"Thiếu {k}_results.json -- báo cáo không được lập khi thiếu bước")
        if d.get("run_id") != man["run_id"]:
            raise SystemExit(f"{k}_results.json thuộc run {d.get('run_id')} khác manifest {man['run_id']}")
        data[k] = d
    return man, data


def sec_header(man):
    cfg = man["config"]
    pre = man.get("preprocess", {})
    env = man["environment"]
    return "\n".join([
        f"# Báo cáo thực nghiệm FKGS v3 — bộ dữ liệu `{man['dataset']}`", "",
        "| Mục | Giá trị |", "|---|---|",
        f"| run_id | `{man['run_id']}` |",
        f"| Mã nguồn (SHA-256 tổng hợp) | `{man['code_sha256'][:16]}…` |",
        f"| Dữ liệu | {man['data']['source']} `{man['data']['path']}` SHA-256 `{(man['data']['sha256'] or '')[:16]}…` |",
        f"| Tiền xử lý | {pre.get('n_raw')} dòng → {pre.get('n_clean')} sau loại {pre.get('n_duplicates_removed')} trùng; nhãn {pre.get('label_counts')}; X trùng khác nhãn: {pre.get('n_conflicting_X')} |",
        f"| Chia dữ liệu | {cfg['kfold']['K']}-fold phân tầng, seed {cfg['kfold']['SEED']}; tập nền N_EXP={cfg['scale']['N_EXP']} |",
        f"| Điền giá trị thiếu | trung vị toàn cục của train (chỉ X), cùng quy tắc train/test |",
        f"| Nhân Sim / Rep | 1−Hamming cùng nhãn, −1 khác nhãn; Rep dùng max(0,Sim) |",
        f"| Mờ hoá | 3 mức theo phân vị {[round(q,3) for q in cfg['fuzzy_quantiles']]} của tập huấn luyện |",
        f"| FISA-P | PAIR_SCOPE=`{cfg['fisa']['PAIR_SCOPE']}`; điểm s = log D1 − log D0; quyết định `{cfg['fisa'].get('DECISION','argmax')}` (tiêu chí `{cfg['fisa'].get('CALIB_CRITERION','')}`, tập xác thực {cfg['fisa'].get('VAL_FRACTION',0)} tập huấn luyện) |",
        f"| Ngân sách | `{cfg['sampling']['BUDGET_MODE']}` (mọi phương pháp chọn đúng ceil(θn) luật) |",
        f"| Môi trường | Python {env['python']}, numpy {env['numpy']}, scipy {env['scipy']}, sklearn {env['sklearn']}, {env['platform']} |",
        f"| Chế độ | {'QUICK (không dùng để báo cáo)' if cfg.get('quick') else 'đầy đủ'} |", ""])


def sec_e1(d):
    rows = d["results"]
    keys = sorted(set((r["n"], r["ratio_requested"]) for r in rows))
    L = ["## E1 — Kiểm chứng cận lý thuyết (RQ1)", "",
         f"{len(rows)} mẫu con từ luật huấn luyện fold 0 (vét cạn). Cận 1−1/e = {d['theoretical_bound']:.4f}.", "",
         "| n | m | m/n thực | Greedy ρ TB / min | SFKGS Rep/OPT_pb min | SFKGS tỉ lệ cục bộ min | SFKGS max\\|g\\| | CFKGS Rep/OPT_pb min | CFKGS min g |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k in keys:
        rr = [r for r in rows if (r["n"], r["ratio_requested"]) == k]
        L.append(f"| {k[0]} | {rr[0]['m']} | {rr[0]['ratio_actual']:.3f} | "
                 f"{np.mean([r['greedy_rho'] for r in rr]):.4f} / {min(r['greedy_rho'] for r in rr):.4f} | "
                 f"{min(r['sfkgs_ratio_pb'] for r in rr):.4f} | {min(r['sfkgs_loc_min'] for r in rr):.4f} | "
                 f"{max(abs(r['sfkgs_g']) for r in rr):.1e} | {min(r['cfkgs_ratio_pb'] for r in rr):.4f} | "
                 f"{min(r['cfkgs_g'] for r in rr):.1e} |")
    v = d["violations"]
    L += ["", "Số vi phạm: " + ", ".join(f"{k}={val}" for k, val in v.items()), ""]
    return "\n".join(L)


def sec_e2(d, figdir):
    S = {(r["method"], r["theta"]): r for r in d["summary"]["per_method"]}
    thetas = C.E2.THETA_VALUES
    methods = ["GreedyFKGS", "SFKGS", "CFKGS", "Random", "StratRandom"]
    fp, ft = S[("FKG_full_pool", 1.0)], S[("FKG_full_train", 1.0)]
    L = ["## E2 — So sánh với FKG đầy đủ và lấy mẫu cơ sở, cùng ngân sách (RQ2)", "",
         "Đơn vị suy luận: fold (n = số fold); Random/StratRandom lấy trung bình theo hạt giống trong fold. "
         "Các fold có tập huấn luyện chồng lấp nên p-value mang tính mô tả trong phạm vi bộ dữ liệu.", "",
         "| Mốc | \\|R\\| | AUC | Balanced Acc | Accuracy | Acc (argmax) | BalAcc (argmax) |", "|---|---|---|---|---|---|---|",
         f"| FKG đầy đủ (tập nền) | {fp['size_mean']:.0f} | {f4(fp['auc_mean'])} ± {f4(fp['auc_sd'])} | {f4(fp['bacc_mean'])} | {f4(fp['acc_mean'])} | {f4(fp.get('acc_argmax_mean'))} | {f4(fp.get('bacc_argmax_mean'))} |",
         f"| FKG đầy đủ (toàn train) | {ft['size_mean']:.0f} | {f4(ft['auc_mean'])} ± {f4(ft['auc_sd'])} | {f4(ft['bacc_mean'])} | {f4(ft['acc_mean'])} | {f4(ft.get('acc_argmax_mean'))} | {f4(ft.get('bacc_argmax_mean'))} |", "",
         "| θ | Phương pháp | \\|S\\| | Rep | Tỉ lệ luật phủ (≥δ) | AUC | Accuracy | Balanced Acc | Lớp+ trong mẫu | t chọn (s) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for t in thetas:
        for m in methods:
            r = S[(m, t)]
            L.append(f"| {t} | {m} | {r['size_mean']:.0f} | {f4(r['rep_mean'])} ± {f4(r['rep_sd'])} | "
                     f"{f4(r['frac_covered_mean'], 3)} | {f4(r['auc_mean'])} ± {f4(r['auc_sd'])} | "
                     f"{f4(r['acc_mean'])} | {f4(r['bacc_mean'])} | {f4(r['pos_share_mean'], 3)} | {f4(r['t_select_mean'], 3)} |")
    L += ["", "Kiểm định Wilcoxon một phía (phương pháp > đối chứng), đơn vị fold, Holm trong từng họ (độ đo × đối chứng):", "",
          "| Độ đo | Đối chứng | θ | Phương pháp | Chênh TB | CI95 (t) | Fold tốt hơn | Tỉ lệ seed bị vượt | p | p Holm |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for x in d["summary"]["tests"]:
        ci = x["ci95"]
        sf = "—" if x["frac_seeds_beaten"] is None else f"{x['frac_seeds_beaten']:.2f}"
        L.append(f"| {x['metric']} | {x['baseline']} | {x['theta']} | {x['method']} | {x['mean_diff']:+.4f} | "
                 f"[{f4(ci[0])}; {f4(ci[1])}] | {x['folds_better']}/{x['n_folds']} | {sf} | "
                 f"{fmt_p(x['p_raw'])} | {fmt_p(x['p_holm']) if x['p_holm'] is not None else 'NA'} |")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for m in methods:
        ax[0].errorbar(thetas, [S[(m, t)]["auc_mean"] for t in thetas], yerr=[S[(m, t)]["auc_sd"] for t in thetas],
                       marker="o", capsize=3, label=m, color=MCOL[m])
        ax[1].plot(thetas, [S[(m, t)]["rep_mean"] for t in thetas], marker="o", label=m, color=MCOL[m])
    ax[0].axhline(fp["auc_mean"], ls="--", c="k", label="FKG đầy đủ (tập nền)")
    ax[0].axhline(ft["auc_mean"], ls=":", c="k", label="FKG đầy đủ (toàn train)")
    ax[0].set(xlabel="θ", ylabel="AUC (test)", title="E2: AUC theo θ")
    ax[1].set(xlabel="θ", ylabel="Rep(S)", title="E2: Rep theo θ")
    for a in ax:
        a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "E2.png"), dpi=140); plt.close(fig)
    L += ["", "![E2](figures/E2.png)", ""]
    return "\n".join(L)


def sec_e3(d, figdir):
    res = d["results"]
    L = ["## E3 — Đường cong theo θ, chọn θ* trên validation (RQ3)", "",
         f"θ* = điểm gối trên mặt Pareto (chi phí = chọn luật + xây FISA + suy diễn; lợi ích = AUC validation). "
         f"Test chỉ dùng để đánh giá tại θ*. Ngưỡng Rep trung bình = {d['rep_mean_threshold']}; "
         f"phủ toàn phần = mọi luật có cov ≥ {d['cover_delta']}.", "",
         "| Phương pháp | θ* theo fold | \\|S\\| TB tại θ* | AUC test tại θ* | Acc / BalAcc test tại θ* | AUC test FKG đầy đủ | Acc / BalAcc FKG đầy đủ | θ nhỏ nhất Rep TB ≥ ngưỡng | θ nhỏ nhất phủ toàn phần |",
         "|---|---|---|---|---|---|---|---|---|"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for m in C.E3.METHODS:
        rr = [r for r in res if r["method"] == m]
        L.append(f"| {m} | {', '.join(str(r['theta_star']) for r in rr)} | "
                 f"{np.mean([r['test_size_at_star'] for r in rr]):.0f} | "
                 f"{np.mean([r['test_auc_at_star'] for r in rr]):.4f} ± {np.std([r['test_auc_at_star'] for r in rr], ddof=1):.4f} | "
                 f"{np.mean([r.get('test_acc_at_star', np.nan) for r in rr]):.4f} / {np.mean([r['test_bacc_at_star'] for r in rr]):.4f} | "
                 f"{np.mean([r['test_auc_full'] for r in rr]):.4f} | "
                 f"{np.mean([r.get('test_acc_full', np.nan) for r in rr]):.4f} / {np.mean([r.get('test_bacc_full', np.nan) for r in rr]):.4f} | "
                 f"{', '.join(str(r['theta_rep_mean']) for r in rr)} | {', '.join(str(r['theta_full_cover']) for r in rr)} |")
        th = [p["theta"] for p in rr[0]["val_curve"]]
        auc = np.mean([[p["auc"] for p in r["val_curve"]] for r in rr], 0)
        rep = np.mean([[p["rep"] for p in r["val_curve"]] for r in rr], 0)
        cov = np.mean([[p["frac_covered"] for p in r["val_curve"]] for r in rr], 0)
        tt = np.mean([[p["t_total"] for p in r["val_curve"]] for r in rr], 0)
        axes[0].plot(th, auc, marker="o", color=MCOL[m], label=m)
        axes[1].plot(th, rep, marker="o", color=MCOL[m], label=f"{m} Rep")
        axes[1].plot(th, cov, marker="x", ls="--", color=MCOL[m], label=f"{m} tỉ lệ phủ")
        axes[2].scatter(tt, auc, color=MCOL[m], label=m)
    axes[0].set(xlabel="θ", ylabel="AUC validation", title="E3: AUC validation theo θ (TB fold)")
    axes[1].axhline(d["rep_mean_threshold"], c="k", ls=":")
    axes[1].set(xlabel="θ", ylabel="giá trị", title="Rep TB và tỉ lệ luật được phủ")
    axes[2].set(xlabel="Thời gian (s)", ylabel="AUC validation", title="Chi phí – lợi ích")
    for a in axes:
        a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(figdir, "E3.png"), dpi=140); plt.close(fig)
    L += ["", "![E3](figures/E3.png)", ""]
    return "\n".join(L)


def sec_e4(d, figdir):
    L = [f"## E4 — So sánh các chiến lược tại θ = {d['theta']}, cùng ngân sách (RQ4)", "",
         "| Phương pháp | \\|S\\| | Rep TB ± ĐLC | Tỉ lệ luật phủ | Lớp+ | Thời gian (ms) |", "|---|---|---|---|---|---|"]
    for s in d["summary"]:
        L.append(f"| {s['method']} | {s['size']:.0f} | {s['rep_mean']:.4f} ± {s['rep_sd']:.4f} | "
                 f"{s['frac_covered']:.3f} | {s['pos_share']:.3f} | {s['time_ms']:.1f} |")
    for key, title in [("rep_fold", "Rep, 5 block = trung bình fold (chính)"),
                       ("rep_block", "Rep, 50 block (mô tả, có điều kiện trên fold)"),
                       ("time_fold", "Thời gian, trung bình fold")]:
        r = d[key]
        L += ["", f"**{title}:** Friedman χ²={r['friedman_stat']:.3f}, p={fmt_p(r['friedman_p'])}, "
                  f"CD={r['cd']:.3f} ({r['n_blocks']} block). Hạng: " +
              ", ".join(f"{m}={x:.2f}" for m, x in zip(d["methods"], r["ranks"])) +
              ". Cặp khác biệt: " + (", ".join(f"{a}–{b}" for a, b, _ in r["significant_pairs"]) or "không có")]
    fig, ax = plt.subplots(figsize=(7, 4))
    ms = d["methods"]
    ax.bar(np.arange(len(ms)) - 0.2, d["rep_fold"]["ranks"], 0.4, label="Hạng Rep (fold)")
    ax.bar(np.arange(len(ms)) + 0.2, d["time_fold"]["ranks"], 0.4, label="Hạng thời gian (fold)")
    ax.set_xticks(np.arange(len(ms))); ax.set_xticklabels(ms, fontsize=8)
    ax.set_ylabel("Hạng TB (1 = tốt nhất)"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(figdir, "E4.png"), dpi=140); plt.close(fig)
    L += ["", "![E4](figures/E4.png)", ""]
    return "\n".join(L)


def sec_pub(d):
    if d is None:
        return ""
    res = d["results"]
    L = ["## PUB — Giao thức của bài công bố: chia cố định 70/30, FKG đầy đủ", "",
         f"Độ chính xác đã công bố của FKG đầy đủ: {d.get('published_acc')}. Ngưỡng hiệu chỉnh được chọn trên tập "
         "xác thực tách 20% từ tập huấn luyện; tập kiểm tra 30% không tham gia chọn ngưỡng.", "",
         "| FISA | Mờ hoá | Số seed | AUC | Acc argmax | BalAcc argmax | Acc (D0>9·D1) | Acc hiệu chỉnh (tiêu chí Acc) | BalAcc hiệu chỉnh (tiêu chí BalAcc) | Độ nhạy / Độ đặc hiệu (hiệu chỉnh BalAcc) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    keys = sorted(set((r["variant"], r["quant"]) for r in res), key=lambda k: (k[0] != "orig", k[1] != "q1090"))
    for v, q in keys:
        rr = [r for r in res if r["variant"] == v and r["quant"] == q]
        m = lambda k1, k2: np.mean([r[k1][k2] for r in rr]); sdv = lambda k1, k2: np.std([r[k1][k2] for r in rr], ddof=1) if len(rr) > 1 else 0.0
        L.append(f"| {'gốc (tổ hợp ba)' if v == 'orig' else 'FISA-P (cặp)'} | {'q10/q90' if q == 'q1090' else 'tam phân vị'} | {len(rr)} | "
                 f"{m('argmax','auc'):.4f} | {m('argmax','acc'):.4f} ± {sdv('argmax','acc'):.4f} | {m('argmax','bacc'):.4f} | "
                 f"{m('ratio9','acc'):.4f} | {m('calib_acc','acc'):.4f} ± {sdv('calib_acc','acc'):.4f} | {m('calib_bacc','bacc'):.4f} | "
                 f"{m('calib_bacc','sens'):.3f} / {m('calib_bacc','spec'):.3f} |")
    return "\n".join(L + [""])


def main():
    man, d = load_all()
    figdir = os.path.join(C.PATHS.RESULTS_DIR, "figures")
    os.makedirs(figdir, exist_ok=True)
    txt = "\n".join([sec_header(man), sec_e1(d["E1"]), sec_e2(d["E2"], figdir),
                     sec_e3(d["E3"], figdir), sec_e4(d["E4"], figdir), sec_pub(load_json("PUB_results.json"))])
    p = os.path.join(C.PATHS.RESULTS_DIR, "report_tong_hop.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"  Đã tạo {p}")


if __name__ == "__main__":
    main()
