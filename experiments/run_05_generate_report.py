"""
experiments/run_05_generate_report.py — Đọc results/E{1..4}_results.json,
sinh bảng CSV/Markdown + biểu đồ PNG, gộp vào report_tong_hop.md.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv

import config as C

OUT = C.PATHS.RESULTS_DIR
FIG_DIR = os.path.join(OUT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _load(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        print(f"  [Bỏ qua] Không tìm thấy {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def report_e1():
    data = _load("E1_results.json")
    if not data:
        return ""
    results = data["results"]
    bound = data["theoretical_bound"]
    n_viol = data["n_violations"]

    rows_by_config = {}
    for r in results:
        key = (r["n"], r["ratio_budget"])
        rows_by_config.setdefault(key, []).append(r["rho"])

    lines = ["| n | m/n | rho trung bình | rho min | rho max | Cận (1-1/e) |",
             "|---|---|---|---|---|---|"]
    for (n, ratio), rhos in sorted(rows_by_config.items()):
        arr = np.array(rhos)
        lines.append(f"| {n} | {ratio} | {arr.mean():.4f} | {arr.min():.4f} | "
                      f"{arr.max():.4f} | {bound:.4f} |")
    md = "\n".join(lines)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    all_rhos = [r["rho"] for r in results]
    ax.hist(all_rhos, bins=20, color="#1f5aa8", edgecolor="white")
    ax.axvline(bound, color="red", linestyle="--", label=f"Cận (1-1/e)={bound:.4f}")
    ax.set_xlabel(r"$\rho = Rep(S_G)/Rep(S^*)$"); ax.set_ylabel("Số lần chạy")
    ax.set_title(f"E1: Phân phối tỉ lệ xấp xỉ ({len(results)} lần chạy, {n_viol} vi phạm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "E1_histogram.png"), dpi=140)
    plt.close(fig)

    summary = (f"**{len(results)} lần chạy, {n_viol} vi phạm cận (1-1/e).** " +
               ("✅ Không vi phạm nào -- đúng lý thuyết Nemhauser." if n_viol == 0
                else "⚠️ Có vi phạm -- cần rà lại cài đặt."))
    return f"{summary}\n\n{md}\n\n![E1](figures/E1_histogram.png)\n"


def report_e2():
    data = _load("E2_results.json")
    if not data:
        return ""
    fields = ["theta", "n_pairs_rep", "rep_greedy_mean", "rep_random_mean",
              "p_rep_holm", "cohens_d_rep", "auc_greedy_mean", "auc_random_mean", "p_auc_holm"]
    headers = ["θ", "n cặp (Rep)", "Rep Greedy", "Rep Random", "p (Holm)",
               "Cohen's d", "AUC Greedy", "AUC Random", "p AUC (Holm)"]
    with open(os.path.join(OUT, "table_E2.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in data:
            w.writerow({k: r.get(k, "") for k in fields})

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in data:
        vals = [str(r.get(k, "")) if not isinstance(r.get(k), float) else f"{r[k]:.4f}" for k in fields]
        lines.append("| " + " | ".join(vals) + " |")
    md = "\n".join(lines)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    thetas = [r["theta"] for r in data]
    x = np.arange(len(thetas))
    ax.bar(x - 0.2, [r["rep_greedy_mean"] for r in data], 0.4, label="GreedyFKGS", color="#1f5aa8")
    ax.bar(x + 0.2, [r["rep_random_mean"] for r in data], 0.4, label="Random", color="#c46e3c")
    ax.set_xticks(x); ax.set_xticklabels([str(t) for t in thetas])
    ax.set_xlabel(r"$\theta$"); ax.set_ylabel("Rep(S)")
    ax.set_title("E2: GreedyFKGS vs Random theo θ"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "E2_comparison.png"), dpi=140)
    plt.close(fig)

    return f"{md}\n\n![E2](figures/E2_comparison.png)\n"


def report_e3():
    data = _load("E3_results.json")
    if not data:
        return ""
    summary = data["summary"]
    fields = ["theta", "rep_mean", "rep_std", "auc_rel_mean", "auc_rel_std", "time_s_mean"]
    headers = ["θ", "Rep (mean)", "std", "AUC tương đối", "std", "Thời gian (s)"]
    with open(os.path.join(OUT, "table_E3.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary:
            w.writerow({k: r.get(k, "") for k in fields})

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in summary:
        vals = [f"{r[k]:.4f}" if isinstance(r.get(k), float) else str(r.get(k)) for k in fields]
        lines.append("| " + " | ".join(vals) + " |")
    md = "\n".join(lines)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    thetas = [r["theta"] for r in summary]

    ax = axes[0]
    ax.plot(thetas, [r["auc_rel_mean"] for r in summary], marker="o", color="#1f5aa8", label="AUC tương đối")
    ax2 = ax.twinx()
    ax2.plot(thetas, [r["time_s_mean"] for r in summary], marker="s", color="#c46e3c", label="Thời gian")
    ax.set_xlabel(r"$\theta$"); ax.set_ylabel("AUC tương đối", color="#1f5aa8")
    ax2.set_ylabel("Thời gian (s)", color="#c46e3c"); ax.set_title("Đường cong kép")

    ax = axes[1]
    ax.plot(thetas, [r["rep_mean"] for r in summary], marker="o", color="#2e7846")
    ax.axhline(C.E3.COVERAGE_THRESHOLD, linestyle="--", color="red",
               label=f"δ={C.E3.COVERAGE_THRESHOLD}")
    ax.set_xlabel(r"$\theta$"); ax.set_ylabel("Rep(S)"); ax.set_title("Rep theo θ"); ax.legend()

    ax = axes[2]
    ax.scatter([r["time_s_mean"] for r in summary], [r["auc_rel_mean"] for r in summary],
               color="#8a4fa8")
    for r in summary:
        ax.annotate(f"{r['theta']}", (r["time_s_mean"], r["auc_rel_mean"]), fontsize=7)
    ax.set_xlabel("Thời gian (s)"); ax.set_ylabel("AUC tương đối"); ax.set_title("Pareto front")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "E3_curves.png"), dpi=140)
    plt.close(fig)

    extra = (f"\n**Theta tối thiểu đạt Rep≥{C.E3.COVERAGE_THRESHOLD}:** "
             f"{data.get('theta_at_coverage_threshold')}  \n"
             f"**Điểm gối (Kneedle):** θ={data.get('knee_theta')}\n")
    return f"{md}\n{extra}\n![E3](figures/E3_curves.png)\n"


def report_e4():
    data = _load("E4_results.json")
    if not data:
        return ""
    methods = data["methods"]
    ranks_rep = data["ranks_rep"]
    ranks_time = data["ranks_time"]

    lines = ["| Phương pháp | Hạng Rep (1=tốt nhất) | Hạng Thời gian (1=nhanh nhất) |",
             "|---|---|---|"]
    for m, rr, rt in zip(methods, ranks_rep, ranks_time):
        lines.append(f"| {m} | {rr:.2f} | {rt:.2f} |")
    md = "\n".join(lines)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(methods))
    ax.bar(x - 0.2, ranks_rep, 0.4, label="Hạng Rep", color="#1f5aa8")
    ax.bar(x + 0.2, ranks_time, 0.4, label="Hạng Thời gian", color="#c46e3c")
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylabel("Hạng trung bình (thấp hơn = tốt hơn)")
    ax.set_title("E4: Xếp hạng kép Rep vs Thời gian"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "E4_ranks.png"), dpi=140)
    plt.close(fig)

    friedman_line = (f"Friedman: stat={data['friedman_stat']:.4f}, p={data['friedman_p']:.6f} "
                      f"(trên {data['n_blocks_total']} block).")
    nemenyi_line = ""
    if data.get("nemenyi"):
        cd = data["nemenyi"]["cd"]
        pairs = data["nemenyi"]["significant_pairs"]
        nemenyi_line = f"\nCritical Difference (Nemenyi)={cd:.4f}. Cặp khác biệt có ý nghĩa: " + \
                       (", ".join(f"{a} vs {b}" for a, b, _ in pairs) if pairs else "không có")

    limitation = ("\n\n**Giới hạn:** 10 block/fold là mẫu con độc lập từ CÙNG một nguồn dữ "
                  "liệu tiểu đường -- kết luận chỉ áp dụng cho bộ dữ liệu này, không suy rộng "
                  "đa lĩnh vực.")
    return f"{friedman_line}{nemenyi_line}{limitation}\n\n{md}\n\n![E4](figures/E4_ranks.png)\n"


def main():
    sections = ["# Báo cáo kết quả thực nghiệm FKGS v2 (E1-E4)\n"]

    e1 = report_e1()
    if e1:
        sections.append(f"## E1: Xác nhận lý thuyết (1-1/e)\n\n{e1}")
    e2 = report_e2()
    if e2:
        sections.append(f"## E2: GreedyFKGS vs Random\n\n{e2}")
    e3 = report_e3()
    if e3:
        sections.append(f"## E3: Đường cong theo θ\n\n{e3}")
    e4 = report_e4()
    if e4:
        sections.append(f"## E4: So sánh 4 chiến lược\n\n{e4}")

    report_path = os.path.join(OUT, "report_tong_hop.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(sections))
    print(f"\n>>> Đã tạo báo cáo tổng hợp: {report_path}")
    print(f">>> Biểu đồ: {FIG_DIR}/, Bảng CSV: {OUT}/table_*.csv")


if __name__ == "__main__":
    main()
