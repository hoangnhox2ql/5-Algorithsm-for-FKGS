"""
report/generate_report.py — Đọc toàn bộ outputs/*.json đã sinh ra từ các
script thực nghiệm, tạo:
  - Bảng kết quả dạng Markdown + CSV cho từng KB
  - Biểu đồ so sánh (PNG) cho từng KB (độ đo chính: config.EVAL.MAIN_METRIC)
  - Một file report_tong_hop.md gộp toàn bộ bảng + chèn ảnh biểu đồ

Chạy: python3 report/generate_report.py [--out THƯ_MỤC]   (sau khi đã chạy xong run_all.py)
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv

import config as C

OUT = C.PATHS.OUTPUT_DIR
FIG_DIR = os.path.join(OUT, "figures")
NAN = float("nan")
METRIC_LABEL = {"accuracy": "Accuracy", "balanced_accuracy": "Độ chính xác cân bằng", "auc": "AUC-ROC"}
# khoá của FISA / FKG-E trong các dòng KB1, KB2, KB6 theo độ đo
KB1_KEYS = {"accuracy": ("fisa_accuracy", "fkge_accuracy_mean"), "balanced_accuracy": ("fisa_bacc", "fkge_bacc_mean"),
            "auc": ("fisa_auc", "fkge_auc_mean")}


def _metric():
    m = getattr(C.EVAL, "MAIN_METRIC", "accuracy")
    return m if m in METRIC_LABEL else "accuracy"


def _load(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        print(f"  [Bỏ qua] Không tìm thấy {path} -- hãy chạy script thực nghiệm tương ứng trước.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_csv(rows, fieldnames, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _md_table(rows, fieldnames, headers=None):
    headers = [h.replace("|", "\\|") for h in (headers or fieldnames)]   # "|R|" không được làm vỡ cột
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        vals = []
        for k in fieldnames:
            v = r.get(k, "")
            if isinstance(v, float):
                v = f"{v:.4f}"
            vals.append(str(v).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _g(r, k):
    v = r.get(k)
    return NAN if v is None else v


def _std_key(k):
    return k[:-5] + "_std" if k.endswith("_mean") else k + "_std"


# ============================================================
# KB1
# ============================================================
def report_kb1():
    data = _load("kb1_results.json")
    if not data:
        return "", ""
    fields = ["dataset", "n_folds", "n_rules", "fisa_accuracy", "fisa_bacc", "fisa_auc", "fisa_f1", "fisa_avg_ms",
              "fisa_seq_avg_ms", "fkge_accuracy_mean", "fkge_accuracy_std", "fkge_bacc_mean", "fkge_bacc_std",
              "fkge_auc_mean", "fkge_auc_std", "fkge_f1_mean", "fkge_agreement_mean", "fkge_dev_mean",
              "fkge_avg_ms_mean", "speedup", "speedup_vs_seq"]
    headers = ["Bộ dữ liệu", "Số fold", "|R|", "FISA Acc", "FISA BalAcc", "FISA AUC", "FISA F1", "FISA bảng tra ms",
               "FISA tuần tự ms", "FKG-E Acc", "std", "FKG-E BalAcc", "std", "FKG-E AUC", "std", "FKG-E F1",
               "Đồng thuận FISA", "Dev", "FKG-E ms/query", "Speedup / bảng tra", "Speedup / tuần tự"]
    _write_csv(data, fields, os.path.join(OUT, "table_KB1.csv"))
    md = _md_table(data, fields, headers)

    metric = _metric(); kf, ke = KB1_KEYS[metric]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names = [d["dataset"] for d in data]
    x = np.arange(len(names))
    w = 0.35
    axes[0].bar(x - w/2, [_g(d, kf) for d in data], w, label="FISA", color="#1f5aa8",
                yerr=[_g(d, kf + "_std") if kf + "_std" in d else 0 for d in data], capsize=4)
    axes[0].bar(x + w/2, [_g(d, ke) for d in data], w, label="FKG-E", color="#c46e3c",
                yerr=[_g(d, _std_key(ke)) for d in data], capsize=4)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    axes[0].set_ylabel(METRIC_LABEL[metric]); axes[0].set_title(f"KB1: {METRIC_LABEL[metric]} FISA vs FKG-E")
    axes[0].legend()

    w = 0.27
    axes[1].bar(x - w, [d["fisa_avg_ms"] for d in data], w, label="FISA (bảng tra)", color="#1f5aa8")
    axes[1].bar(x, [_g(d, "fisa_seq_avg_ms") for d in data], w, label="FISA (tuần tự)", color="#7f9cc9")
    axes[1].bar(x + w, [d["fkge_avg_ms_mean"] for d in data], w, label="FKG-E", color="#c46e3c")
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=15, ha="right", fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("ms / truy vấn (log)"); axes[1].set_title("KB1: Thời gian suy diễn"); axes[1].legend()
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "kb1_comparison.png")
    fig.savefig(fig_path, dpi=140); plt.close(fig)
    return md, "figures/kb1_comparison.png"


# ============================================================
# KB2
# ============================================================
def report_kb2():
    data = _load("kb2_results.json")
    if not data:
        return "", "", ""
    rows, verdict = data["rows"], data["verdict"]
    fields = ["dataset", "n_rules", "fisa_accuracy", "fisa_bacc", "fisa_auc", "fkge_accuracy_mean", "fkge_bacc_mean",
              "fkge_auc_mean", "fisa_avg_ms", "fkge_avg_ms_mean", "speedup"]
    headers = ["Cấu hình", "|R|", "FISA Acc", "FISA BalAcc", "FISA AUC", "FKG-E Acc", "FKG-E BalAcc", "FKG-E AUC",
               "FISA ms/query", "FKG-E ms/query", "Speedup"]
    _write_csv(rows, fields, os.path.join(OUT, "table_KB2.csv"))
    md = _md_table(rows, fields, headers)
    return md, verdict, ""


# ============================================================
# KB3
# ============================================================
def report_kb3():
    data = _load("kb3_results.json")
    if not data:
        return "", ""
    fields = ["d", "accuracy_mean", "accuracy_std", "balanced_accuracy_mean", "balanced_accuracy_std", "auc_mean",
              "auc_std", "avg_ms_mean", "train_time_s_mean", "fisa_accuracy", "fisa_balanced_accuracy", "fisa_auc"]
    headers = ["d", "Accuracy", "std", "BalAcc", "std", "AUC", "std", "ms/query", "TG huấn luyện (s)",
               "FISA Acc", "FISA BalAcc", "FISA AUC"]
    _write_csv(data, fields, os.path.join(OUT, "table_KB3.csv"))
    md = _md_table(data, fields, headers)

    metric = _metric()
    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))
    d_vals = [r["d"] for r in data]
    ax1.errorbar(d_vals, [_g(r, metric + "_mean") for r in data], yerr=[_g(r, metric + "_std") for r in data],
                 marker="o", color="#1f5aa8", label=f"{METRIC_LABEL[metric]} FKG-E")
    ax1.axhline(_g(data[0], "fisa_" + metric), linestyle="--", color="gray", label="FISA (đối chiếu)")
    ax1.set_xlabel("Chiều nhúng d"); ax1.set_ylabel(METRIC_LABEL[metric]); ax1.set_xscale("log", base=2)
    ax1.set_title("KB3: Độ nhạy theo chiều nhúng d"); ax1.legend(loc="lower right")
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "kb3_dim_sensitivity.png")
    fig.savefig(fig_path, dpi=140); plt.close(fig)
    return md, "figures/kb3_dim_sensitivity.png"


# ============================================================
# KB4 (heatmap)
# ============================================================
def report_kb4():
    data = _load("kb4_results.json")
    if not data:
        return "", ""
    lam, beta = data["lambda_grid"], data["beta_grid"]
    hm = np.array(data["heatmap"])
    label = METRIC_LABEL.get(data.get("heatmap_metric", "accuracy"), "Accuracy")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(hm, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(beta))); ax.set_xticklabels(beta)
    ax.set_yticks(range(len(lam))); ax.set_yticklabels(lam)
    ax.set_xlabel(r"$\lambda_P$ (trọng số $L_{pred}$)"); ax.set_ylabel(r"$\lambda_I$ (trọng số $L_{inf}$)")
    ax.set_title(f"KB4: Heatmap {label} theo (λ_I, λ_P)")
    for i in range(len(lam)):
        for j in range(len(beta)):
            ax.text(j, i, f"{hm[i,j]:.3f}", ha="center", va="center",
                     color="white" if hm[i, j] < hm.max() * 0.7 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label=label)
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "kb4_heatmap.png")
    fig.savefig(fig_path, dpi=140); plt.close(fig)

    fields = ["lambda", "beta", "rho", "accuracy_mean", "accuracy_std", "balanced_accuracy_mean", "auc_mean",
              "agreement_mean", "dev_mean"]
    headers = ["λ_I", "λ_P", "ρ = λ_I/λ_P", "Accuracy", "std", "BalAcc", "AUC", "Đồng thuận FISA (R2)",
               "Dev căn cứ luật (R3)"]
    _write_csv(data["rows"], fields, os.path.join(OUT, "table_KB4.csv"))
    md = _md_table(data["rows"], fields, headers)
    return md, "figures/kb4_heatmap.png"


# ============================================================
# KB5
# ============================================================
def report_kb5():
    data = _load("kb5_results.json")
    if not data:
        return "", ""
    fields = ["w", "K", "accuracy_mean", "accuracy_std", "balanced_accuracy_mean", "balanced_accuracy_std",
              "auc_mean", "auc_std"]
    headers = ["w (cửa sổ)", "K (mẫu âm)", "Accuracy", "std", "BalAcc", "std", "AUC", "std"]
    _write_csv(data, fields, os.path.join(OUT, "table_KB5.csv"))
    md = _md_table(data, fields, headers)

    metric = _metric()
    ws = sorted(set(r["w"] for r in data), key=str)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for w in ws:
        sub = [r for r in data if r["w"] == w]
        ax.errorbar([r["K"] for r in sub], [_g(r, metric + "_mean") for r in sub],
                    yerr=[_g(r, metric + "_std") for r in sub], marker="o", capsize=4, label=f"w={w}")
    ax.set_xlabel("Số mẫu âm K"); ax.set_ylabel(METRIC_LABEL[metric])
    ax.set_title("KB5: Độ nhạy theo số mẫu âm K"); ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "kb5_wK_sensitivity.png")
    fig.savefig(fig_path, dpi=140); plt.close(fig)
    return md, "figures/kb5_wK_sensitivity.png"


# ============================================================
# KB6 (đường cong khả năng mở rộng)
# ============================================================
def report_kb6():
    data = _load("kb6_results.json")
    if not data:
        return "", ""
    fields = ["ratio", "n_rules", "fisa_avg_ms", "fisa_seq_avg_ms", "fkge_avg_ms_mean", "fisa_accuracy",
              "fkge_accuracy_mean", "fisa_bacc", "fkge_bacc_mean", "fisa_auc", "fkge_auc_mean"]
    headers = ["Tỉ lệ luật", "|R|", "FISA bảng tra ms", "FISA tuần tự ms", "FKG-E ms", "FISA Acc", "FKG-E Acc",
               "FISA BalAcc", "FKG-E BalAcc", "FISA AUC", "FKG-E AUC"]
    _write_csv(data, fields, os.path.join(OUT, "table_KB6.csv"))
    md = _md_table(data, fields, headers)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    n_rules = [r["n_rules"] for r in data]
    ax.plot(n_rules, [r["fisa_avg_ms"] for r in data], marker="o", label="FISA (bảng tra)", color="#1f5aa8")
    if all("fisa_seq_avg_ms" in r for r in data):
        ax.plot(n_rules, [r["fisa_seq_avg_ms"] for r in data], marker="^", label="FISA (tuần tự)", color="#7f9cc9")
        ax.set_yscale("log")
    ax.plot(n_rules, [r["fkge_avg_ms_mean"] for r in data], marker="s", label="FKG-E", color="#c46e3c")
    ax.set_xlabel("Số luật |R|"); ax.set_ylabel("ms / truy vấn")
    ax.set_title("KB6: Thời gian suy diễn theo |R|"); ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "kb6_scalability.png")
    fig.savefig(fig_path, dpi=140); plt.close(fig)
    return md, "figures/kb6_scalability.png"


# ============================================================
# Ablation
# ============================================================
def report_ablation():
    data = _load("ablation_results.json")
    if not data:
        return "", ""
    fields = ["variant", "accuracy_mean", "balanced_accuracy_mean", "balanced_accuracy_std", "auc_mean", "auc_std",
              "f1_mean", "log_loss_mean", "agreement_fisa_mean", "dev_rule_mean"]
    headers = ["Biến thể", "Accuracy", "BalAcc", "std", "AUC", "std", "F1", "Log-loss", "Đồng thuận FISA (R2)",
               "Dev căn cứ luật (R3)"]
    _write_csv(data, fields, os.path.join(OUT, "table_ablation.csv"))
    md = _md_table(data, fields, headers)

    metric = _metric()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    names = [r["variant"] for r in data]
    x = np.arange(len(names))
    ax2 = ax.twinx()
    ax.bar(x - 0.2, [_g(r, metric + "_mean") for r in data], 0.4, color="#1f5aa8", label=METRIC_LABEL[metric],
           yerr=[_g(r, metric + "_std") for r in data], capsize=3)
    ax2.bar(x + 0.2, [r["log_loss_mean"] for r in data], 0.4, color="#c46e3c", label="Log-loss")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel(METRIC_LABEL[metric], color="#1f5aa8"); ax2.set_ylabel("Log-loss", color="#c46e3c")
    ax.set_title(f"Ablation Study: {METRIC_LABEL[metric]} vs Log-loss")
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "ablation_comparison.png")
    fig.savefig(fig_path, dpi=140); plt.close(fig)
    return md, "figures/ablation_comparison.png"


# ============================================================
# Baseline comparison
# ============================================================
def report_baseline():
    data = _load("baseline_comparison.json")
    if not data:
        return "", ""
    fields = ["method", "accuracy", "balanced_accuracy", "balanced_accuracy_std", "auc", "auc_std", "f1",
              "train_time_s", "avg_ms_per_query"]
    headers = ["Phương pháp", "Accuracy", "BalAcc", "std", "AUC", "std", "F1", "TG huấn luyện (s)", "ms/query"]
    _write_csv(data, fields, os.path.join(OUT, "table_baseline.csv"))
    md = _md_table(data, fields, headers)

    metric = _metric()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    names = [d["method"] for d in data]
    x = np.arange(len(names))
    colors = ["#1f5aa8", "#2e7846", "#c46e3c", "#8a4fa8", "#c0392b"]
    axes[0].bar(x, [_g(d, metric) for d in data], color=colors[:len(names)],
                yerr=[_g(d, metric + "_std") if metric + "_std" in d else 0 for d in data], capsize=4)
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel(METRIC_LABEL[metric]); axes[0].set_title(f"So sánh {METRIC_LABEL[metric]}")

    axes[1].bar(x, [d["avg_ms_per_query"] for d in data], color=colors[:len(names)])
    axes[1].set_xticks(x); axes[1].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("ms / truy vấn"); axes[1].set_title("So sánh tốc độ suy diễn")
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "baseline_comparison.png")
    fig.savefig(fig_path, dpi=140); plt.close(fig)
    return md, "figures/baseline_comparison.png"


def main(out_dir=None):
    global OUT, FIG_DIR
    OUT = out_dir or C.PATHS.OUTPUT_DIR
    FIG_DIR = os.path.join(OUT, "figures")
    os.makedirs(FIG_DIR, exist_ok=True)
    sections = []
    sections.append("# Báo cáo kết quả thực nghiệm FKG-E\n")
    meta = _load("run_meta.json")
    if meta:
        info = (f"**Nguồn dữ liệu:** {meta.get('source')}  \n"
                f"**Cấu hình:** T_ep = {meta.get('T_ep')}; độ đo chính của biểu đồ: "
                f"{METRIC_LABEL.get(meta.get('main_metric'), meta.get('main_metric'))}"
                f"{'; CHẾ ĐỘ NHANH (--quick) — không dùng để báo cáo' if meta.get('quick') else ''}  \n")
        if meta.get("real_data"):
            info += ("Mỗi số liệu FKG-E là trung bình trên các seed trong một fold rồi trung bình giữa các fold; "
                     "cột std là độ lệch chuẩn giữa các fold. Tập kiểm tra BRSET chỉ khoảng 8% dương nên Accuracy "
                     "của bộ phân loại \"luôn đoán âm\" ≈ 0,92 — đọc kết quả theo BalAcc và AUC.\n")
        sections.append(info)

    md, fig = report_kb1()
    if md:
        sections.append(f"## KB1: FKG-E vs FISA trên FKG gốc\n\n{md}\n\n![KB1]({fig})\n")

    md, verdict, _ = report_kb2()
    if md:
        sections.append(f"## KB2: FKG-E vs FISA trên FKGS đã lấy mẫu\n\n{md}\n\n"
                         f"**Kết luận:** {verdict}\n")

    md, fig = report_kb3()
    if md:
        sections.append(f"## KB3: Độ nhạy theo chiều nhúng d\n\n{md}\n\n![KB3]({fig})\n")

    md, fig = report_kb4()
    if md:
        sections.append(f"## KB4: Độ nhạy theo (λ_I, λ_P)\n\n{md}\n\n![KB4]({fig})\n")

    md, fig = report_kb5()
    if md:
        sections.append(f"## KB5: Độ nhạy theo số mẫu âm K\n\n{md}\n\n![KB5]({fig})\n")

    md, fig = report_kb6()
    if md:
        sections.append(f"## KB6: Khả năng mở rộng quy mô\n\n{md}\n\n![KB6]({fig})\n\n"
                         f"**Lưu ý đọc kết quả:** nếu đường FISA (bảng tra) gần như phẳng còn FKG-E "
                         f"tăng theo |R|, đây là hệ quả đúng của việc FISA tiền tính ma trận "
                         f"trọng số một lần (Mệnh đề 3.2.11), KHÔNG phải dấu hiệu lỗi. FISA tuần tự (FISA gốc) "
                         f"tăng tuyến tính theo |R|.\n")

    md, fig = report_ablation()
    if md:
        sections.append(f"## Ablation Study\n\n{md}\n\n![Ablation]({fig})\n\n"
                         f"**Lưu ý:** nếu độ đo phân loại giống nhau giữa các biến thể nhưng "
                         f"Log-loss khác nhau, các thành phần vẫn có đóng góp thật về độ "
                         f"tin cậy dự đoán — không kết luận vội 'không có tác dụng'.\n")

    md, fig = report_baseline()
    if md:
        sections.append(f"## So sánh Baseline (FISA / FKG-E / Node2Vec / TransE / DistMult)\n\n"
                         f"{md}\n\n![Baseline]({fig})\n")

    report_path = os.path.join(OUT, "report_tong_hop.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(sections))
    print(f"\n>>> Đã tạo báo cáo tổng hợp: {report_path}")
    print(f">>> Biểu đồ lưu tại: {FIG_DIR}/")
    print(f">>> Bảng CSV lưu tại: {OUT}/table_*.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="", help="Thư mục chứa *.json kết quả (mặc định config.PATHS.OUTPUT_DIR)")
    main(os.path.abspath(ap.parse_args().out) if ap.parse_args().out else None)
