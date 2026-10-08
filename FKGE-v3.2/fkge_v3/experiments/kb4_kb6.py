"""
experiments/kb4_kb6.py — KB4: quét lưới (λ_I, λ_P), vẽ heatmap (độ đo C.EVAL.MAIN_METRIC).
                         KB6: đường cong thời gian suy diễn theo |R| tăng dần
                              (mô phỏng bằng tỉ lệ luật 20/40/60/80/100% của mỗi fold).
Dữ liệu: experiments/common.py (BRSET thật 5 fold nếu có).
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import config as C
from experiments.common import (brset_folds, describe_source, seeds_for, run_fkge, aggregate, fisa_summary, sweep,
                                FKGE_KEYS)


def run_kb4(lambda_grid=None, beta_grid=None, n_seeds=3):
    lambda_grid = list(lambda_grid or C.KB4_LAMBDA_GRID); beta_grid = list(beta_grid or C.KB4_BETA_GRID)
    metric = C.EVAL.MAIN_METRIC
    print("\n" + "=" * 70)
    print(f"KB4: Quét lưới (λ_I={lambda_grid}, λ_P={beta_grid}) — đánh đổi giữa độ nhất quán với FISA và dự đoán")
    print("=" * 70)
    print(f"Nguồn dữ liệu: {describe_source()}")
    folds = brset_folds()
    # KB4 (Mục 3.2.5.x, Định lý đánh đổi): quét (λ_I, λ_P); tỉ số ρ = λ_I/λ_P quyết định điểm trên mặt Pareto
    configs = [((lam, beta), {"lam_I": lam, "lam_P": beta}) for lam in lambda_grid for beta in beta_grid]
    heatmap = np.zeros((len(lambda_grid), len(beta_grid))); heat_acc = np.zeros_like(heatmap)
    rows = []
    for (lam, beta), ov, E in sweep(folds, configs, n_seeds, "KB4"):
        i, j = lambda_grid.index(lam), beta_grid.index(beta)
        heatmap[i, j], heat_acc[i, j] = E[metric + "_mean"], E["accuracy_mean"]
        rows.append({"lambda": lam, "beta": beta, "rho": lam / beta,
                     "accuracy_mean": E["accuracy_mean"], "accuracy_std": E["accuracy_std"],
                     "balanced_accuracy_mean": E["balanced_accuracy_mean"], "auc_mean": E["auc_mean"],
                     "agreement_mean": E["agreement_fisa_mean"], "dev_mean": E["dev_rule_mean"]})
        print(f"  λ_I={lam:.1f} λ_P={beta:.1f} (ρ={lam / beta:.2f}): accuracy={E['accuracy_mean']:.4f}  "
              f"bacc={E['balanced_accuracy_mean']:.4f}  AUC={E['auc_mean']:.4f}  "
              f"đồng thuận FISA={E['agreement_fisa_mean']:.4f}  Dev={E['dev_rule_mean']:.4f}")
    return {"lambda_grid": lambda_grid, "beta_grid": beta_grid, "heatmap": heatmap.tolist(),
            "heatmap_metric": metric, "heatmap_accuracy": heat_acc.tolist(), "rows": rows, "n_folds": len(folds)}


def run_kb6(ratios=None, n_seeds=3):
    ratios = list(ratios or C.KB6_SAMPLE_RATIOS)
    print("\n" + "=" * 70)
    print(f"KB6: Khả năng mở rộng quy mô — tỉ lệ luật {ratios}")
    print("=" * 70)
    print(f"Nguồn dữ liệu: {describe_source()}")
    # Mỗi tỉ lệ: giữ ngẫu nhiên tỉ lệ đó số luật của mỗi fold (seed 7). Khi chạy JSON và có sẵn các mức nén FKGS thật,
    # file theo quy ước brset_fkgs_rules_<tỉ lệ>.json được dùng thay cho lấy mẫu ngẫu nhiên.
    per_ratio = []
    for ratio in ratios:
        fkgs = C.PATHS.BRSET_FKGS_RULES_FILE.replace(".json", f"_{int(ratio * 100)}.json")
        per_ratio.append((ratio, brset_folds(rule_ratio=ratio, rule_seed=7, fkgs_file=fkgs)))
    n = seeds_for(per_ratio[0][1], n_seeds)
    tasks = [(f, C.FKGE.seed + s, None) for _, folds in per_ratio for f in folds for s in range(n)]
    runs = run_fkge(tasks, "KB6")
    rows = []
    for ratio, folds in per_ratio:
        keys = {f.key for f in folds}
        E = aggregate([(t[0].key, m) for t, m in zip(tasks, runs) if t[0].key in keys], FKGE_KEYS)
        F = fisa_summary(folds)
        row = {
            "ratio": ratio, "n_rules": int(round(np.mean([f.n_rules for f in folds]))), "n_folds": len(folds),
            "rules": folds[0].info.get("rules"),
            "fisa_avg_ms": F["avg_time_per_query_ms_mean"],
            "fisa_seq_avg_ms": F["seq_avg_time_per_query_ms_mean"],
            "fisa_accuracy": F["accuracy_mean"], "fisa_bacc": F["balanced_accuracy_mean"], "fisa_auc": F["auc_mean"],
            "fkge_avg_ms_mean": E["avg_time_per_query_ms_mean"], "fkge_avg_ms_std": E["avg_time_per_query_ms_std"],
            "fkge_accuracy_mean": E["accuracy_mean"], "fkge_bacc_mean": E["balanced_accuracy_mean"],
            "fkge_auc_mean": E["auc_mean"], "fkge_train_time_s_mean": E["train_time_s_mean"],
        }
        print(f"  |R|={row['n_rules']:4d} (tỉ lệ {ratio:.0%}): "
              f"FISA bảng tra={row['fisa_avg_ms']:.4f}ms  FISA tuần tự={row['fisa_seq_avg_ms']:.4f}ms  "
              f"FKG-E={row['fkge_avg_ms_mean']:.4f}ms  | bacc FISA={row['fisa_bacc']:.4f}  FKG-E={row['fkge_bacc_mean']:.4f}")
        rows.append(row)

    # LƯU Ý (Mệnh đề độ phức tạp suy diễn, Chương 3): FISA theo FKG-Pairs có hai cách cài đặt cùng
    # một hàm quyết định. Bản bảng tra tính sẵn W_{c,v,l} khi fit(); mỗi truy vấn tra C(r,3) nút x tối đa
    # 2^3 bộ mức, nên chi phí gần như không phụ thuộc |R|. Bản tuần tự (FISA gốc) duyệt mọi luật trên
    # mọi nút, chi phí O(|R|·C(r,3)). FKG-E tính độ tương đồng với toàn bộ |R| nhúng luật, nên tăng tuyến
    # tính theo |R|: rẻ hơn bản tuần tự, nhưng so với bản bảng tra thì phụ thuộc r và |R|.
    return rows


if __name__ == "__main__":
    os.makedirs(C.PATHS.OUTPUT_DIR, exist_ok=True)
    r4 = run_kb4()
    with open(os.path.join(C.PATHS.OUTPUT_DIR, "kb4_results.json"), "w", encoding="utf-8") as f:
        json.dump(r4, f, ensure_ascii=False, indent=2)
    r6 = run_kb6()
    with open(os.path.join(C.PATHS.OUTPUT_DIR, "kb6_results.json"), "w", encoding="utf-8") as f:
        json.dump(r6, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu kb4_results.json và kb6_results.json vào {C.PATHS.OUTPUT_DIR}")
