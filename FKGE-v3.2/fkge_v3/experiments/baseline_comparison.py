"""
experiments/baseline_comparison.py — Bảng so sánh tổng hợp FKG-E với các
baseline (Mục 3.5.4 / bảng "So sánh tổng hợp FKG-E"):
  - FISA (Chương 2)           : suy luận ký hiệu trực tiếp -- bắt buộc số 1
  - Node2Vec + kNN            : nhúng đồ thị tổng quát, bỏ qua cấu trúc luật
  - TransE + kNN              : KGE chuẩn, bỏ qua cấu trúc luật
  - DistMult + kNN            : KGE chuẩn, bỏ qua cấu trúc luật
  - FKG-E (đề xuất)           : nhúng chuyên biệt cho luật mờ
Dữ liệu: experiments/common.py (BRSET thật 5 fold nếu có); mỗi phương pháp chạy trên cùng các fold.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as C
from models.kge_baselines import Node2VecLite, TransELite, DistMultLite, KNNOnEmbedding
from experiments.common import brset_folds, describe_source, seeds_for, run_fkge, fisa_metrics, aggregate

KEYS = ["accuracy", "balanced_accuracy", "auc", "f1_macro", "train_time_s", "avg_time_per_query_ms"]


def _row(method, runs):
    A = aggregate(runs, KEYS)
    row = {"method": method, "accuracy": A["accuracy_mean"], "accuracy_std": A["accuracy_std"],
           "balanced_accuracy": A["balanced_accuracy_mean"], "balanced_accuracy_std": A["balanced_accuracy_std"],
           "auc": A["auc_mean"], "auc_std": A["auc_std"], "f1": A["f1_macro_mean"],
           "train_time_s": A["train_time_s_mean"], "avg_ms_per_query": A["avg_time_per_query_ms_mean"]}
    print(f"  {method:16s}: acc={row['accuracy']:.4f}  bacc={row['balanced_accuracy']:.4f}  AUC={row['auc']:.4f}  "
          f"f1={row['f1']:.4f}  train={row['train_time_s']:.3f}s  infer={row['avg_ms_per_query']:.4f}ms")
    return row


def run_baseline_comparison(n_seeds=None):
    print("\n" + "=" * 70)
    print("Bảng so sánh tổng hợp: FISA / FKG-E / Node2Vec / TransE / DistMult")
    print("=" * 70)
    print(f"Nguồn dữ liệu: {describe_source()}")
    folds = brset_folds()
    n = seeds_for(folds, n_seeds)
    rows = []

    # --- FISA ---
    runs = []
    for f in folds:
        m = dict(fisa_metrics(f)); m["train_time_s"] = m["fit_time_s"]
        runs.append((f.key, m))
    rows.append(_row("FISA (Chương 2)", runs))

    # --- Node2Vec / TransE / DistMult + kNN ---
    for method, cls in [("Node2Vec + kNN", Node2VecLite), ("TransE + kNN", TransELite), ("DistMult + kNN", DistMultLite)]:
        runs = []
        for f in folds:
            for s in range(n):
                emb = cls(f.fkg, d=C.FKGE.d_e, epochs=15, seed=42 + s).fit()
                r = KNNOnEmbedding(f.fkg, emb.E, f.fkg.token2idx, k=5).evaluate(f.test)
                r["train_time_s"] = emb.train_time_s
                runs.append((f.key, r))
        rows.append(_row(method, runs))

    # --- FKG-E (đề xuất) ---
    tasks = [(f, C.FKGE.seed + s, None) for f in folds for s in range(n)]
    res = run_fkge(tasks, "Baseline FKG-E")
    rows.append(_row("FKG-E (đề xuất)", [(t[0].key, m) for t, m in zip(tasks, res)]))
    for r in rows:
        r["n_folds"] = len(folds)
    return rows


if __name__ == "__main__":
    os.makedirs(C.PATHS.OUTPUT_DIR, exist_ok=True)
    rows = run_baseline_comparison()
    with open(os.path.join(C.PATHS.OUTPUT_DIR, "baseline_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu baseline_comparison.json vào {C.PATHS.OUTPUT_DIR}")
