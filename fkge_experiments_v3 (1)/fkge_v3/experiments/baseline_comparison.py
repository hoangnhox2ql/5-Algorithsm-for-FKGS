"""
experiments/baseline_comparison.py — Bảng so sánh tổng hợp FKG-E với các
baseline (Mục 3.5.4 / bảng "So sánh tổng hợp FKG-E"):
  - FISA (Chương 2)           : suy luận ký hiệu trực tiếp -- bắt buộc số 1
  - Node2Vec + kNN            : nhúng đồ thị tổng quát, bỏ qua cấu trúc luật
  - TransE + kNN              : KGE chuẩn, bỏ qua cấu trúc luật
  - DistMult + kNN            : KGE chuẩn, bỏ qua cấu trúc luật
  - FKG-E (đề xuất)           : nhúng chuyên biệt cho luật mờ
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from data.fkg_io import FKGRuleBase, load_test_samples, generate_synthetic_fkg, generate_synthetic_test_samples
from models.fisa import FISA
from models.fkge import FKGE
from models.kge_baselines import Node2VecLite, TransELite, DistMultLite, KNNOnEmbedding
import config as C


def _get_brset_or_synthetic():
    if os.path.exists(C.PATHS.BRSET_RULES_FILE) and os.path.exists(C.PATHS.BRSET_TEST_FILE):
        fkg = FKGRuleBase.load(C.PATHS.BRSET_RULES_FILE)
        samples = load_test_samples(C.PATHS.BRSET_TEST_FILE)
        print("Đã nạp dữ liệu BRSET THẬT.")
    else:
        print("!! Chưa có BRSET thật -- dùng dữ liệu tổng hợp cho so sánh baseline.")
        fkg = generate_synthetic_fkg(n_rules=250, seed=100, dataset_name="BRSET")
        samples = generate_synthetic_test_samples(fkg, n_samples=200, seed=200)
    n_train = int(0.6 * len(samples))
    return fkg, samples[:n_train], samples[n_train:]


def run_baseline_comparison(n_seeds=C.EVAL.N_SEEDS):
    print("\n" + "=" * 70)
    print("Bảng so sánh tổng hợp: FISA / FKG-E / Node2Vec / TransE / DistMult")
    print("=" * 70)
    fkg, train, test = _get_brset_or_synthetic()
    rows = []

    # --- FISA ---
    fisa = FISA(fkg).fit(val_samples=train)
    res = fisa.evaluate(test)
    rows.append({"method": "FISA (Chương 2)", "accuracy": res["accuracy"], "f1": res["f1_macro"],
                 "train_time_s": res["fit_time_s"], "avg_ms_per_query": res["avg_time_per_query_ms"]})
    print(f"  FISA         : acc={res['accuracy']:.4f}  f1={res['f1_macro']:.4f}  "
          f"train={res['fit_time_s']:.3f}s  infer={res['avg_time_per_query_ms']:.4f}ms")

    # --- Node2Vec + kNN ---
    accs, f1s, tts, its = [], [], [], []
    for s in range(n_seeds):
        n2v = Node2VecLite(fkg, d=C.FKGE.d, epochs=15, seed=42 + s).fit()
        clf = KNNOnEmbedding(fkg, n2v.E, fkg.token2idx, k=5)
        r = clf.evaluate(test)
        accs.append(r["accuracy"]); f1s.append(r["f1_macro"])
        tts.append(n2v.train_time_s); its.append(r["avg_time_per_query_ms"])
    rows.append({"method": "Node2Vec + kNN", "accuracy": float(np.mean(accs)), "f1": float(np.mean(f1s)),
                 "train_time_s": float(np.mean(tts)), "avg_ms_per_query": float(np.mean(its))})
    print(f"  Node2Vec+kNN : acc={np.mean(accs):.4f}  f1={np.mean(f1s):.4f}  "
          f"train={np.mean(tts):.3f}s  infer={np.mean(its):.4f}ms")

    # --- TransE + kNN ---
    accs, f1s, tts, its = [], [], [], []
    for s in range(n_seeds):
        te = TransELite(fkg, d=C.FKGE.d, epochs=15, seed=42 + s).fit()
        clf = KNNOnEmbedding(fkg, te.E, fkg.token2idx, k=5)
        r = clf.evaluate(test)
        accs.append(r["accuracy"]); f1s.append(r["f1_macro"])
        tts.append(te.train_time_s); its.append(r["avg_time_per_query_ms"])
    rows.append({"method": "TransE + kNN", "accuracy": float(np.mean(accs)), "f1": float(np.mean(f1s)),
                 "train_time_s": float(np.mean(tts)), "avg_ms_per_query": float(np.mean(its))})
    print(f"  TransE+kNN   : acc={np.mean(accs):.4f}  f1={np.mean(f1s):.4f}  "
          f"train={np.mean(tts):.3f}s  infer={np.mean(its):.4f}ms")

    # --- DistMult + kNN ---
    accs, f1s, tts, its = [], [], [], []
    for s in range(n_seeds):
        dm = DistMultLite(fkg, d=C.FKGE.d, epochs=15, seed=42 + s).fit()
        clf = KNNOnEmbedding(fkg, dm.E, fkg.token2idx, k=5)
        r = clf.evaluate(test)
        accs.append(r["accuracy"]); f1s.append(r["f1_macro"])
        tts.append(dm.train_time_s); its.append(r["avg_time_per_query_ms"])
    rows.append({"method": "DistMult + kNN", "accuracy": float(np.mean(accs)), "f1": float(np.mean(f1s)),
                 "train_time_s": float(np.mean(tts)), "avg_ms_per_query": float(np.mean(its))})
    print(f"  DistMult+kNN : acc={np.mean(accs):.4f}  f1={np.mean(f1s):.4f}  "
          f"train={np.mean(tts):.3f}s  infer={np.mean(its):.4f}ms")

    # --- FKG-E (đề xuất) ---
    accs, f1s, tts, its = [], [], [], []
    for s in range(n_seeds):
        m = FKGE(fkg, d=C.FKGE.d, w=C.FKGE.w, K_neg=C.FKGE.K_neg,
                 lam_node=C.FKGE.lam_node, beta_rule=C.FKGE.beta_rule,
                 gamma_inf=C.FKGE.gamma_inf, delta_pred=C.FKGE.delta_pred,
                 lr=C.FKGE.lr, epochs=C.FKGE.epochs, seed=C.FKGE.seed + s)
        m.fit(fisa_model=fisa, train_samples=train)
        r = m.evaluate(test)
        accs.append(r["accuracy"]); f1s.append(r["f1_macro"])
        tts.append(m.train_time_s); its.append(r["avg_time_per_query_ms"])
    rows.append({"method": "FKG-E (đề xuất)", "accuracy": float(np.mean(accs)), "f1": float(np.mean(f1s)),
                 "train_time_s": float(np.mean(tts)), "avg_ms_per_query": float(np.mean(its))})
    print(f"  FKG-E        : acc={np.mean(accs):.4f}  f1={np.mean(f1s):.4f}  "
          f"train={np.mean(tts):.3f}s  infer={np.mean(its):.4f}ms")

    return rows


if __name__ == "__main__":
    os.makedirs(C.PATHS.OUTPUT_DIR, exist_ok=True)
    rows = run_baseline_comparison()
    with open(os.path.join(C.PATHS.OUTPUT_DIR, "baseline_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu baseline_comparison.json vào {C.PATHS.OUTPUT_DIR}")
