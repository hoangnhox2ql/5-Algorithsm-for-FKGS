"""
experiments/kb4_kb6.py — KB4: quét lưới (lambda, beta), vẽ heatmap Accuracy.
                         KB6: đường cong thời gian suy diễn theo |R| tăng dần
                              (mô phỏng bằng tỉ lệ mẫu 20/40/60/80/100% từ FKGS).
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from data.fkg_io import FKGRuleBase, load_test_samples, generate_synthetic_fkg, generate_synthetic_test_samples
from models.fisa import FISA, FISASequential
from models.fkge import FKGE
import config as C


def _get_brset_or_synthetic():
    if os.path.exists(C.PATHS.BRSET_RULES_FILE) and os.path.exists(C.PATHS.BRSET_TEST_FILE):
        fkg = FKGRuleBase.load(C.PATHS.BRSET_RULES_FILE)
        samples = load_test_samples(C.PATHS.BRSET_TEST_FILE)
        print("Đã nạp dữ liệu BRSET THẬT.")
    else:
        print("!! Chưa có BRSET thật -- dùng dữ liệu tổng hợp.")
        fkg = generate_synthetic_fkg(n_rules=250, seed=100, dataset_name="BRSET")
        samples = generate_synthetic_test_samples(fkg, n_samples=200, seed=200)
    n_train = int(0.6 * len(samples))
    return fkg, samples[:n_train], samples[n_train:]


def run_kb4(lambda_grid=C.KB4_LAMBDA_GRID, beta_grid=C.KB4_BETA_GRID, n_seeds=3):
    print("\n" + "=" * 70)
    print(f"KB4: Quét lưới (λ_I={lambda_grid}, λ_P={beta_grid}) — đánh đổi giữa độ nhất quán với FISA và dự đoán")
    print("=" * 70)
    fkg, train, test = _get_brset_or_synthetic()
    fisa = FISA(fkg).fit(val_samples=train)
    heatmap = np.zeros((len(lambda_grid), len(beta_grid)))
    rows = []
    for i, lam in enumerate(lambda_grid):
        for j, beta in enumerate(beta_grid):
            accs, agr, devs = [], [], []
            for s in range(n_seeds):
                # KB4 (Mục 3.2.5.x, Định lý đánh đổi): quét (λ_I, λ_P); tỉ số ρ = λ_I/λ_P quyết định điểm trên mặt Pareto
                m = FKGE(fkg, lam_I=lam, lam_P=beta, T_ep=C.FKGE.epochs, seed=C.FKGE.seed + s)
                m.fit(fisa_model=fisa, train_samples=train)
                r = m.evaluate(test)
                accs.append(r["accuracy"]); agr.append(r["agreement_fisa"]); devs.append(r["dev_rule"])
            mean_acc = float(np.mean(accs))
            heatmap[i, j] = mean_acc
            rows.append({"lambda": lam, "beta": beta, "rho": lam / beta, "accuracy_mean": mean_acc,
                         "accuracy_std": float(np.std(accs)), "agreement_mean": float(np.mean(agr)),
                         "dev_mean": float(np.mean(devs))})
            print(f"  λ_I={lam:.1f} λ_P={beta:.1f} (ρ={lam / beta:.2f}): accuracy={mean_acc:.4f}  "
                  f"đồng thuận FISA={np.mean(agr):.4f}  Dev={np.mean(devs):.4f}")
    return {"lambda_grid": lambda_grid, "beta_grid": beta_grid,
            "heatmap": heatmap.tolist(), "rows": rows}


def run_kb6(ratios=C.KB6_SAMPLE_RATIOS, n_seeds=3):
    print("\n" + "=" * 70)
    print(f"KB6: Khả năng mở rộng quy mô — tỉ lệ mẫu {ratios}")
    print("=" * 70)
    fkg_full, train, test = _get_brset_or_synthetic()
    rows = []
    for ratio in ratios:
        if os.path.exists(C.PATHS.BRSET_FKGS_RULES_FILE):
            # Nếu có sẵn nhiều mức nén FKGS thật theo tỉ lệ, nạp đúng file
            # tương ứng (quy ước đặt tên: brset_fkgs_rules_<ratio>.json)
            candidate = C.PATHS.BRSET_FKGS_RULES_FILE.replace(".json", f"_{int(ratio*100)}.json")
            fkg_r = FKGRuleBase.load(candidate) if os.path.exists(candidate) else fkg_full.sample_subset(ratio, seed=7)
        else:
            fkg_r = fkg_full.sample_subset(ratio, seed=7)

        fisa = FISA(fkg_r).fit(val_samples=train)
        res_fisa = fisa.evaluate(test)
        res_seq = FISASequential(fkg_r).fit(val_samples=train).evaluate(test)

        fkge_times, fkge_accs = [], []
        for s in range(n_seeds):
            m = FKGE(fkg_r, d=C.FKGE.d, w=C.FKGE.w, K_neg=C.FKGE.K_neg,
                     lam_node=C.FKGE.lam_node, beta_rule=C.FKGE.beta_rule,
                     gamma_inf=C.FKGE.gamma_inf, delta_pred=C.FKGE.delta_pred,
                     lr=C.FKGE.lr, epochs=C.FKGE.epochs, seed=C.FKGE.seed + s)
            m.fit(fisa_model=fisa, train_samples=train)
            r = m.evaluate(test)
            fkge_times.append(r["avg_time_per_query_ms"])
            fkge_accs.append(r["accuracy"])

        row = {
            "ratio": ratio, "n_rules": len(fkg_r),
            "fisa_avg_ms": res_fisa["avg_time_per_query_ms"],
            "fisa_seq_avg_ms": res_seq["avg_time_per_query_ms"],
            "fisa_bacc": res_fisa["balanced_accuracy"],
            "fkge_avg_ms_mean": float(np.mean(fkge_times)),
            "fkge_avg_ms_std": float(np.std(fkge_times)),
            "fisa_accuracy": res_fisa["accuracy"],
            "fkge_accuracy_mean": float(np.mean(fkge_accs)),
        }
        print(f"  |R|={row['n_rules']:4d} (tỉ lệ {ratio:.0%}): "
              f"FISA bảng tra={row['fisa_avg_ms']:.4f}ms  FISA tuần tự={row['fisa_seq_avg_ms']:.4f}ms  "
              f"FKG-E={row['fkge_avg_ms_mean']:.4f}ms")
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
