# FKGS v3 — mã thực nghiệm Chương 2 (đã sửa theo Báo cáo QA FKGS v2)

## Cài đặt
    pip install numpy pandas scipy scikit-learn matplotlib pytest

## Kiểm thử (bắt buộc đạt trước khi chạy)
    python -m pytest -q tests/          # 46 test hồi quy theo báo cáo QA + bộ đối chiếu legacy

## Chạy thực nghiệm E1–E4 (mỗi lệnh tạo một run mới, có manifest)
    python run_all.py --dataset diabetes      # ~7 phút (1 lõi)
    python run_all.py --dataset heart         # ~1 phút
    python run_all.py --dataset heart --quick --results-suffix quick   # thử nhanh

## Phân tích bổ sung (chạy SAU run_all, ghi vào cùng thư mục kết quả)
    python analysis/sensitivity_pair_scope.py --dataset diabetes   # E6: quy ước miền cặp FISA
    python analysis/ablation_v2_to_v3.py --dataset diabetes        # E6: tách ảnh hưởng các sửa đổi
    python analysis/legacy_build_cost.py --dataset diabetes        # E5: chi phí xây FKG gốc

## Kết quả kèm theo
- results/diabetes/  run 20260926T112708-5f1d4be2
- results/heart/     run 20260926T112613-383fbf90
  (manifest.json, E1–E4_results.json với dữ liệu thô, report_tong_hop.md, figures/, và các
  tệp phân tích bổ sung). Hai run dùng cùng SHA-256 mã nguồn.

## Dữ liệu
- data/diabetes/Data.csv — dữ liệu của dự án (SHA-256 5236d580…, khớp báo cáo QA).
- data/heart/Data.csv — bộ tổng hợp 5 nguồn UCI (918 bản ghi), lấy từ bản sao công khai
  trên GitHub (lcbueno/Heart_failure_scipy_ANOVA_Tukey); cần đối chiếu với bản gốc
  "Heart Failure Prediction" (fedesoriano, Kaggle) trước khi công bố.

Chi tiết sửa lỗi: CHANGELOG_v3.md.
