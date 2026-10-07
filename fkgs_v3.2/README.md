# FKGS v3.2 — mã thực nghiệm Chương 2 (FISA theo đúng mô hình FKG-Pairs của bài báo gốc)

## Cài đặt
    pip install numpy pandas scipy scikit-learn matplotlib pytest

## Kiểm thử (bắt buộc đạt trước khi chạy)
    python -m pytest -q tests/          # 53 test: hồi quy QA, đối chiếu legacy, kiểm định FKG-Pairs

## Chạy thực nghiệm E1–E4 (mỗi lệnh tạo một run mới, có manifest)
    python run_all.py --dataset diabetes      # ~7 phút (1 lõi)
    python run_all.py --dataset heart         # ~1 phút
    python run_all.py --dataset heart --quick --results-suffix quick   # thử nhanh

## Phân tích bổ sung (chạy SAU run_all, ghi vào cùng thư mục kết quả)
    python analysis/sensitivity_node_size.py --dataset diabetes    # độ nhạy theo kích thước nút k = 1, 2, 3
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

## Suy diễn FISA (v3.2)
- `src/fisa_pairs.py`: FKG-Pairs theo đúng công thức (7), (8), (10), (11), (12) của bài báo
  (MTAP 2022) và notebook FKG.ipynb — C̃ gộp CÓ ĐIỀU KIỆN THEO NHÃN; mặc định k = 3 như notebook.
- Tuỳ chọn dòng lệnh: `--node-size 1|2|3` (k), `--decision argmax|ratio9|calibrated`
  (argmax = công thức (12); ratio9 = quy tắc D0 > 9·D1 của notebook; calibrated = ngưỡng trên
  s = log D1 − log D0 chọn trên tập xác thực, mặc định).
- `src/fkg_original.py`, `src/fkg_fast.py`: giữ nguyên notebook (kể cả lỗi bỏ sót luật cuối), chỉ để đối chiếu.
- Ví dụ chạy lại k-fold trên hai bộ dữ liệu:
      python run_all.py --dataset heart
      python run_all.py --dataset diabetes
      python run_all.py --dataset heart --node-size 1 --results-suffix k1       # so sánh k = 1
      python run_all.py --dataset heart --decision argmax --results-suffix argmax
