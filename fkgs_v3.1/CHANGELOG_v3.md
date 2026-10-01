# FKGS v3 — Sửa lỗi theo Báo cáo QA FKGS v2

Bản v3 xây trên mã v2 (19 tệp Python, trùng từng byte với `fkgs_v2.zip`; SHA-256 dữ
liệu tiểu đường `5236d580…` khớp báo cáo QA). Mỗi phát hiện của QA được xử lý như sau.
Cột "Test" là tên hàm trong `tests/test_qa_regression.py`.

## Nghiêm trọng

| Mã | Lỗi | Sửa trong v3 | Test |
|---|---|---|---|
| QA-01 | Điền giá trị thiếu bằng trung vị của lớp của chính hàng test (đọc nhãn test) | `FuzzyFitter` học trung vị toàn cục của train chỉ từ X, dùng cùng quy tắc cho train/test; `transform_features()` không đọc cột nhãn | `test_no_label_leakage_in_imputation`, `test_permuting_test_labels_does_not_change_features`, `test_inference_without_label_column` |
| QA-02 | FISA chỉ cộng cặp j>i: thuộc tính cuối có trọng số 0, phụ thuộc thứ tự cột, lệch ví dụ Chương 1 | `PAIR_SCOPE="global"` (tổng mọi cặp, đúng ví dụ Ch.1: D=6); giữ "adjacent" (Ch.3) và "upper" (v2) để phân tích độ nhạy | `test_fisa_ch1_global_oracle_and_scopes`, `test_fisa_global_is_column_order_invariant`, `test_fisa_upper_matches_v2_hand` |
| QA-03 | Không có raw/manifest liên kết code–run–report | `manifest.json` (run_id, SHA-256 từng tệp mã, SHA-256 dữ liệu, cấu hình hiệu lực, môi trường, thời gian từng bước); mọi JSON mang run_id; REPORT từ chối gộp khác run_id; lưu raw theo fold/seed/block, chỉ số hàng được chọn, chỉ số fold test | chạy thực tế (`results/*/manifest.json`) |

## Trung bình

| Mã | Lỗi | Sửa trong v3 | Test |
|---|---|---|---|
| QA-04 | SFKGS/CFKGS chọn nhiều luật hơn Greedy/Random ở cùng θ | Phân bổ phần dư lớn nhất: tổng đúng bằng ceil(θn); E2/E4 `assert` kích thước và lưu `size`; chế độ `ceil` của v2 giữ để đối chứng | `test_sfkgs_matched_global_budget`, `test_all_methods_same_budget_random_cases`, `test_e4_hand_budgets_equal`, `test_largest_remainder_properties` |
| QA-05 | p-value trên 150 cặp fold×seed / 50 block không độc lập | Đơn vị suy luận chính = fold; seed chỉ báo mô tả (tỉ lệ seed bị vượt); kèm chênh lệch TB, CI95, số fold tốt hơn; E4 Friedman chính trên 5 fold, 50 block chỉ mô tả | `test_wilcoxon_exact_five_positive` |
| QA-06 | Gọi Rep trung bình là "phủ" | `coverage_stats` báo riêng Rep, min cov, tỉ lệ luật có cov ≥ δ; greedy có `coverage_mode="all"`; E3 báo hai ngưỡng khác nhau | `test_coverage_all_mode_is_full_coverage` |
| QA-07 | Đường dẫn dữ liệu sai, tự sinh dữ liệu giả rồi lần sau coi là thật | Cấu hình theo bộ dữ liệu; thiếu file thật → lỗi rõ; dữ liệu giả chỉ khi `--allow-synthetic`, manifest ghi `source` | chạy thực tế |
| QA-08 | Kneedle trên điểm bị trội; thời gian θ=1 gần 0 | `pareto_knee`: lọc Pareto, sắp theo chi phí; chi phí = chọn luật + xây FISA + suy diễn | `test_knee_and_pareto` |
| QA-09 | Mờ hoá/nhân khác mô tả; q50 không dùng | Khai báo tường minh trong `config.py` và manifest; chỉ tính q10, q90 | — |
| QA-10 | Độ phức tạp, bảo chứng mô tả quá rộng; assert "Greedy ≥ Random" | Sửa docstring (O(k n²); SF/CF so với OPT phân bổ); bỏ assert | — |
| QA-11 | Radix legacy: mã -1 của giá trị lạ trùng mã hợp lệ | Tổ hợp chứa giá trị lạ không khớp luật nào (như `fkg_original`) | `test_legacy_matrices_and_unknown_token` |
| QA-12 | Chọn θ trên chính tập test | E3 chia train thành train-trong/validation; θ* chọn trên validation; test chỉ đánh giá tại θ* | chạy thực tế |

## Nhẹ / biên

Rỗng, p = 0, k fold không hợp lệ, ngân sách âm, θ ngoài [0,1], design sai → `ValueError`;
`stratified_subsample` trả đúng số hàng; NaN được điền; Cohen d_z với chênh lệch hằng khác 0
→ ±inf; CD Nemenyi dùng `studentized_range` cho mọi k, α; Holm bác bỏ khi p ≤ α và giữ vị trí
NaN; AUC một lớp → NaN kèm cờ; AUC dùng lề D1−D0 thay xác suất softmax (tránh bão hoà, hoà
giả); E1/E4 không dựng Sim trên toàn tập huấn luyện; `--only` tự chạy PREPROCESS khi cần và
chỉ cho phép khi mã/cấu hình không đổi.

## Điểm cần người chịu trách nhiệm khoa học xác nhận

1. **Quy ước miền cặp của FISA.** v3 chọn "global" theo ví dụ tính tay Chương 1. Phân tích
   độ nhạy (`analysis/sensitivity_pair_scope.py`) cho thấy quy ước này quyết định mức AUC.
2. **Luật quyết định argmax D_l.** Với cả ba quy ước, dự đoán nhãn vẫn gần như luôn là lớp
   đa số (Balanced Accuracy ≈ 0,50) dù điểm lề có AUC cao: số hạng |Val(P_i)→l|/|R| thiên
   về lớp đa số. Không tự sửa công thức; đề xuất hiệu chỉnh ngưỡng lề trên validation.
3. **Thiết kế suy luận.** Với 5 fold, p một phía nhỏ nhất là 1/32 nên sau Holm (15 phép
   kiểm định mỗi họ) không thể có ý nghĩa ở mức 0,05. Muốn kiểm định có độ mạnh cần lặp
   lại việc chia dữ liệu (ví dụ 5×2 CV hoặc 10×10 CV) và mô hình phụ thuộc phù hợp.

---

# FKGS v3.1 — Khắc phục hiện tượng FISA "sụp về lớp đa số"

## Nguyên nhân (đã kiểm chứng bằng thực nghiệm)

1. **Mờ hoá theo phân vị q10/q90** (có từ mã dự án v2, giữ sang v3): khoảng 80% giá trị mỗi
   thuộc tính rơi vào mức Medium, các luật gần như trùng nhau và FKG mất khả năng phân biệt.
   Trên bộ tiểu đường, giao thức 70/30, FISA gốc: AUC 0,684 (q10/q90) so với 0,877 (tam phân vị).
2. **Quy tắc quyết định không thích nghi với tỉ lệ lớp**: D_l = min_i C_il + max_i C_il chứa
   tần suất |Val(P_i)→l|/|R|, nên thiên về lớp đông. Chọn nhãn argmax thì đoán gần như toàn
   lớp đa số; quy tắc cố định D0 > 9·D1 trong mã gốc thì lệch ngược lại (dự đoán dương ~99%).

## Sửa đổi

| Tệp | Sửa đổi |
|---|---|
| `config.py` | `FUZZY_QUANTILES = [1/3, 2/3]` (tam phân vị); `FISA.SCORE="logratio"`, `FISA.DECISION="calibrated"`, `FISA.CALIB_CRITERION="bacc"`, `FISA.VAL_FRACTION=0.2`; lớp `PUB` (giao thức công bố) |
| `src/fisa_corrected.py` | `logratio_one()`: s = log D1 − log D0 (quy tắc D0 > k·D1 ⇔ s < −log k); `calibrate()`: chọn ngưỡng trên tập xác thực; `evaluate()` trả thêm `pred_argmax`, `threshold` |
| `src/kfold_utils.py` | `build_fkg_fold(..., val_fraction)`: tách phân tầng tập xác thực từ tập huấn luyện; bộ mờ hoá và tập nền chỉ xây trên phần còn lại |
| `experiments/common.py` | `evaluate_fisa(..., rules_val)`: hiệu chỉnh ngưỡng trên tập xác thực; báo cả độ đo argmax để đối chiếu |
| `experiments/run_02`, `run_03` | Truyền tập xác thực; E3 báo thêm Acc, BalAcc tại θ* và của FKG đầy đủ |
| `experiments/run_06_published_protocol.py` (mới, bước `PUB`) | Giao thức bài [CT2]: chia cố định 70/30, không loại trùng, FKG đầy đủ; FISA gốc (tổ hợp ba) và FISA-P; mờ hoá q10/q90 và tam phân vị; quy tắc argmax, D0 > 9·D1 và ngưỡng hiệu chỉnh |
| `experiments/run_05_generate_report.py` | Thêm cột Accuracy, mục PUB |
| `tests/test_qa_regression.py` | 2 test mới cho ngưỡng hiệu chỉnh và tương đương s ↔ quy tắc D0 > k·D1 (48/48 đạt) |
| `analysis/diag_fisa.py`, `analysis/published_protocol.py` | Công cụ chẩn đoán độc lập |

Tập kiểm tra không bao giờ được dùng để chọn ngưỡng.
