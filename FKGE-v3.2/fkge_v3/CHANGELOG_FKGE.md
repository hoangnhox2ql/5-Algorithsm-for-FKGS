# FKG-E v3 — cài đặt lại theo Chương 3

Bản v2 (`models/fkge_old_v2.py`, giữ để đối chiếu) khác Chương 3 ở nhiều điểm: skip-gram theo cửa sổ vị trí,
L_node so khớp trọng số μ của tệp cạnh, p_E là trộn phân phối one-hot của luật, không có L_edge/L_A/L_B,
L_rule là SGNS, không có λ_C, không chiếu, Adam/SGD tự viết, trọng số đặt tay (δ_pred = 20).

`models/fkge.py` (mới) cài đặt đúng các định nghĩa và thuật toán của Chương 3 (xem README). Unit test
`tests/test_fkge_ch3.py` (13 test, đều đạt) kiểm tra:
1. ma trận đồng xuất hiện toàn luật, Mệnh đề đồng xuất hiện ↔ (ā, b);
2. P_n, ω đối xứng và thuộc (0, 1];
3. nhúng luật (3.pool), nhúng truy vấn (3.hj)–(3.query), gộp đều α = r/(r+1);
4. p_E là phân phối, ≥ ε/n_L; cận của log-sum-exp; cos_ε0 ∈ (−1, 1);
5. miền giá trị các thành phần, cận dưới L_B, cận L_inf, L_rule, cận c_τ của L_pred;
6. cận dưới L_B đạt được với b̃ (Mệnh đề 3.coer(iii));
7. bất biến quay L(QΘ) = L(Θ);
8. tính bức L ≥ λ_C‖Θ‖², phép chiếu giữ ‖Θ‖ ≤ B₀, λ_C = 0 bị từ chối;
9. ước lượng theo lô không chệch (Thuật toán 3.10);
10. gradient trùng sai phân hữu hạn;
11. giáo viên: argmax p_F^cal trùng quyết định của FISA, e^F chỉ trên luật của ŷ_F;
12. huấn luyện giảm hàm mục tiêu và chuẩn gradient; a_E là phân phối trên luật của nhãn dự đoán;
    Dev ≥ tỉ lệ bất đồng (Mệnh đề quan hệ giữa các ràng buộc);
13. ánh xạ tên tham số cũ.

# FKG-E v3.1 — khớp Chương 3 bản viết lại (word2vec trên A, B) và chạy trên BRSET 5 fold

- `models/fkge.py`: thành phần cấu trúc viết lại theo Chương 3:
  * nút giá trị 𝒱 (không còn token nhãn trong đồ thị); trọng số cạnh a_ii' (3.5) = A^t_jk; phân bố nhãn b_il (3.6);
  * L_edge = SGNS (word2vec) có trọng số a_ii', mẫu âm theo P_n = a_n/|a|; L_node = entropy chéo với b̃ qua u_l;
  * nhúng luật r_k = (α/r)Σz + (1−α)u_out; Θ = {z, c, u};
  * bỏ đồng xuất hiện toàn luật/cửa sổ, ω, L_A, L_B dạng cũ; tên tham số cũ (lam_S, beta_rule) ánh xạ sang λ_E,
    lam_A/lam_B/cooc/w bị bỏ qua kèm cảnh báo.
- `config.FKGE`: λ_E, λ_N, λ_R, λ_I, λ_P, λ_C; bỏ cooc/w. Ablation: bỏ L_edge, bỏ L_node, bỏ cả hai, bỏ L_rule, L_inf,
  L_pred, gộp đều. KB5 chỉ quét K.
- Kiểm thử `tests/test_fkge_ch3_v2.py` (16 test) thay bộ cũ: (3.5) đúng là A của FKG; P_n, b, b̃; nhúng luật dùng u_out;
  nhúng truy vấn; p_E và cận log-sum-exp; **nghiệm bảo toàn cạnh = PMI dịch log K** (tối ưu số từng cặp);
  L_edge phân rã thành các số hạng cặp; cận dưới L_node và trường hợp đạt cận; thành phần không âm, bất biến quay;
  tính bức và phép chiếu; ước lượng theo lô không chệch; gradient = sai phân hữu hạn; giáo viên khớp FISA; giáo viên
  từ mảng (β, T); huấn luyện giảm hàm mục tiêu, a_E đúng giá, Dev ≥ tỉ lệ bất đồng; che luật cùng nhóm; tên cũ.
- `tests/test_brset_folds.py`: thêm kiểm thử đầu-cuối một fold BRSET.
- Tổng: 25 test đạt (gồm đối chiếu FISA với FKGS v3.2 khi đặt FKGS_V32_DIR).
- Kết quả BRSET 5 fold (outputs/brset_kfold.md): FKG-E AUC 0,926 ± 0,025, BalAcc 0,810 ± 0,037; FISA AUC 0,848.

# FKG-E v3.2 — thuật toán huấn luyện khớp Định lý hội tụ (theo kế hoạch chỉnh sửa, mục 9)
- `optimizer = "gd_full"` (mặc định): giảm gradient toàn bộ có chiếu lên hình cầu B₀, độ dài bước chọn bằng tìm kiếm theo
  đường với điều kiện giảm đủ L(Θ⁺) ≤ L(Θ) − (η/2)‖∇L‖² — đúng bất đẳng thức (i) của Định lý hội tụ; `eta_gd = 1.0`,
  `T_ep = 200` vòng, `patience = 20`. Các biến thể theo lô (`amsgrad`, `sgd`) giữ để so sánh, không dùng khi báo cáo.
- Kiểm thử mới: hàm mục tiêu giảm đơn điệu qua mọi vòng, bước dương, tham số nằm trong hình cầu B₀. Tổng 26 test đạt.
- BRSET 5 fold (outputs/brset_kfold.md): FKG-E AUC 0,921 ± 0,033, BalAcc 0,825 ± 0,050, F1 0,508; FISA AUC 0,848;
  chênh lệch AUC dương ở 5/5 fold, p ≈ 0,009 (t cặp hiệu chỉnh Nadeau–Bengio), Wilcoxon p = 0,0625 (giá trị nhỏ nhất với 5 cặp).

# FKG-E v3.3 — chạy toàn bộ KB1–KB6, Ablation, Baseline trên BRSET thật (data_real/BRSET_Data, 5 fold)
- Trước đây chỉ `experiments/brset_kfold.py` dùng dữ liệu thật; `run_all.py` tìm các file JSON không tồn tại nên mọi KB
  chạy trên dữ liệu tổng hợp. Mới: `experiments/common.py` — nguồn dữ liệu chung (BRSET 5 fold > JSON > tổng hợp),
  huấn luyện FKG-E song song (fork), đánh giá/đo thời gian tuần tự, dùng lại cấu hình đã huấn luyện, tổng hợp trung bình
  ± độ lệch chuẩn giữa các fold.
- `brset_kfold.py`: tách `prepare_fold` (giáo viên FISA ngoài phần, τ_F, tập xác thực theo nhóm; tuỳ chọn giữ một tập con
  luật) khỏi `run_fold`; kết quả `run_fold` không đổi.
- KB1–KB6, Ablation, Baseline viết lại trên `common`: chạy theo fold; thêm BalAcc/AUC (tập kiểm tra ~8% dương); KB1 bỏ
  qua bộ Diabetes thiếu file thay vì dùng dữ liệu tổng hợp; KB2/KB6 mô phỏng FKGS bằng tập con luật mỗi fold (ghi rõ
  trong báo cáo); cảnh báo sụp lớp theo độ chính xác cân bằng.
- `models/kge_baselines.py`: Node2Vec dựng đồ thị đồng xuất hiện từ luật khi không có tệp cạnh; kNN báo cáo BalAcc, AUC.
- `models/fkge.py`: gd_full dùng `value_and_grad`, tái sử dụng (L, ∇L) cuối vòng cho đầu vòng sau — nhanh ~1,9 lần,
  tham số và lịch sử huấn luyện trùng từng bit với bản v3.2 (đã kiểm tra trên fold_01 và dữ liệu tổng hợp, cả amsgrad).
- `config.py`: `PATHS.BRSET_FOLDS_DIR`, `FKGE_OUTPUT_DIR`; `EVAL.REAL_FOLDS`, `REAL_SEEDS_PER_FOLD`, `F_C`,
  `N_SEQ_TIMING`, `N_JOBS`, `MAIN_METRIC`. `run_all.py`: `--jobs`, `--folds`, `--seeds`, `--out`; ghi `run_meta.json`.
- `report/generate_report.py`: cột BalAcc/AUC, biểu đồ theo `MAIN_METRIC`, đường FISA tuần tự ở KB1/KB6, ghi nguồn
  dữ liệu; sửa lỗi tiêu đề `|R|` làm vỡ bảng Markdown.
- Kiểm thử mới `tests/test_real_pipeline.py` (7 test): fold thật và bộ nhớ đệm, song song = tuần tự, dùng lại mô hình,
  fold con, đồ thị Node2Vec, tổng hợp giữa fold, quay về dữ liệu tổng hợp. Tổng 33 test (1 bỏ qua khi chưa đặt
  FKGS_V32_DIR).
