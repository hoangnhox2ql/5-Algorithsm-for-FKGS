# Cập nhật FISA theo FKGS v3.2

**Vấn đề.** `models/fisa.py` cũ tính W_{v,l} = Σ support·confidence trên các token tiền đề và gộp theo từng
thuộc tính (nút một thuộc tính, trọng số theo support/confidence). Cách này khác FISA của FKGS v3.2 và của mô hình
FKG-Pairs gốc (nút ba thuộc tính, B^t từ A và M, gộp có điều kiện theo nhãn).

**Thay đổi.**
- `models/fisa.py` viết lại: FKG-Pairs k = 3 theo công thức (7), (8), (3.Wfisa) và suy diễn mờ của Chương 3;
  bảng W dày theo nút (tra bằng tích ngoài độ thuộc, chi phí mỗi truy vấn gần như không đổi theo |R|);
  `FISASequential` (FISA gốc, duyệt tuần tự, O(|R|·C(r,3))) cho cùng D; quy tắc quyết định argmax | ratio9 |
  calibrated; `predict_proba_one` = p_F^cal (Định nghĩa 3.18); `attribution` = a_F (Định nghĩa 3.attrF);
  `evaluate` báo cáo accuracy, F1, độ chính xác cân bằng, AUC, accuracy theo argmax và ratio9, số truy vấn không
  khớp, thời gian. Bản cũ giữ ở `models/fisa_old_v2.py` để đối chiếu.
- `config.FISA`: NODE_SIZE = 3, DECISION = "calibrated", CALIB_CRITERION = "bacc", TEMPERATURE = 1.0.
- `data/fkg_io.generate_synthetic_fkg`: luật tổng hợp có đủ mọi thuộc tính (yêu cầu của FKG-Pairs).
- Thực nghiệm: mọi chỗ dựng FISA dùng `FISA(fkg).fit(val_samples=train)` (ngưỡng chọn trên tập huấn luyện của
  FKG-E, không dùng tập kiểm tra); KB1 và KB6 đo thêm FISA tuần tự và tốc độ của FKG-E so với cả hai cài đặt.
- Báo cáo KB1: thêm cột BalAcc, AUC của FISA, thời gian FISA tuần tự, tốc độ so với bản tuần tự.
- Kiểm thử `tests/test_fisa_v32.py` (4 test đạt): trùng công thức vòng lặp; bảng tra = tuần tự với độ thuộc mờ;
  trùng `src/fisa_pairs.FISAPairs` của FKGS v3.2 (đặt FKGS_V32_DIR); đóng góp luật là phân rã đúng.

**Đo thời gian (10 thuộc tính, dữ liệu tổng hợp):** bảng tra 0,07–0,08 ms/truy vấn với |R| = 250–10 000; bản tuần
tự 0,9 → 14,3 ms/truy vấn (tăng tuyến tính theo |R|).
