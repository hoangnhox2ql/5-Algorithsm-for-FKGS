# Dự án Thực nghiệm FKGS v2 (Chương 2 — 4 thuật toán lấy mẫu)

Hiện thực hóa E1-E4 theo `docs/thiet_ke_thuc_nghiem.md`, **đã điều chỉnh còn
4 thuật toán** (GreedyFKGS, Random, SFKGS, CFKGS) theo cơ sở lý thuyết trong
`docs/Dieu_chinh_5_thuat_toan_Submodular_v4.pdf`, và **xây lại FKG/FKGS theo
từng fold** của k-fold (không dùng một FKG cố định cho mọi fold).

## ⚠️ Đọc trước — các quyết định và phát hiện quan trọng

### 1. Vì sao chỉ 4 thuật toán, không phải 5
`Dieu_chinh_5_thuat_toan_Submodular_v4.pdf` chứng minh: khi sửa 2.3 (Ngẫu
nhiên có chủ đích), 2.6 (Hệ thống), 2.8 (Quả cầu tuyết) để có bảo chứng
$(1-1/e)$, cả ba **suy biến trùng hoàn toàn GreedyFKGS** — không còn là
thuật toán riêng biệt để cài đặt. Chỉ 2.5 (→ **SFKGS**) và 2.7 (→
**CFKGS**) giữ được cấu trúc thiết kế riêng (phân tầng/phân cụm) nhờ tái
sử dụng Định lý 7 (Stratified Greedy) cho phân hoạch tổng quát. Vì vậy bộ
4 thuật toán cuối: **GreedyFKGS, SFKGS, CFKGS** (3 thuật toán có bảo chứng
$(1-1/e)$, đều dùng chung cơ chế tham lam trên bảng `Cov` — Bổ đề 1) +
**Random** (baseline, không có bảo chứng, dùng để đối chứng ở E2).

### 2. Kiến trúc: 3 lớp FISA tách biệt, đừng nhầm lẫn
- `src/fkg_original.py`: sao chép **trung thành** `FKG.ipynb`, **cố tình
  giữ nguyên cả 3 lỗi đã biết** (off-by-one, gán-thay-cộng-dồn, ngưỡng
  9:1). CHỈ dùng để đối chiếu số học, KHÔNG dùng suy diễn thật.
- `src/fkg_fast.py`: vector hóa tương đương (radix-encoding + `np.unique`),
  phải khớp tuyệt đối với bản trên — **đã kiểm định PASS** trên 5 bộ dữ
  liệu khác nhau (`test_validate_fast_vs_original.py`).
- `src/fisa_corrected.py`: FISA **đã sửa đúng** công thức (1.17)-(1.20)
  (bảng tra cứu 3 chiều $W_{i,v,l}$, xem Chương 3), dùng THẬT cho AUC
  downstream ở E2/E3.

### 3. Hạn chế đã biết và CHỦ ĐỘNG CHẤP NHẬN (Hướng A): FISA nhạy với mất cân bằng lớp
Số hạng $|Val(P_i)\to l|$ trong (1.17) đếm tuyệt đối, chia cho $|R|$ tổng
thể (không chia riêng $|R_l|$ từng lớp). Trên dữ liệu lệch lớp mạnh, điều
này khiến $W_{i,v,\text{lớp đa số}}$ lớn hơn **hệ thống** tại mọi giá trị
thuộc tính → FISA có thể sụp về luôn đoán lớp đa số (AUC→0.5). **Đã quyết
định giữ nguyên công thức gốc**, không sửa. `warn_if_fisa_collapsed()`
trong `src/fisa_corrected.py` tự động cảnh báo mỗi khi hiện tượng này xảy
ra trong quá trình chạy — **Rep là chỉ số chính** để đánh giá 4 thuật toán
lấy mẫu (không bị ảnh hưởng bởi vấn đề này); AUC chỉ mang tính tham khảo.

### 4. Lỗi thiết kế thống kê đã tìm và sửa ở E2
Bản đầu tiên lấy **trung bình** 30 seed Random mỗi fold rồi ghép cặp chỉ 5
giá trị cho Wilcoxon — làm giảm cỡ mẫu kiểm định từ 150 xuống 5, gần như
triệt tiêu sức mạnh thống kê (dù Cohen's d rất lớn 12-20, p-value sau Holm
vẫn > 0.05 với n=5). **Đã sửa**: gộp toàn bộ (fold × seed) = 150 cặp thành
1 kiểm định Wilcoxon duy nhất — xác nhận lại cho p<0.0001, khớp đúng với
effect size lớn quan sát được.

### 5. K-fold nằm ở đâu (áp dụng đúng nguyên tắc đã dùng cho FKG-MM/FKG-E)
`src/kfold_utils.py`: `make_stratified_kfold()` chia dữ liệu **thô** theo
Outcome (StratifiedKFold — Diabetes không có nhiều bản ghi/bệnh nhân như
BRSET nên không cần GroupKFold). `build_fkg_fold()` sau đó **mờ hóa (fit
chỉ trên train) + khai phá luật + tính Sim** hoàn toàn MỚI cho mỗi fold —
không tái sử dụng một FKG cố định.

## Cài đặt

```bash
pip install numpy pandas scikit-learn scipy matplotlib --break-system-packages
```

## Chạy

```bash
# Bước 1 — BẮT BUỘC: kiểm định số học fkg_fast.py khớp fkg_original.py
python test_validate_fast_vs_original.py

# Bước 2 — kiểm tra toàn bộ pipeline không lỗi (dùng dữ liệu tổng hợp nếu
# chưa có data/Data.csv thật)
python run_all.py --quick

# Bước 3 — chạy đầy đủ trên dữ liệu thật (đặt Data.csv vào data/ trước)
python run_all.py
```

Chạy từng bước riêng: `python run_all.py --only E1,E4`.

## Chuẩn bị dữ liệu thật

Đặt file `data/Data.csv` với đúng 9 cột: `Pregnancies, Glucose,
BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age,
Outcome`. Nếu chưa có, `run_00_preprocess.py` tự sinh dữ liệu **tổng hợp**
mô phỏng đúng đặc điểm mô tả (0-giả, trùng lặp, mất cân bằng lớp) để bạn
kiểm tra pipeline — **không dùng kết quả từ dữ liệu này để báo cáo**.

## Kết quả

`results/report_tong_hop.md` — bảng + biểu đồ đầy đủ cho E1-E4, dùng trực
tiếp cho luận án.
