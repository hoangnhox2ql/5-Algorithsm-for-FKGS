# Thiết kế chi tiết Thực nghiệm E1–E4 trên Bộ dữ liệu Tiểu đường

**Bộ dữ liệu:** Diabetes_data.world (768,8,2) + Diabetes_Kaggle (2767,9,2) + Healthcare_Diabetes_Kaggle (15000,9,2), ghép thành `Data.csv` — 8 đặc trưng chung (`Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age`), nhãn `Outcome` (0/1).

---

## 0. Đặc điểm dữ liệu cần xử lý trước khi thiết kế thực nghiệm

Trước khi thiết kế bất kỳ thí nghiệm nào, cần rà soát chất lượng dữ liệu — bỏ qua bước này sẽ khiến mọi kết quả sau đó bị nhiễu bởi lỗi dữ liệu chứ không phải do thuật toán.

| Vấn đề phát hiện | Số lượng | Cách xử lý |
|---|---|---|
| Dòng trùng lặp hoàn toàn (do ghép 3 nguồn có phần chồng lấp) | 2.758 / 18.536 (14,9%) | Loại bỏ bằng `drop_duplicates()` trước khi lấy mẫu |
| `Glucose = 0` (bất khả thi sinh học) | 23 dòng | Coi là thiếu dữ liệu → impute bằng median theo lớp |
| `BloodPressure = 0` | 160 dòng | Coi là thiếu dữ liệu → impute bằng median theo lớp |
| `SkinThickness = 0` | 1.027 dòng (5,5%) | Coi là thiếu dữ liệu → impute bằng median theo lớp |
| `Insulin = 0` | 1.704 dòng (9,2%) | Coi là thiếu dữ liệu → impute bằng median theo lớp |
| `BMI = 0` | 50 dòng | Coi là thiếu dữ liệu → impute bằng median theo lớp |
| Mất cân bằng lớp | Outcome=0: 66,4% / Outcome=1: 33,6% | Mọi phép chia (train/test, tầng, block) đều **stratify theo Outcome** |

**Nguyên tắc impute theo lớp (không theo toàn bộ dữ liệu):** tính median riêng cho nhóm Outcome=0 và Outcome=1 rồi điền — tránh làm nhoè khác biệt giữa hai lớp mà chính là tín hiệu học máy cần giữ lại.

---

## 1. Xây dựng tập luật mờ $R$ từ dữ liệu bảng

Vì dữ liệu đầu vào là bảng số (không phải luật mờ có sẵn), bước đầu tiên là **mờ hoá** mỗi bản ghi thành một luật mờ $R_i$:

1. Với mỗi đặc trưng $X_j$ ($j=1..8$), xây 3 tập mờ **Low/Medium/High** theo phân phối Ruspini (tổng độ thuộc = 1 tại mọi điểm), tham số hoá bằng phân vị 10/50/90 của chính cột dữ liệu (tránh outlier chi phối như dùng min/max).
2. Mỗi bản ghi $i$ → vector độ thuộc $\mathbf{x}_i \in \mathbb{R}^{24}$ (8 đặc trưng × 3 tập mờ) — đây là bản sửa "well-defined" của Định nghĩa 2.3 đã phản biện trước đó: dùng trực tiếp **số thực độ thuộc** làm toạ độ, không dùng hàm mờ trừu tượng.
3. Khoảng cách $d(R_i,R_j)$ = Euclid trên $\mathbf{x}_i,\mathbf{x}_j$, chuẩn hoá về $[0,1]$.
4. $\mathrm{Sim}(R_i,R_j) = e^{-\gamma \cdot d(R_i,R_j)}$, $\gamma=3.0$ (giá trị mặc định, xem mục 6 để hiệu chỉnh).
5. Nhãn `Outcome` được giữ **tách riêng** (không hoà vào $d$) — dùng khi cần cho Thuật toán 2.5 (phân tầng theo nhãn).

---

## 2. Giới hạn quy mô — vì sao không dùng cả 18.536 dòng

Ma trận $\mathrm{Sim}$ đầy đủ cần $O(n^2)$ bộ nhớ và GreedyFKGS cần $O(m \cdot n^2)$ phép tính. Với $n=18.536$: ma trận Sim chiếm $\approx$ 2,6 GB (float64), và một lần chạy Greedy ở $\theta=0.5$ ước tính hàng giờ trên phần cứng thông thường — không khả thi để chạy lặp lại hàng chục lần (bắt buộc cho E2/E4 vì cần nhiều seed).

**Giải pháp:** lấy mẫu phân tầng (giữ tỉ lệ Outcome) một tập con $N_{EXP}=1.500$ cho E2–E4 (đủ nhỏ để ma trận Sim vừa bộ nhớ, đủ lớn để phản ánh cấu trúc dữ liệu thật), và một tập kiểm tra cố định $N_{TEST}=3.000$ hoàn toàn tách biệt (không tham gia lấy mẫu), dùng để đánh giá AUC downstream công bằng cho mọi phương pháp.

**Nếu có tài nguyên lớn hơn** (máy chủ RAM lớn, GPU cho phép tính ma trận song song): tăng `N_EXP` trong `config.py`, cân nhắc dùng cấu trúc dữ liệu xấp xỉ (LSH, k-d tree) để tránh tính Sim đầy đủ $O(n^2)$ — đây là hướng mở rộng cho thí nghiệm E5 (scalability) nếu triển khai sau.

---

## 3. E1 — Xác nhận lý thuyết: $\mathrm{Rep}(S_G)/\mathrm{Rep}(S^\ast) \ge (1-1/e)$?

### Thiết kế
Vì $S^\ast$ (nghiệm tối ưu) chỉ tính được bằng brute-force ($O(\binom{n}{m})$ tổ hợp), giới hạn $n \le 18$. Thay vì dùng dữ liệu tổng hợp nhân tạo, ta **lấy mẫu con từ chính dữ liệu tiểu đường thật đã mờ hoá** — giữ tính thực tế thay vì mô phỏng lý tưởng hoá.

| Tham số | Giá trị | Lý do |
|---|---|---|
| $n$ (kích thước tập con) | $\{10, 14, 18\}$ | $\binom{18}{7}\approx 31.824$ vẫn khả thi tính brute-force |
| $m/n$ (tỉ lệ ngân sách) | $\{0.2, 0.4\}$ | Đại diện cho ngân sách nhỏ và trung bình |
| Số lần lặp mỗi cấu hình | 20 (tập con ngẫu nhiên độc lập) | Đủ để ước lượng khoảng tin cậy |
| Tổng số lần chạy | $3 \times 2 \times 20 = 120$ | |

### Tiêu chí đạt/không đạt
Cận $(1-1/e)\approx 0{,}632$ là **chặn dưới đảm bảo tuyệt đối** theo Định lý Nemhauser — không phải trung bình. Tiêu chí: **100% trong 120 lần chạy** phải có $\mathrm{Rep}(S_G)/\mathrm{Rep}(S^\ast) \ge 0{,}632$. Bất kỳ vi phạm nào (dù chỉ 1 lần) là dấu hiệu lỗi cài đặt cần rà soát lại, không phải phản bác định lý.

### Phân tích bổ sung
- Vẽ phân phối tỉ lệ $\rho=\mathrm{Rep}(S_G)/\mathrm{Rep}(S^\ast)$ toàn bộ 120 lần chạy → đánh giá cận có **chặt (tight)** hay **lỏng (loose)** trong thực tế.
- Vẽ đường cong $\Delta(r_t)=\mathrm{Rep}(S_t)-\mathrm{Rep}(S_{t-1})$ theo từng bước Greedy tại 1 cấu hình đại diện → minh hoạ trực quan tính chất lợi ích biên giảm dần (submodularity).

---

## 4. E2 — GreedyFKGS vs Random Sampling (cùng ngân sách)

### Thiết kế ghép cặp (paired design)
```
Với mỗi θ ∈ {0.10, 0.30, 0.50}:
    m = ⌈θ · 1500⌉
    S_greedy = GreedyFKGS(Sim, m)        # tất định — chạy 1 lần
    Với seed = 1..30:
        S_random(seed) = RandomSampling(1500, m, seed)
        Đo Rep(S_random), AUC(S_random) trên test set cố định
    Đo Rep(S_greedy), AUC(S_greedy) — dùng lại cho mọi phép so với 30 seed Random
```

**Lưu ý quan trọng:** GreedyFKGS gần như **tất định** với $\mathrm{Sim}$ liên tục (xác suất tie-break ≈ 0) — không cần lặp 30 lần cho Greedy, chỉ Random mới cần (đây là điểm đã sửa trong quá trình cài đặt, tiết kiệm ~29/30 thời gian tính toán không cần thiết).

### Kiểm định thống kê
- **Wilcoxon signed-rank test** (một phía, $H_1$: Greedy > Random) trên $\mathrm{Rep}$ và trên AUC, riêng cho từng $\theta$.
- **Hiệu chỉnh Holm-Bonferroni** cho 3 kiểm định (3 mức $\theta$) trên cùng một chỉ số.
- **Cohen's d** (effect size, ghép cặp) — quan trọng hơn p-value vì $n=30$ đủ lớn để p-value luôn nhỏ dù hiệu ứng có thể nhỏ.

### Kịch bản diễn giải
- Nếu $\Delta$ (Greedy − Random) **thu hẹp dần** khi $\theta$ tăng: đây là hiện tượng **hợp lý và cần báo cáo trung thực** — khi ngân sách đủ lớn, ngay cả lấy mẫu ngẫu nhiên cũng phủ tốt, nên lợi thế của chọn lọc thông minh giảm dần. Không che giấu hiện tượng này.
- Nếu ở một $\theta$ cụ thể mà $p > 0{,}05$ (Holm-adjusted): báo cáo tường minh, không chỉ chọn các $\theta$ có kết quả đẹp.

---

## 5. E3 — Đường cong đánh đổi theo $\theta$

### Lưới quét
$$\theta \in \{0{,}05;\ 0{,}10;\ 0{,}15;\ 0{,}20;\ 0{,}30;\ 0{,}40;\ 0{,}50;\ 0{,}60;\ 0{,}70;\ 0{,}80;\ 0{,}90;\ 1{,}00\}$$
(dày hơn ở vùng $\theta$ nhỏ vì đây là vùng quan tâm thực tế nhất khi nén mạnh).

### Ba biểu đồ bắt buộc
1. **Đường cong kép** (dual-axis): AUC tương đối (%) so với $\theta=1{,}0$ và thời gian lấy mẫu, cùng trục hoành $\theta$.
2. **$\mathrm{Rep}(S)$ theo $\theta$**, có vạch ngang tại ngưỡng $\delta=0{,}85$ để xác định $\theta$ tối thiểu đạt điều kiện phủ (Định nghĩa 2.12).
3. **Pareto front**: trục X = thời gian lấy mẫu, trục Y = AUC tương đối, mỗi điểm là một $\theta$.

### Xác định điểm gối (knee point)
Dùng xấp xỉ Kneedle (khoảng cách vuông góc lớn nhất tới đường nối điểm đầu–cuối trên đồ thị đã chuẩn hoá $[0,1]^2$) — tránh chọn bằng mắt (chủ quan, khó tái lập).

### Mốc tham chiếu (baseline 100%)
$\mathrm{AUC}(\theta=1{,}0)$ = huấn luyện trên **toàn bộ** tập $N_{EXP}=1.500$ (không lấy mẫu) — đây là "trần" lý thuyết mà không phương pháp nén nào có thể vượt qua (chỉ có thể tiệm cận).

---

## 6. E4 — So sánh 5 chiến lược lấy mẫu (2.3, 2.5, 2.6, 2.7, 2.8) + GreedyFKGS

### ⚠️ Hạn chế cần nêu rõ khi báo cáo
Thiết kế đầy đủ (lý tưởng) cần **nhiều bộ dữ liệu độc lập** làm "block" cho kiểm định Friedman (đúng tinh thần thiết kế gốc dùng 8 bộ dữ liệu — 3 tiểu đường + 5 bệnh tim). Ở đây **chỉ có một nguồn dữ liệu tiểu đường**, nên "block" được thay bằng **10 lần lấy mẫu con độc lập** (kích thước 500 mỗi block) từ pool dữ liệu — đây là một **thỏa hiệp thực nghiệm hợp lý nhưng có giới hạn**: nó đo được độ ổn định của xếp hạng qua các mẫu con khác nhau của **cùng một phân phối dữ liệu**, KHÔNG đo được khả năng tổng quát hoá qua các lĩnh vực/loại luật khác nhau. Kết luận rút ra từ E4 phiên bản này chỉ nên phát biểu ở mức: *"trên bộ dữ liệu tiểu đường đã dùng"*, không suy rộng "trên mọi loại dữ liệu".

### Cách hiện thực hoá 5 chiến lược trên dữ liệu bảng (không có đồ thị/thực thể tường minh)

| Thuật toán | Cách thích nghi cho dữ liệu bảng |
|---|---|
| 2.3 (gốc) | Rút gọn thành Random Sampling thuần (không lọc cục bộ) — đại diện cho baseline "ngẫu nhiên có kiểm tra tối thiểu" |
| 2.5 (Stratified) | Phân tầng theo `Outcome`, Greedy độc lập trong từng tầng, phân bổ ngân sách theo tỉ lệ |
| 2.6 (Systematic, đã sửa) | Sắp theo `order_by = Glucose` (sort_key), cửa sổ $2k=10$ lân cận theo thứ tự, dùng bản `Candidates`-based while đã sửa lỗi điều kiện dừng |
| 2.7 (Cluster) | Gom cụm bằng KMeans ($M_c=10$) trên vector độ thuộc (thay cho connected_components/louvain — không có đồ thị luật tường minh trên dữ liệu bảng), Greedy trong từng cụm theo PPS |
| 2.8 (Snowball) | Xây đồ thị k-NN trên $\mathrm{Sim}$ làm đồ thị "giới thiệu" thay thế (không có thực thể chung tường minh), BFS lan truyền với `branching_cap=8`, `w_max=20`, có vòng lặp tái gieo đã sửa |

**Đây là các giả định thay thế cần nêu rõ trong luận án** — nếu dữ liệu gốc (luật mờ có cấu trúc quan hệ/đồ thị tường minh) khác với dữ liệu bảng phẳng này, hành vi thực tế của 2.7/2.8 có thể khác đáng kể.

### Kiểm định
1. **Friedman test** trên ma trận $\mathrm{Rep}$ (10 block × 6 phương pháp, gồm cả GreedyFKGS làm mốc).
2. Nếu $p<0{,}05$: **Nemenyi post-hoc** + Critical Difference (CD) để xác định các cặp khác biệt có ý nghĩa.
3. Vẽ **biểu đồ xếp hạng kép**: hạng theo $\mathrm{Rep}$ và hạng theo thời gian — minh hoạ trade-off (không có phương pháp nào thắng tuyệt đối cả hai tiêu chí).

---

## 7. Quy trình chạy tổng thể

```
run_00_preprocess.py          # BẮT BUỘC chạy đầu tiên
        │
        ├──► run_01_E1_theory_validation.py
        ├──► run_02_E2_greedy_vs_random.py
        ├──► run_03_E3_theta_curve.py
        ├──► run_04_E4_five_strategies.py
        │        (4 nhánh trên độc lập nhau, chạy song song được)
        │
        ▼
run_05_generate_report.py     # BẮT BUỘC chạy sau cùng (đọc kết quả E1-E4)
```

Toàn bộ có thể chạy bằng `python run_all.py` (script điều phối, tự dừng nếu một bước lỗi).

---

## 8. Checklist kiểm soát chất lượng trước khi đưa vào luận án

- [ ] Đã kiểm tra 0 vi phạm cận $(1-1/e)$ trong toàn bộ lần chạy E1 (nếu có vi phạm → rà lại cài đặt `greedy_fkgs`, không phải phản bác định lý)
- [ ] Đã báo cáo p-value **sau hiệu chỉnh Holm-Bonferroni**, không dùng p-value thô khi so sánh nhiều $\theta$/nhiều thuật toán
- [ ] Đã nêu rõ giới hạn "10 block từ 1 nguồn dữ liệu" ở E4, không diễn giải như tổng quát hoá đa lĩnh vực
- [ ] Đã nêu rõ các giả định thay thế cho 2.7/2.8 (KMeans, k-NN graph) khi dữ liệu gốc không có cấu trúc đồ thị tường minh
- [ ] Đã dùng cùng tập `N_TEST` cố định cho mọi phương pháp khi đo AUC (không train/test riêng cho từng phương pháp)
- [ ] Đã kiểm tra imputation theo lớp (không theo toàn bộ dữ liệu) để tránh làm nhoè tín hiệu Outcome
