# Bộ thực nghiệm FKG-E (KB1–KB6, Ablation, Baseline)

Bộ code này hiện thực hóa đầy đủ thiết kế thực nghiệm định lượng cho FKG-E
(Mục 3.5.4 luận án), dùng FKG-MM trên BRSET (hoặc FKGS đã lấy mẫu ở
Chương 2) làm đầu vào, so sánh với FISA và ba baseline nhúng đồ thị tri
thức chuẩn (Node2Vec, TransE, DistMult).

## ⚠️ ĐỌC TRƯỚC KHI CHẠY — 3 phát hiện quan trọng trong quá trình xây dựng

Trong lúc viết và **kiểm thử thật** bộ code này (không chỉ viết cho chạy
được mà chủ động dò lỗi bằng dữ liệu tổng hợp có cấu trúc biết trước), đã
phát hiện và sửa 3 vấn đề — bạn cần biết để không hiểu nhầm khi chạy trên
dữ liệu thật của mình:

### 1. Mất cân bằng giữa các thành phần hàm mất mát (đã sửa)
Ban đầu, `L_SGNS` (có hàng trăm cặp token/batch) áp đảo hoàn toàn
`L_pred`/`L_inf` (chỉ vài mẫu/batch) về độ lớn tuyệt đối — dù cả hai có
trọng số λ/β/γ/δ = nhau, `L_SGNS` chiếm tới **99.5% tổng loss**. Hậu quả:
mô hình học representation đồng xuất hiện tốt nhưng **không học được tín
hiệu phân loại**, khiến $p_E$ suy biến về đúng tỉ lệ nhãn trung bình
(luôn đoán lớp đa số) bất kể siêu tham số. **Đã sửa** bằng cách chuẩn hóa
mỗi thành phần loss theo số phần tử của nó (trung bình, không phải tổng)
trong `models/fkge.py`. Sau khi sửa, cần **`delta_pred` lớn hơn đáng kể**
so với `beta_rule`/`lam_node` (giá trị mặc định trong `config.py` đã được
hiệu chỉnh dựa trên thực nghiệm: `delta_pred=20.0` so với `beta_rule=1.0`).

**Việc bạn cần làm trên dữ liệu thật:** mọi lần đánh giá đều tự động in
cảnh báo nếu accuracy không vượt rõ baseline "luôn đoán lớp đa số" (xem
`models/fisa.py::warn_if_not_beating_majority`). Nếu thấy cảnh báo này
trên BRSET thật, **tăng `delta_pred`/`gamma_inf`** trước tiên (thử
`[5, 10, 20, 50]`), sau đó mới tăng `lr`/`epochs`.

### 2. Seed chưa kiểm soát toàn bộ tính ngẫu nhiên (đã sửa)
`np.random.choice` (negative sampling) ban đầu dùng trạng thái ngẫu nhiên
**toàn cục**, không bị chi phối bởi tham số `seed` truyền vào từng model
— khiến việc lặp `n_seeds` để đo phương sai trở nên vô nghĩa (độ lệch
chuẩn luôn ra 0.0000 một cách đáng ngờ). **Đã sửa** bằng cách gọi
`random.seed()`/`np.random.seed()` ở đầu `fit()`.

### 3. FISA gần như phẳng theo |R|, FKG-E mới là đại lượng tăng (KB6)
Giả thuyết "ngây thơ" ban đầu (FISA tăng tuyến tính theo số luật, FKG-E
phẳng) **không đúng với cách cài đặt FISA có tiền tính ma trận trọng số
W một lần** (đúng Mệnh đề 3.2.11 trong Chương 3): sau khi `fit()`, chi phí
suy diễn FISA mỗi truy vấn gần như không đổi theo `|R|`; ngược lại, FKG-E
phải so khớp với toàn bộ `|R|` luật mỗi truy vấn nên **chi phí FKG-E mới
là đại lượng tăng theo `|R|`**. Kết quả KB6 cần được đọc theo đúng chiều
này — đây không phải lỗi.

---

## 1. Cài đặt

```bash
pip install numpy scikit-learn matplotlib --break-system-packages
```

Không cần PyTorch/TensorFlow — toàn bộ FKG-E được cài bằng NumPy thuần,
gradient viết tay và **đã kiểm chứng bằng gradient checking** (so khớp
đạo hàm giải tích với sai phân hữu hạn, xem `models/fkge.py::_gradient_check`).

## 2. Cấu trúc thư mục

```
fkge_v2/
├── config.py                    # TOÀN BỘ tham số — sửa đường dẫn dữ liệu ở đây
├── data/
│   └── fkg_io.py                 # Định dạng dữ liệu chuẩn + sinh dữ liệu tổng hợp
├── models/
│   ├── fisa.py                   # FISA theo FKGS v3.2 (FKG-Pairs, k = 3): FISA (bảng tra), FISASequential (FISA gốc)
│   ├── fisa_old_v2.py            # bản FISA cũ (W = support·confidence) — chỉ để đối chiếu, không dùng
│   ├── fkge.py                   # FKG-E: SGNS + Node/Inf/Pred Loss (đã gradient-check)
│   └── kge_baselines.py          # Node2Vec-lite, TransE-lite, DistMult-lite + kNN
├── experiments/
│   ├── kb1_kb2.py                 # KB1: FKG-E vs FISA trên FKG gốc (3 bộ dữ liệu)
│   │                               # KB2: FKG-E vs FISA trên FKGS đã nén
│   ├── kb3_kb5.py                 # KB3: quét chiều nhúng d
│   │                               # KB5: quét (cửa sổ w, số mẫu âm K)
│   ├── kb4_kb6.py                 # KB4: quét lưới (λ, β) -> heatmap
│   │                               # KB6: đường cong khả năng mở rộng theo |R|
│   ├── ablation.py                 # Ablation: Rule-only / Node-only / Uniform / Full
│   └── baseline_comparison.py      # So sánh FISA/FKG-E/Node2Vec/TransE/DistMult
├── report/
│   └── generate_report.py          # Sinh bảng CSV/Markdown + biểu đồ PNG
├── run_all.py                       # Script tổng — chạy toàn bộ pipeline
└── outputs/                          # Kết quả (JSON, CSV, PNG, report_tong_hop.md)
```

## 3. Chuẩn bị dữ liệu đầu vào (BẮT BUỘC trước khi có kết quả thật)

> **BRSET đã chia 5 fold (`data_real/BRSET_Data/fold_XX/*.csv`) được dùng TỰ ĐỘNG cho mọi kịch bản**
> (KB1–KB6, Ablation, Baseline) — không cần chuyển sang JSON. Xem mục
> [Chạy toàn bộ KB trên BRSET thật](#chạy-toàn-bộ-kb1kb6-ablation-baseline-trên-brset-thật-v33).
> Các mục 3.1–3.3 dưới đây chỉ cần cho dữ liệu dạng JSON (ví dụ hai bộ Diabetes của KB1).

Nếu bạn CHƯA chuẩn bị dữ liệu thật, **mọi script vẫn chạy được** — chúng
tự động dùng dữ liệu tổng hợp (synthetic) và **in cảnh báo rõ ràng** mỗi
lần dùng dữ liệu giả, để bạn không nhầm kết quả demo với kết quả thật.

### 3.1. Định dạng file luật (rules)

Tạo thư mục `data_real/` và đặt các file JSON theo định dạng sau (xem chi
tiết trong docstring của `data/fkg_io.py`):

```json
{
  "meta": {"dataset": "BRSET", "n_classes": 2, "class_names": ["No-DR", "DR"]},
  "vocab": ["Age-Low", "Age-Medium", "Age-High", "HbA1c-High", "class-DR", "class-No-DR"],
  "rules": [
    {
      "id": 0,
      "antecedent_tokens": ["Age-High", "HbA1c-High"],
      "consequent_token": "class-DR",
      "confidence": 0.87,
      "support": 0.12
    }
  ],
  "edges": [
    {"u": "Age-High", "v": "HbA1c-High", "mu": 0.63}
  ]
}
```

`rules` xuất từ chính pipeline khai phá luật FKG-MM/FKGS bạn đã xây ở
Chương 2–3 (đầu ra Wang–Mendel/M-CFIS). `edges` xuất từ ma trận quan hệ
mờ `A`/`ω_uv` (Mục 3.3.4). Nếu bạn chưa có bước xuất JSON, viết một script
nhỏ chuyển đổi từ cấu trúc luật nội bộ hiện có sang đúng định dạng trên —
đây là điểm tích hợp DUY NHẤT cần làm để dùng dữ liệu thật.

### 3.2. Định dạng file test (mẫu đã mờ hoá, có nhãn)

```json
{
  "samples": [
    {"membership": {"Age-High": 0.8, "HbA1c-High": 0.9}, "label": "class-DR"}
  ]
}
```

### 3.3. Đặt đúng đường dẫn trong `config.py`

Sửa 6 đường dẫn trong `class PATHS` — trỏ tới các file bạn vừa tạo:
`BRSET_RULES_FILE`, `DIABETES_KAGGLE_RULES_FILE`,
`HEALTHCARE_DIABETES_RULES_FILE`, `BRSET_FKGS_RULES_FILE` (luật đã lấy
mẫu bởi FKGS — Chương 2), và 3 file `*_TEST_FILE` tương ứng.

Với KB6 (cần nhiều mức nén 20/40/60/80/100%), nếu bạn có sẵn 5 file FKGS
thật ở các tỉ lệ khác nhau, đặt tên theo quy ước
`brset_fkgs_rules_20.json`, `..._40.json`, ... — script sẽ tự nhận diện.
Nếu không có, script tự lấy mẫu ngẫu nhiên từ FKG-MM đầy đủ để mô phỏng
(kém chính xác hơn dùng FKGS thật, nhưng đủ để có đường cong tham khảo).

## 4. Cách chạy

### Bước 1 — Kiểm tra pipeline chạy đúng (bắt buộc, ~5-10 phút)

```bash
python3 run_all.py --quick
```

Chạy với T_ep = 15 và lưới quét nhỏ (trên BRSET thật: 2 fold đầu, khoảng 3 phút), dùng dữ liệu tổng hợp nếu
chưa có dữ liệu thật. Mục đích DUY NHẤT là xác nhận không có lỗi runtime — **không dùng
kết quả ở bước này để báo cáo**.

### Bước 2 — Chạy từng model riêng để kiểm tra (tuỳ chọn)

```bash
python3 models/fisa.py              # Kiểm tra FISA (bảng tra và tuần tự phải trùng nhau)
FKGS_V32_DIR=/đường/dẫn/fkgs_v3.2 python3 -m pytest -q tests/   # đối chiếu với src/fisa_pairs.py của FKGS v3.2
python3 models/fkge.py              # Gradient checking + huấn luyện thử FKG-E
python3 models/kge_baselines.py     # Kiểm tra Node2Vec/TransE/DistMult
```

### Bước 3 — Chạy đầy đủ trên dữ liệu thật

```bash
python3 run_all.py
```

Chạy toàn bộ KB1→KB6→Ablation→Baseline rồi tự động sinh báo cáo. Trên BRSET thật (5 fold × khoảng 1.778 luật,
T_ep = 200, 1 seed mỗi fold, lưới mặc định) cần khoảng 215 lần huấn luyện FKG-E — **ước tính khoảng 2 giờ** với 4
tiến trình (KB4 chiếm hơn một nửa). Có thể chạy từng phần:

```bash
python3 run_all.py --only kb1,kb2      # chỉ chạy KB1, KB2
python3 run_all.py --only kb6          # chỉ chạy KB6
python3 run_all.py --jobs 4 --seeds 1 --folds fold_01,fold_02,fold_03,fold_04,fold_05 --out outputs
```

### Bước 4 — Sinh lại báo cáo (nếu đã có sẵn outputs/*.json)

```bash
python3 report/generate_report.py                 # hoặc --out <thư mục kết quả>
```

## 5. Đọc kết quả

Sau khi chạy xong, thư mục `outputs/` chứa:
- `*.json` — kết quả thô từng KB
- `table_*.csv` — bảng dạng CSV (mở bằng Excel)
- `figures/*.png` — biểu đồ so sánh
- **`report_tong_hop.md`** — báo cáo tổng hợp đầy đủ bảng + biểu đồ, dùng
  trực tiếp để trích vào Mục 3.5.4 của luận án

## 6. Các điểm cần LUÔN kiểm tra trước khi tin dùng một con số

1. **So với baseline lớp đa số**: mọi accuracy in ra kèm cảnh báo tự động
   nếu không vượt rõ baseline này. Đừng bỏ qua cảnh báo.
2. **So giữa FISA và FKG-E trên đúng cùng fold, cùng seed dữ liệu**: script
   đã đảm bảo điều này, nhưng nếu bạn sửa code, giữ nguyên tính chất này.
3. **Std giữa các seed phải khác 0** (trừ trường hợp đặc biệt): nếu thấy
   `std=0.0000` một cách hệ thống trên dữ liệu thật, nghi ngờ ngay có vấn
   đề tương tự phát hiện #2 ở trên.
4. **KB6**: đọc đúng chiều — FKG-E là đường tăng, FISA là đường phẳng, đây
   là hành vi ĐÚNG chứ không phải bug (xem phát hiện #3 ở trên).


## Suy diễn FISA (cập nhật theo FKGS v3.2)

`models/fisa.py` cài đặt FISA đúng như `src/fisa_pairs.py` của FKGS v3.2 và Chương 2–3 luận án:

- **Mô hình FKG-Pairs, nút gồm k = 3 thuộc tính** (`config.FISA.NODE_SIZE`): A^t trên bộ 4 thuộc tính (7),
  B^t_c = (ΣA^t)·min M^t_i trên nút (8), bảng W_{c,v,l} = Σ B^t_c chỉ trên các luật **có nhãn l** (3.Wfisa),
  suy diễn mờ W̃_{c,l}(x) = Σ_v Π μ_{v_j}(x_j) W_{c,v,l}, D_l = max_c + min_c. Với độ thuộc rời rạc, kết quả
  trùng FKGS v3.2 (kiểm thử `tests/test_fisa_v32.py`).
- **Quy tắc quyết định** (`config.FISA.DECISION`): `calibrated` (mặc định, ngưỡng trên s = log D1 − log D0
  chọn trên tập huấn luyện của FKG-E, không dùng tập kiểm tra), `argmax` (công thức (12) của FKG-Pairs),
  `ratio9` (quy tắc D0 > 9·D1 của notebook gốc). `evaluate()` báo cáo cả ba, cùng độ chính xác cân bằng và AUC.
- **Phân phối giáo viên p_F^cal** (`predict_proba_one`) = softmax((log(D_l+ε0) + β_l)/T), β theo ngưỡng hiệu
  chỉnh, nên argmax của p_F^cal trùng quy tắc quyết định.
- **Đóng góp luật a_F** (`attribution`): phân rã chính xác D của nhãn dự đoán qua hai nút đạt min/max
  (Định nghĩa 3.attrF), dùng cho độ lệch căn cứ luật (R3).
- **Hai cài đặt cùng hàm quyết định:** `FISA` (bảng tra, chi phí mỗi truy vấn gần như không đổi theo |R|) và
  `FISASequential` (FISA gốc duyệt tuần tự, O(|R|·C(r,3))). KB1 và KB6 báo cáo thời gian của cả hai; tốc độ của
  FKG-E được so với cả hai (ràng buộc (R4) dùng bản tuần tự).
- **Yêu cầu dữ liệu:** mỗi luật phải có một nhãn ngôn ngữ cho **mọi** thuộc tính (luật sinh từ mẫu như FKG-MM/FKGS);
  luật thiếu thuộc tính chỉ tham gia các nút mà nó có đủ thuộc tính (có cảnh báo). Bộ sinh dữ liệu tổng hợp đã
  được sửa để sinh luật đầy đủ thuộc tính.

## FKG-E theo Chương 3 (bản v3)

`models/fkge.py` cài đặt đúng Chương 3 (Mục 3.2.5, Thuật toán 3.10–3.11); gradient bằng `autograd`
(`pip install -r requirements.txt`).

- **Hàm mục tiêu (3.28):** λ_S L_SGNS + λ_E L_edge + λ_N(λ_A L_A + λ_B L_B^comp) + λ_R L_rule + λ_I L_inf
  + λ_P L_pred + λ_C‖Θ‖², với các thành phần đã chuẩn hoá thang đo: L_SGNS chia |C| và dùng phân phối mẫu âm
  P_n = c_j/|C|; đồng xuất hiện **toàn luật** (3.cooc); ω đối xứng hoá; đích b̃ làm trơn (ε_b > 0); p_E dùng
  log-sum-exp theo lớp (τ_c); L_rule = KL(a_F ‖ w_{ŷ_F}) với a_F là căn cứ luật của FISA (FKGS v3.2);
  p_F^cal = softmax((log(D+ε0)+β)/T), β theo ngưỡng của FISA, T chọn trên tập xác thực.
- **Tham số** (`config.FKGE`): d_e, K, α, τ, τ_c, ε, ε0, ε_b, λ_S..λ_C (mặc định mọi λ = 1, λ_C = 10⁻³ > 0),
  η, optimizer (AMSGrad hoặc SGD, đều chiếu lên hình cầu B₀), ε_g, b, b_p, T_ep, patience. Tên cũ của bản v2
  (d, K_neg, lam_node, beta_rule, gamma_inf, delta_pred, lr, epochs) vẫn được chấp nhận.
- **Thuật toán 3.10:** giảm gradient toàn bộ có chiếu với tìm kiếm bước (mặc định `gd_full`); tách 20% tập huấn luyện làm tập xác thực; (biến thể theo lô:
  (|E⁺|/b_p, |P_A|/b_p, L_B đầy đủ); chiếu lên hình cầu bán kính B₀ = √(L(Θ₀)/λ_C); dừng khi ‖∇L‖ ≤ ε_g hoặc
  dừng sớm theo L_pred; chọn ngưỡng τ_E trên tập xác thực.
- **Thuật toán 3.11:** `predict_one` (nhúng truy vấn → s_k → s̄_l → p_E → ngưỡng τ_E), `explain_one` (a_E và
  K_act luật). `evaluate` báo cáo thêm độ đồng thuận với FISA (R2) và độ lệch căn cứ luật Dev (R3).
- **Kịch bản:** KB4 quét (λ_I, λ_P) — tức ρ = λ_I/λ_P của Định lý đánh đổi; KB5 so sánh đồng xuất hiện toàn
  luật với cửa sổ w = 2; Ablation bỏ lần lượt L_pred, L_inf, L_rule, nhóm cấu trúc, gộp đều, cửa sổ.
- **Kiểm thử:** `python -m pytest -q tests/` — 17 test (13 cho FKG-E, 4 cho FISA).

## Chạy trên BRSET (FRB 5 fold theo ID bệnh nhân)

Đặt dữ liệu tại `data_real/BRSET_Data/fold_01 … fold_05/{TrainDataRule.csv, TestDataRule.csv}`
(16 cột mức ngôn ngữ 1–5 và cột nhãn 1/2; mỗi dòng TrainDataRule là một luật FRB). Sau đó:

    python -m pytest -q tests/                 # 33 test (1 bỏ qua nếu chưa đặt FKGS_V32_DIR)
    python experiments/brset_kfold.py          # 5 fold, khoảng 1–2 phút
    python experiments/brset_kfold.py --quick  # kiểm tra nhanh

Kết quả: `outputs/brset_kfold.json`, `outputs/brset_kfold.md`. Các điều chỉnh để chạy được với dữ liệu này:
- `data/brset_folds.py`: bộ nạp CSV → FKGRuleBase (token `F{j}-L{v}`, nhãn `class-1/2`), mẫu one-hot (Crisp);
  gom nhóm các dòng trùng (tập huấn luyện đã cân bằng lớp bằng lặp mẫu, khoảng 650–790 dòng trùng mỗi fold).
- Chống rò rỉ khi luật sinh từ chính mẫu huấn luyện: (i) giáo viên FISA tính chéo theo 5 phần chia theo nhóm
  (Định nghĩa 3.18); (ii) FKG-E loại luật cùng nhóm với mẫu khi tính mất mát và khi chọn ngưỡng/dừng sớm
  (`FKGE._mask`); (iii) tập xác thực FKG-E và các phần tính chéo đều chia theo nhóm.
- `FKGE.fit(..., teacher_data=...)`: nhận giáo viên tính sẵn (D ngoài phần, a_F, ŷ_F, β).
- `FISA.attribution` được vector hoá (nhanh hơn khoảng 100 lần trên 1.778 luật × 16 thuộc tính).

## Chạy toàn bộ KB1–KB6, Ablation, Baseline trên BRSET thật (v3.3)

Khi có `data_real/BRSET_Data/fold_XX`, `run_all.py` và từng script trong `experiments/` tự chạy trên BRSET thật
(`experiments/common.py`); thứ tự ưu tiên nguồn dữ liệu: BRSET 5 fold > JSON trong `config.PATHS` > dữ liệu tổng hợp.

    python3 run_all.py --quick      # kiểm tra nhanh trên 2 fold đầu (~3 phút)
    python3 run_all.py              # đầy đủ 5 fold (~2 giờ với 4 tiến trình)

- **Cùng quy trình chống rò rỉ với `brset_kfold.py`** (`prepare_fold`): mỗi fold chỉ chuẩn bị một lần — giáo viên FISA
  tính chéo theo nhóm, ngưỡng τ_F trên điểm ngoài phần, tập xác thực FKG-E 20% nhóm, FKG-E che luật cùng nhóm.
  Mọi kịch bản và mọi phương pháp dùng đúng các fold đó; tập kiểm tra chỉ dùng để đánh giá cuối cùng.
- **Tổng hợp:** mỗi cấu hình chạy `config.EVAL.REAL_SEEDS_PER_FOLD` seed (mặc định 1) trên mỗi fold; bảng báo cáo
  trung bình ± độ lệch chuẩn **giữa các fold**. JSON KB1 có thêm `per_fold`.
- **Độ đo:** tập kiểm tra BRSET chỉ khoảng 8% dương (accuracy của "luôn đoán âm" ≈ 0,92), nên mọi bảng có thêm
  BalAcc và AUC; biểu đồ/heatmap dùng `config.EVAL.MAIN_METRIC` (mặc định độ chính xác cân bằng). Cảnh báo "sụp về
  một lớp" dựa trên độ chính xác cân bằng thay vì so accuracy với lớp đa số.
- **KB1:** chỉ BRSET; hai bộ Diabetes chạy khi có file JSON tương ứng, nếu thiếu thì **bỏ qua** (không thay bằng dữ
  liệu tổng hợp). **KB2/KB6:** chưa có luật FKGS thật cho từng fold nên FKGS được **mô phỏng** bằng cách giữ ngẫu nhiên
  30% (KB2) hoặc 20–100% (KB6) số luật của mỗi fold, mẫu huấn luyện giữ nguyên; giáo viên FISA tính lại trên cơ sở
  luật con. Báo cáo ghi rõ "FKGS mô phỏng". **KB5:** mô hình theo Chương 3 không có cửa sổ w, chỉ quét K.
- **Baseline:** FRB BRSET không kèm tệp cạnh nên Node2Vec tự dựng đồ thị đồng xuất hiện tiền đề từ luật; kNN báo
  cáo thêm BalAcc, AUC (điểm = tỉ lệ phiếu lớp dương).
- **Tốc độ:** huấn luyện song song `config.EVAL.N_JOBS` tiến trình (`--jobs`; thông lượng bão hoà từ khoảng 4 tiến
  trình do giới hạn băng thông bộ nhớ); đánh giá và đo thời gian suy diễn chạy tuần tự trong tiến trình chính để phép
  đo không bị nhiễu bởi tải song song. Cấu hình đã huấn luyện được dùng lại giữa các kịch bản (cấu hình mặc định xuất
  hiện ở KB1, KB2, KB3, KB4, KB5, KB6, Ablation, Baseline). `FKGE.fit` (gd_full) dùng `value_and_grad` và tái sử dụng
  gradient cuối vòng t cho đầu vòng t + 1: nhanh khoảng 1,9 lần, kết quả trùng từng bit với bản trước.
- **Tuỳ chọn `run_all.py`:** `--jobs N`, `--folds fold_01,fold_02`, `--seeds N` (số seed mỗi fold), `--out THƯ_MỤC`
  (hoặc biến môi trường `FKGE_OUTPUT_DIR`). `outputs/run_meta.json` ghi nguồn dữ liệu và cấu hình của lần chạy.
