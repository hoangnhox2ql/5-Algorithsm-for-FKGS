"""
config.py — Cấu hình trung tâm cho toàn bộ bộ thực nghiệm FKG-E (KB1-KB6,
Ablation, Baseline), khớp đúng ký hiệu và công thức trong Chương 3 luận án
(Mục 3.4 Mô hình FKG-E, Mục 3.5.4 Kịch bản đánh giá FKG-E).

CHỈNH SỬA DUY NHẤT BẠN CẦN LÀM trước khi chạy trên dữ liệu thật:
  1. PATHS.FKG_MM_RULES_FILE  -> trỏ tới file luật đã khai phá từ FKG-MM (BRSET)
  2. PATHS.FKGS_RULES_FILE    -> trỏ tới file luật đã lấy mẫu bởi FKGS (Chương 2)
  3. PATHS.DIABETES_*_FILE    -> trỏ tới 2 bộ luật Diabetes-Kaggle,
                                  Healthcare-Diabetes-Kaggle (đã khai phá luật)
Định dạng file luật: xem docstring trong data/fkg_io.py.
"""
import os

# ============================================================
# ĐƯỜNG DẪN DỮ LIỆU — SỬA THEO MÔI TRƯỜNG THẬT CỦA BẠN
# ============================================================
class PATHS:
    ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT, "data_real")   # nơi đặt file luật thật
    OUTPUT_DIR = os.path.join(ROOT, "outputs")

    # --- Input bắt buộc cho KB1 (3 bộ dữ liệu) ---
    BRSET_RULES_FILE = os.path.join(DATA_DIR, "brset_fkgmm_rules.json")
    DIABETES_KAGGLE_RULES_FILE = os.path.join(DATA_DIR, "diabetes_kaggle_rules.json")
    HEALTHCARE_DIABETES_RULES_FILE = os.path.join(DATA_DIR, "healthcare_diabetes_rules.json")

    # --- Input bắt buộc cho KB2 (luật đã lấy mẫu bởi FKGS, Chương 2) ---
    BRSET_FKGS_RULES_FILE = os.path.join(DATA_DIR, "brset_fkgs_rules.json")

    # --- Dữ liệu đánh giá (mẫu test đã mờ hoá, có nhãn thật) ---
    BRSET_TEST_FILE = os.path.join(DATA_DIR, "brset_test.json")
    DIABETES_KAGGLE_TEST_FILE = os.path.join(DATA_DIR, "diabetes_kaggle_test.json")
    HEALTHCARE_DIABETES_TEST_FILE = os.path.join(DATA_DIR, "healthcare_diabetes_test.json")


# ============================================================
# SUY DIỄN FISA — đúng như FKGS v3.2 (mô hình FKG-Pairs, models/fisa.py)
# ============================================================
class FISA:
    NODE_SIZE = 3              # k: số thuộc tính trong một nút (notebook gốc và FKGS v3.2: 3)
    DECISION = "calibrated"    # "argmax" (công thức (12) FKG-Pairs) | "ratio9" (notebook: D0 > 9*D1)
                               # | "calibrated" (ngưỡng trên s = log D1 - log D0 chọn trên tập xác thực)
    CALIB_CRITERION = "bacc"   # tiêu chí chọn ngưỡng: "bacc" | "acc"
    TEMPERATURE = 1.0          # nhiệt độ T của p_F^cal (Định nghĩa 3.18)


# ============================================================
# THAM SỐ FKG-E MẶC ĐỊNH (khớp Mục 3.4 luận án)
# ============================================================
class FKGE:
    """Tham số FKG-E theo Chương 3 (Mục 3.2.5, Thuật toán 3.10–3.11). Sau khi chuẩn hoá thang đo các thành
    phần (Nhận xét 3.scale), mặc định mọi trọng số λ = 1; sự đánh đổi được khảo sát qua ρ = λ_I/λ_P."""
    d_e = 32            # số chiều nhúng
    K = 5               # số mẫu âm của SGNS
    alpha = 0.7         # trọng số tiền đề trong nhúng luật (3.pool); gộp đều: pooling="mean"
    tau = 0.1           # nhiệt độ của p_E (3.pE)
    tau_c = 0.1         # hệ số làm trơn log-sum-exp theo lớp; điều khiển độ tập trung của a_E (ràng buộc R3)
    eps = 0.05          # làm trơn p_E: p_E >= eps/n_L
    eps0 = 1e-6         # hằng số của cos_ε0 (3.cos)
    eps_b = 0.05        # làm trơn đích b̃ (bắt buộc > 0, Mệnh đề 3.coer)
    lam_S = 1.0; lam_E = 1.0; lam_N = 1.0; lam_A = 1.0; lam_B = 1.0
    lam_R = 1.0; lam_I = 1.0; lam_P = 1.0
    lam_C = 1e-3        # bắt buộc > 0 (tính bức, Mệnh đề 3.coer)
    eta = 0.01          # tốc độ học
    optimizer = "amsgrad"   # "amsgrad" (rồi chiếu) | "sgd" (gradient ngẫu nhiên có chiếu, Định lý 3.sgd)
    eps_g = 1e-4        # dừng khi ‖∇L‖ <= eps_g (Hệ quả 3.stop)
    b = 64              # kích thước lô mẫu
    b_p = 256           # kích thước lô cặp token
    T_ep = 100          # số epoch tối đa
    patience = 10       # dừng sớm theo L_pred trên tập xác thực
    sigma = 0.1         # độ lệch chuẩn khởi tạo
    val_fraction = 0.2  # tỉ lệ tập xác thực tách từ tập huấn luyện
    cooc = "full"       # đồng xuất hiện toàn luật (3.cooc); "window" chỉ dùng để đối chứng (KB5)
    w = 2               # cửa sổ khi cooc="window"
    seed = 42
    # Tên cũ (bản v2) — giữ để các kịch bản cũ chạy được; ánh xạ sang tên Chương 3 trong models/fkge.py
    d = d_e; K_neg = K; lam_node = lam_N; beta_rule = lam_S; gamma_inf = lam_I; delta_pred = lam_P
    lr = eta; epochs = T_ep; alpha_pool = alpha; batch_size = b


# ============================================================
# KB3: Độ nhạy theo số chiều nhúng
# ============================================================
KB3_DIMS = [8, 16, 32, 64, 128]

# ============================================================
# KB4: Độ nhạy theo (lambda, beta) — quét lưới
# ============================================================
KB4_LAMBDA_GRID = [0.1, 0.3, 0.5, 0.7, 1.0]
KB4_BETA_GRID = [0.1, 0.3, 0.5, 0.7, 1.0]

# ============================================================
# KB5: Độ nhạy theo (w, K) skip-gram
# ============================================================
KB5_W_GRID = [None, 2]          # None = đồng xuất hiện toàn luật (3.cooc); số = cửa sổ vị trí (đối chứng)
KB5_K_GRID = [2, 5, 10]

# ============================================================
# KB6: Khả năng mở rộng quy mô — tỉ lệ mẫu từ FKGS
# ============================================================
KB6_SAMPLE_RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0]

# ============================================================
# Đánh giá / kiểm định thống kê
# ============================================================
class EVAL:
    N_SEEDS = 5               # số seed lặp lại mỗi cấu hình (giảm phương sai)
    K_FOLD = 5                # k-fold cross-validation
    CONFIDENCE = 0.95
