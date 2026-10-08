"""
config.py — Cấu hình trung tâm cho toàn bộ bộ thực nghiệm FKG-E (KB1-KB6,
Ablation, Baseline), khớp đúng ký hiệu và công thức trong Chương 3 luận án
(Mục 3.4 Mô hình FKG-E, Mục 3.5.4 Kịch bản đánh giá FKG-E).

Nguồn dữ liệu (experiments/common.py chọn theo thứ tự ưu tiên):
  1. PATHS.BRSET_FOLDS_DIR    -> BRSET thật, FRB đã chia 5 fold theo ID bệnh nhân
                                  (fold_XX/TrainDataRule.csv, TestDataRule.csv; xem data/brset_folds.py).
                                  Có thư mục này thì MỌI kịch bản KB1–KB6, Ablation, Baseline chạy trên nó.
  2. PATHS.BRSET_RULES_FILE + BRSET_TEST_FILE (JSON, định dạng trong data/fkg_io.py) -> một lần chia 60/40.
  3. Không có cả hai -> dữ liệu tổng hợp (chỉ để kiểm tra pipeline, có cảnh báo).
Hai bộ Diabetes của KB1 chỉ chạy khi có file JSON tương ứng (PATHS.DIABETES_*_FILE); khi đang chạy BRSET thật,
bộ nào thiếu file sẽ bị bỏ qua chứ không thay bằng dữ liệu tổng hợp.
"""
import os

# ============================================================
# ĐƯỜNG DẪN DỮ LIỆU — SỬA THEO MÔI TRƯỜNG THẬT CỦA BẠN
# ============================================================
class PATHS:
    ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT, "data_real")   # nơi đặt file luật thật
    OUTPUT_DIR = os.environ.get("FKGE_OUTPUT_DIR", os.path.join(ROOT, "outputs"))

    # --- BRSET thật: FRB 5 fold theo ID bệnh nhân (ưu tiên số 1 cho mọi kịch bản) ---
    BRSET_FOLDS_DIR = os.path.join(DATA_DIR, "BRSET_Data")

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
    lam_E = 1.0         # bảo toàn cạnh: SGNS (word2vec) trên trọng số cạnh A
    lam_N = 1.0         # bảo toàn nút: phân bố nhãn của mỗi nút giá trị (hàng của B)
    lam_R = 1.0; lam_I = 1.0; lam_P = 1.0
    lam_C = 1e-3        # bắt buộc > 0 (tính bức, Mệnh đề 3.coer)
    eta = 0.01          # tốc độ học
    optimizer = "gd_full"   # "gd_full": giảm gradient toàn bộ có chiếu (Thuật toán 3.10, khớp Định lý hội tụ)
                            # | "amsgrad", "sgd": biến thể theo lô (chỉ để so sánh)
    eta_gd = 1.0        # bước khởi đầu của tìm kiếm theo đường (quy tắc giảm đủ) cho gd_full
    eps_g = 1e-4        # dừng khi ‖∇L‖ <= eps_g (Hệ quả 3.stop)
    b = 64              # kích thước lô mẫu
    b_p = 256           # kích thước lô cặp token
    T_ep = 200          # số vòng lặp tối đa (mỗi vòng một bước gradient toàn bộ với gd_full)
    patience = 20       # dừng sớm theo L_pred trên tập xác thực
    sigma = 0.1         # độ lệch chuẩn khởi tạo
    val_fraction = 0.2  # tỉ lệ tập xác thực tách từ tập huấn luyện
    seed = 42
    # Tên cũ (bản v2) — giữ để các kịch bản cũ chạy được; ánh xạ sang tên Chương 3 trong models/fkge.py
    d = d_e; K_neg = K; lam_node = lam_N; beta_rule = lam_E; gamma_inf = lam_I; delta_pred = lam_P
    w = None            # không còn dùng (giữ để kịch bản cũ chạy được)
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
KB5_W_GRID = [None]             # mô hình theo Chương 3 không dùng cửa sổ; KB5 chỉ quét số mẫu âm K
KB5_K_GRID = [2, 5, 10]

# ============================================================
# KB6: Khả năng mở rộng quy mô — tỉ lệ mẫu từ FKGS
# ============================================================
KB6_SAMPLE_RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0]

# ============================================================
# Đánh giá / kiểm định thống kê
# ============================================================
class EVAL:
    N_SEEDS = 5               # số seed lặp lại mỗi cấu hình khi chỉ có MỘT lần chia (dữ liệu tổng hợp/JSON)
    K_FOLD = 5                # k-fold cross-validation
    CONFIDENCE = 0.95
    # --- BRSET thật (5 fold) ---
    REAL_FOLDS = None         # None = mọi fold_* trong PATHS.BRSET_FOLDS_DIR; hoặc ví dụ ["fold_01", "fold_02"]
    REAL_SEEDS_PER_FOLD = 1   # số seed FKG-E mỗi fold; độ lệch chuẩn báo cáo là độ lệch chuẩn GIỮA CÁC FOLD
    F_C = 5                   # số phần tính chéo giáo viên FISA theo nhóm (Định nghĩa 3.18)
    N_SEQ_TIMING = 50         # số truy vấn đo thời gian FISA tuần tự (chậm, O(|R|·C(r,3)) mỗi truy vấn)
    N_JOBS = max(1, min(4, (os.cpu_count() or 2) // 2))   # số tiến trình huấn luyện FKG-E song song (thông lượng
                                                          # bão hoà từ ~4 do giới hạn băng thông bộ nhớ)
    MAIN_METRIC = "balanced_accuracy"   # độ đo vẽ biểu đồ/heatmap: "balanced_accuracy" | "auc" | "accuracy"
                                        # (tập kiểm tra BRSET chỉ ~8% dương: accuracy của "luôn đoán âm" ≈ 0,92)
