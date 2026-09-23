"""
config.py — Cấu hình trung tâm dự án FKGS v2 (Chương 2, thực nghiệm E1-E4
trên bộ dữ liệu Tiểu đường), đúng theo docs/thiet_ke_thuc_nghiem.md và
Dieu_chinh_5_thuat_toan_Submodular_v4.pdf.

THAY ĐỔI CĂN BẢN so với bản mô tả gốc trong thiet_ke_thuc_nghiem.md:
  1. CHỈ 4 thuật toán lấy mẫu (không phải 5): GreedyFKGS, Random, SFKGS,
     CFKGS. Bỏ Sys-FKGS (2.6) và SB-FKGS (2.8) vì phiên bản đã sửa để có
     bảo chứng (1-1/e) của CẢ HAI thuật toán này suy biến TRÙNG HOÀN TOÀN
     với GreedyFKGS (xem bảng tóm tắt đầu Dieu_chinh_..._v4.pdf) — không
     còn là thuật toán riêng biệt để cài đặt/so sánh.
  2. FKG và FKGS đều được XÂY DỰNG LẠI THEO TỪNG FOLD của k-fold (mờ hoá,
     khai phá luật, tính Sim/Rep, chạy 4 thuật toán lấy mẫu) — không dùng
     một FKG cố định rồi mới chia fold ở bước đánh giá cuối, tránh rò rỉ
     dữ liệu qua bước fitting (cùng nguyên tắc đã áp dụng cho FKG-MM/FKG-E
     ở Chương 3).
  3. Sim(R_i,R_j) dùng ĐÚNG Định nghĩa 2.2 (Hamming trên thuộc tính đã mờ
     hoá phạm trù + điều kiện cùng nhãn Outcome), KHÔNG dùng Euclid liên
     tục trên vector Ruspini như bản v1 (đã xác nhận không khớp FKG.ipynb).
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))


class PATHS:
    DATA_DIR = os.path.join(ROOT, "data")
    RESULTS_DIR = os.path.join(ROOT, "results")
    DATA_CSV = os.path.join(DATA_DIR, "Data.csv")
    # Ba nguồn gốc trước khi ghép — chỉ cần nếu bạn build lại Data.csv từ đầu
    SRC_PIMA = os.path.join(DATA_DIR, "diabetes_pima.csv")
    SRC_KAGGLE = os.path.join(DATA_DIR, "diabetes_kaggle.csv")
    SRC_HEALTHCARE = os.path.join(DATA_DIR, "healthcare_diabetes_kaggle.csv")


# ============================================================
# Đặc trưng và mờ hoá (Mục 1, thiet_ke_thuc_nghiem.md)
# ============================================================
FEATURE_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
LABEL_COLUMN = "Outcome"

# Cột có giá trị 0 = thiếu dữ liệu về mặt sinh học (Mục 0 của thiet_ke_thuc_nghiem.md)
ZERO_AS_MISSING_COLUMNS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Nhãn ngôn ngữ mặc định: Low/Medium/High (3 mức, theo phân vị 10/50/90 của
# CHÍNH TẬP TRAIN của fold — không phải toàn bộ dữ liệu, xem src/fuzzification.py)
FUZZY_LEVELS = ["Low", "Medium", "High"]
FUZZY_QUANTILES = [0.10, 0.50, 0.90]

# BMI dùng ngưỡng lâm sàng chuẩn (WHO) thay vì phân vị dữ liệu — 4 mức
BMI_CLINICAL_LEVELS = ["Underweight", "Normal", "Overweight", "Obese"]
BMI_CLINICAL_BOUNDARIES = [18.5, 25.0, 30.0]  # <18.5 / 18.5-25 / 25-30 / >=30


# ============================================================
# Quy mô thực nghiệm (Mục 2, thiet_ke_thuc_nghiem.md)
# ============================================================
class SCALE:
    N_EXP = 1500      # tập con cho E2-E4 (lấy mẫu phân tầng theo Outcome, TRONG train fold)
    N_TEST = 3000      # tập kiểm tra AUC cố định, tách biệt hoàn toàn


# ============================================================
# K-FOLD (bổ sung theo yêu cầu — không có trong thiet_ke_thuc_nghiem.md gốc)
# ============================================================
class KFOLD:
    K = 5
    SEED = 42
    # Diabetes: mỗi dòng là 1 bệnh nhân độc lập (không có nhiều bản ghi/bệnh
    # nhân như BRSET), nên dùng StratifiedKFold (không cần GroupKFold theo
    # patient_id) — vẫn PHẢI xây lại FKG/FKGS trong từng fold để tránh rò
    # rỉ tham số mờ hoá/khai phá luật, dù không có group cần bảo vệ.


# ============================================================
# E1 — Xác nhận lý thuyết (Mục 3)
# ============================================================
class E1:
    N_VALUES = [10, 14, 18]
    RATIO_VALUES = [0.2, 0.4]
    N_REPEATS = 20


# ============================================================
# E2 — GreedyFKGS vs Random (Mục 4)
# ============================================================
class E2:
    THETA_VALUES = [0.10, 0.30, 0.50]
    N_SEEDS_RANDOM = 30


# ============================================================
# E3 — Đường cong theta (Mục 5)
# ============================================================
class E3:
    THETA_GRID = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                  0.60, 0.70, 0.80, 0.90, 1.00]
    COVERAGE_THRESHOLD = 0.85   # delta, Định nghĩa 2.12


# ============================================================
# E4 — So sánh 4 chiến lược (Mục 6, đã SỬA: 4 không phải 5/6)
# ============================================================
class E4:
    N_BLOCKS = 10
    BLOCK_SIZE = 500
    THETA = 0.20                 # tỉ lệ mẫu dùng chung để so sánh 4 thuật toán
    N_STRATA = 2                  # SFKGS: phân tầng theo Outcome (2 lớp)
    ALLOC_METHOD = "proportional"  # SFKGS: proportional | neyman | equal
    N_CLUSTERS = 10                # CFKGS: KMeans trên vector mờ hoá phạm trù
    CLUSTER_DESIGN = "two_stage"    # CFKGS: one_stage | two_stage
    METHODS = ["GreedyFKGS", "Random", "SFKGS", "CFKGS"]  # ĐÚNG 4, không phải 5/6


# ============================================================
# Thống kê
# ============================================================
class STATS:
    ALPHA = 0.05
    HOLM_CORRECTION = True
