"""
config.py — Cấu hình trung tâm FKGS v3 (sửa theo Báo cáo QA FKGS v2).

Một mã nguồn chạy cho HAI bộ dữ liệu, chọn bằng `set_dataset(name)` hoặc
`python run_all.py --dataset diabetes|heart`. Mọi đặc tả có ảnh hưởng tới kết
quả được khai báo TƯỜNG MINH ở đây (QA-09) và được ghi vào run manifest.

ĐẶC TẢ ĐÃ CHỐT TRONG v3 (xem CHANGELOG_v3.md):
  * Mờ hoá: phân hoạch CỨNG 3 mức theo phân vị q10/q90 của tập huấn luyện
    (không phải Ruspini); BMI dùng 4 mức WHO; thuộc tính phạm trù giữ nguyên.
  * Giá trị thiếu (0 phi sinh lý hoặc NaN): điền trung vị TOÀN CỤC của tập
    huấn luyện (chỉ dùng X), áp dụng cùng một quy tắc cho train và test —
    KHÔNG dùng nhãn của hàng đang biến đổi (sửa QA-01).
  * Nhân tương đồng: Sim = 1 - Hamming chuẩn hoá nếu cùng nhãn, -1 nếu khác
    nhãn; hàm đại diện dùng a(x,s) = max(0, Sim) (sửa mô tả QA-09).
  * FISA: B^t_{il} = (tổng A^t_{jk} trên MỌI cặp j<k của luật t) *
    |Val(P_i)->l| / |R| — đúng cách đọc công thức (1.17) và ví dụ Chương 1
    (PAIR_SCOPE="global"). Hai quy ước khác được giữ để phân tích độ nhạy:
    "adjacent" (chỉ cặp chứa i) và "upper" (j>i, cách cài v2). (QA-02)
  * Ngân sách: tổng số luật của SFKGS/CFKGS ĐÚNG BẰNG ceil(theta*n) như
    GreedyFKGS/Random (phân bổ phần dư lớn nhất) (sửa QA-04).
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET = None  # được đặt bởi set_dataset()

DATASETS = {
    "diabetes": dict(
        csv="data/diabetes/Data.csv",
        features=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                  "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"],
        label="Outcome",
        zero_missing=["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
        categorical=[],
        clinical={"BMI": ([18.5, 25.0, 30.0], ["Underweight", "Normal", "Overweight", "Obese"])},
        description="3 nguồn (Pima/Data.world, Kaggle, Healthcare-Kaggle), 8 thuộc tính",
    ),
    "heart": dict(
        csv="data/heart/Data.csv",
        features=["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
                  "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"],
        label="HeartDisease",
        zero_missing=["RestingBP", "Cholesterol"],
        categorical=["Sex", "ChestPainType", "FastingBS", "RestingECG", "ExerciseAngina", "ST_Slope"],
        clinical={},
        description="5 nguồn UCI (Cleveland, Hungarian, Switzerland, Long Beach VA, Statlog), 11 thuộc tính",
    ),
}

# ---- Các biến được set_dataset() gán lại ----
FEATURE_COLUMNS = []
LABEL_COLUMN = None
ZERO_AS_MISSING_COLUMNS = []
CATEGORICAL_COLUMNS = []
CLINICAL_BINS = {}

FUZZY_LEVELS = ["Low", "Medium", "High"]
# v3.1: mờ hoá theo TAM PHÂN VỊ của tập huấn luyện (mỗi mức ~1/3 số mẫu).
# Điểm cắt q10/q90 của v3 dồn ~80% giá trị vào mức Medium, làm FISA mất khả năng
# phân biệt (nguyên nhân thứ nhất của hiện tượng sụp về lớp đa số).
FUZZY_QUANTILES = [1/3, 2/3]
MISSING_CATEGORY = "Missing"            # giá trị phạm trù thiếu


class PATHS:
    DATA_CSV = None
    RESULTS_DIR = None
    DATA_CLEAN = None


class RUN:
    ALLOW_SYNTHETIC = False   # QA-07: mặc định KHÔNG tự sinh dữ liệu giả
    RUN_ID = None             # đặt ở run_all; mọi JSON kết quả mang run_id này


class SCALE:
    N_EXP = 1500              # tập nền lấy mẫu trong mỗi fold (nếu train lớn hơn)


class KFOLD:
    K = 5
    SEED = 42


class FISA:
    PAIR_SCOPE = "global"     # "global" | "adjacent" | "upper" (QA-02)
    SCORE = "logratio"        # v3.1: điểm s = log D1 - log D0 (tổng quát hoá quy tắc D0 > k*D1)
    DECISION = "calibrated"   # "argmax" (k=1) | "calibrated": ngưỡng trên s chọn trên tập xác thực
    CALIB_CRITERION = "bacc"  # "bacc" (Youden) | "acc"
    VAL_FRACTION = 0.2        # tỉ lệ tập xác thực tách từ tập huấn luyện của fold


class SAMPLING:
    BUDGET_MODE = "exact"     # "exact" (sum = ceil(theta*n)) | "ceil" (v2, chỉ để đối chứng)
    ALLOC_METHOD = "proportional"
    N_CLUSTERS_PER_LABEL = 5  # CFKGS: số cụm trong MỖI nhãn (tổng 10 với 2 nhãn, như v2)
    CLUSTER_WITHIN_LABEL = True


class E1:
    N_VALUES = [10, 14, 18]
    RATIO_VALUES = [0.2, 0.4]
    N_REPEATS = 20


class E2:
    THETA_VALUES = [0.05, 0.10, 0.20, 0.30, 0.50]
    N_SEEDS_RANDOM = 30


class E3:
    THETA_GRID = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
                  0.60, 0.70, 0.80, 0.90, 1.00]
    REP_MEAN_THRESHOLD = 0.85   # ngưỡng ĐẠI DIỆN TRUNG BÌNH (không phải phủ toàn phần) (QA-06)
    COVER_DELTA = 0.85          # ngưỡng phủ từng luật: tỉ lệ luật có cov >= delta
    INNER_VAL_FRACTION = 0.25   # chọn theta trên validation bên trong train (QA-12)
    INNER_SEED = 7
    METHODS = ["GreedyFKGS", "SFKGS", "CFKGS"]


class E4:
    N_BLOCKS = 10
    BLOCK_SIZE = 500
    THETA = 0.20
    METHODS = ["GreedyFKGS", "Random", "SFKGS", "CFKGS", "StratRandom"]


class PUB:
    """v3.1: giao thức của bài công bố [CT2] -- chia cố định 70/30, không loại trùng."""
    SEEDS = [42, 1, 2, 3, 4]
    PUBLISHED_ACC = {"diabetes": 0.7613}


class STATS:
    ALPHA = 0.05              # bác bỏ khi p <= alpha (quy ước thống nhất)
    N_BOOTSTRAP = 10000
    CI_LEVEL = 0.95


def set_dataset(name):
    """Chọn bộ dữ liệu; gán lại các biến cấu hình phụ thuộc bộ dữ liệu."""
    global DATASET, FEATURE_COLUMNS, LABEL_COLUMN, ZERO_AS_MISSING_COLUMNS
    global CATEGORICAL_COLUMNS, CLINICAL_BINS
    if name not in DATASETS:
        raise ValueError(f"Bộ dữ liệu không hỗ trợ: {name!r}; chọn một trong {list(DATASETS)}")
    d = DATASETS[name]
    DATASET = name
    FEATURE_COLUMNS = list(d["features"])
    LABEL_COLUMN = d["label"]
    ZERO_AS_MISSING_COLUMNS = list(d["zero_missing"])
    CATEGORICAL_COLUMNS = list(d["categorical"])
    CLINICAL_BINS = dict(d["clinical"])
    PATHS.DATA_CSV = os.path.join(ROOT, d["csv"])
    PATHS.RESULTS_DIR = os.path.join(ROOT, "results", name)
    PATHS.DATA_CLEAN = os.path.join(PATHS.RESULTS_DIR, "Data_clean.csv")
    return d


def snapshot():
    """Cấu hình HIỆU LỰC (sau --quick, --dataset...) để ghi vào manifest."""
    def cls(c):
        return {k: v for k, v in vars(c).items() if not k.startswith("_")}
    return dict(
        dataset=DATASET, features=FEATURE_COLUMNS, label=LABEL_COLUMN,
        zero_missing=ZERO_AS_MISSING_COLUMNS, categorical=CATEGORICAL_COLUMNS,
        clinical={k: [list(v[0]), list(v[1])] for k, v in CLINICAL_BINS.items()},
        fuzzy_levels=FUZZY_LEVELS, fuzzy_quantiles=FUZZY_QUANTILES,
        scale=cls(SCALE), kfold=cls(KFOLD), fisa=cls(FISA), sampling=cls(SAMPLING),
        e1=cls(E1), e2=cls(E2), e3=cls(E3), e4=cls(E4), stats=cls(STATS), pub=cls(PUB),
        allow_synthetic=RUN.ALLOW_SYNTHETIC,
    )
