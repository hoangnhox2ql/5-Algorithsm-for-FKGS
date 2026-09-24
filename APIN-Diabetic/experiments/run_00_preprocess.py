"""
experiments/run_00_preprocess.py — Tiền xử lý theo đúng Mục 0
thiet_ke_thuc_nghiem.md: loại dòng trùng lặp, báo cáo tỉ lệ giá trị 0 bất
khả thi sinh học (việc impute thực sự diễn ra TRONG TỪNG FOLD ở
src/fuzzification.py, không ở đây, để tránh rò rỉ).

Nếu chưa có data/Data.csv thật, tự động sinh dữ liệu TỔNG HỢP mô phỏng đúng
cấu trúc mô tả (8 đặc trưng, có 0-giả cần impute, có dòng trùng lặp, mất cân
bằng lớp ~66/34) để bạn kiểm tra toàn bộ pipeline E1-E4 không lỗi trước khi
có dữ liệu thật.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import config as C


def generate_synthetic_diabetes(n_total=6000, seed=42, duplicate_ratio=0.15):
    """Sinh dữ liệu tổng hợp mô phỏng ĐÚNG các đặc điểm đã liệt kê ở Mục 0
    thiet_ke_thuc_nghiem.md: 8 đặc trưng, 0-giả cần impute, trùng lặp, mất
    cân bằng lớp. Outcome phụ thuộc THẬT vào các đặc trưng (không nhiễu
    hoàn toàn) để GreedyFKGS/SFKGS/CFKGS có tín hiệu thật để khai thác.
    """
    rng = np.random.RandomState(seed)
    n_unique = int(n_total * (1 - duplicate_ratio))

    glucose = rng.normal(120, 30, n_unique).clip(50, 250)
    bmi = rng.normal(31, 7, n_unique).clip(15, 60)
    age = rng.randint(21, 81, n_unique)
    preg = rng.poisson(2.5, n_unique).clip(0, 15)
    bp = rng.normal(72, 12, n_unique).clip(30, 130)
    skin = rng.normal(25, 10, n_unique).clip(0, 70)
    insulin = rng.normal(90, 85, n_unique).clip(0, 500)
    dpf = rng.uniform(0.05, 2.2, n_unique)

    # Outcome phụ thuộc thật vào Glucose, BMI, Age (giống mô hình dịch tễ thực tế)
    logit = -6.0 + 0.025 * glucose + 0.08 * bmi + 0.03 * age + 0.15 * preg
    prob = 1 / (1 + np.exp(-logit))
    outcome = (rng.rand(n_unique) < prob).astype(int)

    df = pd.DataFrame({
        "Pregnancies": preg, "Glucose": glucose, "BloodPressure": bp,
        "SkinThickness": skin, "Insulin": insulin, "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, "Age": age, "Outcome": outcome,
    })

    # Tạo giá trị 0-giả (thiếu dữ liệu) đúng tỉ lệ đã mô tả
    for col, ratio in [("Glucose", 0.002), ("BloodPressure", 0.02),
                        ("SkinThickness", 0.06), ("Insulin", 0.10), ("BMI", 0.005)]:
        mask = rng.rand(n_unique) < ratio
        df.loc[mask, col] = 0

    # Nhân bản một phần để tạo dòng trùng lặp (mô phỏng việc ghép 3 nguồn chồng lấp)
    n_dup = n_total - n_unique
    dup_idx = rng.choice(n_unique, n_dup, replace=True)
    df_dup = df.iloc[dup_idx].copy()
    df_final = pd.concat([df, df_dup], ignore_index=True)
    df_final = df_final.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df_final


def preprocess(df: pd.DataFrame):
    n_before = len(df)
    df_dedup = df.drop_duplicates().reset_index(drop=True)
    n_after = len(df_dedup)
    n_removed = n_before - n_after

    print(f"Dòng trước loại trùng: {n_before}")
    print(f"Dòng trùng lặp bị loại: {n_removed} ({n_removed/n_before*100:.1f}%)")
    print(f"Dòng sau loại trùng: {n_after}")

    print("\nTỉ lệ giá trị 0 (bất khả thi sinh học) theo cột:")
    for col in C.ZERO_AS_MISSING_COLUMNS:
        if col in df_dedup.columns:
            n_zero = (df_dedup[col] == 0).sum()
            print(f"  {col:28s}: {n_zero:5d} dòng ({n_zero/n_after*100:.2f}%)")

    label_dist = df_dedup[C.LABEL_COLUMN].value_counts(normalize=True).sort_index()
    print(f"\nPhân bố nhãn Outcome: {dict(label_dist.round(4))}")

    return df_dedup


def main():
    os.makedirs(C.PATHS.DATA_DIR, exist_ok=True)
    if os.path.exists(C.PATHS.DATA_CSV):
        print(f"Đã tìm thấy {C.PATHS.DATA_CSV} -- nạp dữ liệu thật.")
        df = pd.read_csv(C.PATHS.DATA_CSV)
    else:
        print(f"!! CHƯA TÌM THẤY {C.PATHS.DATA_CSV}.")
        print("!! Sinh DỮ LIỆU TỔNG HỢP để kiểm tra pipeline -- KHÔNG dùng "
              "kết quả từ dữ liệu này cho báo cáo chính thức.")
        df = generate_synthetic_diabetes()
        df.to_csv(C.PATHS.DATA_CSV, index=False)
        print(f"Đã lưu dữ liệu tổng hợp vào {C.PATHS.DATA_CSV} "
              "(xoá file này và đặt Data.csv thật vào đúng vị trí khi sẵn sàng).")

    df_clean = preprocess(df)
    out_path = os.path.join(C.PATHS.DATA_DIR, "Data_clean.csv")
    df_clean.to_csv(out_path, index=False)
    print(f"\nĐã lưu dữ liệu đã tiền xử lý vào {out_path}")
    return df_clean


if __name__ == "__main__":
    main()
