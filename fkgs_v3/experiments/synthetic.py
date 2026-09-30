"""
experiments/synthetic.py — Dữ liệu TỔNG HỢP chỉ để kiểm tra pipeline (v3).
Chỉ được dùng khi chạy với --allow-synthetic; manifest ghi source="synthetic".
KHÔNG dùng cho báo cáo khoa học (QA-07).
"""
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


def generate_synthetic(n_total=6000, seed=42, duplicate_ratio=0.15):
    if C.DATASET != "diabetes":
        raise NotImplementedError("Dữ liệu tổng hợp chỉ hỗ trợ lược đồ tiểu đường")
    return generate_synthetic_diabetes(n_total, seed, duplicate_ratio)


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


