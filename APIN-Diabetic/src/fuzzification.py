"""
src/fuzzification.py — Mờ hoá phạm trù (categorical), đúng Mục 1 của
docs/thiet_ke_thuc_nghiem.md và đúng cách FKG.ipynb xử lý (mỗi thuộc tính
-> ĐÚNG MỘT nhãn ngôn ngữ rời rạc, không phải vector độ thuộc liên tục
kiểu Ruspini của bản v1 đã bị phản bác).

NGUYÊN TẮC CHỐNG RÒ RỈ (Chương 3, Mục 3.5.1, áp dụng lại ở đây): tham số
mờ hoá (các ngưỡng phân vị 10/50/90, hoặc median dùng để impute) PHẢI được
ước lượng CHỈ từ tập train của fold hiện tại (FuzzyFitter.fit), rồi áp
dụng NGUYÊN VẸN cho cả train lẫn test của đúng fold đó (FuzzyFitter.transform)
— không bao giờ fit lại trên test.
"""
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


class FuzzyFitter:
    """Học tham số mờ hoá (median impute theo lớp + ngưỡng phân vị/lâm sàng)
    CHỈ từ dữ liệu train, rồi áp dụng cho bất kỳ tập dữ liệu nào (train hoặc
    test của CÙNG fold)."""

    def __init__(self, feature_columns=None, label_column=None):
        self.feature_columns = feature_columns or C.FEATURE_COLUMNS
        self.label_column = label_column or C.LABEL_COLUMN
        self.impute_median_ = {}      # {col: {0: median0, 1: median1}}
        self.quantile_bounds_ = {}     # {col: [q10, q50, q90]}  (trừ BMI)

    def fit(self, df_train: pd.DataFrame):
        df = df_train.copy()
        # 1) Học median-theo-lớp để impute các cột có 0 = thiếu (chỉ từ train)
        for col in C.ZERO_AS_MISSING_COLUMNS:
            if col not in df.columns:
                continue
            self.impute_median_[col] = {}
            for cls in df[self.label_column].unique():
                sub = df[(df[self.label_column] == cls) & (df[col] != 0)]
                med = sub[col].median() if len(sub) > 0 else df[df[col] != 0][col].median()
                self.impute_median_[col][cls] = med

        # 2) Impute NGAY trên bản sao để tính phân vị đúng dữ liệu đã sạch
        df_clean = self._apply_impute(df)

        # 3) Học ngưỡng phân vị 10/50/90 cho mọi cột TRỪ BMI (dùng ngưỡng lâm sàng cố định)
        for col in self.feature_columns:
            if col == "BMI":
                continue
            qs = df_clean[col].quantile(C.FUZZY_QUANTILES).values
            self.quantile_bounds_[col] = qs
        return self

    def _apply_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col, med_by_class in self.impute_median_.items():
            mask_zero = df[col] == 0
            if not mask_zero.any():
                continue
            for cls, med in med_by_class.items():
                m = mask_zero & (df[self.label_column] == cls)
                df.loc[m, col] = med
            # Trường hợp nhãn không xuất hiện trong med_by_class (hiếm, an toàn):
            remaining = df[col] == 0
            if remaining.any():
                fallback = np.mean(list(med_by_class.values()))
                df.loc[remaining, col] = fallback
        return df

    def _fuzzify_value(self, col, val):
        """Trả về nhãn ngôn ngữ rời rạc (str) cho MỘT giá trị số của MỘT cột."""
        if col == "BMI":
            b = C.BMI_CLINICAL_BOUNDARIES
            if val < b[0]:
                return "Underweight"
            elif val < b[1]:
                return "Normal"
            elif val < b[2]:
                return "Overweight"
            else:
                return "Obese"
        q10, q50, q90 = self.quantile_bounds_[col]
        if val <= q10:
            return "Low"
        elif val <= q90:
            return "Medium"
        else:
            return "High"

    def transform(self, df: pd.DataFrame):
        """Trả về DataFrame mới với các cột đặc trưng đã thay bằng nhãn
        ngôn ngữ rời rạc (str), cột nhãn giữ nguyên. Impute trước, mờ hoá sau,
        đều dùng tham số đã fit từ train."""
        df_clean = self._apply_impute(df)
        out = pd.DataFrame(index=df_clean.index)
        for col in self.feature_columns:
            out[col] = df_clean[col].apply(lambda v: self._fuzzify_value(col, v))
        out[self.label_column] = df_clean[self.label_column].values
        return out


def rules_from_fuzzy_df(df_fuzzy: pd.DataFrame, feature_columns=None, label_column=None):
    """Chuyển DataFrame đã mờ hoá phạm trù thành list các 'luật' dạng
    list[str] (đúng định dạng FKG.ipynb: các cột đặc trưng + cột nhãn ở
    cuối), dùng trực tiếp cho src/fkg_original.py / fkg_fast.py."""
    feature_columns = feature_columns or C.FEATURE_COLUMNS
    label_column = label_column or C.LABEL_COLUMN
    rules = []
    for _, row in df_fuzzy.iterrows():
        r = [row[c] for c in feature_columns] + [int(row[label_column])]
        rules.append(r)
    return rules


if __name__ == "__main__":
    # Tự kiểm thử: sinh dữ liệu tổng hợp nhỏ, xác nhận fit/transform không lỗi
    # và impute + mờ hoá cho ra đúng số mức mong đợi.
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({
        "Pregnancies": rng.randint(0, 10, n),
        "Glucose": np.where(rng.rand(n) < 0.05, 0, rng.normal(120, 30, n).clip(40, 200)),
        "BloodPressure": np.where(rng.rand(n) < 0.05, 0, rng.normal(70, 12, n).clip(40, 120)),
        "SkinThickness": np.where(rng.rand(n) < 0.1, 0, rng.normal(25, 10, n).clip(5, 60)),
        "Insulin": np.where(rng.rand(n) < 0.15, 0, rng.normal(100, 80, n).clip(0, 400)),
        "BMI": rng.normal(31, 7, n).clip(15, 55),
        "DiabetesPedigreeFunction": rng.uniform(0.05, 2.0, n),
        "Age": rng.randint(21, 80, n),
        "Outcome": rng.randint(0, 2, n),
    })
    n_train = 150
    df_train, df_test = df.iloc[:n_train], df.iloc[n_train:]

    fitter = FuzzyFitter().fit(df_train)
    fuzzy_train = fitter.transform(df_train)
    fuzzy_test = fitter.transform(df_test)

    print("Ví dụ 3 dòng train đã mờ hoá:")
    print(fuzzy_train.head(3).to_string())
    print("\nCác mức xuất hiện ở cột Glucose (train):", sorted(fuzzy_train["Glucose"].unique()))
    print("Các mức xuất hiện ở cột BMI (train):", sorted(fuzzy_train["BMI"].unique()))

    assert set(fuzzy_train["Glucose"].unique()) <= set(C.FUZZY_LEVELS)
    assert set(fuzzy_train["BMI"].unique()) <= set(C.BMI_CLINICAL_LEVELS)
    assert (fitter.transform(df_train)["Glucose"] != "").all()

    rules = rules_from_fuzzy_df(fuzzy_train)
    print(f"\nSố luật sinh ra: {len(rules)}, ví dụ luật đầu: {rules[0]}")
    assert len(rules) == len(fuzzy_train)
    print("\nKiểm thử fuzzification: OK")
