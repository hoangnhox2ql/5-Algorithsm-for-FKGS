"""
src/fuzzification.py — Mờ hoá phạm trù (v3).

SỬA SO VỚI v2
  * QA-01: v2 điền giá trị thiếu của MỖI HÀNG bằng trung vị của lớp mà hàng đó
    thuộc về — tức đọc nhãn thật của mẫu test. v3 học trung vị toàn cục của
    tập huấn luyện CHỈ từ X và áp dụng cùng một quy tắc cho mọi hàng; phép
    biến đổi đặc trưng không cần (và không đọc) cột nhãn.
  * Suy diễn không có nhãn: `transform_features()` chạy được khi dữ liệu không
    có cột nhãn (QA Stage 2: KeyError 'Outcome').
  * NaN: được coi là thiếu và được điền như số 0 phi sinh lý; thuộc tính phạm
    trù thiếu nhận nhãn "Missing". Sau khi điền, mọi giá trị số phải hữu hạn,
    nếu không sẽ báo lỗi (v2 lặng lẽ đổi NaN thành 'High'/'Obese').
  * Chỉ tính hai phân vị thực sự dùng (q10, q90); q50 của v2 bị bỏ vì không
    tham gia phép mờ hoá (QA-09).
"""
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C


class FuzzyFitter:
    def __init__(self, feature_columns=None, label_column=None, categorical=None,
                 zero_missing=None, clinical=None):
        self.feature_columns = list(feature_columns or C.FEATURE_COLUMNS)
        self.label_column = label_column or C.LABEL_COLUMN
        self.categorical = set(C.CATEGORICAL_COLUMNS if categorical is None else categorical)
        self.zero_missing = list(C.ZERO_AS_MISSING_COLUMNS if zero_missing is None else zero_missing)
        self.clinical = dict(C.CLINICAL_BINS if clinical is None else clinical)
        self.numeric = [c for c in self.feature_columns if c not in self.categorical]
        self.median_ = {}
        self.bounds_ = {}
        self.fitted_ = False

    # ---------------------------------------------------------------
    def _check_columns(self, df):
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Thiếu cột đặc trưng: {missing}")

    def _missing_mask(self, s, col):
        m = s.isna()
        if col in self.zero_missing:
            m = m | (s == 0)
        return m

    def fit(self, df_train: pd.DataFrame):
        if df_train is None or len(df_train) == 0:
            raise ValueError("FuzzyFitter.fit: tập huấn luyện rỗng")
        self._check_columns(df_train)
        for col in self.numeric:
            s = pd.to_numeric(df_train[col], errors="coerce").astype(float)
            valid = s[~self._missing_mask(s, col)]
            if len(valid) == 0:
                raise ValueError(f"Cột {col}: không có giá trị hợp lệ để học trung vị")
            self.median_[col] = float(valid.median())
        X = self._impute(df_train)
        for col in self.numeric:
            if col in self.clinical:
                continue
            q = X[col].quantile(C.FUZZY_QUANTILES).values
            self.bounds_[col] = (float(q[0]), float(q[1]))
        self.fitted_ = True
        return self

    def _impute(self, df):
        """Điền giá trị thiếu của các cột số bằng trung vị toàn cục của train.
        KHÔNG đọc cột nhãn."""
        out = pd.DataFrame(index=df.index)
        for col in self.numeric:
            s = pd.to_numeric(df[col], errors="coerce").astype(float)
            s = s.where(~self._missing_mask(s, col), self.median_[col])
            if not np.isfinite(s.values).all():
                raise ValueError(f"Cột {col}: còn giá trị không hữu hạn sau khi điền")
            out[col] = s
        for col in self.categorical:
            s = df[col]
            out[col] = s.where(~s.isna(), C.MISSING_CATEGORY).astype(str)
        return out

    def _fuzzify_numeric(self, col, v):
        if col in self.clinical:
            bounds, labels = self.clinical[col]
            for b, lab in zip(bounds, labels):
                if v < b:
                    return lab
            return labels[-1]
        q10, q90 = self.bounds_[col]
        if v <= q10:
            return C.FUZZY_LEVELS[0]
        elif v <= q90:
            return C.FUZZY_LEVELS[1]
        return C.FUZZY_LEVELS[2]

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Biến đổi CHỈ đặc trưng — dùng được cho dữ liệu suy diễn không nhãn."""
        if not self.fitted_:
            raise RuntimeError("FuzzyFitter chưa fit")
        self._check_columns(df)
        X = self._impute(df)
        out = pd.DataFrame(index=df.index)
        for col in self.feature_columns:
            if col in self.categorical:
                out[col] = X[col].astype(str)
            else:
                out[col] = [self._fuzzify_numeric(col, v) for v in X[col].values]
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Đặc trưng đã mờ hoá + cột nhãn (nếu có) được sao chép nguyên vẹn.
        Nhãn KHÔNG tham gia bất kỳ phép tính nào trên đặc trưng."""
        out = self.transform_features(df)
        if self.label_column in df.columns:
            out[self.label_column] = df[self.label_column].values
        return out


def rules_from_fuzzy_df(df_fuzzy: pd.DataFrame, feature_columns=None, label_column=None):
    """Mỗi hàng -> luật [giá trị ngôn ngữ..., nhãn:int]."""
    feature_columns = feature_columns or C.FEATURE_COLUMNS
    label_column = label_column or C.LABEL_COLUMN
    if label_column not in df_fuzzy.columns:
        raise ValueError("rules_from_fuzzy_df cần cột nhãn; dùng transform_features cho suy diễn")
    F = df_fuzzy[feature_columns].values.tolist()
    y = df_fuzzy[label_column].astype(int).tolist()
    return [list(f) + [lab] for f, lab in zip(F, y)]
