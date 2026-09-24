"""
src/kfold_utils.py — K-fold cho FKG và FKGS: với MỖI fold, mờ hoá (fit trên
train), khai phá luật (chỉ từ train), tính Sim/Rep TRÊN TẬP TRAIN của fold,
rồi chạy 4 thuật toán lấy mẫu TRÊN CHÍNH tập luật train của fold đó — không
dùng một FKG/FKGS cố định chung cho mọi fold.

Diabetes: mỗi dòng là 1 bệnh nhân độc lập (không có nhiều bản ghi/bệnh nhân
như BRSET) nên dùng StratifiedKFold (theo Outcome), KHÔNG cần GroupKFold
theo patient_id -- nhưng vẫn PHẢI xây lại toàn bộ pipeline mờ hoá + khai phá
luật trong từng fold để tránh rò rỉ tham số (cùng nguyên tắc áp dụng cho
FKG-MM/FKG-E ở Chương 3, Mục 3.5.1).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import StratifiedKFold
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

import config as C
from src.fuzzification import FuzzyFitter, rules_from_fuzzy_df
from src.fkg_sim_rep import compute_sim_matrix


def make_stratified_kfold(df: pd.DataFrame, k=None, seed=None, label_column=None):
    """Trả về list[(train_idx, test_idx)] theo StratifiedKFold trên Outcome.
    `df` phải có index mặc định 0..n-1 (reset_index trước khi gọi nếu cần)."""
    k = k or C.KFOLD.K
    seed = seed if seed is not None else C.KFOLD.SEED
    label_column = label_column or C.LABEL_COLUMN
    y = df[label_column].values

    if _HAS_SKLEARN:
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        folds = list(skf.split(np.zeros(len(df)), y))
    else:
        folds = _manual_stratified_kfold(y, k, seed)

    _verify_stratified_balance(y, folds)
    return folds


def _manual_stratified_kfold(y, k, seed):
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    fold_of_idx = np.zeros(len(y), dtype=int)
    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        for pos, i in enumerate(idx_c):
            fold_of_idx[i] = pos % k
    folds = []
    for f in range(k):
        test_idx = np.where(fold_of_idx == f)[0]
        train_idx = np.where(fold_of_idx != f)[0]
        folds.append((train_idx, test_idx))
    return folds


def _verify_stratified_balance(y, folds, max_ratio_diff=0.15):
    """Kiểm tra tự động: tỉ lệ lớp 1 trong mỗi fold-test không được lệch
    quá xa tỉ lệ tổng thể -- nếu lệch nhiều, cảnh báo (không dừng chương
    trình, vì với Neyman/nhỏ có thể lệch chút, nhưng phải BIẾT)."""
    overall_ratio = y.mean()
    for f, (_, test_idx) in enumerate(folds):
        r = y[test_idx].mean()
        if abs(r - overall_ratio) > max_ratio_diff:
            print(f"  !! CẢNH BÁO: fold {f} có tỉ lệ Outcome=1 là {r:.3f}, lệch "
                  f"{abs(r-overall_ratio):.3f} so với tổng thể {overall_ratio:.3f} "
                  f"(ngưỡng cảnh báo {max_ratio_diff}).")


def build_fkg_fold(df_train: pd.DataFrame, df_test: pd.DataFrame,
                    n_sample=None, seed=0):
    """Xây FKG cho MỘT fold: fit mờ hoá trên train, transform cả train/test,
    (tuỳ chọn) lấy mẫu phân tầng n_sample luật từ train (N_EXP), tính Sim.
    Trả về dict {rules_train, rules_test, sim, fitter}."""
    fitter = FuzzyFitter().fit(df_train)
    fuzzy_train = fitter.transform(df_train)
    fuzzy_test = fitter.transform(df_test)

    rules_train_full = rules_from_fuzzy_df(fuzzy_train)
    rules_test = rules_from_fuzzy_df(fuzzy_test)

    if n_sample is not None and n_sample < len(rules_train_full):
        rules_train = _stratified_subsample(rules_train_full, n_sample, seed)
    else:
        rules_train = rules_train_full

    sim = compute_sim_matrix(rules_train)
    return {
        "rules_train": rules_train, "rules_test": rules_test,
        "sim": sim, "fitter": fitter,
        "n_rules_train_full": len(rules_train_full),
    }


def _stratified_subsample(rules, n_sample, seed):
    """Lấy mẫu phân tầng theo nhãn (cột cuối) giữ đúng tỉ lệ lớp -- dùng để
    thu nhỏ N_EXP từ toàn bộ train fold, đúng Mục 2 thiet_ke_thuc_nghiem.md."""
    rng = np.random.RandomState(seed)
    labels = np.array([r[-1] for r in rules])
    classes = np.unique(labels)
    selected = []
    for c in classes:
        idx_c = np.where(labels == c)[0]
        n_c = int(round(n_sample * len(idx_c) / len(rules)))
        n_c = min(n_c, len(idx_c))
        chosen = rng.choice(idx_c, n_c, replace=False)
        selected.extend(chosen.tolist())
    rng.shuffle(selected)
    return [rules[i] for i in selected[:n_sample]]


if __name__ == "__main__":
    # Kiểm thử với dữ liệu tổng hợp mô phỏng cấu trúc Diabetes
    rng = np.random.RandomState(0)
    n = 400
    df = pd.DataFrame({
        "Pregnancies": rng.randint(0, 10, n),
        "Glucose": rng.normal(120, 30, n).clip(40, 200),
        "BloodPressure": rng.normal(70, 12, n).clip(40, 120),
        "SkinThickness": rng.normal(25, 10, n).clip(5, 60),
        "Insulin": rng.normal(100, 80, n).clip(0, 400),
        "BMI": rng.normal(31, 7, n).clip(15, 55),
        "DiabetesPedigreeFunction": rng.uniform(0.05, 2.0, n),
        "Age": rng.randint(21, 80, n),
        "Outcome": rng.choice([0, 1], n, p=[0.65, 0.35]),
    })

    folds = make_stratified_kfold(df, k=5, seed=42)
    print(f"Đã chia {len(folds)} fold từ {len(df)} dòng.")
    for f, (train_idx, test_idx) in enumerate(folds):
        print(f"  Fold {f}: train={len(train_idx)}, test={len(test_idx)}, "
              f"tỉ lệ Outcome=1 (test)={df.iloc[test_idx]['Outcome'].mean():.3f}")

    # Xây FKG cho fold 0, xác nhận luật khác nhau nếu build lại với seed khác
    train_idx, test_idx = folds[0]
    df_train, df_test = df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    result = build_fkg_fold(df_train, df_test, n_sample=100, seed=1)
    print(f"\nFold 0: {result['n_rules_train_full']} luật đầy đủ -> lấy mẫu "
          f"{len(result['rules_train'])} luật, Sim shape={result['sim'].shape}")
    assert len(result["rules_train"]) == 100
    assert result["sim"].shape == (100, 100)
    print("\nKiểm thử kfold_utils: OK")
