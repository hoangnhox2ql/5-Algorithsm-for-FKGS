"""
experiments/run_00_preprocess.py — Nạp dữ liệu (có kiểm soát nguồn gốc) và loại
trùng (v3). Không tự sinh dữ liệu giả trừ khi RUN.ALLOW_SYNTHETIC (QA-07).
Ghi Data_clean.csv vào thư mục kết quả của lần chạy và trả thông tin tiền xử lý.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
from experiments.common import load_raw_data, preprocess, results_path


def main():
    df, prov = load_raw_data()
    clean, info = preprocess(df)
    os.makedirs(C.PATHS.RESULTS_DIR, exist_ok=True)
    clean.to_csv(C.PATHS.DATA_CLEAN, index=False)
    print(f"  Nguồn: {prov['source']} ({prov['path']}), sha256={prov['sha256']}")
    print(f"  {info['n_raw']} dòng -> {info['n_clean']} sau loại {info['n_duplicates_removed']} trùng; "
          f"nhãn {info['label_counts']}; X trùng khác nhãn: {info['n_conflicting_X']}")
    return prov, info


if __name__ == "__main__":
    main()
