"""
src/fisa_corrected.py — Lớp tương thích ngược (v3.2).

Từ v3.2, suy diễn FISA dùng cho mọi độ đo downstream được cài trong src/fisa_pairs.py theo
đúng mô hình FKG-Pairs của bài báo gốc và notebook FKG.ipynb (gộp C̃ có điều kiện theo nhãn,
công thức (10)). Bản FISA-P v3/v3.1 trước đây gộp trọng số KHÔNG điều kiện theo nhãn và dùng
"miền cặp" (global/adjacent/upper); cách này khác công thức (10) nên đã bị thay thế.

Tên FISACorrected được giữ để các script cũ chạy được; tham số pair_scope (nếu truyền)
bị bỏ qua kèm cảnh báo.
"""
import warnings
from src.fisa_pairs import FISAPairs, DECISIONS  # noqa: F401


class FISACorrected(FISAPairs):
    def __init__(self, rules, pair_scope=None, node_size=None):
        if pair_scope is not None:
            warnings.warn("pair_scope không còn được dùng từ v3.2 (FISA theo FKG-Pairs); bỏ qua.",
                          DeprecationWarning, stacklevel=2)
        super().__init__(rules, node_size=node_size)
