# FISA (FKG-Pairs, FKGS v3.2) và FKG-E (Chương 3) trên BRSET — 5 fold theo ID bệnh nhân

Cấu hình: F_c = 5 phần tính chéo; tập xác thực FKG-E = 20% nhóm; T_ep = 10; tham số FKG-E theo config.FKGE.

| Độ đo | FISA (trung bình ± SD) | FISA CI95 | FKG-E (trung bình ± SD) | FKG-E CI95 |
|---|---|---|---|---|
| AUC-ROC | 0.8660 ± 0.0000 | [nan; nan] | 0.9568 ± 0.0000 | [nan; nan] |
| Độ chính xác cân bằng | 0.5986 ± 0.0000 | [nan; nan] | 0.8691 ± 0.0000 | [nan; nan] |
| Độ nhạy | 0.9000 ± 0.0000 | [nan; nan] | 0.9500 ± 0.0000 | [nan; nan] |
| Độ đặc hiệu | 0.2973 ± 0.0000 | [nan; nan] | 0.7883 ± 0.0000 | [nan; nan] |
| F1 lớp dương | 0.1856 ± 0.0000 | [nan; nan] | 0.4419 ± 0.0000 | [nan; nan] |
| Độ chính xác | 0.3471 ± 0.0000 | [nan; nan] | 0.8017 ± 0.0000 | [nan; nan] |

| FISA — quy tắc khác | Giá trị |
|---|---|
| Acc / BalAcc theo argmax (công thức (12)) | 0.4421 / 0.6505 |
| Acc theo D0 > 9·D1 (notebook gốc) | 0.0826 |

| FKG-E — ràng buộc Bài toán 3.1A | Giá trị |
|---|---|
| Đồng thuận nhãn với FISA (R2) | 0.5207 ± 0.0000 |
| Độ lệch căn cứ luật Dev (R3) | 0.5698 ± 0.0000 |
| ms/truy vấn: FISA bảng tra / FISA tuần tự / FKG-E (R4) | 0.571 / 13.424 / 0.131 |

## Theo fold

| Fold | n_train (nhóm) | n_test (dương) | FISA AUC | FISA BalAcc | FKG-E AUC | FKG-E BalAcc | Đồng thuận | Dev | Epoch (dừng) |
|---|---|---|---|---|---|---|---|---|---|
| fold_01 | 1778 (1074) | 242 (20) | 0.8660 | 0.5986 | 0.9568 | 0.8691 | 0.5207 | 0.5698 | 10 (T_ep) |

Ghi chú: dữ liệu chỉ có mức ngôn ngữ rời rạc nên suy diễn là trường hợp Crisp. Tập huấn luyện đã được cân bằng lớp bằng lặp mẫu; các dòng trùng được gom nhóm để mọi phép chia nội bộ giữ chúng trong cùng một phần, và FKG-E loại luật của chính mẫu (cùng nhóm) khi tính mất mát.
