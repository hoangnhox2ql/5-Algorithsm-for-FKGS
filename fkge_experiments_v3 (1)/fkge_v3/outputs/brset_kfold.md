# FISA (FKG-Pairs, FKGS v3.2) và FKG-E (Chương 3) trên BRSET — 5 fold theo ID bệnh nhân

Cấu hình: F_c = 5 phần tính chéo; tập xác thực FKG-E = 20% nhóm; T_ep = 100; tham số FKG-E theo config.FKGE.

| Độ đo | FISA (trung bình ± SD) | FISA CI95 | FKG-E (trung bình ± SD) | FKG-E CI95 |
|---|---|---|---|---|
| AUC-ROC | 0.8484 ± 0.0319 | [0.8088; 0.8880] | 0.9322 ± 0.0274 | [0.8982; 0.9662] |
| Độ chính xác cân bằng | 0.6232 ± 0.0932 | [0.5075; 0.7389] | 0.8506 ± 0.0374 | [0.8042; 0.8970] |
| Độ nhạy | 0.6684 ± 0.3674 | [0.2123; 1.1245] | 0.8453 ± 0.0800 | [0.7459; 0.9446] |
| Độ đặc hiệu | 0.5780 ± 0.3594 | [0.1318; 1.0242] | 0.8560 ± 0.0634 | [0.7772; 0.9347] |
| F1 lớp dương | 0.2346 ± 0.1656 | [0.0290; 0.4402] | 0.4979 ± 0.0916 | [0.3842; 0.6115] |
| Độ chính xác | 0.5853 ± 0.3055 | [0.2060; 0.9646] | 0.8551 ± 0.0555 | [0.7863; 0.9240] |

| FISA — quy tắc khác | Giá trị |
|---|---|
| Acc / BalAcc theo argmax (công thức (12)) | 0.7126 / 0.5885 |
| Acc theo D0 > 9·D1 (notebook gốc) | 0.0820 |

| FKG-E — ràng buộc Bài toán 3.1A | Giá trị |
|---|---|
| Đồng thuận nhãn với FISA (R2) | 0.6474 ± 0.2168 |
| Độ lệch căn cứ luật Dev (R3) | 0.4553 ± 0.1964 |
| ms/truy vấn: FISA bảng tra / FISA tuần tự / FKG-E (R4) | 0.552 / 13.518 / 0.134 |

## Theo fold

| Fold | n_train (nhóm) | n_test (dương) | FISA AUC | FISA BalAcc | FKG-E AUC | FKG-E BalAcc | Đồng thuận | Dev | Epoch (dừng) |
|---|---|---|---|---|---|---|---|---|---|
| fold_01 | 1778 (1074) | 242 (20) | 0.8660 | 0.5986 | 0.9568 | 0.8691 | 0.5207 | 0.5698 | 11 (early) |
| fold_02 | 1776 (1121) | 242 (19) | 0.8892 | 0.6514 | 0.9580 | 0.9070 | 0.5124 | 0.5918 | 11 (early) |
| fold_03 | 1776 (1021) | 242 (19) | 0.8539 | 0.5241 | 0.9173 | 0.8342 | 0.8347 | 0.2916 | 11 (early) |
| fold_04 | 1780 (1115) | 241 (20) | 0.8126 | 0.7683 | 0.9354 | 0.8275 | 0.9253 | 0.1973 | 11 (early) |
| fold_05 | 1778 (990) | 241 (19) | 0.8204 | 0.5735 | 0.8936 | 0.8152 | 0.4440 | 0.6260 | 11 (early) |

Ghi chú: dữ liệu chỉ có mức ngôn ngữ rời rạc nên suy diễn là trường hợp Crisp. Tập huấn luyện đã được cân bằng lớp bằng lặp mẫu; các dòng trùng được gom nhóm để mọi phép chia nội bộ giữ chúng trong cùng một phần, và FKG-E loại luật của chính mẫu (cùng nhóm) khi tính mất mát.
