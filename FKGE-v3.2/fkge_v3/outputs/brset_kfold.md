# FISA (FKG-Pairs, FKGS v3.2) và FKG-E (Chương 3) trên BRSET — 5 fold theo ID bệnh nhân

Cấu hình: F_c = 5 phần tính chéo; tập xác thực FKG-E = 20% nhóm; T_ep = 200; tham số FKG-E theo config.FKGE.

| Độ đo | FISA (trung bình ± SD) | FISA CI95 | FKG-E (trung bình ± SD) | FKG-E CI95 |
|---|---|---|---|---|
| AUC-ROC | 0.8484 ± 0.0319 | [0.8088; 0.8880] | 0.9209 ± 0.0326 | [0.8804; 0.9614] |
| Độ chính xác cân bằng | 0.6232 ± 0.0932 | [0.5075; 0.7389] | 0.8253 ± 0.0503 | [0.7629; 0.8876] |
| Độ nhạy | 0.6684 ± 0.3674 | [0.2123; 1.1245] | 0.7721 ± 0.1218 | [0.6209; 0.9233] |
| Độ đặc hiệu | 0.5780 ± 0.3594 | [0.1318; 1.0242] | 0.8784 ± 0.0732 | [0.7875; 0.9693] |
| F1 lớp dương | 0.2346 ± 0.1656 | [0.0290; 0.4402] | 0.5084 ± 0.1088 | [0.3734; 0.6435] |
| Độ chính xác | 0.5853 ± 0.3055 | [0.2060; 0.9646] | 0.8700 ± 0.0623 | [0.7926; 0.9474] |

| FISA — quy tắc khác | Giá trị |
|---|---|
| Acc / BalAcc theo argmax (công thức (12)) | 0.7126 / 0.5885 |
| Acc theo D0 > 9·D1 (notebook gốc) | 0.0820 |

| FKG-E — ràng buộc Bài toán 3.1A | Giá trị |
|---|---|
| Đồng thuận nhãn với FISA (R2) | 0.6673 ± 0.2272 |
| Độ lệch căn cứ luật Dev (R3) | 0.4028 ± 0.2109 |
| ms/truy vấn: FISA bảng tra / FISA tuần tự / FKG-E (R4) | 0.610 / 14.165 / 0.144 |

## Theo fold

| Fold | n_train (nhóm) | n_test (dương) | FISA AUC | FISA BalAcc | FKG-E AUC | FKG-E BalAcc | Đồng thuận | Dev | Epoch (dừng) |
|---|---|---|---|---|---|---|---|---|---|
| fold_01 | 1778 (1074) | 242 (20) | 0.8660 | 0.5986 | 0.9387 | 0.8486 | 0.5331 | 0.5234 | 200 (T_ep) |
| fold_02 | 1776 (1121) | 242 (19) | 0.8892 | 0.6514 | 0.9450 | 0.8919 | 0.4959 | 0.5736 | 126 (early) |
| fold_03 | 1776 (1021) | 242 (19) | 0.8539 | 0.5241 | 0.9314 | 0.7603 | 0.9091 | 0.1758 | 200 (T_ep) |
| fold_04 | 1780 (1115) | 241 (20) | 0.8126 | 0.7683 | 0.9252 | 0.8298 | 0.9212 | 0.1699 | 184 (early) |
| fold_05 | 1778 (990) | 241 (19) | 0.8204 | 0.5735 | 0.8642 | 0.7956 | 0.4772 | 0.5715 | 200 (T_ep) |

Ghi chú: dữ liệu chỉ có mức ngôn ngữ rời rạc nên suy diễn là trường hợp Crisp. Tập huấn luyện đã được cân bằng lớp bằng lặp mẫu; các dòng trùng được gom nhóm để mọi phép chia nội bộ giữ chúng trong cùng một phần, và FKG-E loại luật của chính mẫu (cùng nhóm) khi tính mất mát.
