# Báo cáo thực nghiệm FKGS v3 — bộ dữ liệu `diabetes`

| Mục | Giá trị |
|---|---|
| run_id | `20261003T141426-beb12322` |
| Mã nguồn (SHA-256 tổng hợp) | `3ac46dee1f5551d8…` |
| Dữ liệu | real `data\diabetes\Data.csv` SHA-256 `5236d580bcdeef7e…` |
| Tiền xử lý | 18536 dòng → 15778 sau loại 2758 trùng; nhãn {'0': 10506, '1': 5272}; X trùng khác nhãn: 0 |
| Chia dữ liệu | 5-fold phân tầng, seed 42; tập nền N_EXP=1500 |
| Điền giá trị thiếu | trung vị toàn cục của train (chỉ X), cùng quy tắc train/test |
| Nhân Sim / Rep | 1−Hamming cùng nhãn, −1 khác nhãn; Rep dùng max(0,Sim) |
| Mờ hoá | 3 mức theo phân vị [0.333, 0.667] của tập huấn luyện |
| FISA (FKG-Pairs) | k = `3` (gộp C̃ có điều kiện theo nhãn, công thức (10)); điểm s = log D1 − log D0; quyết định `calibrated` (tiêu chí `bacc`, tập xác thực 0.2 tập huấn luyện) |
| Ngân sách | `exact` (mọi phương pháp chọn đúng ceil(θn) luật) |
| Môi trường | Python 3.11.5, numpy 2.2.2, scipy 1.15.1, sklearn 1.6.1, Windows-10-10.0.19045-SP0 |
| Chế độ | đầy đủ |

## E1 — Kiểm chứng cận lý thuyết (RQ1)

120 mẫu con từ luật huấn luyện fold 0 (vét cạn). Cận 1−1/e = 0.6321.

| n | m | m/n thực | Greedy ρ TB / min | SFKGS Rep/OPT_pb min | SFKGS tỉ lệ cục bộ min | SFKGS max\|g\| | CFKGS Rep/OPT_pb min | CFKGS min g |
|---|---|---|---|---|---|---|---|---|
| 10 | 2 | 0.200 | 1.0000 / 1.0000 | 0.9767 | 0.9767 | 1.1e-16 | 1.0000 | -1.1e-16 |
| 10 | 4 | 0.400 | 0.9940 / 0.9483 | 0.9483 | 0.9032 | 1.1e-16 | 1.0000 | -1.1e-16 |
| 14 | 3 | 0.214 | 0.9879 / 0.9403 | 0.9403 | 0.9200 | 1.1e-16 | 1.0000 | 1.1e-16 |
| 14 | 6 | 0.429 | 0.9908 / 0.9663 | 0.9551 | 0.9200 | 1.1e-16 | 0.9881 | -1.1e-16 |
| 18 | 4 | 0.222 | 0.9884 / 0.9684 | 0.9684 | 0.9434 | 1.1e-16 | 1.0000 | 1.1e-16 |
| 18 | 7 | 0.389 | 0.9903 / 0.9732 | 0.9732 | 0.9429 | 1.1e-16 | 0.9818 | -1.1e-16 |

Số vi phạm: greedy=0, sfkgs=0, cfkgs=0, sfkgs_decomposition=0, cfkgs_negative_g=0, chain=0, classdev_sfkgs=0, classdev_cfkgs=0, greedy_above_opt=0, size_mismatch=0

## E2 — So sánh với FKG đầy đủ và lấy mẫu cơ sở, cùng ngân sách (RQ2)

Đơn vị suy luận: fold (n = số fold); Random/StratRandom lấy trung bình theo hạt giống trong fold. Các fold có tập huấn luyện chồng lấp nên p-value mang tính mô tả trong phạm vi bộ dữ liệu.

| Mốc | \|R\| | AUC | Balanced Acc | Accuracy | Acc (argmax) | BalAcc (argmax) |
|---|---|---|---|---|---|---|
| FKG đầy đủ (tập nền) | 1500 | 0.9031 ± 0.0058 | 0.8272 | 0.8243 | 0.7732 | 0.6761 |
| FKG đầy đủ (toàn train) | 10097 | 0.9079 ± 0.0039 | 0.8303 | 0.8188 | 0.7656 | 0.6603 |

| θ | Phương pháp | \|S\| | Rep | Tỉ lệ luật phủ (≥δ) | AUC | Accuracy | Balanced Acc | Lớp+ trong mẫu | t chọn (s) |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | GreedyFKGS | 75 | 0.8018 ± 0.0019 | 0.436 | 0.8956 ± 0.0102 | 0.8037 | 0.8210 | 0.341 | 2.169 |
| 0.05 | SFKGS | 75 | 0.8018 ± 0.0019 | 0.436 | 0.8943 ± 0.0109 | 0.8038 | 0.8194 | 0.333 | 0.690 |
| 0.05 | CFKGS | 75 | 0.7936 ± 0.0013 | 0.409 | 0.8903 ± 0.0048 | 0.8036 | 0.8124 | 0.333 | 0.062 |
| 0.05 | Random | 75 | 0.7619 ± 0.0021 | 0.274 | 0.8639 ± 0.0049 | 0.7785 | 0.7888 | 0.338 | 0.000 |
| 0.05 | StratRandom | 75 | 0.7627 ± 0.0014 | 0.277 | 0.8673 ± 0.0057 | 0.7799 | 0.7922 | 0.333 | 0.000 |
| 0.1 | GreedyFKGS | 150 | 0.8505 ± 0.0019 | 0.666 | 0.8969 ± 0.0094 | 0.8012 | 0.8201 | 0.344 | 4.435 |
| 0.1 | SFKGS | 150 | 0.8505 ± 0.0019 | 0.666 | 0.8965 ± 0.0074 | 0.8022 | 0.8203 | 0.333 | 1.606 |
| 0.1 | CFKGS | 150 | 0.8442 ± 0.0019 | 0.633 | 0.8953 ± 0.0052 | 0.8050 | 0.8151 | 0.333 | 0.072 |
| 0.1 | Random | 150 | 0.8132 ± 0.0013 | 0.458 | 0.8818 ± 0.0042 | 0.7964 | 0.8069 | 0.336 | 0.000 |
| 0.1 | StratRandom | 150 | 0.8137 ± 0.0013 | 0.461 | 0.8839 ± 0.0055 | 0.7959 | 0.8076 | 0.333 | 0.000 |
| 0.2 | GreedyFKGS | 300 | 0.8980 ± 0.0020 | 0.867 | 0.8986 ± 0.0074 | 0.8066 | 0.8210 | 0.348 | 8.504 |
| 0.2 | SFKGS | 300 | 0.8979 ± 0.0019 | 0.867 | 0.8978 ± 0.0063 | 0.8094 | 0.8204 | 0.333 | 2.804 |
| 0.2 | CFKGS | 300 | 0.8933 ± 0.0013 | 0.837 | 0.8980 ± 0.0045 | 0.8156 | 0.8204 | 0.333 | 0.086 |
| 0.2 | Random | 300 | 0.8645 ± 0.0012 | 0.676 | 0.8923 ± 0.0052 | 0.8045 | 0.8151 | 0.334 | 0.000 |
| 0.2 | StratRandom | 300 | 0.8647 ± 0.0009 | 0.678 | 0.8944 ± 0.0058 | 0.8070 | 0.8184 | 0.333 | 0.000 |
| 0.3 | GreedyFKGS | 450 | 0.9239 ± 0.0015 | 0.941 | 0.9012 ± 0.0046 | 0.8198 | 0.8238 | 0.364 | 11.610 |
| 0.3 | SFKGS | 450 | 0.9239 ± 0.0015 | 0.940 | 0.9016 ± 0.0062 | 0.8146 | 0.8223 | 0.333 | 3.796 |
| 0.3 | CFKGS | 450 | 0.9207 ± 0.0014 | 0.918 | 0.9025 ± 0.0050 | 0.8139 | 0.8243 | 0.333 | 0.100 |
| 0.3 | Random | 450 | 0.8959 ± 0.0009 | 0.796 | 0.8962 ± 0.0057 | 0.8078 | 0.8188 | 0.334 | 0.000 |
| 0.3 | StratRandom | 450 | 0.8960 ± 0.0011 | 0.797 | 0.8974 ± 0.0049 | 0.8087 | 0.8192 | 0.333 | 0.001 |
| 0.5 | GreedyFKGS | 750 | 0.9587 ± 0.0010 | 1.000 | 0.9061 ± 0.0054 | 0.8213 | 0.8304 | 0.354 | 17.004 |
| 0.5 | SFKGS | 750 | 0.9587 ± 0.0010 | 1.000 | 0.9069 ± 0.0051 | 0.8208 | 0.8324 | 0.333 | 5.570 |
| 0.5 | CFKGS | 750 | 0.9584 ± 0.0011 | 0.997 | 0.9039 ± 0.0068 | 0.8186 | 0.8269 | 0.333 | 0.127 |
| 0.5 | Random | 750 | 0.9374 ± 0.0008 | 0.912 | 0.9001 ± 0.0060 | 0.8131 | 0.8230 | 0.333 | 0.001 |
| 0.5 | StratRandom | 750 | 0.9374 ± 0.0007 | 0.913 | 0.9011 ± 0.0058 | 0.8149 | 0.8242 | 0.333 | 0.001 |

Kiểm định Wilcoxon một phía (phương pháp > đối chứng), đơn vị fold, Holm trong từng họ (độ đo × đối chứng):

| Độ đo | Đối chứng | θ | Phương pháp | Chênh TB | CI95 (t) | Fold tốt hơn | Tỉ lệ seed bị vượt | p | p Holm |
|---|---|---|---|---|---|---|---|---|---|
| rep | Random | 0.05 | GreedyFKGS | +0.0399 | [0.0388; 0.0411] | 5/5 | 1.00 | 0.0312 | 0.4688 |
| rep | Random | 0.05 | SFKGS | +0.0399 | [0.0388; 0.0411] | 5/5 | 1.00 | 0.0312 | 0.4688 |
| rep | Random | 0.05 | CFKGS | +0.0317 | [0.0292; 0.0342] | 5/5 | 1.00 | 0.0312 | 0.4688 |
| rep | Random | 0.1 | GreedyFKGS | +0.0373 | [0.0353; 0.0392] | 5/5 | 1.00 | 0.0312 | 0.4688 |
| rep | Random | 0.1 | SFKGS | +0.0372 | [0.0353; 0.0391] | 5/5 | 1.00 | 0.0312 | 0.4688 |
| rep | Random | 0.1 | CFKGS | +0.0309 | [0.0289; 0.0330] | 5/5 | 1.00 | 0.0312 | 0.4688 |
| rep | Random | 0.2 | GreedyFKGS | +0.0335 | [0.0323; 0.0347] | 5/5 | 1.00 | 0.0312 | 0.4688 |
| rep | Random | 0.2 | SFKGS | +0.0334 | [0.0323; 0.0346] | 5/5 | 1.00 | 0.0312 | 0.4688 |
| rep | Random | 0.2 | CFKGS | +0.0288 | [0.0279; 0.0297] | 5/5 | 1.00 | 0.0312 | 0.4688 |
| rep | Random | 0.3 | GreedyFKGS | +0.0280 | [0.0272; 0.0288] | 5/5 | 1.00 | 0.0312 | 0.4688 |
| rep | Random | 0.3 | SFKGS | +0.0280 | [0.0272; 0.0288] | 5/5 | 1.00 | 0.0312 | 0.4688 |
| rep | Random | 0.3 | CFKGS | +0.0248 | [0.0241; 0.0255] | 5/5 | 1.00 | 0.0312 | 0.4688 |
| rep | Random | 0.5 | GreedyFKGS | +0.0214 | [0.0205; 0.0223] | 5/5 | 1.00 | 0.0312 | 0.4688 |
| rep | Random | 0.5 | SFKGS | +0.0214 | [0.0205; 0.0223] | 5/5 | 1.00 | 0.0312 | 0.4688 |
| rep | Random | 0.5 | CFKGS | +0.0211 | [0.0200; 0.0221] | 5/5 | 1.00 | 0.0312 | 0.4688 |
| rep | StratRandom | 0.05 | GreedyFKGS | +0.0391 | [0.0381; 0.0401] | 5/5 | 1.00 | 0.0312 | 0.4375 |
| rep | StratRandom | 0.05 | SFKGS | +0.0391 | [0.0381; 0.0401] | 5/5 | 1.00 | 0.0312 | 0.4375 |
| rep | StratRandom | 0.05 | CFKGS | +0.0308 | [0.0295; 0.0321] | 5/5 | 1.00 | 0.0312 | 0.4375 |
| rep | StratRandom | 0.1 | GreedyFKGS | +0.0368 | [0.0350; 0.0386] | 5/5 | 1.00 | 0.0312 | 0.4375 |
| rep | StratRandom | 0.1 | SFKGS | +0.0368 | [0.0350; 0.0386] | 5/5 | 1.00 | 0.0211 | 0.3163 |
| rep | StratRandom | 0.1 | CFKGS | +0.0305 | [0.0292; 0.0318] | 5/5 | 1.00 | 0.0312 | 0.4375 |
| rep | StratRandom | 0.2 | GreedyFKGS | +0.0333 | [0.0316; 0.0349] | 5/5 | 1.00 | 0.0312 | 0.4375 |
| rep | StratRandom | 0.2 | SFKGS | +0.0332 | [0.0316; 0.0348] | 5/5 | 1.00 | 0.0312 | 0.4375 |
| rep | StratRandom | 0.2 | CFKGS | +0.0285 | [0.0276; 0.0295] | 5/5 | 1.00 | 0.0312 | 0.4375 |
| rep | StratRandom | 0.3 | GreedyFKGS | +0.0279 | [0.0272; 0.0286] | 5/5 | 1.00 | 0.0312 | 0.4375 |
| rep | StratRandom | 0.3 | SFKGS | +0.0279 | [0.0272; 0.0286] | 5/5 | 1.00 | 0.0312 | 0.4375 |
| rep | StratRandom | 0.3 | CFKGS | +0.0247 | [0.0239; 0.0254] | 5/5 | 1.00 | 0.0312 | 0.4375 |
| rep | StratRandom | 0.5 | GreedyFKGS | +0.0213 | [0.0205; 0.0221] | 5/5 | 1.00 | 0.0312 | 0.4375 |
| rep | StratRandom | 0.5 | SFKGS | +0.0213 | [0.0205; 0.0221] | 5/5 | 1.00 | 0.0312 | 0.4375 |
| rep | StratRandom | 0.5 | CFKGS | +0.0210 | [0.0200; 0.0220] | 5/5 | 1.00 | 0.0312 | 0.4375 |
| auc | Random | 0.05 | GreedyFKGS | +0.0317 | [0.0247; 0.0388] | 5/5 | 0.96 | 0.0312 | 0.4688 |
| auc | Random | 0.05 | SFKGS | +0.0305 | [0.0224; 0.0385] | 5/5 | 0.96 | 0.0312 | 0.4688 |
| auc | Random | 0.05 | CFKGS | +0.0264 | [0.0177; 0.0351] | 5/5 | 0.93 | 0.0312 | 0.4688 |
| auc | Random | 0.1 | GreedyFKGS | +0.0152 | [0.0067; 0.0236] | 5/5 | 0.87 | 0.0312 | 0.4688 |
| auc | Random | 0.1 | SFKGS | +0.0147 | [0.0088; 0.0205] | 5/5 | 0.86 | 0.0312 | 0.4688 |
| auc | Random | 0.1 | CFKGS | +0.0135 | [0.0102; 0.0169] | 5/5 | 0.91 | 0.0312 | 0.4688 |
| auc | Random | 0.2 | GreedyFKGS | +0.0063 | [-0.0009; 0.0136] | 4/5 | 0.75 | 0.0625 | 0.4688 |
| auc | Random | 0.2 | SFKGS | +0.0055 | [0.0001; 0.0109] | 4/5 | 0.71 | 0.0625 | 0.4688 |
| auc | Random | 0.2 | CFKGS | +0.0057 | [0.0009; 0.0105] | 5/5 | 0.71 | 0.0312 | 0.4688 |
| auc | Random | 0.3 | GreedyFKGS | +0.0051 | [0.0017; 0.0085] | 5/5 | 0.75 | 0.0312 | 0.4688 |
| auc | Random | 0.3 | SFKGS | +0.0054 | [-0.0005; 0.0113] | 4/5 | 0.75 | 0.0625 | 0.4688 |
| auc | Random | 0.3 | CFKGS | +0.0064 | [0.0014; 0.0113] | 5/5 | 0.82 | 0.0312 | 0.4688 |
| auc | Random | 0.5 | GreedyFKGS | +0.0060 | [-0.0009; 0.0129] | 5/5 | 0.85 | 0.0312 | 0.4688 |
| auc | Random | 0.5 | SFKGS | +0.0068 | [-0.0001; 0.0138] | 5/5 | 0.88 | 0.0312 | 0.4688 |
| auc | Random | 0.5 | CFKGS | +0.0038 | [-0.0032; 0.0109] | 4/5 | 0.71 | 0.0938 | 0.4688 |
| auc | StratRandom | 0.05 | GreedyFKGS | +0.0283 | [0.0226; 0.0340] | 5/5 | 0.93 | 0.0312 | 0.4688 |
| auc | StratRandom | 0.05 | SFKGS | +0.0271 | [0.0205; 0.0337] | 5/5 | 0.91 | 0.0312 | 0.4688 |
| auc | StratRandom | 0.05 | CFKGS | +0.0230 | [0.0142; 0.0318] | 5/5 | 0.87 | 0.0312 | 0.4688 |
| auc | StratRandom | 0.1 | GreedyFKGS | +0.0130 | [0.0072; 0.0189] | 5/5 | 0.84 | 0.0312 | 0.4688 |
| auc | StratRandom | 0.1 | SFKGS | +0.0125 | [0.0089; 0.0162] | 5/5 | 0.81 | 0.0312 | 0.4688 |
| auc | StratRandom | 0.1 | CFKGS | +0.0114 | [0.0054; 0.0174] | 5/5 | 0.78 | 0.0312 | 0.4688 |
| auc | StratRandom | 0.2 | GreedyFKGS | +0.0042 | [-0.0015; 0.0099] | 4/5 | 0.65 | 0.0938 | 0.4688 |
| auc | StratRandom | 0.2 | SFKGS | +0.0034 | [-0.0008; 0.0076] | 4/5 | 0.61 | 0.0625 | 0.4688 |
| auc | StratRandom | 0.2 | CFKGS | +0.0036 | [0.0002; 0.0070] | 5/5 | 0.65 | 0.0312 | 0.4688 |
| auc | StratRandom | 0.3 | GreedyFKGS | +0.0039 | [0.0012; 0.0066] | 5/5 | 0.69 | 0.0312 | 0.4688 |
| auc | StratRandom | 0.3 | SFKGS | +0.0042 | [-0.0012; 0.0096] | 4/5 | 0.72 | 0.0625 | 0.4688 |
| auc | StratRandom | 0.3 | CFKGS | +0.0051 | [0.0009; 0.0094] | 4/5 | 0.76 | 0.0625 | 0.4688 |
| auc | StratRandom | 0.5 | GreedyFKGS | +0.0051 | [-0.0018; 0.0119] | 5/5 | 0.77 | 0.0312 | 0.4688 |
| auc | StratRandom | 0.5 | SFKGS | +0.0059 | [-0.0010; 0.0127] | 5/5 | 0.82 | 0.0312 | 0.4688 |
| auc | StratRandom | 0.5 | CFKGS | +0.0029 | [-0.0041; 0.0099] | 4/5 | 0.64 | 0.2188 | 0.4688 |
| auc | FKG_full_pool | 0.05 | GreedyFKGS | -0.0074 | [-0.0136; -0.0013] | 1/5 | — | 0.9688 | 1.0000 |
| auc | FKG_full_pool | 0.05 | SFKGS | -0.0087 | [-0.0167; -0.0007] | 1/5 | — | 0.9688 | 1.0000 |
| auc | FKG_full_pool | 0.05 | CFKGS | -0.0128 | [-0.0198; -0.0057] | 0/5 | — | 1.0000 | 1.0000 |
| auc | FKG_full_pool | 0.1 | GreedyFKGS | -0.0061 | [-0.0123; 0.0001] | 1/5 | — | 0.9688 | 1.0000 |
| auc | FKG_full_pool | 0.1 | SFKGS | -0.0066 | [-0.0100; -0.0032] | 0/5 | — | 1.0000 | 1.0000 |
| auc | FKG_full_pool | 0.1 | CFKGS | -0.0077 | [-0.0132; -0.0022] | 0/5 | — | 1.0000 | 1.0000 |
| auc | FKG_full_pool | 0.2 | GreedyFKGS | -0.0044 | [-0.0093; 0.0004] | 1/5 | — | 0.9688 | 1.0000 |
| auc | FKG_full_pool | 0.2 | SFKGS | -0.0052 | [-0.0084; -0.0021] | 0/5 | — | 1.0000 | 1.0000 |
| auc | FKG_full_pool | 0.2 | CFKGS | -0.0050 | [-0.0087; -0.0014] | 0/5 | — | 1.0000 | 1.0000 |
| auc | FKG_full_pool | 0.3 | GreedyFKGS | -0.0018 | [-0.0047; 0.0011] | 1/5 | — | 0.9375 | 1.0000 |
| auc | FKG_full_pool | 0.3 | SFKGS | -0.0015 | [-0.0067; 0.0038] | 2/5 | — | 0.7812 | 1.0000 |
| auc | FKG_full_pool | 0.3 | CFKGS | -0.0005 | [-0.0049; 0.0038] | 2/5 | — | 0.5938 | 1.0000 |
| auc | FKG_full_pool | 0.5 | GreedyFKGS | +0.0031 | [-0.0034; 0.0095] | 4/5 | — | 0.1562 | 1.0000 |
| auc | FKG_full_pool | 0.5 | SFKGS | +0.0039 | [-0.0025; 0.0103] | 5/5 | — | 0.0312 | 0.4688 |
| auc | FKG_full_pool | 0.5 | CFKGS | +0.0009 | [-0.0057; 0.0075] | 3/5 | — | 0.3125 | 1.0000 |

![E2](figures/E2.png)

## E3 — Đường cong theo θ, chọn θ* trên validation (RQ3)

θ* = điểm gối trên mặt Pareto (chi phí = chọn luật + xây FISA + suy diễn; lợi ích = AUC validation). Test chỉ dùng để đánh giá tại θ*. Ngưỡng Rep trung bình = 0.85; phủ toàn phần = mọi luật có cov ≥ 0.85.

| Phương pháp | θ* theo fold | \|S\| TB tại θ* | AUC test tại θ* | Acc / BalAcc test tại θ* | AUC test FKG đầy đủ | Acc / BalAcc FKG đầy đủ | θ nhỏ nhất Rep TB ≥ ngưỡng | θ nhỏ nhất phủ toàn phần |
|---|---|---|---|---|---|---|---|---|
| GreedyFKGS | 0.4, 0.4, 0.05, 0.15, 0.05 | 315 | 0.8997 ± 0.0100 | 0.8071 / 0.8233 | 0.9031 | 0.8243 / 0.8272 | 0.15, 0.1, 0.1, 0.15, 0.1 | 0.4, 0.4, 0.4, 0.4, 0.4 |
| SFKGS | 0.4, 0.4, 0.05, 0.2, 0.05 | 330 | 0.8967 ± 0.0089 | 0.8068 / 0.8205 | 0.9031 | 0.8243 / 0.8272 | 0.15, 0.1, 0.1, 0.15, 0.1 | 0.5, 0.5, 0.5, 0.5, 0.5 |
| CFKGS | 0.5, 0.7, 0.3, 0.5, 1.0 | 900 | 0.9033 ± 0.0081 | 0.8193 / 0.8243 | 0.9031 | 0.8243 / 0.8272 | 0.15, 0.15, 0.15, 0.15, 0.15 | 0.6, 0.5, 0.6, 0.6, 0.5 |

![E3](figures/E3.png)

## E4 — So sánh các chiến lược tại θ = 0.2, cùng ngân sách (RQ4)

| Phương pháp | \|S\| | Rep TB ± ĐLC | Tỉ lệ luật phủ | Lớp+ | Thời gian (ms) |
|---|---|---|---|---|---|
| GreedyFKGS | 100 | 0.8544 ± 0.0031 | 0.621 | 0.341 | 230.8 |
| Random | 100 | 0.8159 ± 0.0047 | 0.442 | 0.345 | 0.1 |
| SFKGS | 100 | 0.8542 ± 0.0032 | 0.620 | 0.330 | 22.9 |
| CFKGS | 100 | 0.8468 ± 0.0036 | 0.585 | 0.330 | 14.7 |
| StratRandom | 100 | 0.8149 ± 0.0045 | 0.437 | 0.330 | 0.2 |

**Rep, 5 block = trung bình fold (chính):** Friedman χ²=19.360, p=6.68e-04, CD=2.728 (5 block). Hạng: GreedyFKGS=1.00, Random=4.20, SFKGS=2.00, CFKGS=3.00, StratRandom=4.80. Cặp khác biệt: GreedyFKGS–Random, GreedyFKGS–StratRandom, SFKGS–StratRandom

**Rep, 50 block (mô tả, có điều kiện trên fold):** Friedman χ²=187.942, p=1.47e-39, CD=0.863 (50 block). Hạng: GreedyFKGS=1.32, Random=4.42, SFKGS=1.68, CFKGS=3.00, StratRandom=4.58. Cặp khác biệt: GreedyFKGS–Random, GreedyFKGS–CFKGS, GreedyFKGS–StratRandom, Random–SFKGS, Random–CFKGS, SFKGS–CFKGS, SFKGS–StratRandom, CFKGS–StratRandom

**Thời gian, trung bình fold:** Friedman χ²=20.000, p=4.99e-04, CD=2.728 (5 block). Hạng: GreedyFKGS=5.00, Random=1.00, SFKGS=4.00, CFKGS=3.00, StratRandom=2.00. Cặp khác biệt: GreedyFKGS–Random, GreedyFKGS–StratRandom, Random–SFKGS

![E4](figures/E4.png)

## PUB — Giao thức của bài công bố: chia cố định 70/30, FKG đầy đủ

Độ chính xác đã công bố của FKG đầy đủ: 0.7613. Ngưỡng hiệu chỉnh được chọn trên tập xác thực tách 20% từ tập huấn luyện; tập kiểm tra 30% không tham gia chọn ngưỡng.

| FISA | Mờ hoá | Số seed | AUC | Acc argmax | BalAcc argmax | Acc (D0>9·D1) | Acc hiệu chỉnh (tiêu chí Acc) | BalAcc hiệu chỉnh (tiêu chí BalAcc) | Độ nhạy / Độ đặc hiệu (hiệu chỉnh BalAcc) |
|---|---|---|---|---|---|---|---|---|---|
| FKG-Pairs k=3 | q10/q90 | 1 | 0.6837 | 0.6648 ± 0.0000 | 0.5005 | 0.3390 | 0.6564 ± 0.0000 | 0.6322 | 0.895 / 0.369 |
| FKG-Pairs k=1 | tam phân vị | 5 | 0.8556 | 0.6649 ± 0.0006 | 0.5006 | 0.4597 | 0.8016 ± 0.0153 | 0.7777 | 0.824 / 0.732 |
| FKG-Pairs k=3 | tam phân vị | 5 | 0.8791 | 0.7569 ± 0.0081 | 0.6539 | 0.5573 | 0.8021 ± 0.0064 | 0.8063 | 0.848 / 0.765 |
| Notebook gốc (k=3, nguyên trạng) | tam phân vị | 5 | 0.8791 | 0.7569 ± 0.0081 | 0.6539 | 0.5573 | 0.8021 ± 0.0064 | 0.8063 | 0.848 / 0.765 |
