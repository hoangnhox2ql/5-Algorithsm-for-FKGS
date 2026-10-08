"""
data/brset_folds.py — Nạp cơ sở luật FRB của BRSET đã chia 5 fold theo ID bệnh nhân.

Cấu trúc thư mục (đúng như bộ dữ liệu được cung cấp):
    BRSET_Data/fold_01/TrainDataRule.csv, TestDataRule.csv, ..., fold_05/...
Mỗi tệp CSV: dòng tiêu đề "0,1,...,16"; cột 0..15 là nhãn ngôn ngữ (mã 1..5) của 16 thuộc tính hợp nhất
(ảnh + lâm sàng), cột cuối là nhãn chẩn đoán (1 = không DR, 2 = DR). Mỗi dòng TrainDataRule là một luật
của FRB (luật sinh từ mẫu, có đủ mọi thuộc tính, đúng yêu cầu của FKG-Pairs); TestDataRule là mẫu kiểm tra.

Chuyển đổi sang định dạng của bộ thực nghiệm:
  * token tiền đề "F{j:02d}-L{v}" (thuộc tính j, mức v), token nhãn "class-{nhãn}";
  * luật: antecedent_tokens (16 token), consequent_token, id = chỉ số dòng, group = mã của bộ giá trị
    (các dòng trùng nhau — do cân bằng lớp bằng lặp mẫu — có cùng group);
  * mẫu: membership one-hot (độ thuộc 1 cho mức đã cho), label; mẫu huấn luyện mang thêm rule_id và group
    để mô hình loại luật của chính mẫu (và các bản sao của nó) khi tính mất mát — tránh rò rỉ nhãn.
Dữ liệu chỉ có mức ngôn ngữ rời rạc, nên suy diễn là trường hợp Crisp của suy diễn mờ (Mệnh đề 3.fisamm(i)).
"""
import os
import csv
from data.fkg_io import FKGRuleBase


def _read(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], [r for r in rows[1:] if r and any(x.strip() for x in r)]
    return header, [[int(float(x)) for x in r] for r in body]


def list_folds(root):
    return sorted(d for d in os.listdir(root) if d.startswith("fold_") and os.path.isdir(os.path.join(root, d)))


def load_fold(root, fold, n_levels=5):
    fdir = os.path.join(root, fold)
    h_tr, train = _read(os.path.join(fdir, "TrainDataRule.csv"))
    h_te, test = _read(os.path.join(fdir, "TestDataRule.csv"))
    if len(h_tr) != len(h_te):
        raise ValueError(f"{fold}: số cột Train ({len(h_tr)}) khác Test ({len(h_te)})")
    m = len(h_tr) - 1
    for name, rows in [("Train", train), ("Test", test)]:
        bad = [r for r in rows if len(r) != m + 1]
        if bad:
            raise ValueError(f"{fold}/{name}: {len(bad)} dòng sai số cột")
        vals = {v for r in rows for v in r[:-1]}
        if not vals <= set(range(1, n_levels + 1)):
            raise ValueError(f"{fold}/{name}: mã mức ngoài 1..{n_levels}: {sorted(vals - set(range(1, n_levels + 1)))}")
    attrs = [f"F{j:02d}" for j in range(m)]
    labels = sorted({r[-1] for r in train} | {r[-1] for r in test})
    vocab = [f"{a}-L{v}" for a in attrs for v in range(1, n_levels + 1)] + [f"class-{c}" for c in labels]
    sig = {}
    rules, tr_samples = [], []
    for i, r in enumerate(train):
        g = sig.setdefault(tuple(r), len(sig))
        toks = [f"{attrs[j]}-L{r[j]}" for j in range(m)]
        rules.append({"id": i, "antecedent_tokens": toks, "consequent_token": f"class-{r[-1]}",
                      "support": 1.0 / len(train), "confidence": 1.0, "group": g})
        tr_samples.append({"membership": {t: 1.0 for t in toks}, "label": f"class-{r[-1]}", "rule_id": i, "group": g})
    te_samples = [{"membership": {f"{attrs[j]}-L{r[j]}": 1.0 for j in range(m)}, "label": f"class-{r[-1]}"} for r in test]
    fkg = FKGRuleBase(vocab, rules, [], {"dataset": f"BRSET {fold}", "n_classes": len(labels),
                                          "class_names": [f"class-{c}" for c in labels]})
    info = {"fold": fold, "n_attr": m, "n_train": len(train), "n_test": len(test), "n_groups": len(sig),
            "n_duplicate_rows": len(train) - len(sig),
            "train_pos": sum(r[-1] == labels[-1] for r in train), "test_pos": sum(r[-1] == labels[-1] for r in test)}
    return fkg, tr_samples, te_samples, info


def subset_rulebase(fkg, keep_ids):
    keep = set(keep_ids)
    return FKGRuleBase(fkg.vocab, [r for r in fkg.rules if r["id"] in keep], [], dict(fkg.meta))
