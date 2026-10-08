"""
data/fkg_io.py — Định dạng dữ liệu chuẩn cho một cơ sở luật FKG/FKG-MM/FKGS,
và bộ nạp/lưu tương ứng. Cũng cung cấp bộ SINH DỮ LIỆU TỔNG HỢP (synthetic)
để kiểm thử toàn bộ pipeline (KB1-KB6, ablation, baseline) trước khi có dữ
liệu BRSET/Diabetes thật.

ĐỊNH DẠNG FILE LUẬT (JSON), khớp Định nghĩa 3.2 (Chương 3):
{
  "meta": {"dataset": "BRSET", "n_classes": 2, "class_names": ["No-DR","DR"]},
  "vocab": ["Age-Low", "Age-Medium", "Age-High", "HbA1c-Low", ..., "class-DR"],
  "rules": [
      {
        "id": 0,
        "antecedent_tokens": ["Age-High", "HbA1c-High", "LesionArea-High"],
        "consequent_token": "class-DR",
        "confidence": 0.87,
        "support": 0.12
      },
      ...
  ],
  "edges": [   # dùng cho Node Loss (mu_ij, Định nghĩa 3.2.2 / Mục 3.3.4)
      {"u": "Age-High", "v": "HbA1c-High", "mu": 0.63},
      ...
  ]
}

ĐỊNH DẠNG FILE TEST (JSON):
{
  "samples": [
      {
        "membership": {"Age-High": 0.8, "Age-Medium": 0.2, "HbA1c-High": 0.9, ...},
        "label": "class-DR"
      },
      ...
  ]
}
"""
import json
import os
import random
import math


class FKGRuleBase:
    """Cấu trúc dữ liệu trong bộ nhớ cho một cơ sở luật FKG đã khai phá."""

    def __init__(self, vocab, rules, edges, meta=None):
        self.vocab = list(vocab)                       # danh sách token (str)
        self.token2idx = {t: i for i, t in enumerate(self.vocab)}
        self.rules = rules                              # list[dict]
        self.edges = edges                               # list[dict {u,v,mu}]
        self.meta = meta or {}
        self.class_tokens = sorted({r["consequent_token"] for r in rules})

    def __len__(self):
        return len(self.rules)

    def n_tokens(self):
        return len(self.vocab)

    @staticmethod
    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return FKGRuleBase(d["vocab"], d["rules"], d.get("edges", []), d.get("meta", {}))

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "meta": self.meta, "vocab": self.vocab,
                "rules": self.rules, "edges": self.edges,
            }, f, ensure_ascii=False, indent=2)

    def sample_subset(self, ratio, seed=0):
        """Lấy mẫu ngẫu nhiên ratio% số luật — dùng mô phỏng KB6 khi chưa có
        đủ 5 mức nén thật từ FKGS Chương 2."""
        rng = random.Random(seed)
        n_keep = max(1, int(math.ceil(len(self.rules) * ratio)))
        kept = rng.sample(self.rules, n_keep)
        used_tokens = set()
        for r in kept:
            used_tokens.update(r["antecedent_tokens"])
            used_tokens.add(r["consequent_token"])
        new_vocab = [t for t in self.vocab if t in used_tokens]
        new_edges = [e for e in self.edges if e["u"] in used_tokens and e["v"] in used_tokens]
        return FKGRuleBase(new_vocab, kept, new_edges, dict(self.meta))


def load_test_samples(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return d["samples"]


def save_test_samples(samples, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"samples": samples}, f, ensure_ascii=False, indent=2)


# ============================================================
# BỘ SINH DỮ LIỆU TỔNG HỢP — dùng để kiểm thử toàn bộ pipeline
# ============================================================
def generate_synthetic_fkg(n_attrs=6, n_labels_per_attr=3, n_rules=200,
                            n_classes=2, seed=42, dataset_name="SYNTH"):
    """Sinh một FKG tổng hợp có cấu trúc THỰC (không random thuần túy):
    mỗi thuộc tính có n_labels_per_attr nhãn ngôn ngữ; luật được sinh sao
    cho tồn tại một mối liên hệ thống kê thật giữa tổ hợp tiền đề và nhãn
    lớp (để accuracy suy luận có ý nghĩa kiểm thử, không phải nhiễu ngẫu
    nhiên hoàn toàn).
    """
    rng = random.Random(seed)
    attrs = [f"A{i}" for i in range(n_attrs)]
    levels = ["Low", "Medium", "High"][:n_labels_per_attr]
    vocab = [f"{a}-{l}" for a in attrs for l in levels]
    class_tokens = [f"class-{c}" for c in range(n_classes)]
    vocab = vocab + class_tokens

    # Quy luật ẩn thật: nếu đa số thuộc tính "High" -> class-1, ngược lại class-0
    # (n_classes=2 mặc định); tổng quát hoá cho n_classes>2 bằng chia đều.
    rules = []
    for rid in range(n_rules):
        # Mỗi luật sinh từ một mẫu: có đủ một nhãn ngôn ngữ cho MỌI thuộc tính (yêu cầu của
        # mô hình FKG-Pairs mà FISA trong FKGS v3.2 cài đặt).
        n_ante = n_attrs
        chosen_attrs = list(attrs)
        ante_tokens = []
        high_count = 0
        for a in chosen_attrs:
            lvl = rng.choice(levels)
            if lvl == "High":
                high_count += 1
            ante_tokens.append(f"{a}-{lvl}")
        # Quy luật ẩn + nhiễu 15% để không "quá dễ" (giống dữ liệu thật có nhiễu)
        ratio = high_count / n_ante
        true_class = min(n_classes - 1, int(ratio * n_classes))
        if rng.random() < 0.15:
            true_class = rng.randrange(n_classes)
        conf = round(0.6 + 0.35 * rng.random(), 3)
        supp = round(0.02 + 0.20 * rng.random(), 3)
        rules.append({
            "id": rid,
            "antecedent_tokens": ante_tokens,
            "consequent_token": f"class-{true_class}",
            "confidence": conf,
            "support": supp,
        })

    # Cạnh đồng xuất hiện mu_ij: ước lượng tần suất đồng xuất hiện trong luật
    from collections import defaultdict
    co = defaultdict(int)
    solo = defaultdict(int)
    for r in rules:
        toks = r["antecedent_tokens"]
        for t in toks:
            solo[t] += 1
        for i in range(len(toks)):
            for j in range(len(toks)):
                if i != j:
                    co[(toks[i], toks[j])] += 1
    edges = []
    for (u, v), c in co.items():
        mu = c / (solo[u] + 1e-9)
        edges.append({"u": u, "v": v, "mu": round(min(mu, 1.0), 4)})

    meta = {"dataset": dataset_name, "n_classes": n_classes,
            "class_names": class_tokens}
    return FKGRuleBase(vocab, rules, edges, meta)


def generate_synthetic_test_samples(fkg: FKGRuleBase, n_samples=300, seed=123):
    """Sinh mẫu test có nhãn thật NHẤT QUÁN với quy luật ẩn đã dùng để sinh
    luật ở generate_synthetic_fkg (đếm số thuộc tính 'High' được mờ hoá)."""
    rng = random.Random(seed)
    attrs = sorted({t.rsplit("-", 1)[0] for t in fkg.vocab if not t.startswith("class-")})
    levels = ["Low", "Medium", "High"]
    n_classes = fkg.meta.get("n_classes", 2)
    samples = []
    for _ in range(n_samples):
        membership = {}
        high_count = 0
        for a in attrs:
            # random.dirichlet-lite: gán độ thuộc mờ cho 3 mức, tổng ~ không cần = 1
            raw = [rng.random() for _ in levels]
            s = sum(raw)
            degs = [x / s for x in raw]
            dominant = levels[degs.index(max(degs))]
            if dominant == "High":
                high_count += 1
            for lvl, d in zip(levels, degs):
                tok = f"{a}-{lvl}"
                if tok in fkg.token2idx:
                    membership[tok] = round(d, 4)
        ratio = high_count / max(1, len(attrs))
        true_class = min(n_classes - 1, int(ratio * n_classes))
        if rng.random() < 0.15:
            true_class = rng.randrange(n_classes)
        samples.append({"membership": membership, "label": f"class-{true_class}"})
    return samples


if __name__ == "__main__":
    # Tự kiểm tra nhanh: sinh + lưu + nạp lại
    fkg = generate_synthetic_fkg()
    print(f"Sinh FKG tổng hợp: {len(fkg)} luật, {fkg.n_tokens()} token, "
          f"{len(fkg.edges)} cạnh")
    samples = generate_synthetic_test_samples(fkg, n_samples=50)
    print(f"Sinh {len(samples)} mẫu test")
    fkg.save("/tmp/test_fkg.json")
    fkg2 = FKGRuleBase.load("/tmp/test_fkg.json")
    assert len(fkg2) == len(fkg), "Lỗi round-trip save/load!"
    print("Round-trip save/load: OK")
