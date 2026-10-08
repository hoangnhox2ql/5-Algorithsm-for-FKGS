"""
tests/test_brset_folds.py — Kiểm tra bộ nạp FRB BRSET 5 fold và cơ chế chống rò rỉ khi luật sinh từ chính mẫu.
Bỏ qua nếu chưa có data_real/BRSET_Data.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest
import config as C
from data.brset_folds import list_folds, load_fold, subset_rulebase
from models.fisa import FISA
from models.fkge import FKGE
from experiments.brset_kfold import group_partition

ROOT = os.path.join(C.PATHS.DATA_DIR, "BRSET_Data")
pytestmark = pytest.mark.skipif(not os.path.isdir(ROOT), reason="chưa có data_real/BRSET_Data")


def test_all_folds_load_and_are_consistent():
    folds = list_folds(ROOT)
    assert len(folds) == 5
    for f in folds:
        fkg, tr, te, info = load_fold(ROOT, f)
        assert info["n_attr"] == 16 and len(fkg.class_tokens) == 2
        assert all(len(r["antecedent_tokens"]) == 16 for r in fkg.rules)       # luật đầy đủ thuộc tính
        assert all(t in fkg.token2idx for s in te for t in s["membership"])
        assert info["train_pos"] * 2 == info["n_train"]                         # tập huấn luyện đã cân bằng


def test_group_partition_keeps_duplicates_together():
    fkg, tr, te, info = load_fold(ROOT, list_folds(ROOT)[0])
    g = np.array([s["group"] for s in tr]); parts = group_partition(g, 5, 0)
    for grp in np.unique(g):
        assert len(set(parts[g == grp])) == 1


def test_mask_excludes_own_rule_and_duplicates():
    fkg, tr, te, info = load_fold(ROOT, list_folds(ROOT)[0])
    m = FKGE(fkg, d_e=4, T_ep=1)
    G = m._mask(tr[:50])
    for i, s in enumerate(tr[:50]):
        own = [k for k, R in enumerate(fkg.rules) if R["group"] == s["group"]]
        assert set(np.where(G[i])[0]) == set(own) and s["rule_id"] in own
    assert m._mask(te[:10]) is None                                            # mẫu kiểm tra không bị che


def test_fisa_runs_on_fold_and_subset():
    fkg, tr, te, info = load_fold(ROOT, list_folds(ROOT)[0])
    f = FISA(fkg, decision="argmax").fit()
    r = f.evaluate(te[:30])
    assert r["n_unseen"] == 0 and 0 <= r["accuracy"] <= 1
    sub = subset_rulebase(fkg, [R["id"] for R in fkg.rules[:500]])
    a = FISA(sub, decision="argmax").fit().attribution(te[0]["membership"])
    assert abs(sum(a.values()) - 1) < 1e-9 and all(k < 500 for k in a)


def test_end_to_end_one_fold_smoke():
    """Chạy trọn quy trình của experiments/brset_kfold.py trên fold đầu (T_ep = 1) và kiểm tra đầu ra."""
    from experiments.brset_kfold import run_fold
    cfg = {"F_c": 3, "val_fraction": 0.2, "seed": 1, "n_seq_timing": 5, "T_ep": 1}
    out = run_fold(ROOT, list_folds(ROOT)[0], cfg)
    for model in ("FISA", "FKGE"):
        r = out[model]
        assert 0.0 <= r["auc"] <= 1.0 and 0.0 <= r["balanced_accuracy"] <= 1.0
        assert r["tp"] + r["fn"] == out["info"]["test_pos"]
        assert r["tp"] + r["tn"] + r["fp"] + r["fn"] == out["info"]["n_test"]
    assert 0.0 <= out["FKGE"]["dev_rule"] <= 1.0
    assert out["FKGE"]["dev_rule"] >= 1 - out["FKGE"]["agreement_fisa"] - 1e-9
