"""
tests/test_real_pipeline.py — Kiểm tra experiments/common.py: nguồn dữ liệu chung của KB1–KB6/Ablation/Baseline
trên BRSET thật (5 fold), huấn luyện song song cho kết quả trùng tuần tự, dùng lại mô hình, fold con (mô phỏng FKGS),
tổng hợp giữa các fold và quay về dữ liệu tổng hợp khi không có dữ liệu thật.
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pytest
import config as C
from experiments import common
from models.kge_baselines import Node2VecLite

REAL = os.path.isdir(C.PATHS.BRSET_FOLDS_DIR)
needs_real = pytest.mark.skipif(not REAL, reason="chưa có data_real/BRSET_Data")
QUALITY = ["accuracy", "balanced_accuracy", "auc", "log_loss", "agreement_fisa", "dev_rule", "sensitivity", "specificity"]


@pytest.fixture
def one_fold(monkeypatch):
    monkeypatch.setattr(C.EVAL, "REAL_FOLDS", ["fold_01"])
    monkeypatch.setattr(C.FKGE, "T_ep", 2)
    monkeypatch.setattr(common, "_STATES", {})
    return common.brset_folds()[0]


@needs_real
def test_brset_folds_are_real_and_cached():
    assert common.using_real_data()
    folds = common.brset_folds()
    assert [f.name for f in folds] == common.real_fold_names() and all(f.kind == "real" for f in folds)
    assert all(a is b for a, b in zip(folds, common.brset_folds()))          # chuẩn bị một lần mỗi tiến trình
    f = folds[0]
    assert f.fisa.threshold is not None and f.n_rules == f.info["n_train"]
    tr, va = f.fit_kwargs["train_samples"], f.fit_kwargs["val_samples"]
    assert not {s["group"] for s in tr} & {s["group"] for s in va}           # tập xác thực tách theo nhóm


@needs_real
def test_other_datasets_not_synthesized_when_real(tmp_path):
    assert common.other_dataset_folds("X", str(tmp_path / "a.json"), str(tmp_path / "b.json"), 1) == []


@needs_real
def test_parallel_equals_sequential_and_reuse(one_fold, monkeypatch):
    tasks = [(one_fold, 42, None), (one_fold, 43, {"d_e": 8})]
    monkeypatch.setattr(C.EVAL, "N_JOBS", 1)
    seq = common.run_fkge(tasks, "test")
    monkeypatch.setattr(common, "_STATES", {})
    monkeypatch.setattr(C.EVAL, "N_JOBS", 2)
    par = common.run_fkge(tasks, "test")
    for a, b in zip(seq, par):
        for k in QUALITY:
            assert a[k] == pytest.approx(b[k], abs=1e-12), k
    n_before = len(common._STATES)
    again = common.run_fkge([(one_fold, 42, {"d_e": C.FKGE.d_e, "K": C.FKGE.K})], "test")   # trùng mặc định
    assert len(common._STATES) == n_before
    for k in QUALITY:
        assert again[0][k] == pytest.approx(seq[0][k], abs=1e-12)


@needs_real
def test_rule_subset_fold(one_fold):
    sub = common.brset_folds(rule_ratio=0.2, rule_seed=7)[0]
    assert sub is not one_fold and sub.info["rules"] == "random_0.2"
    assert sub.n_rules == math.ceil(0.2 * one_fold.n_rules)
    assert len(sub.train) == len(one_fold.train)                              # mẫu huấn luyện giữ nguyên
    assert sub.fit_kwargs["teacher_data"]["eF"].shape[1] == sub.n_rules
    m = common.run_fkge([(sub, 42, None)], "test")[0]
    assert 0 <= m["balanced_accuracy"] <= 1 and 0 <= m["dev_rule"] <= 1


@needs_real
def test_node2vec_graph_from_rules(one_fold):
    n2v = Node2VecLite(one_fold.fkg, d=4, epochs=1)
    assert one_fold.fkg.edges == [] and len(n2v.adj) > 0
    assert all(len(v) == len(set(v)) for v in n2v.adj.values())


def test_aggregate_folds_then_seeds():
    runs = [("a", {"x": 1.0}), ("a", {"x": 3.0}), ("b", {"x": 6.0})]
    A = common.aggregate(runs, ["x"])
    assert A["x_mean"] == pytest.approx(4.0) and A["x_std"] == pytest.approx(np.std([2.0, 6.0], ddof=1))
    B = common.aggregate([("a", {"x": 1.0}), ("a", {"x": 3.0})], ["x"])
    assert B["x_mean"] == pytest.approx(2.0) and B["x_std"] == pytest.approx(1.0)


def test_falls_back_to_synthetic_without_real_data(monkeypatch, tmp_path):
    monkeypatch.setattr(C.PATHS, "BRSET_FOLDS_DIR", str(tmp_path / "none"))
    monkeypatch.setattr(C.PATHS, "BRSET_RULES_FILE", str(tmp_path / "none.json"))
    folds = common.brset_folds()
    assert len(folds) == 1 and folds[0].kind == "synthetic"
    assert common.seeds_for(folds, 3) == 3 and "TỔNG HỢP" in common.describe_source()
