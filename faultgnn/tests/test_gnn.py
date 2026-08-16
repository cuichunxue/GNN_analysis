# -*- coding: utf-8 -*-
"""PyTorch Geometric GNNモジュールの回帰テスト。

torch / torch_geometric が未インストールの環境（既定のrequirements.txtのみ）でも
メインのテストスイートが壊れないよう、importorskip でこのファイル全体をスキップする。
"""
from argparse import Namespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from gnn_data import (  # noqa: E402
    ATTR_VALUE,
    FAULT,
    HAS_ATTR,
    REV_HAS_ATTR,
    SIMILAR_TO,
    attach_fault_similarity_edges,
    build_bipartite_hetero_data,
    split_cause_labels,
)
from gnn_model import HeteroGNNEncoder  # noqa: E402
from train_gnn import run_link_prediction, run_node_classification  # noqa: E402


def make_args(**overrides):
    defaults = dict(
        hidden_dim=32,
        num_layers=2,
        dropout=0.3,
        epochs=60,
        lr=0.01,
        weight_decay=5e-4,
        patience=25,
        val_size=0.15,
        min_shared_attrs=3,
        neg_ratio=1.0,
        link_decoder="dot",
        precision_k=10,
        seed=0,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


# ----------------------------------------------------------------- グラフ構築
def test_bipartite_graph_shapes(analyzer):
    data = build_bipartite_hetero_data(analyzer)
    assert data[FAULT].num_nodes == len(analyzer.df)
    assert data[ATTR_VALUE].num_nodes == len(analyzer.vocab)
    assert data[HAS_ATTR].edge_index.shape[1] == analyzer.X_raw.nnz
    assert data[REV_HAS_ATTR].edge_index.shape[1] == analyzer.X_raw.nnz


def test_cause_never_appears_in_the_graph(analyzer):
    """原因列がノード/エッジ構造に混ざるとラベルリークになる。"""
    assert not any(v.startswith(f"{analyzer.cause_column}::") for v in analyzer.vocab)
    data = build_bipartite_hetero_data(analyzer)
    assert data[ATTR_VALUE].num_nodes == len(analyzer.vocab)


def test_similarity_edges_are_symmetric_with_no_self_loops(analyzer):
    data = build_bipartite_hetero_data(analyzer)
    data, diag = attach_fault_similarity_edges(data, analyzer, min_shared_attrs=3)
    ei = data[SIMILAR_TO].edge_index
    pairs = set(map(tuple, ei.t().tolist()))
    assert all(a != b for a, b in pairs)
    assert all((b, a) in pairs for a, b in pairs)
    assert diag["n_positive_pairs"] > 0
    assert 0.0 < diag["density"] < 1.0


def test_similarity_edges_reject_out_of_range_threshold(analyzer):
    data = build_bipartite_hetero_data(analyzer)
    with pytest.raises(ValueError):
        attach_fault_similarity_edges(data, analyzer, min_shared_attrs=1)
    with pytest.raises(ValueError):
        attach_fault_similarity_edges(data, analyzer, min_shared_attrs=len(analyzer.attr_columns) + 1)


# ----------------------------------------------------------------- 分割
def test_cause_split_matches_baseline_test_set(analyzer):
    train_idx, val_idx, test_idx, y_full, classes = split_cause_labels(analyzer, random_state=0)
    baseline = analyzer.evaluate_cause_model(random_state=0)
    assert len(test_idx) == baseline["n_test"]
    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0

    train2, val2, test2, _, _ = split_cause_labels(analyzer, random_state=0)
    assert np.array_equal(np.sort(train_idx), np.sort(train2))
    assert np.array_equal(np.sort(test_idx), np.sort(test2))


# ----------------------------------------------------------------- エンコーダ
def test_encoder_output_is_unaffected_by_similarity_edges(analyzer):
    """メッセージパッシングは has_attr/rev_has_attr のみを使う（similar_to は無視される）ことを、
    PyGの内部実装ではなく実際の出力の一致で検証する。"""
    data = build_bipartite_hetero_data(analyzer)
    with_sim, _ = attach_fault_similarity_edges(data, analyzer, min_shared_attrs=3)

    torch.manual_seed(0)
    enc = HeteroGNNEncoder(num_attr_values=len(analyzer.vocab), hidden_dim=16, num_layers=2, dropout=0.0)
    enc.eval()
    with torch.no_grad():
        out_without = enc(data)[FAULT].clone()
        out_with = enc(with_sim)[FAULT]
    assert torch.equal(out_without, out_with)


def test_encoder_rejects_single_layer():
    with pytest.raises(ValueError):
        HeteroGNNEncoder(num_attr_values=10, num_layers=1)


# ----------------------------------------------------------------- ①ノード分類
def test_node_classification_beats_majority_baseline(analyzer):
    baseline = analyzer.evaluate_cause_model(random_state=0)
    result = run_node_classification(analyzer, make_args())
    assert result["top1"] > baseline["rows"][0]["top1"]
    assert result["top1"] <= result["top3"] <= 1.0
    assert result["n_train"] + result["n_val"] + result["n_test"] == baseline["n_train"] + baseline["n_test"]


def test_node_classification_is_deterministic(analyzer):
    args = make_args(dropout=0.0, epochs=30)
    r1 = run_node_classification(analyzer, args)
    r2 = run_node_classification(analyzer, args)
    assert r1["top1"] == r2["top1"] and r1["top3"] == r2["top3"]


# ----------------------------------------------------------------- ②リンク予測
def test_link_prediction_auc_well_above_chance(analyzer):
    result = run_link_prediction(analyzer, make_args())
    assert result["auc"] > 0.65
    assert 0.0 <= result["ap"] <= 1.0
    assert result["n_positive_pairs"] > 0
