# -*- coding: utf-8 -*-
"""PyTorch Geometric 用のグラフ構築・分割ロジック。

`graph_engine.FaultAnalyzer` が持つ Fault×属性値 incidence 行列を、そのまま
`HeteroData` に写像する。以下の設計上の制約は `graph_engine.py` と揃えてある:

  - `attr_value` ノードは `analyzer.vocab`（原因列を含まない）からのみ構築するため、
    原因ラベルがメッセージパッシングに混ざることは構造的にあり得ない（ラベルリーク防止）。
  - `fault` ノードには学習可能な埋め込みテーブルを持たせない。リンク予測の教師信号が
    Fault-Fault ペアであるため、Faultごとに自由なパラメータを与えると「構造から学習する」
    のではなく「ペアを丸暗記する」動作になってしまう。`fault` の初期特徴はゼロベクトルとし、
    表現は近傍（attr_value）からのメッセージパッシングのみで決まるようにする。

Fault-Fault の類似度エッジ（`similar_to`）はリンク予測の**教師信号としてのみ**使い、
メッセージパッシングには使わない（`HeteroGNNEncoder` はこの edge type を参照しない）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

FAULT = "fault"
ATTR_VALUE = "attr_value"
HAS_ATTR = ("fault", "has_attr", "attr_value")
REV_HAS_ATTR = ("attr_value", "rev_has_attr", "fault")
SIMILAR_TO = ("fault", "similar_to", "fault")


def build_bipartite_hetero_data(analyzer) -> HeteroData:
    """`analyzer.X_raw`（Fault×属性値 incidence）から二部グラフを構築する。

    ノードID:
      - 'fault' のインデックス == `analyzer.df` の行位置（== `analyzer.fault_ids` の順序）
      - 'attr_value' のインデックス == `analyzer.vocab_index` の順序
    原因列には一切触れない（`analyzer.vocab` が既に原因列を除外して構築されているため）。
    """
    data = HeteroData()
    data[FAULT].num_nodes = len(analyzer.df)
    data[ATTR_VALUE].num_nodes = len(analyzer.vocab)

    coo = analyzer.X_raw.tocoo()
    edge_index = torch.as_tensor(np.stack([coo.row, coo.col]), dtype=torch.long)
    data[HAS_ATTR].edge_index = edge_index

    # has_attr / rev_has_attr は2つのノード型をまたぐ関係なので ToUndirected() は
    # 新しい逆関係 rev_has_attr を安全に作る（同一ノード型どうしの関係を対称化する
    # 挙動とは異なるため、similar_to を追加する前のこの時点で一度だけ呼ぶ）。
    data = ToUndirected()(data)
    return data


def attach_fault_similarity_edges(data: HeteroData, analyzer, min_shared_attrs: int = 3):
    """共有属性数 >= min_shared_attrs の Fault ペアを正例として (fault,similar_to,fault) に追加する。

    メッセージパッシングには使わない（`HeteroGNNEncoder.forward` はこの edge type を
    参照しないため、リンク予測の学習対象と符号化に使う構造が漏れなく分離されている）。

    戻り値: (data, diagnostics)。diagnostics には正例数・密度・平均次数を含む
    （閾値が強すぎて0件になっていないか等をCLI側で表示するため）。
    """
    n_attr = len(analyzer.attr_columns)
    if not (2 <= min_shared_attrs <= n_attr):
        raise ValueError(f"min_shared_attrs は 2〜{n_attr} の範囲で指定してください（指定値: {min_shared_attrs}）")

    X = analyzer.X_raw
    shared = (X @ X.T).tocoo()  # (i,j) = i番目とj番目のFaultが共有する属性値の数
    mask = (shared.row < shared.col) & (shared.data >= min_shared_attrs)
    i, j = shared.row[mask], shared.col[mask]
    if len(i) == 0:
        raise ValueError(
            f"min_shared_attrs={min_shared_attrs} では正例となるFaultペアが0件でした。"
            "閾値を下げてください。"
        )

    pairs = torch.as_tensor(np.stack([i, j]), dtype=torch.long)
    sym = torch.cat([pairs, pairs.flip(0)], dim=1)  # 対称化（i->j と j->i の両方向）
    data[SIMILAR_TO].edge_index = sym

    n_fault = len(analyzer.df)
    total_pairs = n_fault * (n_fault - 1) // 2
    n_pos = int(len(i))
    diagnostics = {
        "n_faults": n_fault,
        "min_shared_attrs": min_shared_attrs,
        "n_positive_pairs": n_pos,
        "total_pairs": total_pairs,
        "density": n_pos / total_pairs if total_pairs else 0.0,
        "avg_degree": (2 * n_pos) / n_fault if n_fault else 0.0,
    }
    return data, diagnostics


def split_cause_labels(analyzer, test_size: float = 0.3, val_size: float = 0.15, random_state: int = 0):
    """原因ラベルの train/val/test 分割。

    test集合は `FaultAnalyzer.evaluate_cause_model()`（graph_engine.py の該当ロジック）と
    同じ random_state・同じ層化ロジックで再現するため、**厳密に同一の test 集合**になる
    （比較表のGNN行を他の行と正しく比較できるようにするため）。実装は
    graph_engine.py の evaluate_cause_model と同期を保つこと。

    戻り値: (train_idx, val_idx, test_idx, y_code_full, classes)
      - train_idx/val_idx/test_idx: analyzer.df の行インデックス（互いに素）
      - y_code_full: 全Faultぶんのラベルコード配列（長さ len(analyzer.df)）。
        ラベルが無い/空の行は -1 になる（学習・評価では対応するインデックスを使わないため
        マスクとして機能する）。
      - classes: y_code_full の値からクラス名への対応（np.unique の第1戻り値）
    """
    if not analyzer.cause_column:
        raise ValueError("原因列がないため分割できません")

    labels_all = analyzer.df[analyzer.cause_column]
    idx = np.where(pd.notna(labels_all) & (labels_all.astype(str).str.strip() != ""))[0]
    if len(idx) < 30:
        raise ValueError("評価には最低30件の原因ラベルが必要です")

    y = labels_all.iloc[idx].astype(str).to_numpy()
    classes, y_code = np.unique(y, return_inverse=True)

    pos = np.arange(len(idx))
    counts = np.bincount(y_code)
    n_test = int(round(len(idx) * test_size))
    stratify = y_code if (counts.min() >= 2 and n_test >= len(classes)) else None
    try:
        tr_pos, te_pos = train_test_split(pos, test_size=test_size, random_state=random_state, stratify=stratify)
    except ValueError:
        tr_pos, te_pos = train_test_split(pos, test_size=test_size, random_state=random_state)
    test_idx = idx[te_pos]

    # train側をさらに学習/検証に分割する（testはここでは一切触れない）
    tr_counts = np.bincount(y_code[tr_pos]) if len(tr_pos) else np.array([])
    n_val = int(round(len(tr_pos) * val_size))
    stratify2 = y_code[tr_pos] if (len(tr_counts) and tr_counts.min() >= 2 and n_val >= len(classes)) else None
    try:
        tr2_pos, val_pos = train_test_split(
            np.arange(len(tr_pos)), test_size=val_size, random_state=random_state, stratify=stratify2
        )
    except ValueError:
        tr2_pos, val_pos = train_test_split(np.arange(len(tr_pos)), test_size=val_size, random_state=random_state)
    train_idx = idx[tr_pos[tr2_pos]]
    val_idx = idx[tr_pos[val_pos]]

    y_code_full = np.full(len(analyzer.df), -1, dtype=np.int64)
    y_code_full[idx] = y_code
    return train_idx, val_idx, test_idx, y_code_full, classes
