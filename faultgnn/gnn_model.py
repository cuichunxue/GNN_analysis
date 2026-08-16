# -*- coding: utf-8 -*-
"""異種グラフGNN本体（PyTorch Geometric）。

`fault`-`attr_value` の二部グラフ上で2層のメッセージパッシングを行い、Fault埋め込みを
学習する。原因ラベルとFault-Fault類似度は、この埋め込みの上に載せる2つの独立したヘッド
（ノード分類・リンク予測）としてのみ使う。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

from gnn_data import ATTR_VALUE, FAULT, HAS_ATTR, REV_HAS_ATTR


class HeteroGNNEncoder(nn.Module):
    """fault/attr_value の二部グラフ専用エンコーダ。

    - attr_value ノードの初期特徴は学習可能な埋め込みテーブル。
    - fault ノードの初期特徴はゼロベクトル（学習可能な個別パラメータを持たせない。
      理由は gnn_data.py のモジュールdocstring参照）。
    - `data` に similar_to など他の edge type が含まれていても、このクラスは
      HAS_ATTR / REV_HAS_ATTR しか参照しないため、それらの存在が出力に影響することはない
      （tests/test_gnn.py の test_encoder_ignores_similarity_edges で保証）。
    """

    def __init__(self, num_attr_values: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers は2以上が必要です（fault→attr_value→fault の2ホップが最小構成のため）")
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.attr_embedding = nn.Embedding(num_attr_values, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                HeteroConv(
                    {
                        HAS_ATTR: SAGEConv((-1, -1), hidden_dim),
                        REV_HAS_ATTR: SAGEConv((-1, -1), hidden_dim),
                    },
                    aggr="sum",
                )
            )

    def forward(self, data) -> dict:
        n_fault = data[FAULT].num_nodes
        x_dict = {
            FAULT: torch.zeros(n_fault, self.hidden_dim, device=self.attr_embedding.weight.device),
            ATTR_VALUE: self.attr_embedding.weight,
        }
        edge_index_dict = {
            HAS_ATTR: data[HAS_ATTR].edge_index,
            REV_HAS_ATTR: data[REV_HAS_ATTR].edge_index,
        }
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < len(self.convs) - 1:
                x_dict = {k: F.dropout(F.relu(v), p=self.dropout, training=self.training) for k, v in x_dict.items()}
        return x_dict


class CausePredictionHead(nn.Module):
    """Fault埋め込み -> 原因クラスのロジット（ノード分類）。"""

    def __init__(self, hidden_dim: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, n_classes)

    def forward(self, fault_emb: torch.Tensor) -> torch.Tensor:
        return self.linear(fault_emb)


class LinkPredictionHead(nn.Module):
    """Fault埋め込みペア -> 類似リンクのロジット。

    mode="dot": 内積のみ（パラメータなし、教師データが少ない場合の過学習リスクが低い）。
    mode="mlp": 2層MLP（表現力は高いがパラメータが増える）。
    """

    def __init__(self, hidden_dim: int, mode: str = "dot"):
        super().__init__()
        if mode not in ("dot", "mlp"):
            raise ValueError("mode は 'dot' か 'mlp' を指定してください")
        self.mode = mode
        if mode == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
            )

    def forward(self, fault_emb: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_label_index[0], edge_label_index[1]
        a, b = fault_emb[src], fault_emb[dst]
        if self.mode == "dot":
            return (a * b).sum(dim=-1)
        return self.mlp(torch.cat([a, b], dim=-1)).squeeze(-1)
