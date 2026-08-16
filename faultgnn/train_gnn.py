# -*- coding: utf-8 -*-
"""PyTorch Geometric によるGNN学習・評価CLI（スタンドアロン、Flaskアプリには未統合）。

実行例:
    pip install -r requirements-gnn.txt
    python3 train_gnn.py --task both

このスクリプトは graph_engine.py / app.py の挙動を一切変更しない。既存の埋め込みkNN・
勾配ブースティングとの比較用の、独立した実験モジュールという位置づけ。
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling

from gnn_data import (
    FAULT,
    SIMILAR_TO,
    attach_fault_similarity_edges,
    build_bipartite_hetero_data,
    split_cause_labels,
)
from gnn_model import CausePredictionHead, HeteroGNNEncoder, LinkPredictionHead
from graph_engine import FaultAnalyzer
from sample_data import build_sample_dataframe


def _load_analyzer(args):
    if args.csv:
        for enc in ("utf-8-sig", "cp932", "utf-16"):
            try:
                df = pd.read_csv(args.csv, encoding=enc)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        else:
            raise SystemExit(f"CSVを読み込めませんでした: {args.csv}")
    else:
        df = build_sample_dataframe(base_date="2026-07-01")
    return FaultAnalyzer(df)


def _init_encoder(num_attr_values, args):
    enc = HeteroGNNEncoder(
        num_attr_values=num_attr_values,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    return enc


def _resolve_device(args) -> torch.device:
    """--device auto（既定）なら GPU(CUDA) があれば優先して使う。"""
    want = getattr(args, "device", "auto")
    if want == "cpu":
        return torch.device("cpu")
    if want == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("GPU(CUDA)が利用できません。--device cpu を指定するか、GPU環境で実行してください。")
        return torch.device("cuda")
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _seed_everything(seed: int, device: torch.device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# ① ノード分類（原因予測）
# ----------------------------------------------------------------------
def run_node_classification(analyzer: FaultAnalyzer, args) -> dict:
    device = _resolve_device(args)
    _seed_everything(args.seed, device)
    if not analyzer.cause_column:
        raise ValueError("原因列がないためノード分類は実行できません")

    data = build_bipartite_hetero_data(analyzer).to(device)
    train_idx, val_idx, test_idx, y_full, classes = split_cause_labels(
        analyzer, val_size=args.val_size, random_state=args.seed
    )
    train_idx = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    val_idx = torch.as_tensor(val_idx, dtype=torch.long, device=device)
    test_idx = torch.as_tensor(test_idx, dtype=torch.long, device=device)
    y = torch.as_tensor(y_full, dtype=torch.long, device=device)

    encoder = _init_encoder(len(analyzer.vocab), args).to(device)
    head = CausePredictionHead(args.hidden_dim, len(classes)).to(device)
    with torch.no_grad():  # lazy SAGEConv パラメータを最適化器構築前に確定させる（data と同じ device で materialize される）
        encoder(data)
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    best_val, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(args.epochs):
        encoder.train()
        head.train()
        opt.zero_grad()
        logits = head(encoder(data)[FAULT])
        loss = F.cross_entropy(logits[train_idx], y[train_idx])
        loss.backward()
        opt.step()

        encoder.eval()
        head.eval()
        with torch.no_grad():
            logits = head(encoder(data)[FAULT])
            val_top1 = float((logits[val_idx].argmax(1) == y[val_idx]).float().mean())
        if val_top1 > best_val:
            best_val = val_top1
            best_state = (
                {k: v.clone() for k, v in encoder.state_dict().items()},
                {k: v.clone() for k, v in head.state_dict().items()},
            )
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    if best_state:
        encoder.load_state_dict(best_state[0])
        head.load_state_dict(best_state[1])
    encoder.eval()
    head.eval()
    with torch.no_grad():
        logits = head(encoder(data)[FAULT])
        top1 = float((logits[test_idx].argmax(1) == y[test_idx]).float().mean())
        top3_idx = logits[test_idx].argsort(dim=1, descending=True)[:, :3]
        top3 = float((top3_idx == y[test_idx].unsqueeze(1)).any(dim=1).float().mean())

    return {
        "model": "GNN（PyTorch Geometric, GraphSAGE）",
        "top1": round(top1, 3),
        "top3": round(top3, 3),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "epochs_run": epoch + 1,
        "device": str(device),
    }


# ----------------------------------------------------------------------
# ② リンク予測（③類似Fault検索の代替）
# ----------------------------------------------------------------------
def run_link_prediction(analyzer: FaultAnalyzer, args) -> dict:
    device = _resolve_device(args)
    _seed_everything(args.seed, device)
    np.random.seed(args.seed)

    data = build_bipartite_hetero_data(analyzer)
    data, diag = attach_fault_similarity_edges(data, analyzer, min_shared_attrs=args.min_shared_attrs)

    # 分割は CPU 上で行い（PyG推奨）、分割後にまとめて device へ転送する
    splitter = RandomLinkSplit(
        num_val=0.1,
        num_test=0.2,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=args.neg_ratio,
        edge_types=[SIMILAR_TO],
    )
    train_data, val_data, test_data = splitter(data)
    data = data.to(device)
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    n_fault = data[FAULT].num_nodes

    encoder = _init_encoder(len(analyzer.vocab), args).to(device)
    head = LinkPredictionHead(args.hidden_dim, mode=args.link_decoder).to(device)
    with torch.no_grad():
        encoder(train_data)
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    def _eval(split_data):
        encoder.eval()
        head.eval()
        with torch.no_grad():
            fault_emb = encoder(split_data)[FAULT]
            logits = head(fault_emb, split_data[SIMILAR_TO].edge_label_index)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = split_data[SIMILAR_TO].edge_label.cpu().numpy()
        auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan")
        ap = average_precision_score(labels, probs) if len(set(labels)) > 1 else float("nan")
        return auc, ap

    best_val, best_state, bad_epochs = -1.0, None, 0
    pos_train_index = train_data[SIMILAR_TO].edge_label_index
    for epoch in range(args.epochs):
        encoder.train()
        head.train()
        opt.zero_grad()
        fault_emb = encoder(train_data)[FAULT]
        # 学習時は毎エポック負例を再サンプリングする（固定1組より汎化しやすい）
        neg_index = negative_sampling(
            train_data[SIMILAR_TO].edge_index, num_nodes=n_fault, num_neg_samples=pos_train_index.size(1)
        )
        edge_label_index = torch.cat([pos_train_index, neg_index], dim=1)
        edge_label = torch.cat(
            [
                torch.ones(pos_train_index.size(1), device=device),
                torch.zeros(neg_index.size(1), device=device),
            ]
        )
        logits = head(fault_emb, edge_label_index)
        loss = F.binary_cross_entropy_with_logits(logits, edge_label)
        loss.backward()
        opt.step()

        val_auc, _ = _eval(val_data)
        if val_auc > best_val:
            best_val = val_auc
            best_state = (
                {k: v.clone() for k, v in encoder.state_dict().items()},
                {k: v.clone() for k, v in head.state_dict().items()},
            )
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    if best_state:
        encoder.load_state_dict(best_state[0])
        head.load_state_dict(best_state[1])
    test_auc, test_ap = _eval(test_data)

    # Precision@k: 全体の共起関係（transductive）に対する検索品質の参考値。
    # 上のAUC/APが汎化性能の正式な指標で、これは③類似Fault検索との比較用の参考指標。
    encoder.eval()
    head.eval()
    with torch.no_grad():
        fault_emb = encoder(data)[FAULT]
        sims = fault_emb @ fault_emb.T
        sims.fill_diagonal_(-float("inf"))
        k = min(args.precision_k, n_fault - 1)
        topk = sims.argsort(dim=1, descending=True)[:, :k].cpu()

    pos_pairs = set(map(tuple, data[SIMILAR_TO].edge_index.cpu().t().tolist()))
    hits = sum(1 for f in range(n_fault) for c in topk[f].tolist() if (f, c) in pos_pairs)
    precision_at_k = hits / (n_fault * k) if n_fault and k else 0.0

    return {
        "auc": round(float(test_auc), 3),
        "ap": round(float(test_ap), 3),
        "precision_at_k": round(precision_at_k, 3),
        "k": k,
        "n_train_edges": int(pos_train_index.size(1)),
        "n_val_edges": int(val_data[SIMILAR_TO].edge_label_index.size(1) // 2),
        "n_test_edges": int(test_data[SIMILAR_TO].edge_label_index.size(1) // 2),
        "epochs_run": epoch + 1,
        "device": str(device),
        **diag,
    }


# ----------------------------------------------------------------------
def _print_node_result(analyzer, result):
    baseline = analyzer.evaluate_cause_model(random_state=0)
    print("\n=== ① 原因予測 (Node Classification) ===")
    print(f"test set は evaluate_cause_model() のベースラインと同一（random_state=0, n_test={result['n_test']}）\n")
    print(f"{'model':<38}{'top1':>8}{'top3':>8}")
    print("-" * 54)
    for row in baseline["rows"]:
        top3 = f"{row['top3']:.3f}" if row["top3"] is not None else "-"
        print(f"{row['model']:<38}{row['top1']:>8.3f}{top3:>8}")
    print(f"{result['model']:<38}{result['top1']:>8.3f}{result['top3']:>8.3f}")
    print("-" * 54)
    print(f"n_train={result['n_train']}  n_val={result['n_val']}  n_test={result['n_test']}  "
          f"(train/valはEarly Stopping用。testは学習・停止判定のどちらにも使っていない)")


def _print_link_result(result):
    print("\n=== ③ 類似Fault検索 の代替 (Link Prediction) ===")
    print(
        f"min_shared_attrs={result['min_shared_attrs']} -> "
        f"positive pairs={result['n_positive_pairs']:,} "
        f"({result['density']*100:.1f}% density, avg degree={result['avg_degree']:.1f})"
    )
    print(
        f"train edges={result['n_train_edges']:,}   "
        f"val edges={result['n_val_edges']:,} (+同数の負例)   "
        f"test edges={result['n_test_edges']:,} (+同数の負例)\n"
    )
    print(f"  AUC-ROC            {result['auc']}")
    print(f"  Average Precision  {result['ap']}")
    print(f"  Precision@{result['k']}         {result['precision_at_k']}  (全体の共起関係に対する参考指標)")


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["node", "link", "both"], default="both")
    p.add_argument("--csv", default=None, help="CSVファイルパス（省略時はサンプルデータ）")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--min-shared-attrs", type=int, default=3)
    p.add_argument("--neg-ratio", type=float, default=1.0)
    p.add_argument("--link-decoder", choices=["dot", "mlp"], default="dot")
    p.add_argument("--precision-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="auto（既定）はGPU(CUDA)が使えれば優先して使う。cpu/cudaで明示指定も可能。",
    )
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    analyzer = _load_analyzer(args)
    device = _resolve_device(args)
    print(f"データ: {len(analyzer.df)}件 / 属性列 {analyzer.attr_columns} / 属性値 {len(analyzer.vocab)}種")
    print(f"device: {device}" + ("（GPU利用可）" if device.type == "cuda" else "（CPU実行。GPUは検出されませんでした）"))

    if args.task in ("node", "both"):
        _print_node_result(analyzer, run_node_classification(analyzer, args))
    if args.task in ("link", "both"):
        _print_link_result(run_link_prediction(analyzer, args))


if __name__ == "__main__":
    sys.exit(main() or 0)
