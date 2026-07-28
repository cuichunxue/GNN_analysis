# -*- coding: utf-8 -*-
"""スケール特性の計測スクリプト。

    python3 benchmark.py            # 1,000 / 5,000 / 20,000 件
    python3 benchmark.py 100000     # 件数を指定

計測対象は「構築」「原因予測」「パターン発見」「精度検証」「類似検索」。
性能改善を主張する変更を入れたら、このスクリプトの出力を根拠として添えること。
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from graph_engine import FaultAnalyzer

COLUMNS = {"装置": 12, "部品": 30, "症状": 20, "Error": 40, "条件": 8, "工程": 10}
N_CAUSES = 15


def make_df(n, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.choice([f"{c}{i}" for i in range(k)], n) for c, k in COLUMNS.items()})
    df["原因"] = rng.choice([f"原因{i}" for i in range(N_CAUSES)], n)
    df["対策"] = "点検"
    df["発生日"] = pd.Timestamp("2026-07-01") - pd.to_timedelta(rng.integers(0, 500, n), unit="D")
    return df


def timed(label, fn):
    t0 = time.perf_counter()
    fn()
    dt = time.perf_counter() - t0
    print(f"  {label:<16} {dt:7.2f}s")
    return dt


def run(n):
    print(f"\n=== {n:,} 件 ===")
    df = make_df(n)
    holder = {}
    timed("構築", lambda: holder.setdefault("a", FaultAnalyzer(df)))
    a = holder["a"]
    timed("原因予測", lambda: a.predict_cause({"装置": "装置0", "部品": "部品1"}))
    timed("パターン発見", lambda: a.discover_patterns())
    timed("精度検証", lambda: a.evaluate_cause_model())
    timed("類似検索", lambda: a.find_similar(fault_index=0))
    print(f"  埋め込み: {a.E.shape} {a.E.dtype} / 属性値 {len(a.vocab)}種 / 非ゼロ {a.X.nnz:,}")


if __name__ == "__main__":
    sizes = [int(x) for x in sys.argv[1:]] or [1000, 5000, 20000]
    for n in sizes:
        run(n)
