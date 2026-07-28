# -*- coding: utf-8 -*-
"""解析エンジンの回帰テスト。

旧版で実際に混入していた不具合（NaN が属性値になる / クエリ埋め込みが
保存済み埋め込みと別空間 / パラメータ無検証 など）を再発防止として固定する。
"""
import numpy as np
import pandas as pd
import pytest

from graph_engine import FaultAnalyzer, normalize_values


def make_df(n=120, seed=0, with_date=True):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "装置": rng.choice(["装置A", "装置B", "装置C"], n),
            "部品": rng.choice(["Motor", "Bearing", "Sensor"], n),
            "症状": rng.choice(["振動", "異音", "発熱"], n),
            "Error": rng.choice(["E101", "E205", "E310"], n),
            "条件": rng.choice(["高温", "通常", "夜間"], n),
            "工程": rng.choice(["組立", "検査"], n),
            "原因": rng.choice(["摩耗", "軸ずれ", "潤滑不足"], n),
            "対策": "点検",
            "備考": "テスト",
        }
    )
    if with_date:
        df["発生日"] = pd.Timestamp("2026-07-01") - pd.to_timedelta(rng.integers(0, 300, n), unit="D")
    return df


# ----------------------------------------------------------------- 正規化
def test_normalize_values_treats_blank_and_nan_as_missing():
    s = pd.Series(["装置A", " 装置A ", "", "   ", None, np.nan, 5])
    out = normalize_values(s)
    assert out.iloc[0] == out.iloc[1] == "装置A"  # 前後空白は同一値に寄せる
    assert out.iloc[2] is np.nan or pd.isna(out.iloc[2])
    assert pd.isna(out.iloc[3]) and pd.isna(out.iloc[4]) and pd.isna(out.iloc[5])
    assert out.iloc[6] == "5"


def test_missing_values_never_become_an_attribute_value():
    """旧版は astype(str) で NaN を "nan" という属性値として扱っていた。"""
    df = make_df(60)
    df.loc[df.index[:20], "条件"] = np.nan
    a = FaultAnalyzer(df)
    assert not any(v.endswith("::nan") or v.endswith("::None") for v in a.vocab)
    surges = a.detect_surges(recent_days=100)
    assert all("nan" not in item["pattern"] for item in surges["items"])


# ----------------------------------------------------------------- 構築
def test_rejects_too_small_input():
    with pytest.raises(ValueError):
        FaultAnalyzer(pd.DataFrame({"装置": ["A"]}))


def test_high_cardinality_column_is_dropped_but_not_on_small_data():
    df = make_df(80)
    df["装置"] = [f"ID{i}" for i in range(len(df))]  # ID列相当
    a = FaultAnalyzer(df)
    assert "装置" not in a.attr_columns
    assert any(d["column"] == "装置" for d in a.dropped_columns)

    small = make_df(10)  # 10件では比率ルールを適用しない（全列除外バグの再発防止）
    assert len(FaultAnalyzer(small).attr_columns) == 6


def test_duplicate_attr_columns_are_deduplicated():
    df = make_df(40)
    a = FaultAnalyzer(df, attr_columns=["装置", "装置", "部品"])
    assert a.attr_columns == ["装置", "部品"]


def test_cause_column_is_excluded_from_embedding_space():
    """原因列が埋め込みに混ざるとラベルリークになる。"""
    a = FaultAnalyzer(make_df(60))
    assert not any(v.startswith("原因::") for v in a.vocab)


def test_summary_reports_warnings_for_missing_columns():
    df = make_df(40, with_date=False).drop(columns=["原因"])
    s = FaultAnalyzer(df).summary()
    assert any("原因列" in w for w in s["warnings"])
    assert any("日付列" in w for w in s["warnings"])


# ----------------------------------------------------------------- 埋め込み
def test_embeddings_are_discriminative():
    """旧v1は全Faultペアの類似度が 0.9986〜1.0000 に潰れていた。"""
    a = FaultAnalyzer(make_df(200))
    S = a.E @ a.E.T
    off = S[~np.eye(len(S), dtype=bool)]
    assert off.max() - off.min() > 0.5


def test_similarity_correlates_with_attribute_overlap():
    a = FaultAnalyzer(make_df(200))
    cols = a.attr_columns
    rng = np.random.default_rng(1)
    pairs = rng.integers(0, len(a.df), size=(400, 2))
    overlap, sim = [], []
    for i, j in pairs:
        if i == j:
            continue
        overlap.append(sum(a.df[c].iat[i] == a.df[c].iat[j] for c in cols))
        sim.append(float(a.E[i] @ a.E[j]))
    assert np.corrcoef(overlap, sim)[0, 1] > 0.8


def test_query_embedding_lives_in_the_same_space_as_stored_rows():
    """旧版はクエリだけ平滑化前の空間にあり、既存Faultと同じ属性を入力しても
    自分自身がトップに来ない train/serve skew があった。"""
    a = FaultAnalyzer(make_df(150))
    for row in (0, 7, 42):
        attrs = {c: a.df[c].iat[row] for c in a.attr_columns}
        q, matched = a.embed_query(attrs)
        assert len(matched) == len(a.attr_columns)
        assert float(q @ a.E[row]) > 0.99


# ----------------------------------------------------------------- ① 原因予測
def test_predict_cause_returns_normalised_candidates(analyzer):
    r = analyzer.predict_cause({"部品": "Motor", "条件": "高温"})
    assert r["candidates"]
    assert r["matched_attrs"] == ["部品", "条件"]
    assert sum(c["probability"] for c in r["candidates"]) <= 100.5
    assert all(c["evidence"] for c in r["candidates"])


def test_predict_cause_rejects_empty_and_unknown_attrs(analyzer):
    with pytest.raises(ValueError, match="1つ以上"):
        analyzer.predict_cause({})
    with pytest.raises(ValueError, match="存在しない"):
        analyzer.predict_cause({"部品": "存在しない部品"})


def test_predict_cause_without_cause_column():
    a = FaultAnalyzer(make_df(40).drop(columns=["原因"]))
    with pytest.raises(ValueError):
        a.predict_cause({"部品": "Motor"})


def test_predict_cause_finds_the_planted_correlation(analyzer):
    """サンプルデータでは Motor×高温 → 摩耗 を仕込んである。"""
    r = analyzer.predict_cause({"部品": "Motor", "条件": "高温"})
    assert r["candidates"][0]["cause"] == "摩耗"


# ----------------------------------------------------------------- 評価
def test_evaluate_reports_three_models(analyzer):
    r = analyzer.evaluate_cause_model()
    assert [row["model"] for row in r["rows"]] == [
        "多数決ベースライン",
        "埋め込みkNN（本アプリ①）",
        "勾配ブースティング（CatBoost相当）",
    ]
    assert r["n_train"] + r["n_test"] == len(analyzer.df)
    for row in r["rows"]:
        assert 0.0 <= row["top1"] <= 1.0
    knn, gb = r["rows"][1], r["rows"][2]
    assert knn["top1"] > r["rows"][0]["top1"]  # ベースラインは超える
    assert knn["top3"] >= knn["top1"] and gb["top3"] >= gb["top1"]


def test_evaluate_requires_enough_labels():
    with pytest.raises(ValueError, match="30件"):
        FaultAnalyzer(make_df(20)).evaluate_cause_model()


def test_evaluate_falls_back_when_stratification_impossible():
    """1件しかないクラスがあると層化分割は失敗する。例外ではなく通常分割に落ちる。"""
    df = make_df(60)
    df.loc[df.index[0], "原因"] = "レア原因"
    r = FaultAnalyzer(df).evaluate_cause_model()
    assert r["n_test"] > 0


# ----------------------------------------------------------------- ② パターン
def test_discover_patterns_shape(analyzer):
    r = analyzer.discover_patterns()
    assert r["clusters"] and r["chosen_k"] >= 2
    assert sum(c["size"] for c in r["clusters"]) == len(analyzer.df)
    assert r["communities"] and "n_occurrences" in r["communities"][0]
    assert r["network"]["nodes"] and r["network"]["edges"]
    assert all(n["community"] >= -1 for n in r["network"]["nodes"])


def test_discover_patterns_clamps_out_of_range_parameters(analyzer):
    """検証しないと contamination=0.9 で sklearn が例外、n_clusters=十万で事実上のDoS。"""
    r = analyzer.discover_patterns(n_clusters=10**6, contamination=0.99, recent_days=10**9)
    assert 2 <= r["chosen_k"] <= 30
    # recent_days は上限で丸められ、比較期間が取れないことが理由付きで返る
    assert r["surges"]["available"] is False
    assert analyzer.detect_surges(recent_days=10**9)["items"] == []
    assert analyzer.discover_patterns(contamination=-1)["anomalies"] is not None


def test_anomaly_scores_are_sorted(analyzer):
    scores = [a["anomaly_score"] for a in analyzer.discover_patterns()["anomalies"]]
    assert scores == sorted(scores, reverse=True)


# ----------------------------------------------------------------- 急増検知
def test_surge_detection_finds_the_planted_spike(analyzer):
    """サンプルデータは 装置B×E205×夜間/高湿度 を直近60日に集中させてある。"""
    s = analyzer.detect_surges(recent_days=90)
    assert s["available"]
    significant = [i["pattern"] for i in s["items"] if i["significant"]]
    assert any("E205" in p for p in significant)
    assert all(i["p_value"] <= 1.0 and i["q_value"] <= 1.0 for i in s["items"])


def test_surge_detection_flags_small_counts_as_not_significant():
    """件数が少ないだけの偶然のゆらぎを「×24倍の急増」として上位に出さない。"""
    rng = np.random.default_rng(3)
    n = 400
    df = pd.DataFrame(
        {
            "装置": rng.choice(["装置A", "装置B"], n),
            "部品": rng.choice(["Motor", "Bearing"], n),
            "原因": rng.choice(["摩耗", "軸ずれ"], n),
            "発生日": pd.Timestamp("2026-07-01") - pd.to_timedelta(rng.integers(0, 300, n), unit="D"),
        }
    )
    s = FaultAnalyzer(df).detect_surges(recent_days=60)
    assert not [i for i in s["items"] if i["significant"]]


def test_surge_detection_unavailable_without_dates():
    a = FaultAnalyzer(make_df(60, with_date=False))
    assert a.detect_surges()["available"] is False


def test_bh_qvalues_are_monotone():
    q = FaultAnalyzer._bh_qvalues([0.001, 0.02, 0.3, 0.9])
    assert q == sorted(q)
    assert all(0.0 <= x <= 1.0 for x in q)


# ----------------------------------------------------------------- ③ 類似検索
def test_find_similar_by_id_excludes_self_and_is_sorted(analyzer):
    rows = analyzer.find_similar(fault_index=5, top_k=10)
    assert len(rows) == 10
    assert "F00005" not in [r["fault_id"] for r in rows]
    sims = [r["similarity"] for r in rows]
    assert sims == sorted(sims, reverse=True)


def test_find_similar_by_attrs_retrieves_the_same_row(analyzer):
    row = 11
    attrs = {c: analyzer.df[c].iat[row] for c in analyzer.attr_columns}
    rows = analyzer.find_similar(attrs=attrs, top_k=5)
    assert analyzer.fault_ids[row] in [r["fault_id"] for r in rows]
    assert rows[0]["similarity"] > 95


def test_find_similar_validates_input(analyzer):
    with pytest.raises(ValueError):
        analyzer.find_similar(attrs={})
    with pytest.raises(ValueError):
        analyzer.find_similar(fault_index=10**9)


def test_top_k_is_capped(analyzer):
    assert len(analyzer.find_similar(fault_index=0, top_k=10**6)) <= 100


# ----------------------------------------------------------------- 検索補助
def test_fault_id_search(analyzer):
    r = analyzer.search_fault_ids("f0001", limit=50)
    assert r["total"] == len(analyzer.df)
    assert all("F0001" in f for f in r["fault_ids"])
    assert analyzer.index_of_fault_id("F00003") == 3
    assert analyzer.index_of_fault_id("存在しない") is None


def test_attr_options_are_strings(analyzer):
    for col, vals in analyzer.attr_value_options().items():
        assert all(isinstance(v, str) for v in vals)
