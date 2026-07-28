# -*- coding: utf-8 -*-
"""Fault Management 解析エンジン

構成は「Fault × 属性値 の incidence 行列 → TF-IDF → TruncatedSVD →
Fault-Fault 共有属性グラフ上での近傍平滑化（1ホップのメッセージパッシング）」。
学習される重みを持たないため厳密には GNN ではない（README の「限界」を参照）。

主要な設計上の制約:
  - n×n の類似度行列を実体化しない。S E = X (Xᵀ E) の結合則で O(nnz·d) に保つ。
  - 埋め込みは属性列のみから構成する。原因列は検索結果の集計にのみ使い、
    埋め込みには入れない（ラベルリークの防止）。
  - クエリ埋め込みは保存済み埋め込みと同じ平滑化を通す（train/serve skew の防止）。
"""
from __future__ import annotations

import logging
import threading
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from networkx.algorithms.community import greedy_modularity_communities
from scipy.stats import poisson
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

log = logging.getLogger(__name__)

DEFAULT_ATTR_COLUMNS = ["装置", "部品", "症状", "Error", "条件", "工程"]
DEFAULT_CAUSE_COLUMN = "原因"
DEFAULT_ACTION_COLUMN = "対策"
DEFAULT_TEXT_COLUMN = "備考"
DEFAULT_DATE_COLUMN = "発生日"

# 高カーディナリティ列の除外閾値（ID列などがグラフを爆発させるのを防ぐ）
MAX_CARDINALITY_RATIO = 0.5
MAX_CARDINALITY_ABS = 500
# 比率ルールを適用する最小行数。小規模データで全列が誤除外されるのを防ぐ。
CARDINALITY_RATIO_MIN_ROWS = 50

# 解析パラメータの許容範囲（APIからの過大な入力による DoS を防ぐ）
MAX_CLUSTERS = 30
MIN_CONTAMINATION, MAX_CONTAMINATION = 0.005, 0.5
MAX_RECENT_DAYS = 3650
MAX_TOP_K = 100
MAX_KNN = 200

# クラスタ数探索を全件で回すと大規模データで支配的なコストになるため、
# 探索はサブサンプル上で行い、確定した k だけ全件で1回学習する。
KSEARCH_SAMPLE = 5000
SILHOUETTE_SAMPLE = 2000
# 精度評価は行数に上限を設ける（UIからの同期リクエストで走るため）
EVAL_MAX_ROWS = 30000


def _clip(value, lo, hi):
    return max(lo, min(hi, value))


def normalize_values(s: pd.Series) -> pd.Series:
    """属性列を「文字列 or NaN」に正規化する。

    - 欠損・空文字・空白のみの値はすべて NaN に統一する
      （旧版は astype(str) により NaN が "nan" という属性値として扱われていた）。
    - それ以外は前後空白を除去した文字列にする。数値とその文字列表現、
      および " 装置A" と "装置A" が別の属性値に分裂するのを防ぐ。
    """
    try:
        t = s.astype("string")
    except (TypeError, ValueError):
        t = s.map(lambda v: pd.NA if pd.isna(v) else str(v)).astype("string")
    t = t.str.strip()
    t = t.mask(t == "", pd.NA)
    return t.astype(object).where(t.notna(), np.nan)


class FaultAnalyzer:
    def __init__(
        self,
        df: pd.DataFrame,
        attr_columns=None,
        cause_column=DEFAULT_CAUSE_COLUMN,
        action_column=DEFAULT_ACTION_COLUMN,
        text_column=DEFAULT_TEXT_COLUMN,
        date_column=DEFAULT_DATE_COLUMN,
        n_components: int = 24,
        smooth_hops: int = 1,
        smooth_alpha: float = 0.3,
    ):
        if df is None or len(df) < 2:
            raise ValueError("解析には2行以上のデータが必要です")

        self.df = df.reset_index(drop=True).copy()
        cols = list(self.df.columns)

        self.cause_column = cause_column if cause_column in cols else None
        self.action_column = action_column if action_column in cols else None
        self.text_column = text_column if text_column in cols else None
        self.date_column = date_column if date_column in cols else None

        reserved = {self.cause_column, self.action_column, self.text_column, self.date_column}
        # dict.fromkeys で順序を保ったまま重複した列指定を除去する
        cand = [c for c in dict.fromkeys(attr_columns or DEFAULT_ATTR_COLUMNS) if c in cols and c not in reserved]
        if not cand:
            raise ValueError("属性列として使える列がありません")

        # 属性列は解析前に正規化する（以降の集計・グラフ構築はすべてこの値を使う）
        for c in cand:
            self.df[c] = normalize_values(self.df[c])

        self.attr_columns, self.dropped_columns = [], []
        apply_ratio = len(self.df) >= CARDINALITY_RATIO_MIN_ROWS
        for c in cand:
            nu = int(self.df[c].nunique(dropna=True))
            if nu == 0:
                self.dropped_columns.append({"column": c, "n_unique": 0, "reason": "値が空"})
                continue
            too_many = nu > MAX_CARDINALITY_ABS or (apply_ratio and nu > MAX_CARDINALITY_RATIO * len(self.df))
            if too_many:
                self.dropped_columns.append({"column": c, "n_unique": nu, "reason": "値の種類が多すぎる"})
            else:
                self.attr_columns.append(c)
        if not self.attr_columns:
            raise ValueError("有効な属性列がありません（全列が空、または値の種類が多すぎます）")

        if self.date_column:
            self.df[self.date_column] = pd.to_datetime(self.df[self.date_column], errors="coerce")
            if self.df[self.date_column].isna().all():
                self.date_column = None

        self.fault_ids = [f"F{i:05d}" for i in self.df.index]
        self._fault_id_index = {fid: i for i, fid in enumerate(self.fault_ids)}
        self.n_components = int(n_components)
        self.smooth_hops = int(smooth_hops)
        self.smooth_alpha = float(smooth_alpha)

        # 解析結果は複数リクエストから共有されうるため、遅延キャッシュはロックで保護する
        self._cache = {}
        self._lock = threading.RLock()

        self._build_features()
        self._smooth_embeddings()

        self.n_graph_nodes = int(len(self.df) + len(self.vocab))
        self.n_graph_edges = int(self.X_raw.nnz)

    # ------------------------------------------------------------------
    # 特徴量
    # ------------------------------------------------------------------
    @staticmethod
    def _nid(col, val):
        return f"{col}::{val}"

    def _incidence(self, columns):
        """Fault × 属性値 の 0/1 疎行列を factorize で一括構築する。

        旧版は df.iterrows() / df.iat による行単位ループで、20,000件の共起集計に
        約1.7秒（うち iat が約8割）かかっていた。列単位のベクトル化で解消する。
        """
        n = len(self.df)
        vocab, vindex = [], {}
        row_parts, col_parts = [], []
        for c in columns:
            s = self.df[c]
            if s.dtype != object:
                s = normalize_values(s)
            codes, uniques = pd.factorize(s, use_na_sentinel=True)
            base = len(vocab)
            for v in uniques:
                key = self._nid(c, v)
                vindex[key] = len(vocab)
                vocab.append(key)
            present = codes >= 0
            row_parts.append(np.nonzero(present)[0])
            col_parts.append(codes[present] + base)

        rows = np.concatenate(row_parts) if row_parts else np.empty(0, dtype=int)
        cols_ = np.concatenate(col_parts) if col_parts else np.empty(0, dtype=int)
        M = sp.csr_matrix(
            (np.ones(len(rows), dtype=np.float64), (rows, cols_)),
            shape=(n, max(len(vocab), 1)),
        )
        return M, vocab, vindex

    def _build_features(self):
        X, vocab, vindex = self._incidence(self.attr_columns)
        if len(vocab) < 2:
            raise ValueError("属性値の種類が少なすぎます（2種類以上必要です）")
        self.vocab, self.vocab_index = vocab, vindex
        self.X_raw = X

        self.tfidf = TfidfTransformer()
        self.X = self.tfidf.fit_transform(X)

        k = int(max(1, min(self.n_components, len(vocab) - 1, len(self.df) - 1)))
        self.svd = TruncatedSVD(n_components=k, random_state=0)
        self.E_base = self._l2(self.svd.fit_transform(self.X)).astype(np.float32)

    @staticmethod
    def _l2(M):
        M = np.asarray(M, dtype=np.float64)
        n = np.linalg.norm(M, axis=-1, keepdims=True)
        return M / np.maximum(n, 1e-12)

    def _smooth_embeddings(self):
        """Fault-Fault 共有属性グラフ上でのメッセージパッシング。

        概念上は S = X Xᵀ（Fault間類似度）を行正規化した P で E <- (1-a)E + a P E。
        S を実体化すると O(n^2) メモリで破綻する（5,000件で77%密、20,000件でOOM）ため、
        結合則 S E = X (Xᵀ E) を使い n×n を作らずに計算する。X は行L2正規化済みなので
        diag(S)=1 であり、自己ループ除去は -E で足りる。計算量は O(nnz·d)。

        各ホップの中間表現を保持し、クエリ側にも同じ演算を適用する
        （旧版はクエリだけ平滑化前の空間に置かれており、既存Faultと同一の属性を
        入力しても類似度が 1.0 にならない train/serve skew があった）。
        """
        E = self.E_base.astype(np.float64)
        self._E_levels = [self.E_base]
        if self.smooth_hops <= 0 or self.smooth_alpha <= 0:
            self.E = self.E_base
            return

        X = self.X
        ones = np.ones((X.shape[0], 1))
        deg = np.asarray(X @ (X.T @ ones)).ravel() - 1.0  # 自己類似度(=1)を除く
        deg[deg <= 1e-12] = 1.0
        inv = (1.0 / deg)[:, None]
        for _ in range(self.smooth_hops):
            agg = X @ (X.T @ E) - E  # = S_offdiag @ E （n×n を作らない）
            E = self._l2((1 - self.smooth_alpha) * E + self.smooth_alpha * (inv * agg))
            self._E_levels.append(E.astype(np.float32))
        self.E = self._E_levels[-1]

    # ------------------------------------------------------------------
    def embed_query(self, attrs: dict):
        """新規Faultの属性辞書を、保存済み埋め込みと同じ空間へ写像する。

        戻り値は (ベクトル, 認識できた属性名のリスト)。認識できた属性が0個なら (None, [])。
        """
        cols_, matched = [], []
        for c, v in (attrs or {}).items():
            if v is None:
                continue
            val = str(v).strip()
            if val == "":
                continue
            key = self._nid(c, val)
            if key in self.vocab_index:
                cols_.append(self.vocab_index[key])
                matched.append(c)
        if not cols_:
            return None, []

        x = sp.csr_matrix((np.ones(len(cols_)), ([0] * len(cols_), cols_)), shape=(1, len(self.vocab)))
        x = self.tfidf.transform(x)
        q = self._l2(self.svd.transform(x))[0]

        if self.smooth_hops > 0 and self.smooth_alpha > 0:
            sims = np.asarray((self.X @ x.T).todense()).ravel()  # クエリと全Faultの類似度
            total = float(sims.sum())
            if total > 1e-12:
                for level in range(self.smooth_hops):
                    agg = (sims @ self._E_levels[level].astype(np.float64)) / total
                    q = self._l2((1 - self.smooth_alpha) * q + self.smooth_alpha * agg)
        return q.astype(np.float32), matched

    # ------------------------------------------------------------------
    # ① 原因予測（埋め込みは属性のみから構成されるためラベルリークなし）
    # ------------------------------------------------------------------
    def predict_cause(self, attrs: dict, top_k=5, knn=25):
        if not self.cause_column:
            raise ValueError("データに原因列がありません")
        top_k = int(_clip(int(top_k), 1, MAX_TOP_K))
        knn = int(_clip(int(knn), 1, MAX_KNN))

        if not any(v not in (None, "") for v in (attrs or {}).values()):
            raise ValueError("属性を1つ以上選択してください")
        q, matched = self.embed_query(attrs)
        if q is None:
            raise ValueError("指定された属性値が過去データに1件も存在しないため予測できません")

        sims = self.E @ q
        order = self._top_indices(sims, knn)
        w, ev = Counter(), defaultdict(list)
        causes = self.df[self.cause_column]
        for i in order:
            i = int(i)
            c = causes.iat[i]
            if pd.isna(c) or c == "":
                continue
            s = max(float(sims[i]), 0.0)
            w[str(c)] += s + 1e-9
            ev[str(c)].append({"fault_id": self.fault_ids[i], "similarity": round(s * 100, 1)})

        total = sum(w.values())
        candidates = [
            {
                "cause": c,
                "probability": round(100 * v / total, 1),
                "n_neighbors": len(ev[c]),
                "evidence": sorted(ev[c], key=lambda e: -e["similarity"])[:3],
            }
            for c, v in sorted(w.items(), key=lambda x: -x[1])[:top_k]
        ] if total else []
        return {"candidates": candidates, "matched_attrs": matched, "n_neighbors": int(len(order))}

    @staticmethod
    def _top_indices(scores, k):
        """スコア上位 k 件のインデックスを降順で返す（全体ソートを避ける）。"""
        k = int(min(max(int(k), 1), len(scores)))
        if k >= len(scores):
            return np.argsort(-scores)
        part = np.argpartition(-scores, k - 1)[:k]
        return part[np.argsort(-scores[part])]

    # ------------------------------------------------------------------
    # 精度評価: ホールドアウトで Top-1/Top-3 をベースライン・GBDTと比較
    # ------------------------------------------------------------------
    def evaluate_cause_model(self, test_size=0.3, knn=25, random_state=0):
        if not self.cause_column:
            raise ValueError("原因列がないため評価できません")
        knn = int(_clip(int(knn), 1, MAX_KNN))

        labels_all = self.df[self.cause_column]
        idx = np.where(pd.notna(labels_all) & (labels_all.astype(str).str.strip() != ""))[0]
        if len(idx) < 30:
            raise ValueError("評価には最低30件の原因ラベルが必要です")

        sampled = False
        if len(idx) > EVAL_MAX_ROWS:
            rng = np.random.default_rng(random_state)
            idx = np.sort(rng.choice(idx, EVAL_MAX_ROWS, replace=False))
            sampled = True

        y = labels_all.iloc[idx].astype(str).to_numpy()
        classes, y_code = np.unique(y, return_inverse=True)
        n_classes = len(classes)
        if n_classes < 2:
            raise ValueError("原因が1種類しかないため評価できません")

        pos = np.arange(len(idx))
        counts = np.bincount(y_code)
        n_test = int(round(len(idx) * test_size))
        stratify = y_code if (counts.min() >= 2 and n_test >= n_classes) else None
        try:
            tr_pos, te_pos = train_test_split(
                pos, test_size=test_size, random_state=random_state, stratify=stratify
            )
        except ValueError:  # 層化が成立しない構成では通常分割にフォールバック
            tr_pos, te_pos = train_test_split(pos, test_size=test_size, random_state=random_state)
        tr, te = idx[tr_pos], idx[te_pos]
        ytr_code, yte_code = y_code[tr_pos], y_code[te_pos]

        maj = int(np.bincount(ytr_code, minlength=n_classes).argmax())
        base1 = float(np.mean(yte_code == maj))
        knn1, knn3 = self._knn_holdout(tr, te, ytr_code, yte_code, n_classes, knn)
        gb1, gb3 = self._gbdt_holdout(tr, te, ytr_code, yte_code)

        return {
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_classes": int(n_classes),
            "sampled": sampled,
            "note": "埋め込みは検証分を含む全件から教師なしで構成されています。原因ラベルは"
            "学習分しか使っていないためラベルリークはありませんが、kNN側がやや有利な"
            "transductive 設定である点は割り引いて解釈してください。",
            "rows": [
                {"model": "多数決ベースライン", "top1": round(base1, 3), "top3": None},
                {"model": "埋め込みkNN（本アプリ①）", "top1": round(knn1, 3), "top3": round(knn3, 3)},
                {"model": "勾配ブースティング（CatBoost相当）", "top1": round(gb1, 3), "top3": round(gb3, 3)},
            ],
        }

    def _knn_holdout(self, tr, te, ytr_code, yte_code, n_classes, knn):
        """検証×学習の類似度をチャンクで処理し、クラス別重みをベクトル化して集計する。

        全件の類似度行列を一度に作ると O(|te|·|tr|) で破綻する（10万件で15.6GiB要求）。
        """
        Etr, Ete = self.E[tr].astype(np.float32), self.E[te].astype(np.float32)
        kk = int(min(knn, len(tr)))
        hit1 = hit3 = 0
        chunk = max(1, int(4_000_000 // max(len(tr), 1)))
        for s0 in range(0, len(te), chunk):
            S = Ete[s0 : s0 + chunk] @ Etr.T
            m = S.shape[0]
            if kk < S.shape[1]:
                part = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
            else:
                part = np.tile(np.arange(S.shape[1]), (m, 1))
            w = np.take_along_axis(S, part, axis=1).clip(min=0) + 1e-9
            votes = np.zeros((m, n_classes), dtype=np.float64)
            np.add.at(votes, (np.arange(m)[:, None], ytr_code[part]), w)
            top3 = np.argsort(-votes, axis=1)[:, :3]
            gt = yte_code[s0 : s0 + m]
            hit1 += int((top3[:, 0] == gt).sum())
            hit3 += int((top3 == gt[:, None]).any(axis=1).sum())
            del S
        return hit1 / len(te), hit3 / len(te)

    def _gbdt_holdout(self, tr, te, ytr_code, yte_code):
        """カテゴリ特徴の勾配ブースティング。エンコーダは学習分だけで学習する
        （旧版は全件で fit しており、検証分のカテゴリ語彙が学習側に漏れていた）。"""
        attrs = self.df[self.attr_columns].astype(str)
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=np.nan
        )
        Xtr = enc.fit_transform(attrs.iloc[tr])
        Xte = enc.transform(attrs.iloc[te])  # 未知カテゴリは NaN（＝欠損）として扱われる
        # early_stopping は既定の "auto"（1万件超で有効）のまま。random_state 固定で
        # 内部の検証分割も再現可能になるため、決定性を保ったまま大規模データで速い。
        gb = HistGradientBoostingClassifier(
            categorical_features=list(range(len(self.attr_columns))), random_state=0
        ).fit(Xtr, ytr_code)
        proba = gb.predict_proba(Xte)
        ranked = gb.classes_[np.argsort(-proba, axis=1)[:, :3]]
        gb1 = float(np.mean(ranked[:, 0] == yte_code))
        gb3 = float(np.mean((ranked == yte_code[:, None]).any(axis=1)))
        return gb1, gb3

    # ------------------------------------------------------------------
    # ② パターン発見
    # ------------------------------------------------------------------
    def discover_patterns(self, n_clusters=None, contamination=0.08, recent_days=90):
        contamination = float(_clip(float(contamination), MIN_CONTAMINATION, MAX_CONTAMINATION))
        recent_days = int(_clip(int(recent_days), 1, MAX_RECENT_DAYS))
        X = self.E.astype(np.float64)
        n = len(X)

        chosen_k, sil_scores, labels = self._cluster_labels(X, n, n_clusters)
        return {
            "clusters": self._cluster_profiles(labels, chosen_k),
            "anomalies": self._anomalies(X, n, contamination),
            "silhouette": sil_scores,
            "chosen_k": chosen_k,
            "communities": self._communities(),
            "network": self._network(),
            "surges": self.detect_surges(recent_days),
        }

    def _cluster_labels(self, X, n, n_clusters):
        """クラスタ数をシルエット係数で選び、そのラベルを返す。

        探索はサブサンプル上で行う。旧版は全件で k=2..10 を試したうえ、確定後に
        もう一度全件で学習し直しており、20,000件で 11.4 秒かかっていた。
        """
        if n_clusters:
            k = int(_clip(int(n_clusters), 2, max(2, min(MAX_CLUSTERS, n - 1))))
            return k, [], KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        if n < 20:
            return None, [], None

        rng = np.random.default_rng(0)
        Xs = X if n <= KSEARCH_SAMPLE else X[np.sort(rng.choice(n, KSEARCH_SAMPLE, replace=False))]
        sil_scores, best = [], (-1.0, None)
        for k in range(2, min(11, len(Xs) // 5 + 1)):
            lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(Xs)
            if len(set(lab)) < 2:
                continue
            s = float(silhouette_score(Xs, lab, sample_size=min(SILHOUETTE_SAMPLE, len(Xs)), random_state=0))
            sil_scores.append({"k": k, "silhouette": round(s, 3)})
            if s > best[0]:
                best = (s, k)
        k = best[1]
        if not k:
            return None, sil_scores, None
        return k, sil_scores, KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)

    def _global_dist(self):
        """全体の属性値構成比。クラスタごとに再計算するとクラスタ数に比例して
        全件スキャンが走るため、一度だけ計算してキャッシュする。"""
        with self._lock:
            if "gdist" not in self._cache:
                self._cache["gdist"] = {
                    c: self.df[c].value_counts(normalize=True, dropna=True) for c in self.attr_columns
                }
            return self._cache["gdist"]

    def _cluster_profiles(self, labels, chosen_k):
        if labels is None or not chosen_k:
            return []
        gdist = self._global_dist()
        clusters = []
        for c in range(chosen_k):
            ix = np.where(labels == c)[0]
            if not len(ix):
                continue
            sub = self.df.iloc[ix]
            feats = []
            for col in self.attr_columns:
                vc = sub[col].value_counts(normalize=True, dropna=True)
                gl = gdist[col]
                for v, p in vc.head(2).items():
                    lift = float(p) / float(gl.get(v, 1e-9))
                    if lift > 1.2 and p > 0.2:
                        feats.append(
                            {
                                "col": col,
                                "val": str(v),
                                "share": round(float(p) * 100, 1),
                                "lift": round(lift, 2),
                            }
                        )
            feats.sort(key=lambda f: -f["lift"])
            top_cause = None
            if self.cause_column:
                cv = sub[self.cause_column].value_counts(dropna=True)
                if len(cv):
                    top_cause = f"{cv.index[0]}({int(cv.iloc[0])}/{len(ix)}件)"
            clusters.append(
                {
                    "cluster_id": int(c),
                    "size": int(len(ix)),
                    "features": feats[:6],
                    "top_cause": top_cause,
                    "fault_ids": [self.fault_ids[i] for i in ix[:12]],
                }
            )
        clusters.sort(key=lambda c: -c["size"])
        return clusters

    def _anomalies(self, X, n, contamination, limit=20):
        if n < 20:
            return []
        iso = IsolationForest(contamination=contamination, random_state=0).fit(X)
        score = -iso.score_samples(X)
        ax = np.where(iso.predict(X) == -1)[0]
        ax = ax[np.argsort(-score[ax])][:limit]
        out = []
        for i in ax:
            i = int(i)
            r = self.df.iloc[i]
            out.append(
                {
                    "fault_id": self.fault_ids[i],
                    "anomaly_score": round(float(score[i]), 3),
                    "attrs": {c: str(r[c]) for c in self.attr_columns if pd.notna(r.get(c))},
                    "cause": str(r[self.cause_column])
                    if self.cause_column and pd.notna(r.get(self.cause_column))
                    else None,
                }
            )
        return out

    # ------------------------------------------------------------------
    # 共起グラフ / コミュニティ
    # ------------------------------------------------------------------
    def _cooccurrence(self):
        """属性値どうしの共起行列。C = Bᵀ B（B は Fault×属性値 の 0/1 行列）。"""
        with self._lock:
            if "cooc" not in self._cache:
                cols = self.attr_columns + ([self.cause_column] if self.cause_column else [])
                B, vocab, _ = self._incidence(cols)
                counts = np.asarray(B.sum(0)).ravel()
                self._cache["cooc"] = (vocab, counts, (B.T @ B).tocsr())
            return self._cache["cooc"]

    def _top_subgraph(self, max_nodes):
        """出現頻度上位ノードだけの共起部分行列を取り出す。

        旧版は全非ゼロ要素を Python で走査していた（属性値3,000種なら最大900万回）。
        疎行列のスライスで必要な部分だけを取り出す。
        """
        vocab, counts, C = self._cooccurrence()
        top = np.argsort(-counts)[: min(max_nodes, len(vocab))]
        return vocab, counts, top, C[top][:, top].tocoo()

    def _attr_graph(self, max_nodes=80):
        with self._lock:
            if "ag" in self._cache:
                return self._cache["ag"]
        vocab, counts, top, sub = self._top_subgraph(max_nodes)
        H = nx.Graph()
        for gi in top:
            H.add_node(vocab[gi], count=int(counts[gi]))
        for a, b, w in zip(sub.row, sub.col, sub.data):
            if a < b and w > 0:
                H.add_edge(vocab[top[a]], vocab[top[b]], weight=float(w))
        with self._lock:
            self._cache["ag"] = H
        return H

    def _communities(self):
        with self._lock:
            if "comm" in self._cache:
                return self._cache["comm"]
        H = self._attr_graph()
        try:
            coms = list(greedy_modularity_communities(H, weight="weight")) if H.number_of_edges() else []
        except Exception:  # pragma: no cover - 収束しない稀なケース
            log.warning("community detection failed", exc_info=True)
            coms = []
        n2c, out = {}, []
        for ci, com in enumerate(coms):
            members = sorted(com, key=lambda x: -H.nodes[x].get("count", 0))
            for x in members:
                n2c[x] = ci
            out.append(
                {
                    "community_id": ci,
                    "size": len(members),
                    # 属性値の出現回数の合計。1 Fault が複数の属性値を持つため Fault 件数ではない
                    # （旧版は n_faults という名前で件数のように見せていた）。
                    "n_occurrences": int(sum(H.nodes[x].get("count", 0) for x in members)),
                    "key_attrs": [x.replace("::", ": ") for x in members[:10]],
                }
            )
        out.sort(key=lambda c: -c["n_occurrences"])
        out = out[:10]
        with self._lock:
            self._cache["comm"] = out
            self._cache["n2c"] = n2c
        return out

    def _network(self, max_nodes=60, max_edges=300):
        with self._lock:
            if "net" in self._cache:
                return self._cache["net"]
        self._communities()
        with self._lock:
            n2c = self._cache.get("n2c", {})
        vocab, counts, top, sub = self._top_subgraph(max_nodes)
        edges = sorted(
            [(int(top[a]), int(top[b]), float(w)) for a, b, w in zip(sub.row, sub.col, sub.data) if a < b and w > 0],
            key=lambda e: -e[2],
        )[:max_edges]
        used = {x for a, b, _ in edges for x in (a, b)}
        nodes = []
        for gi in sorted(used):
            col, val = vocab[gi].split("::", 1)
            nodes.append(
                {
                    "id": vocab[gi],
                    "label": str(val),
                    "type": col,
                    "count": int(counts[gi]),
                    "community": n2c.get(vocab[gi], -1),
                }
            )
        net = {
            "nodes": nodes,
            "edges": [{"from": vocab[a], "to": vocab[b], "weight": int(w)} for a, b, w in edges],
        }
        with self._lock:
            self._cache["net"] = net
        return net

    # ------------------------------------------------------------------
    # 急増パターン検知
    # ------------------------------------------------------------------
    def detect_surges(self, recent_days=90, min_recent=3, min_lift=2.0, fdr=0.1):
        """直近期間で構成比が増えた属性（単独／2属性の組合せ）を検出する。

        旧版は構成比の比（リフト）だけで判定していたため、直近3件しかない組合せが
        「×24倍」として上位に並びうる状態だった。ここでは帰無仮説「直近の発生率は
        以前と同じ」の下でのポアソン上側確率を p値として計算し、Benjamini-Hochberg
        法で多重比較を補正したうえで有意フラグを立てる。
        """
        recent_days = int(_clip(int(recent_days), 1, MAX_RECENT_DAYS))
        if not self.date_column:
            return {"available": False, "reason": "日付列がないため急増検知は実行できません", "items": []}
        d = self.df[self.date_column]
        tmax = d.max()
        if pd.isna(tmax):
            return {"available": False, "reason": "日付をパースできませんでした", "items": []}

        cut = tmax - pd.Timedelta(days=recent_days)
        rec, prev = (d > cut), (d <= cut)
        nr, npv = int(rec.sum()), int(prev.sum())
        if nr < min_recent or npv < min_recent:
            return {"available": False, "reason": "比較に十分な期間データがありません", "items": []}

        keys = []
        for i, ca in enumerate(self.attr_columns):
            keys.append((ca, self.df[ca]))
            for cb in self.attr_columns[i + 1 :]:
                # どちらかが欠損している行は組合せとして数えない
                pair = (self.df[ca].astype(str) + " × " + self.df[cb].astype(str)).where(
                    self.df[ca].notna() & self.df[cb].notna()
                )
                keys.append((f"{ca}×{cb}", pair))

        items = []
        for label, series in keys:
            rc = series[rec].value_counts(dropna=True)
            pc = series[prev].value_counts(dropna=True)
            for value, cnt in rc.items():
                cnt = int(cnt)
                if cnt < min_recent:
                    continue
                prev_cnt = int(pc.get(value, 0))
                p_rec = cnt / nr
                # ラプラス平滑化した以前の発生率。新規出現でも有限のリフトになる。
                rate_prev = (prev_cnt + 0.5) / (npv + 1.0)
                lift = p_rec / rate_prev
                if lift < min_lift:
                    continue
                items.append(
                    {
                        "pattern": f"{label}: {value}",
                        "recent_count": cnt,
                        "prev_count": prev_cnt,
                        "recent_share": round(p_rec * 100, 2),
                        "prev_share": round(prev_cnt / npv * 100, 2),
                        "lift": "新規出現" if prev_cnt == 0 else round(float(lift), 2),
                        "lift_value": round(float(lift), 2),
                        "p_value": float(poisson.sf(cnt - 1, max(nr * rate_prev, 1e-12))),
                    }
                )

        for it, q in zip(items, self._bh_qvalues([x["p_value"] for x in items])):
            it["q_value"] = round(q, 5)
            it["significant"] = bool(q <= fdr)
            it["p_value"] = round(it["p_value"], 6)
        items.sort(key=lambda x: (not x["significant"], x["q_value"], -x["lift_value"]))
        return {
            "available": True,
            "recent_days": recent_days,
            "n_recent": nr,
            "n_prev": npv,
            "n_tested": len(items),
            "fdr": fdr,
            "items": items[:15],
        }

    @staticmethod
    def _bh_qvalues(pvals):
        """Benjamini-Hochberg 法で p値を q値（FDR補正）に変換する。"""
        m = len(pvals)
        if m == 0:
            return []
        p = np.asarray(pvals, dtype=float)
        order = np.argsort(p)
        ranked = p[order] * m / (np.arange(m) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        q = np.empty(m)
        q[order] = np.clip(ranked, 0.0, 1.0)
        return q.tolist()

    # ------------------------------------------------------------------
    # ③ 類似検索
    # ------------------------------------------------------------------
    def find_similar(self, fault_index=None, attrs=None, top_k=10):
        top_k = int(_clip(int(top_k), 1, MAX_TOP_K))
        if fault_index is not None:
            fault_index = int(fault_index)
            if not 0 <= fault_index < len(self.df):
                raise ValueError("Fault ID が見つかりません")
            q = self.E[fault_index]
        else:
            if not any(v not in (None, "") for v in (attrs or {}).values()):
                raise ValueError("属性を1つ以上選択してください")
            q, _ = self.embed_query(attrs)
            if q is None:
                raise ValueError("指定された属性値が過去データに1件も存在しません")

        sims = self.E @ q
        out = []
        for i in self._top_indices(sims, top_k + (1 if fault_index is not None else 0)):
            i = int(i)
            if fault_index is not None and i == fault_index:
                continue
            r = self.df.iloc[i]
            item = {
                "fault_id": self.fault_ids[i],
                "similarity": round(float(sims[i]) * 100, 1),
                "attrs": {c: str(r[c]) for c in self.attr_columns if pd.notna(r.get(c))},
            }
            for key, col in (
                ("cause", self.cause_column),
                ("action", self.action_column),
                ("note", self.text_column),
            ):
                if col and pd.notna(r.get(col)):
                    item[key] = str(r[col])
            if self.date_column and pd.notna(r.get(self.date_column)):
                item["date"] = str(pd.to_datetime(r[self.date_column]).date())
            out.append(item)
            if len(out) >= top_k:
                break
        return out

    def index_of_fault_id(self, fault_id):
        """Fault ID → 行番号。list.index() の O(n) 走査を避ける。"""
        return self._fault_id_index.get(fault_id)

    def search_fault_ids(self, query="", limit=200):
        """Fault ID の部分一致検索。全件（最大20万件）をブラウザへ返さないための入口。"""
        limit = int(_clip(int(limit), 1, 1000))
        q = (query or "").strip().upper()
        hits = self.fault_ids[:limit] if not q else [f for f in self.fault_ids if q in f][:limit]
        return {
            "fault_ids": hits,
            "total": len(self.fault_ids),
            "truncated": len(self.fault_ids) > len(hits),
        }

    # ------------------------------------------------------------------
    def attr_value_options(self, max_values=500):
        return {
            c: sorted(self.df[c].dropna().astype(str).unique().tolist())[:max_values]
            for c in self.attr_columns
        }

    def summary(self):
        warnings = []
        if not self.cause_column:
            warnings.append("原因列がないため「①原因予測」と精度検証は使用できません。")
        if not self.date_column:
            warnings.append("日付列がないため「急増検知」は使用できません。")
        for c in self.dropped_columns:
            warnings.append(f"列「{c['column']}」を属性から除外しました（{c['reason']} / {c['n_unique']}種）。")
        return {
            "n_faults": int(len(self.df)),
            "n_nodes": self.n_graph_nodes,
            "n_edges": self.n_graph_edges,
            "n_attr_values": int(len(self.vocab)),
            "attr_columns": self.attr_columns,
            "cause_column": self.cause_column,
            "date_column": self.date_column,
            "dropped_columns": self.dropped_columns,
            "embed_dim": int(self.E.shape[1]),
            "warnings": warnings,
        }
