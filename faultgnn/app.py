# -*- coding: utf-8 -*-
"""Fault Management 解析アプリ（Flask）

運用・セキュリティ上の要点:
  - secret_key は環境変数、debug はデフォルト無効
  - アップロードサイズ・行数の上限
  - セッションストアは LRU + TTL（無制限に増え続けるメモリリークの防止）
  - 全APIで例外を捕捉し、スタックトレースを外部に出さない
  - すべてのAPIパラメータをサーバ側で範囲検証する（過大な値による DoS の防止）
  - 状態を変更するAPIには CSRF トークンを要求する
"""
from __future__ import annotations

import functools
import hmac
import io
import logging
import os
import secrets
import threading
import time
import uuid
from collections import OrderedDict

import pandas as pd
from flask import Flask, jsonify, render_template, request, session

from graph_engine import (
    DEFAULT_ACTION_COLUMN,
    DEFAULT_ATTR_COLUMNS,
    DEFAULT_CAUSE_COLUMN,
    DEFAULT_DATE_COLUMN,
    DEFAULT_TEXT_COLUMN,
    MAX_CLUSTERS,
    MAX_CONTAMINATION,
    MAX_RECENT_DAYS,
    MAX_TOP_K,
    MIN_CONTAMINATION,
    FaultAnalyzer,
)
from sample_data import build_sample_dataframe

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger(__name__)

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "32"))
MAX_ROWS = int(os.environ.get("MAX_ROWS", "200000"))
MAX_ATTR_COLUMNS = int(os.environ.get("MAX_ATTR_COLUMNS", "8"))
STORE_CAPACITY = int(os.environ.get("STORE_CAPACITY", "32"))
STORE_TTL_SEC = int(os.environ.get("STORE_TTL_SEC", str(60 * 60)))
ENCODINGS = ("utf-8-sig", "cp932", "utf-16")

app = Flask(__name__)
_secret = os.environ.get("SECRET_KEY")
if not _secret:
    # プロセスごとに別の鍵になるため、複数ワーカー構成ではセッションが成立しない。
    log.warning(
        "SECRET_KEY が未設定のため一時鍵を生成しました。"
        "複数ワーカー/再起動をまたぐ運用では必ず環境変数で設定してください。"
    )
    _secret = secrets.token_hex(32)
app.secret_key = _secret
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("SESSION_COOKIE_SECURE", "0") == "1",
)
# Flask 3 では JSON_SORT_KEYS 設定は無効。キーをソートすると attr_options の順序が
# 列順から辞書順に変わり、入力フォームの並びがデータの列順と食い違う。
app.json.sort_keys = False

# CDN(cdnjs) 以外のスクリプト読み込みとインラインスクリプトを禁止する。
# esc() によるエスケープが破られた場合の二重の防御。
CSP = (
    "default-src 'self'; "
    "script-src 'self' https://cdnjs.cloudflare.com; "
    "style-src 'self' https://cdnjs.cloudflare.com 'unsafe-inline'; "
    "font-src 'self' https://cdnjs.cloudflare.com; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "base-uri 'none'; "
    "form-action 'self'; "
    "frame-ancestors 'none'"
)


class TTLStore:
    """LRU + TTL 付きのプロセス内ストア。複数ワーカー構成では Redis 等に置き換えること。"""

    def __init__(self, capacity, ttl):
        self._d = OrderedDict()
        self._cap, self._ttl = capacity, ttl
        self._lock = threading.Lock()

    def _purge(self):
        now = time.time()
        for k in [k for k, (t, _) in self._d.items() if now - t > self._ttl]:
            self._d.pop(k, None)
        while len(self._d) > self._cap:
            self._d.popitem(last=False)

    def put(self, k, v):
        with self._lock:
            self._d[k] = (time.time(), v)
            self._d.move_to_end(k)
            self._purge()

    def get(self, k):
        with self._lock:
            self._purge()
            if k not in self._d:
                return None
            self._d.move_to_end(k)
            return self._d[k][1]

    def __len__(self):
        with self._lock:
            return len(self._d)


STORE = TTLStore(STORE_CAPACITY, STORE_TTL_SEC)


# ----------------------------------------------------------------------
# セッション / CSRF
# ----------------------------------------------------------------------
def _csrf_token():
    token = session.get("csrf")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf"] = token
    return token


@app.before_request
def _csrf_protect():
    if request.method not in ("POST", "PUT", "PATCH", "DELETE"):
        return None
    expected, sent = session.get("csrf"), request.headers.get("X-CSRF-Token", "")
    if not expected or not hmac.compare_digest(expected, sent):
        return jsonify({"ok": False, "error": "セッションが無効です。ページを再読み込みしてください。"}), 403
    return None


@app.after_request
def _security_headers(resp):
    resp.headers.setdefault("Content-Security-Policy", CSP)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "same-origin")
    if request.path.startswith("/api/"):
        resp.headers.setdefault("Cache-Control", "no-store")
    return resp


def _analyzer():
    token = session.get("token")
    a = STORE.get(token) if token else None
    if a is None:
        raise LookupError("データが読み込まれていません（またはセッションが期限切れです）。再度読み込んでください。")
    return a


def _register(a):
    token = str(uuid.uuid4())
    STORE.put(token, a)
    session["token"] = token


def _loaded_payload(a):
    return jsonify(
        {
            "ok": True,
            "summary": a.summary(),
            "attr_options": a.attr_value_options(),
            "preview": a.df.head(8).astype(str).to_dict(orient="records"),
            "columns": list(map(str, a.df.columns)),
        }
    )


def api_guard(fn):
    """例外を握りつぶしてスタックトレースの外部漏洩を防ぐデコレータ。"""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except LookupError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        except MemoryError:
            log.exception("out of memory in %s", fn.__name__)
            return jsonify({"ok": False, "error": "データが大きすぎて処理できませんでした。"}), 507
        except Exception:
            log.exception("unhandled error in %s", fn.__name__)
            return jsonify({"ok": False, "error": "サーバー内部エラーが発生しました。"}), 500

    return wrapper


# ----------------------------------------------------------------------
# パラメータ検証
# ----------------------------------------------------------------------
def _payload():
    d = request.get_json(silent=True)
    return d if isinstance(d, dict) else {}


def _int_param(d, key, default, lo, hi, allow_none=False):
    v = d.get(key, default)
    if v in (None, "") and allow_none:
        return None
    if v in (None, ""):
        v = default
    try:
        v = int(float(v))
    except (TypeError, ValueError):
        raise ValueError(f"パラメータ {key} は整数で指定してください")
    return max(lo, min(hi, v))


def _float_param(d, key, default, lo, hi):
    v = d.get(key, default)
    if v in (None, ""):
        v = default
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"パラメータ {key} は数値で指定してください")
    if v != v:  # NaN
        raise ValueError(f"パラメータ {key} は数値で指定してください")
    return max(lo, min(hi, v))


def _attrs_param(d, key="attrs"):
    a = d.get(key)
    if a in (None, ""):
        return {}
    if not isinstance(a, dict):
        raise ValueError("attrs はオブジェクトで指定してください")
    return {str(k): ("" if v is None else str(v)) for k, v in list(a.items())[:64]}


# ----------------------------------------------------------------------
# ルーティング
# ----------------------------------------------------------------------
@app.errorhandler(413)
def too_large(e):
    return jsonify({"ok": False, "error": f"ファイルが大きすぎます（上限{MAX_UPLOAD_MB}MB）"}), 413


@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": "存在しないAPIです"}), 404
    return e


@app.route("/")
def index():
    return render_template("index.html", csrf_token=_csrf_token())


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True, "sessions": len(STORE)})


@app.route("/api/load_sample", methods=["POST"])
@api_guard
def load_sample():
    a = FaultAnalyzer(build_sample_dataframe())
    _register(a)
    return _loaded_payload(a)


def _read_csv(raw: bytes):
    """文字コードを推定しつつ CSV を読む。デコード失敗・解析失敗の両方で次を試す。"""
    last_error = None
    for enc in ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except (UnicodeDecodeError, UnicodeError, pd.errors.ParserError) as e:
            last_error = e
            continue
        except pd.errors.EmptyDataError:
            raise ValueError("CSVが空です")
    log.info("csv decode failed: %s", last_error)
    raise ValueError("CSVを読み込めませんでした（UTF-8 / Shift_JIS のいずれかで保存してください）")


@app.route("/api/upload", methods=["POST"])
@api_guard
def upload():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "ファイルが見つかりません"}), 400
    raw = request.files["file"].read()
    if not raw:
        return jsonify({"ok": False, "error": "ファイルが空です"}), 400

    df = _read_csv(raw)
    if df.empty:
        return jsonify({"ok": False, "error": "CSVにデータ行がありません"}), 400
    if len(df) > MAX_ROWS:
        return jsonify({"ok": False, "error": f"行数が上限({MAX_ROWS:,}行)を超えています"}), 400

    reserved = {DEFAULT_CAUSE_COLUMN, DEFAULT_ACTION_COLUMN, DEFAULT_TEXT_COLUMN, DEFAULT_DATE_COLUMN}
    attrs = [c for c in DEFAULT_ATTR_COLUMNS if c in df.columns]
    if not attrs:
        attrs = [c for c in df.columns if c not in reserved][:MAX_ATTR_COLUMNS]

    a = FaultAnalyzer(df, attr_columns=attrs)
    _register(a)
    return _loaded_payload(a)


@app.route("/api/predict_cause", methods=["POST"])
@api_guard
def predict_cause():
    d = _payload()
    result = _analyzer().predict_cause(_attrs_param(d), top_k=_int_param(d, "top_k", 5, 1, MAX_TOP_K))
    return jsonify({"ok": True, "result": result})


@app.route("/api/evaluate", methods=["POST"])
@api_guard
def evaluate():
    return jsonify({"ok": True, "result": _analyzer().evaluate_cause_model()})


@app.route("/api/discover_patterns", methods=["POST"])
@api_guard
def discover_patterns():
    d = _payload()
    result = _analyzer().discover_patterns(
        n_clusters=_int_param(d, "n_clusters", None, 2, MAX_CLUSTERS, allow_none=True),
        contamination=_float_param(d, "contamination", 0.08, MIN_CONTAMINATION, MAX_CONTAMINATION),
        recent_days=_int_param(d, "recent_days", 90, 1, MAX_RECENT_DAYS),
    )
    return jsonify({"ok": True, "result": result})


@app.route("/api/find_similar", methods=["POST"])
@api_guard
def find_similar():
    d = _payload()
    a = _analyzer()
    top_k = _int_param(d, "top_k", 10, 1, MAX_TOP_K)
    fault_id = d.get("fault_id")
    if fault_id:
        idx = a.index_of_fault_id(str(fault_id))
        if idx is None:
            return jsonify({"ok": False, "error": f"Fault ID '{fault_id}' が見つかりません"}), 400
        return jsonify({"ok": True, "result": a.find_similar(fault_index=idx, top_k=top_k)})
    return jsonify({"ok": True, "result": a.find_similar(attrs=_attrs_param(d), top_k=top_k)})


@app.route("/api/fault_ids", methods=["GET"])
@api_guard
def fault_ids():
    """全件返すと20万件の <option> でブラウザが固まるため、検索＋件数上限で返す。"""
    limit = max(1, min(1000, int(request.args.get("limit", 200) or 200)))
    return jsonify({"ok": True, **_analyzer().search_fault_ids(request.args.get("q", ""), limit=limit)})


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
