# -*- coding: utf-8 -*-
"""HTTP API のテスト（セッション・CSRF・入力検証・エラー処理）。"""
import io
import re

import pytest

import app as app_module


@pytest.fixture
def client():
    app_module.app.config.update(TESTING=True)
    app_module.STORE = app_module.TTLStore(8, 60)
    with app_module.app.test_client() as c:
        yield c


def csrf(client):
    """トップページを開いて CSRF トークンを得る（ブラウザと同じ手順）。"""
    html = client.get("/").get_data(as_text=True)
    return re.search(r'name="csrf-token" content="([^"]+)"', html).group(1)


def post(client, url, json=None, token=None, **kw):
    headers = {"X-CSRF-Token": token if token is not None else csrf(client)}
    return client.post(url, json=json, headers=headers, **kw)


def load_sample(client):
    r = post(client, "/api/load_sample")
    assert r.status_code == 200 and r.get_json()["ok"]
    return r.get_json()


CSV = (
    "装置,部品,症状,Error,条件,工程,原因,対策,備考,発生日\n"
    + "\n".join(
        f"装置{'AB'[i % 2]},Motor,振動,E10{i % 3},高温,組立,摩耗,交換,メモ,2026-0{i % 6 + 1}-01"
        for i in range(40)
    )
    + "\n"
)


# ----------------------------------------------------------------- 基本
def test_index_serves_page_with_csrf_and_security_headers(client):
    r = client.get("/")
    assert r.status_code == 200
    assert 'name="csrf-token"' in r.get_data(as_text=True)
    assert "script-src" in r.headers["Content-Security-Policy"]
    assert r.headers["X-Content-Type-Options"] == "nosniff"


def test_healthz(client):
    assert client.get("/healthz").get_json()["ok"] is True


def test_unknown_api_returns_json_404(client):
    assert client.get("/api/nope").get_json()["ok"] is False


# ----------------------------------------------------------------- CSRF
def test_post_without_csrf_token_is_rejected(client):
    assert client.post("/api/load_sample").status_code == 403


def test_post_with_wrong_csrf_token_is_rejected(client):
    assert post(client, "/api/load_sample", token="wrong").status_code == 403


# ----------------------------------------------------------------- セッション
def test_apis_require_loaded_data(client):
    token = csrf(client)
    for url in ("/api/predict_cause", "/api/evaluate", "/api/discover_patterns", "/api/find_similar"):
        r = post(client, url, json={}, token=token)
        assert r.status_code == 400 and "読み込まれていません" in r.get_json()["error"]
    assert client.get("/api/fault_ids").status_code == 400


def test_expired_session_reports_a_readable_error(client):
    load_sample(client)
    app_module.STORE = app_module.TTLStore(8, 60)  # ストアだけ揮発した状態を再現
    r = post(client, "/api/find_similar", json={"fault_id": "F00001"})
    assert r.status_code == 400 and "セッション" in r.get_json()["error"]


# ----------------------------------------------------------------- 読み込み
def test_load_sample_returns_summary_and_options(client):
    d = load_sample(client)
    assert d["summary"]["n_faults"] > 300
    assert set(d["summary"]["attr_columns"]) >= {"装置", "部品"}
    assert d["preview"] and d["columns"]


def test_upload_csv(client):
    r = post(
        client,
        "/api/upload",
        data={"file": (io.BytesIO(CSV.encode("utf-8")), "f.csv")},
        content_type="multipart/form-data",
    )
    d = r.get_json()
    assert d["ok"] and d["summary"]["n_faults"] == 40


def test_upload_cp932_csv(client):
    r = post(
        client,
        "/api/upload",
        data={"file": (io.BytesIO(CSV.encode("cp932")), "f.csv")},
        content_type="multipart/form-data",
    )
    assert r.get_json()["ok"]


def test_upload_rejects_empty_and_garbage(client):
    token = csrf(client)
    r = post(client, "/api/upload", data={}, content_type="multipart/form-data", token=token)
    assert r.status_code == 400
    r = post(
        client,
        "/api/upload",
        data={"file": (io.BytesIO(b""), "f.csv")},
        content_type="multipart/form-data",
        token=token,
    )
    assert r.status_code == 400 and r.get_json()["ok"] is False


def test_upload_does_not_leak_internal_details(client):
    r = post(
        client,
        "/api/upload",
        data={"file": (io.BytesIO(b"\x00\x01\x02\x03"), "f.csv")},
        content_type="multipart/form-data",
    )
    body = r.get_json()
    assert body["ok"] is False
    assert "Traceback" not in body["error"] and "pandas" not in body["error"]


# ----------------------------------------------------------------- 解析API
def test_predict_cause(client):
    load_sample(client)
    d = post(client, "/api/predict_cause", json={"attrs": {"部品": "Motor", "条件": "高温"}}).get_json()
    assert d["ok"] and d["result"]["candidates"][0]["cause"] == "摩耗"


def test_predict_cause_reports_input_errors(client):
    load_sample(client)
    r = post(client, "/api/predict_cause", json={"attrs": {}})
    assert r.status_code == 400 and "1つ以上" in r.get_json()["error"]


def test_discover_patterns_with_extreme_parameters_is_clamped(client):
    load_sample(client)
    d = post(
        client,
        "/api/discover_patterns",
        json={"n_clusters": 999999, "contamination": 5, "recent_days": -1},
    ).get_json()
    assert d["ok"] and d["result"]["chosen_k"] <= 30


def test_non_numeric_parameters_return_400_not_500(client):
    load_sample(client)
    r = post(client, "/api/discover_patterns", json={"n_clusters": "たくさん"})
    assert r.status_code == 400 and r.get_json()["ok"] is False


def test_malformed_json_body_is_tolerated(client):
    load_sample(client)
    r = client.post(
        "/api/predict_cause",
        data="{not json",
        content_type="application/json",
        headers={"X-CSRF-Token": csrf(client)},
    )
    assert r.status_code == 400 and r.get_json()["ok"] is False


def test_attrs_must_be_an_object(client):
    load_sample(client)
    r = post(client, "/api/find_similar", json={"attrs": ["装置A"]})
    assert r.status_code == 400


def test_find_similar_by_id_and_unknown_id(client):
    load_sample(client)
    d = post(client, "/api/find_similar", json={"fault_id": "F00002", "top_k": 5}).get_json()
    assert d["ok"] and len(d["result"]) == 5
    r = post(client, "/api/find_similar", json={"fault_id": "F99999"})
    assert r.status_code == 400 and "見つかりません" in r.get_json()["error"]


def test_evaluate(client):
    load_sample(client)
    d = post(client, "/api/evaluate").get_json()
    assert d["ok"] and len(d["result"]["rows"]) == 3


def test_fault_ids_are_paged_and_searchable(client):
    load_sample(client)
    d = client.get("/api/fault_ids?limit=10").get_json()
    assert len(d["fault_ids"]) == 10 and d["truncated"] and d["total"] > 300
    d = client.get("/api/fault_ids?q=F0001").get_json()
    assert all("F0001" in f for f in d["fault_ids"])
    # 上限を超える limit を指定してもサーバ側で丸める
    d = client.get("/api/fault_ids?limit=99999").get_json()
    assert len(d["fault_ids"]) <= 1000


# ----------------------------------------------------------------- XSS
def test_csv_payloads_are_returned_as_data_not_markup(client):
    """CSVの値がそのままHTMLとして解釈されないこと（描画側は esc() を通す）。"""
    payload = "<img src=x onerror=alert(1)>"
    csv = "装置,部品,原因\n" + "\n".join(f'"{payload}",Motor,摩耗' for _ in range(10)) + "\n"
    post(
        client,
        "/api/upload",
        data={"file": (io.BytesIO(csv.encode("utf-8")), "x.csv")},
        content_type="multipart/form-data",
    )
    d = post(client, "/api/find_similar", json={"attrs": {"装置": payload}}).get_json()
    assert d["ok"] and d["result"][0]["attrs"]["装置"] == payload
    assert "text/html" not in d and client.get("/api/fault_ids").headers["Content-Type"].startswith(
        "application/json"
    )


# ----------------------------------------------------------------- ストア
def test_store_evicts_by_capacity():
    store = app_module.TTLStore(2, 60)
    store.put("a", 1)
    store.put("b", 2)
    store.put("c", 3)
    assert store.get("a") is None and store.get("c") == 3


def test_store_expires_by_ttl():
    store = app_module.TTLStore(4, 0)
    store.put("a", 1)
    assert store.get("a") is None
