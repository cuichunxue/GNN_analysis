"use strict";

// XSS対策: CSV由来の文字列は必ずこの関数を通してから innerHTML に入れる。
// サーバ側の Content-Security-Policy と合わせた二重の防御になっている。
function esc(s) {
  if (s === null || s === undefined) return "";
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

const $ = (s) => document.querySelector(s);
const el = (t, c, h) => { const e = document.createElement(t); if (c) e.className = c; if (h !== undefined) e.innerHTML = h; return e; };
const CSRF = document.querySelector('meta[name="csrf-token"]').content;

let ATTR_OPTIONS = {};

// ---------- 通信 ----------
class ApiError extends Error {}

async function request(url, opts) {
  let res;
  try {
    res = await fetch(url, opts);
  } catch (e) {
    throw new ApiError("サーバーに接続できませんでした。");
  }
  let data = null;
  try {
    data = await res.json();
  } catch (e) {
    // 500 のHTMLエラーページなど、JSON以外が返ってきた場合
    throw new ApiError(`サーバーエラーが発生しました（HTTP ${res.status}）。`);
  }
  if (!data || data.ok !== true) {
    throw new ApiError((data && data.error) || `処理に失敗しました（HTTP ${res.status}）。`);
  }
  return data;
}

const postJSON = (url, body) =>
  request(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-CSRF-Token": CSRF },
    body: JSON.stringify(body || {}),
  });

const getJSON = (url) => request(url, { method: "GET" });

// 実行中はボタンを無効化する（重い解析の多重実行を防ぐ）
async function withBusy(btn, target, label, fn) {
  const old = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin me-1"></i>${esc(label)}`;
  if (target) $(target).innerHTML = `<div class="text-muted small"><i class="fa-solid fa-spinner fa-spin me-1"></i>${esc(label)}</div>`;
  try {
    await fn();
  } catch (e) {
    const msg = e instanceof ApiError ? e.message : "予期しないエラーが発生しました。";
    if (target) $(target).innerHTML = `<div class="alert alert-danger mb-0">${esc(msg)}</div>`;
    else setStatus(esc(msg), "danger");
  } finally {
    btn.disabled = false;
    btn.innerHTML = old;
  }
}

function setStatus(html, type = "info") {
  $("#loadStatus").innerHTML = `<div class="alert alert-${type} py-2 mb-0">${html}</div>`;
}

// 描画中の例外で、それまでに描画済みの結果まで消してしまわないようにする
function safeRender(box, label, fn) {
  try {
    fn();
  } catch (e) {
    console.error(e);
    box.appendChild(el("div", "alert alert-warning py-2 small", `${esc(label)}の描画に失敗しました。`));
  }
}

// CDN（Bootstrap / FontAwesome / vis-network）に到達できない閉域網でも
// 解析機能自体は動く。読み込めなかったことだけ利用者に伝える。
window.addEventListener("load", () => {
  const missing = [];
  if (typeof bootstrap === "undefined") missing.push("Bootstrap");
  if (typeof vis === "undefined") missing.push("vis-network（ネットワーク図）");
  if (!missing.length) return;
  const w = $("#cdnWarning");
  w.className = "container pt-3";
  w.innerHTML = `<div class="alert alert-warning py-2 mb-0 small">
    外部CDNから ${esc(missing.join(" / "))} を読み込めませんでした。解析機能は動作しますが、
    見た目と一部の可視化が制限されます。閉域網ではREADMEの手順でCDN資産をローカルに配置してください。</div>`;
});

// ---------- データ読込 ----------
$("#btnSample").addEventListener("click", () =>
  withBusy($("#btnSample"), null, "生成中...", async () => {
    handleLoad(await postJSON("/api/load_sample"));
  }));

$("#btnUpload").addEventListener("click", () => {
  const f = $("#csvFile");
  if (!f.files.length) return setStatus("CSVファイルを選択してください。", "warning");
  const fd = new FormData();
  fd.append("file", f.files[0]);
  withBusy($("#btnUpload"), null, "解析中...", async () => {
    handleLoad(await request("/api/upload", { method: "POST", headers: { "X-CSRF-Token": CSRF }, body: fd }));
  });
});

function handleLoad(d) {
  const s = d.summary;
  let msg = `<i class="fa-solid fa-circle-check me-1"></i>読込完了：${esc(s.n_faults)}件 / ${esc(s.n_nodes)}ノード / ${esc(s.n_edges)}エッジ（埋め込み${esc(s.embed_dim)}次元）`;
  (s.warnings || []).forEach((w) => { msg += `<br><span class="small">※ ${esc(w)}</span>`; });
  setStatus(msg, "success");

  ATTR_OPTIONS = d.attr_options || {};
  renderSummary(s);
  renderPreview(d.columns, d.preview);
  buildForm("#causeAttrForm", "cause_");
  buildForm("#simNewBlock", "sim_");
  refreshFaultIds("");
  ["#causeResult", "#evalResult", "#patternResult", "#similarResult"].forEach((sel) => { $(sel).innerHTML = ""; });
  $("#networkSection").classList.add("d-none");
  $("#dataSummary").classList.remove("d-none");
  $("#analysisCard").classList.remove("d-none");
}

function renderSummary(s) {
  const w = $("#summaryCards");
  w.innerHTML = "";
  [["Fault件数", s.n_faults], ["属性値の種類", s.n_attr_values], ["ノード数", s.n_nodes], ["エッジ数", s.n_edges],
   ["埋め込み次元", s.embed_dim], ["原因列", s.cause_column || "なし"], ["日付列", s.date_column || "なし"]]
    .forEach(([l, v]) => {
      const c = el("div", "col-auto summary-chip");
      c.innerHTML = `<div class="val">${esc(v)}</div><div class="lbl">${esc(l)}</div>`;
      w.appendChild(c);
    });
}

function renderPreview(cols, rows) {
  $("#previewTable").innerHTML =
    "<thead><tr>" + cols.map((c) => `<th>${esc(c)}</th>`).join("") + "</tr></thead><tbody>" +
    rows.map((r) => "<tr>" + cols.map((c) => `<td>${esc(r[c])}</td>`).join("") + "</tr>").join("") + "</tbody>";
}

function buildForm(sel, prefix) {
  const w = $(sel);
  w.innerHTML = "";
  Object.entries(ATTR_OPTIONS).forEach(([col, vals], i) => {
    const id = `${prefix}${i}`;
    const d = el("div", "col-md-4");
    d.innerHTML = `<label class="form-label small text-muted" for="${id}">${esc(col)}</label>
      <select class="form-select" id="${id}" data-col="${esc(col)}">
      <option value="">（未選択）</option>${vals.map((v) => `<option value="${esc(v)}">${esc(v)}</option>`).join("")}</select>`;
    w.appendChild(d);
  });
}

// 列名をそのままDOM idにすると記号入りの列名でセレクタが壊れるため、data-col から集める
function collect(prefix) {
  const a = {};
  document.querySelectorAll(`[id^="${prefix}"][data-col]`).forEach((s) => {
    if (s.value) a[s.dataset.col] = s.value;
  });
  return a;
}

// ---------- Fault ID 検索 ----------
// 全件を <option> に展開するとデータ次第でブラウザが固まるため、サーバ側で絞り込む
let faultIdTimer = null;
async function refreshFaultIds(q) {
  try {
    const d = await getJSON(`/api/fault_ids?q=${encodeURIComponent(q || "")}&limit=200`);
    $("#faultIdList").innerHTML = d.fault_ids.map((f) => `<option value="${esc(f)}"></option>`).join("");
    $("#faultIdHint").textContent = d.truncated
      ? `全${d.total}件中 ${d.fault_ids.length}件を表示（入力すると絞り込みます）`
      : `全${d.total}件`;
    if (!$("#simFaultId").value && d.fault_ids.length) $("#simFaultId").value = d.fault_ids[0];
  } catch (e) {
    $("#faultIdHint").textContent = "";
  }
}
$("#simFaultId").addEventListener("input", (e) => {
  clearTimeout(faultIdTimer);
  const v = e.target.value;
  faultIdTimer = setTimeout(() => refreshFaultIds(v), 250);
});

// ---------- タブ ----------
document.querySelectorAll("#methodTabs .nav-link").forEach((b) => {
  b.addEventListener("click", () => {
    document.querySelectorAll("#methodTabs .nav-link").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    document.querySelectorAll(".tab-pane").forEach((p) => p.classList.add("d-none"));
    $("#pane-" + b.dataset.tab).classList.remove("d-none");
  });
});
document.querySelectorAll('input[name="simMode"]').forEach((r) =>
  r.addEventListener("change", () => {
    const existing = $("#simModeExisting").checked;
    $("#simExistingBlock").classList.toggle("d-none", !existing);
    $("#simNewBlock").classList.toggle("d-none", existing);
  }));

// ---------- ① 原因予測 ----------
$("#btnPredictCause").addEventListener("click", () =>
  withBusy($("#btnPredictCause"), "#causeResult", "予測中...", async () => {
    const d = await postJSON("/api/predict_cause", { attrs: collect("cause_"), top_k: 5 });
    const r = d.result;
    const box = $("#causeResult");
    box.innerHTML = "";
    if (!r.candidates.length) {
      box.innerHTML = `<div class="alert alert-warning mb-0">近傍に原因が記録されたFaultがありませんでした。</div>`;
      return;
    }
    const given = Object.keys(collect("cause_"));
    const unknown = given.filter((c) => !r.matched_attrs.includes(c));
    if (unknown.length) {
      box.appendChild(el("div", "alert alert-warning py-2 small",
        `過去データに存在しない属性は無視しました: ${esc(unknown.join(", "))}`));
    }
    r.candidates.forEach((c) => {
      const card = el("div", "result-card");
      const width = Math.max(0, Math.min(100, Number(c.probability) || 0));
      card.innerHTML = `<div class="d-flex justify-content-between align-items-center mb-2">
          <div class="fw-bold fs-5">${esc(c.cause)}</div>
          <div class="fs-5 fw-bold" style="color:#4f46e5">${esc(c.probability)}%</div></div>
        <div class="cause-bar-wrap mb-2"><div class="cause-bar" style="width:${width}%"></div></div>
        <div class="small-muted mb-1">根拠となった類似過去Fault（近傍${esc(c.n_neighbors)}件中の上位）：</div>
        <div>${c.evidence.map((e) => `<span class="attr-tag">${esc(e.fault_id)} ${esc(e.similarity)}%</span>`).join("")}</div>`;
      box.appendChild(card);
    });
  }));

// ---------- 精度評価 ----------
$("#btnEvaluate").addEventListener("click", () =>
  withBusy($("#btnEvaluate"), "#evalResult", "ホールドアウト評価を実行中...", async () => {
    const r = (await postJSON("/api/evaluate")).result;
    $("#evalResult").innerHTML = `
      <div class="small-muted mb-2">学習${esc(r.n_train)}件 / 検証${esc(r.n_test)}件 / 原因${esc(r.n_classes)}クラス${r.sampled ? "（大規模データのためサブサンプルで評価）" : ""}</div>
      <table class="table table-sm table-bordered align-middle">
        <thead><tr><th>モデル</th><th>Top-1 正解率</th><th>Top-3 正解率</th></tr></thead>
        <tbody>${r.rows.map((x) => `<tr><td>${esc(x.model)}</td>
          <td><b>${esc(x.top1)}</b></td><td>${x.top3 === null ? "-" : "<b>" + esc(x.top3) + "</b>"}</td></tr>`).join("")}</tbody>
      </table>
      <div class="alert alert-warning py-2 small mb-2">
        勾配ブースティングが埋め込みkNNと同等以上なら、原因予測は表形式モデルを本番採用し、
        グラフ/埋め込みは②パターン発見・③類似検索に使うのが合理的です。</div>
      <div class="small-muted">${esc(r.note)}</div>`;
  }));

// ---------- ② パターン発見 ----------
$("#btnDiscover").addEventListener("click", () =>
  withBusy($("#btnDiscover"), "#patternResult", "解析中...", async () => {
    const r = (await postJSON("/api/discover_patterns", {
      n_clusters: $("#nClusters").value || null,
      contamination: $("#contamination").value || 0.08,
      recent_days: $("#recentDays").value || 90,
    })).result;

    const box = $("#patternResult");
    box.innerHTML = "";
    safeRender(box, "急増パターン", () => renderSurges(box, r.surges));
    safeRender(box, "頻出パターン", () => renderClusters(box, r));
    safeRender(box, "異常検知", () => renderAnomalies(box, r.anomalies));
    safeRender(box, "コミュニティ", () => renderCommunities(box, r.communities));
    safeRender(box, "共起ネットワーク", () => renderNetwork(r.network));
  }));

function renderSurges(box, s) {
  box.appendChild(el("h6", "cluster-title mb-2", '<i class="fa-solid fa-arrow-trend-up me-1"></i>直近の急増パターン'));
  if (!s.available) {
    box.appendChild(el("div", "alert alert-secondary py-2 small", esc(s.reason)));
    return;
  }
  if (!s.items.length) {
    box.appendChild(el("div", "alert alert-success py-2 small", "急増している属性・組合せは検出されませんでした。"));
    return;
  }
  const t = el("div", "table-responsive mb-4");
  t.innerHTML = `<div class="small-muted mb-1">直近${esc(s.recent_days)}日(${esc(s.n_recent)}件) vs それ以前(${esc(s.n_prev)}件)　/
    候補${esc(s.n_tested)}件をBenjamini-Hochberg法(FDR=${esc(s.fdr)})で多重比較補正</div>
    <table class="table table-sm table-bordered align-middle"><thead><tr>
    <th>パターン</th><th>直近件数</th><th>以前の構成比</th><th>直近の構成比</th><th>増加倍率</th><th>q値</th></tr></thead><tbody>
    ${s.items.map((x) => `<tr class="${x.significant ? "" : "text-muted"}"><td>${esc(x.pattern)}
      ${x.significant ? '<span class="badge bg-danger ms-1">有意</span>' : '<span class="badge bg-secondary ms-1">参考</span>'}</td>
      <td>${esc(x.recent_count)}</td><td>${esc(x.prev_share)}%</td><td><b>${esc(x.recent_share)}%</b></td>
      <td><span class="badge badge-anomaly">${esc(x.lift)}</span></td><td>${esc(x.q_value)}</td></tr>`).join("")}</tbody></table>
    <div class="small-muted">※ q値は「実際には急増していないのに検出してしまう」期待割合。件数が少ないだけの偶然のゆらぎは「参考」に落ちます。</div>`;
  box.appendChild(t);
}

function renderClusters(box, r) {
  const kInfo = r.chosen_k ? `（k=${esc(r.chosen_k)}${r.silhouette.length ? " をシルエット係数で自動選択" : ""}）` : "";
  box.appendChild(el("h6", "cluster-title mb-2 mt-4", `<i class="fa-solid fa-layer-group me-1"></i>頻出パターン${kInfo}`));
  if (!r.clusters.length) {
    box.appendChild(el("div", "text-muted small mb-4", "クラスタリングを行うには件数が不足しています。"));
    return;
  }
  const cr = el("div", "row g-3 mb-2");
  r.clusters.forEach((c) => {
    const col = el("div", "col-md-6");
    col.innerHTML = `<div class="result-card h-100">
      <div class="d-flex justify-content-between"><div class="fw-bold">クラスタ #${esc(c.cluster_id)}</div>
      <span class="badge bg-secondary">${esc(c.size)}件</span></div>
      <div class="mt-2">${c.features.length
        ? c.features.map((f) => `<span class="attr-tag">${esc(f.col)}: ${esc(f.val)}
          <b>${esc(f.share)}%</b> <span style="color:#b91c1c">×${esc(f.lift)}</span></span>`).join("")
        : '<span class="small-muted">際立った特徴はありません</span>'}</div>
      ${c.top_cause ? `<div class="small-muted mt-2">主な原因: ${esc(c.top_cause)}</div>` : ""}</div>`;
    cr.appendChild(col);
  });
  box.appendChild(cr);
  box.appendChild(el("div", "small-muted mb-4", "※ ×n はリフト値（全体平均比での出現しやすさ）。値が大きいほどそのクラスタ固有の特徴です。"));
}

function renderAnomalies(box, anomalies) {
  box.appendChild(el("h6", "cluster-title mb-2", '<i class="fa-solid fa-triangle-exclamation me-1"></i>外れ値Fault（異常検知）'));
  if (!anomalies.length) {
    box.appendChild(el("div", "text-muted small mb-4", "明確な外れ値は検出されませんでした。"));
    return;
  }
  const t = el("div", "table-responsive mb-4");
  t.innerHTML = `<table class="table table-sm table-bordered align-middle">
    <thead><tr><th>Fault ID</th><th>属性の組合せ</th><th>原因</th><th>異常スコア</th></tr></thead><tbody>
    ${anomalies.map((a) => `<tr><td><span class="badge badge-anomaly">${esc(a.fault_id)}</span></td>
      <td>${Object.entries(a.attrs).map(([k, v]) => `<span class="attr-tag">${esc(k)}: ${esc(v)}</span>`).join("")}</td>
      <td>${esc(a.cause || "-")}</td><td>${esc(a.anomaly_score)}</td></tr>`).join("")}</tbody></table>`;
  box.appendChild(t);
}

function renderCommunities(box, communities) {
  box.appendChild(el("h6", "cluster-title mb-2", '<i class="fa-solid fa-people-group me-1"></i>関連コミュニティ（属性値のまとまり）'));
  if (!communities || !communities.length) {
    box.appendChild(el("div", "text-muted small mb-4", "コミュニティは検出されませんでした。"));
    return;
  }
  const cr = el("div", "row g-3 mb-4");
  communities.forEach((c) => {
    const col = el("div", "col-md-6");
    col.innerHTML = `<div class="result-card h-100">
      <div class="d-flex justify-content-between align-items-center">
        <div class="fw-bold"><span class="legend-dot" style="background:${colorFor(c.community_id)}"></span>コミュニティ #${esc(c.community_id)}</div>
        <span class="badge bg-secondary">${esc(c.size)}属性値 / のべ${esc(c.n_occurrences)}出現</span></div>
      <div class="mt-2">${c.key_attrs.map((k) => `<span class="attr-tag">${esc(k)}</span>`).join("")}</div></div>`;
    cr.appendChild(col);
  });
  box.appendChild(cr);
}

// ---------- 共起ネットワーク ----------
let NET = null;
const PALETTE = ["#4f46e5", "#059669", "#d97706", "#db2777", "#0891b2", "#7c3aed", "#ca8a04", "#dc2626", "#16a34a", "#2563eb"];
const colorFor = (c) => (c === -1 || c === null || c === undefined ? "#9ca3af" : PALETTE[c % PALETTE.length]);

function renderNetwork(nd) {
  const section = $("#networkSection"), box = $("#networkGraph"), lg = $("#networkLegend");
  section.classList.remove("d-none");
  if (typeof vis === "undefined") {
    box.innerHTML = '<div class="d-flex align-items-center justify-content-center h-100 text-muted small p-3 text-center">'
      + 'ネットワーク描画ライブラリ(vis-network)を読み込めなかったため、この図は表示できません。<br>'
      + '上の解析結果（急増パターン・クラスタ・異常検知・コミュニティ）は有効です。</div>';
    lg.innerHTML = "";
    return;
  }
  if (!nd || !nd.nodes.length) {
    if (NET) { NET.destroy(); NET = null; }
    box.innerHTML = '<div class="d-flex align-items-center justify-content-center h-100 text-muted small">描画できる共起関係がありません。</div>';
    lg.innerHTML = "";
    return;
  }
  lg.innerHTML = [...new Set(nd.nodes.map((n) => n.community))].sort((a, b) => a - b)
    .map((c) => `<span class="attr-tag" style="border-left:4px solid ${colorFor(c)}">${c === -1 ? "その他" : "コミュニティ #" + esc(c)}</span>`).join("");

  // vis-network の title/label はテキストとして扱われるため、ここでは esc 不要
  const nodes = new vis.DataSet(nd.nodes.map((n) => ({
    id: n.id, label: n.label,
    title: `${n.type}: ${n.label}\n出現数: ${n.count}\nコミュニティ: ${n.community === -1 ? "その他" : "#" + n.community}`,
    value: n.count, color: { background: colorFor(n.community), border: "#333" }, font: { size: 13 },
  })));
  const mx = Math.max(...nd.edges.map((e) => e.weight), 1);
  const edges = new vis.DataSet(nd.edges.map((e) => ({
    from: e.from, to: e.to, value: e.weight, title: `共起回数: ${e.weight}`,
    color: { color: "#c7cae8", opacity: 0.3 + 0.5 * (e.weight / mx) },
  })));
  if (NET) NET.destroy();
  box.innerHTML = "";
  NET = new vis.Network(box, { nodes, edges }, {
    nodes: { shape: "dot", scaling: { min: 10, max: 40 }, borderWidth: 1 },
    edges: { smooth: { type: "continuous" }, scaling: { min: 1, max: 8 } },
    physics: { stabilization: { iterations: 150 }, barnesHut: { gravitationalConstant: -4000, springLength: 130, springConstant: 0.02 } },
    interaction: { hover: true, tooltipDelay: 100 },
  });
}

// ---------- ③ 類似検索 ----------
$("#btnFindSimilar").addEventListener("click", () =>
  withBusy($("#btnFindSimilar"), "#similarResult", "検索中...", async () => {
    const p = { top_k: parseInt($("#simTopK").value, 10) || 10 };
    if ($("#simModeExisting").checked) p.fault_id = $("#simFaultId").value.trim();
    else p.attrs = collect("sim_");
    const rows = (await postJSON("/api/find_similar", p)).result;
    if (!rows.length) {
      $("#similarResult").innerHTML = `<div class="alert alert-warning mb-0">結果が見つかりませんでした。</div>`;
      return;
    }
    $("#similarResult").innerHTML = `<div class="table-responsive"><table class="table table-sm table-bordered align-middle">
      <thead><tr><th>類似度</th><th>Fault ID</th><th>発生日</th><th>属性</th><th>原因</th><th>対策</th></tr></thead><tbody>
      ${rows.map((r) => `<tr><td><span class="badge badge-sim">${esc(r.similarity)}%</span></td>
        <td>${esc(r.fault_id)}</td><td class="small-muted">${esc(r.date || "-")}</td>
        <td>${Object.entries(r.attrs).map(([k, v]) => `<span class="attr-tag">${esc(k)}: ${esc(v)}</span>`).join("")}</td>
        <td>${esc(r.cause || "-")}</td><td class="small-muted">${esc(r.action || "-")}</td></tr>`).join("")}
      </tbody></table></div>`;
  }));
