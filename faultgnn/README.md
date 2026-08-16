# Fault Management Analyzer

Fault Management（設備故障管理）データから、以下の3手法を1つのFlask Webアプリで実行できます。

| 手法 | やること | 内部アルゴリズム |
|---|---|---|
| ① 原因予測 | 新規Faultの属性から原因候補を確率付きで推定 | 埋め込み近傍のkNN重み付き投票 |
| ② 未知パターン発見 | 頻出パターン・急増中の未知パターンを発見 | KMeans / IsolationForest / コミュニティ検出 / ポアソン検定＋FDR補正 |
| ③ 類似Fault検索 | 今回のFaultと似た過去事例を原因・対策とともに検索 | 埋め込みのコサイン類似度 |

## セットアップ

```bash
python3 -m venv venv
source venv/bin/activate            # Windowsは venv\Scripts\activate
pip install -r requirements.txt
export SECRET_KEY="$(python3 -c 'import secrets;print(secrets.token_hex(32))')"
python app.py
```

ブラウザで `http://127.0.0.1:5000` を開いてください。

### 環境変数

| 変数 | 既定値 | 用途 |
|---|---|---|
| `SECRET_KEY` | 起動ごとにランダム | セッション署名鍵。**本番では必ず固定値を設定**（未設定だと再起動・複数ワーカーでセッションが切れる） |
| `HOST` / `PORT` | `127.0.0.1` / `5000` | 待ち受け先 |
| `FLASK_DEBUG` | `0` | `1` でデバッグモード（本番では使用しない） |
| `SESSION_COOKIE_SECURE` | `0` | HTTPS配信時は `1` |
| `MAX_UPLOAD_MB` / `MAX_ROWS` | `32` / `200000` | アップロード上限 |
| `STORE_CAPACITY` / `STORE_TTL_SEC` | `32` / `3600` | セッションストアの上限と保持時間 |

## 使い方

1. **STEP1**: Fault管理CSVをアップロードするか、「サンプルデータで試す」でデモデータを生成します。
   - 推奨列名: `装置, 部品, 症状, Error, 条件, 工程, 原因, 対策, 備考, 発生日`
   - 列名が異なっても、`原因/対策/備考/発生日` 以外の列を属性列として自動推定します。
   - `原因` 列がない場合は①原因予測と精度検証、`発生日` 列がない場合は急増検知が無効になります（画面に理由を表示します）。
   - ID列のような高カーディナリティ列は自動的に属性から除外します。
2. **STEP2**: タブで3手法を切り替えて実行します。
   - **① 原因予測**: 属性を選んで実行。原因候補が確率順に、根拠となった過去Faultとともに表示されます。
     併せて「精度検証」を必ず実行してください（後述）。
   - **② 未知パターン発見**: 急増パターン・頻出クラスタ・外れ値Fault・関連コミュニティ・共起ネットワークを表示します。
   - **③ 類似Fault検索**: 既存Fault IDまたは新規属性から、類似事例を原因・対策付きで表示します。

## アルゴリズム

```
Fault × 属性値 の incidence 行列
  → TF-IDF（レアな属性値ほど情報量を大きく）
  → TruncatedSVD（24次元）
  → Fault-Fault 共有属性グラフ上での近傍平滑化（1ホップのメッセージパッシング）
```

- 平滑化は概念上 `E ← (1-α)E + α·P·E`（P は Fault間類似度 `S = X Xᵀ` の行正規化）ですが、
  `S` を実体化すると O(n²) メモリで破綻するため、結合則 `S E = X (Xᵀ E)` により **O(nnz·d)** で計算します。
- 埋め込みは**属性列のみ**から構成します。原因列は検索後の集計にのみ使い、ラベルリークを避けます。
- 新規Faultのクエリにも保存済み埋め込みと同じ平滑化を適用します（train/serve skew の回避）。
- ②のクラスタ数はシルエット係数で自動選択、クラスタの特徴は最頻値ではなく**リフト値**（全体平均比）で表示します。
- 急増検知は「直近N日 vs それ以前」の構成比を比較し、ポアソン上側確率を p値として
  Benjamini-Hochberg 法で多重比較補正します。**件数が少ないだけの偶然のゆらぎは「参考」に落ちます。**

## テストとベンチマーク

```bash
pip install -r requirements-dev.txt
pytest                      # 54件（エンジンの回帰テスト + HTTP APIのテスト。GNNモジュール未導入でも実行可）
python3 benchmark.py        # 1,000 / 5,000 / 20,000件 のスケール計測
python3 benchmark.py 100000 # 件数指定
```

計測値（この環境での参考値）:

| 件数 | 構築 | パターン発見 | 精度検証 | 類似検索 |
|---|---|---|---|---|
| 20,000 | 0.6秒 | 4.0秒 | 1.2秒 | 0.01秒未満 |
| 100,000 | 2.1秒 | 7.2秒 | 1.4秒 | 0.01秒未満 |

## GNNモジュール（オプション、PyTorch Geometric）

Web アプリ本体（`app.py`）が使う手法は、学習される重みを持たない軽量な埋め込み手法です
（下記「本番投入前に検討すべき点」参照）。それとは別に、**実際に学習する異種グラフGNN**を
`torch` / `torch_geometric` で実装したスタンドアロン実験モジュールを同梱しています。
**Flaskアプリの `/api/*` には統合していません**（あくまで比較・検証用のCLIスクリプトです）。

```bash
pip install -r requirements-gnn.txt   # torch / torch_geometric を追加インストール（CPU想定）
python3 train_gnn.py --task both      # ノード分類・リンク予測の両方を学習・評価
```

### アーキテクチャ

- `fault`（故障）と `attr_value`（属性値）の2種類のノードからなる二部グラフ（`HeteroData`）。
  `attr_value` は `graph_engine.FaultAnalyzer.vocab` をそのまま使うため、**原因列は構造的に
  含まれません**（本体アプリと同じくラベルリークを防止）。
- `attr_value` ノードは学習可能な埋め込みテーブル、`fault` ノードは学習可能パラメータを
  持たせずゼロベクトルから開始し、`SAGEConv` によるメッセージパッシング（2層）のみで
  表現を獲得します。故障ごとの自由パラメータを持たせないのは、リンク予測の教師信号が
  Fault-Faultペアであるため、そこに自由度を与えると「構造から学習」ではなく「ペアの丸暗記」に
  なってしまうからです。
- **① ノード分類（原因予測の代替）**: `fault` ノードの埋め込みから原因クラスを分類。
  test集合は本体アプリの `evaluate_cause_model()` と**同一の分割**（`random_state=0`）にしてあり、
  比較表に1行追加する形でそのまま横並び比較できます。
- **② リンク予測（③類似Fault検索の代替）**: 属性値を`min_shared_attrs`個以上共有するFaultペアを
  正例として学習します。**メッセージパッシングにはこの類似度エッジを一切使いません**
  （構造的な分離は `tests/test_gnn.py::test_encoder_output_is_unaffected_by_similarity_edges` で保証）。

### 実測値（同梱サンプルデータ、既定パラメータ、CPU）

```
model                                     top1    top3
------------------------------------------------------
多数決ベースライン                                0.298       -
埋め込みkNN（本アプリ①）                           0.488   0.802
勾配ブースティング（CatBoost相当）                    0.496   0.835
GNN（PyTorch Geometric, GraphSAGE）        0.496   0.769
------------------------------------------------------
リンク予測: AUC-ROC 0.96 / Average Precision 0.96 / Precision@10 0.95
実行時間: 約7秒（torchのインポート込み、学習自体は数秒）
```

Top-1は勾配ブースティングと同水準まで伸びましたが、Top-3は表形式モデルにわずかに劣ります。
**この規模・この特徴量では表形式モデルを明確に上回るわけではない**という、既存の検証結果と
整合する正直な結果です。リンク予測（類似Fault検索の代替）は良好な精度が出ていますが、
これは「属性を多く共有するFaultほどリンクを引く」という定義そのものが学習しやすい構造の
タスクであることに留意してください（実際に役立つ「類似」の定義かどうかは別途検証が必要です）。

主な `train_gnn.py` オプション: `--task {node,link,both}` `--csv PATH`（省略時はサンプルデータ）
`--epochs` `--hidden-dim` `--min-shared-attrs` `--link-decoder {dot,mlp}` `--seed`
`--device {auto,cpu,cuda}`（既定は `auto`。**GPU(CUDA)が使える環境では自動的に優先して使用**します。
`pip install torch` は既定でCUDA対応wheelが入るため、追加設定なしでGPUを検出・利用できます。
明示的にCPUで実行したい場合は `--device cpu` を指定してください）

## セキュリティ

- CSV由来の文字列はすべて `esc()` を通してから描画します（保存型XSS対策）。
  加えてサーバが `Content-Security-Policy` を返し、インライン/外部スクリプトの実行元を制限します。
- 状態を変更するAPI（POST）は `X-CSRF-Token` ヘッダを要求します。トークンはトップページの
  `<meta name="csrf-token">` から取得してください（curl等でAPIを直接叩く場合も同様）。
- APIパラメータはすべてサーバ側で範囲検証します（過大なクラスタ数などによるDoS対策）。
- 例外のスタックトレースは外部に出さず、サーバログにのみ記録します。

## 閉域網（オフライン）で使う場合

Bootstrap / FontAwesome / vis-network をCDNから読み込んでいます。到達できない場合は
警告を表示したうえで解析機能自体は動作しますが、見た目とネットワーク図が制限されます。
資産をローカルに置く場合:

```bash
cd static && mkdir -p vendor && cd vendor
curl -O https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.3/css/bootstrap.min.css
curl -O https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.3/js/bootstrap.bundle.min.js
curl -O https://cdnjs.cloudflare.com/ajax/libs/vis-network/10.1.0/standalone/umd/vis-network.min.js
# FontAwesome はフォント本体も必要なため配布zipを展開して配置する
```

そのうえで `templates/index.html` のCDN URLを `/static/vendor/...` に書き換えてください。

## ファイル構成

```
faultgnn/
├── app.py                 # Flaskアプリ本体・APIエンドポイント
├── graph_engine.py        # 埋め込み・3手法のコアロジック
├── sample_data.py         # デモ用サンプルデータ生成
├── benchmark.py           # スケール計測
├── gnn_data.py            # [オプション] GNN用グラフ構築・分割（PyTorch Geometric）
├── gnn_model.py           # [オプション] 異種グラフGNN本体
├── train_gnn.py           # [オプション] GNN学習・評価CLI（Flaskアプリには未統合）
├── requirements.txt / requirements-dev.txt / requirements-gnn.txt
├── templates/index.html
├── static/{app.js, style.css}
└── tests/{conftest.py, test_engine.py, test_api.py, test_gnn.py}
```

## 本番投入前に検討すべき点

1. **Web アプリ本体（`app.py`）が使う手法は、真の意味でのGNNではありません。** TF-IDF+SVD に
   1ホップの近傍平滑化を加えたもので、学習される重みを持ちません。実際に学習する異種グラフGNN
   （GraphSAGEベース）を PyTorch Geometric で実装したスタンドアロンモジュールを
   `train_gnn.py` として同梱しています（上記「GNNモジュール」参照）。ただし実測が示すとおり、
   この規模・この特徴量では勾配ブースティングを明確に超えるわけではなく、Flaskアプリへの
   統合は行っていません。
2. **原因予測は表形式モデルに劣後します。** 同梱サンプルでの実測は
   多数決ベースライン Top-1 0.298 / 埋め込みkNN 0.488（Top-3 0.802）/ 勾配ブースティング 0.496（Top-3 0.835）。
   原因予測は表形式モデルを本番採用し、グラフ/埋め込みは②パターン発見・③類似検索に使うのが合理的です。
   **必ず実データで「精度を検証する」を実行し、判断してください。**
3. **自由記述テキストが未活用。** 日本語Sentence-BERT等で「現象・原因・処置内容」をベクトル化し
   特徴量に連結するのが次の一手です。
4. **CatBoost未使用。** 実行環境の制約から `HistGradientBoostingClassifier` で代替しています。
5. **プロセス内ストア。** 複数ワーカー/複数インスタンス構成では Redis 等への置き換えが必須です。
6. **認証・監査ログなし。** 障害データは機微情報です。社内展開時は認証必須です。
7. **精度そのものが実用水準とは限りません。** Top-1が約0.50、Top-3が約0.84というのはサンプルデータでの値です。
   「Top-3で担当者に候補提示する」運用が成立するかを実データで必ず確認してください。

## 同梱サンプルデータについて

`sample_data.py` は「Motor×高温で摩耗が多発」「装置B×E205×夜間×高湿度が最近急増している未知パターン」
を模した相関構造をあえて仕込んだデモ用データ（約400件）を生成します。
②の急増検知でこの装置B×E205パターンが「有意」として検出されることを確認できます。
`python3 sample_data.py` で `sample_fault_data.csv` を再生成できます。
