# `/v1/systemone` のワイヤ形式と sokudan の対応

`sokudan serve`（`sokudan/serve/systemone.py`、`sokudan/serve/wire.py`）が話す形式の記録です。

- **読んだもの**（2026-09-27 取得）:
  - TypeSafe の公開ドキュメント: [docs.typesafe.ai/api](https://docs.typesafe.ai/api)、[primitives](https://docs.typesafe.ai/primitives) 以下の Choice / Score / Noul、[confidence](https://docs.typesafe.ai/confidence)、[state](https://docs.typesafe.ai/concepts/state)、JavaScript SDK の型のページ（`SystemOneRequestPayload`、`ChoiceResponse`、`ScoreResponse`、`NoulResponse`、`Usage`、`ChoiceCriteria`、`ScoreCriteria`）。
  - 公開実装の README: [InterfazeAI/lev](https://github.com/InterfazeAI/lev)（Apache-2.0）、[jaredpalmer/kev](https://github.com/jaredpalmer/kev)（Apache-2.0）、[NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)（Apache-2.0）。
- **使っていないもの**: TypeSafe の SDK（`typesafe_sdk`）はインストールも import もしていません。TypeSafe の API も呼んでいません。互換性の確認は、素の HTTP と上の公開例で行いました（`tests/test_systemone_server.py`、例は `tests/fixtures/systemone_wire_examples.json`）。
- 課題の指示にあった `interfaze-ai/lev` は GitHub では 404 で、README は `InterfazeAI/lev` にありました（HF の重みは `interfaze-ai/lev`）。

## 1. リクエスト

`POST /v1/systemone`、`Content-Type: application/json`。

| フィールド | 読み取った形 | 出典 | sokudan の対応 |
|---|---|---|---|
| `Authorization` | `Bearer <API_KEY>`。無い・不正は 401 | API ref | **受けるだけで検査しない**（ローカル用）。`/health` の `auth` に明記 |
| `state` | 必須。string / object / array。SDK の型は `null` も許す | API ref、SDK `SystemOneRequestPayload` | string はそのまま。object / array は **`Agent.predict` に変更せず渡し**、既存の描画（object は最上位のキーごとに `key: value` の行、array は 1 要素 1 行、`{role, content}` は `role: content`）で読む。どの経路かを各レスポンスの `sokudan.state_format` に、意味を `/health` の `state_rendering` に出す。`null` は 422 |
| `model` | 必須の string（`"jev-latest"` など） | API ref | **任意**。受けて無視し、読み込んだモデルで答える。レスポンスの `model` は常に `sokudan-ja-310m` |
| `questions` | 必須。`map<string, Question>`。キーは利用者が決め、モデルには渡らない | API ref | 1〜64 問（`SOKUDAN_MAX_QUESTIONS`）。空のキーは 422。**全質問を `Agent.predict` の 1 回の呼び出しで処理** |
| `questions.*.type` | `"noul"` / `"choice"` / `"score"` | API ref | 同じ 3 つ。加えて sokudan 独自の `"bool"` を `"noul"` と同じに扱う。それ以外は 422 |
| `questions.*.instructions` | 必須。string / object / array（object なら質問とデータを別フィールドに） | API ref。SDK の型では任意・`null` 可 | 必須。object は `key: value` の行、array は 1 要素 1 行に描画（入れ子の値は JSON）。空は 422 |
| `noul` の `criteria` | 任意。`{"true"?: 説明, "false"?: 説明}`、各説明は string / object / array | API ref、SDK `NoulQuestion` | sokudan の bool の 2 つのマーカー文に足す: `はい: <true の説明>` / `いいえ: <false の説明>`。**精度への影響は測定していない**。`true` / `false` 以外のキーは 422 |
| `choice` の `criteria` | 必須。`map<選択肢, 説明 or null>`、説明は string / object / array。最大 255 | API ref、`ChoiceCriteria` | 選択肢名 → 説明の文字列。`null` は空文字にし、**選択肢名だけのマーカー文**になる（既存の描画: 説明があれば `名前: 説明`）。順序は送られたまま。1〜255。空・空白の選択肢名は 422 |
| `score` の `criteria` | 必須。順序付きの水準の配列（低い方から）。2 以上、API は 10 まで | API ref、Score の頁。SDK の型では各水準 `null` 可 | 水準テキストの配列。2〜10 水準。`null` や空の水準は 422（sokudan は水準をテキストで読むため）。描画後に同じ文字列になる水準は、sokudan の parser が 422 を返す |
| 質問の未知のキー | 記載なし | — | 無視する（Laya の `labels` など、クライアント独自のキーで壊さないため） |

## 2. レスポンス

| フィールド | 読み取った形 | 出典 | sokudan の対応 |
|---|---|---|---|
| `model` | 答えたモデル（`"jev-1.13.0"`） | API ref | `"sokudan-ja-310m"`。読み込んだ参照（`GeneLab/sokudan-ja-310m` など）は `/health` の `model_ref` |
| `answers` | `map<質問 id, Answer>`、同じキー | API ref | 同じキー |
| noul の答え | `{"type": "noul", "noul": P(yes)}`。confidence は無い | API ref、`NoulResponse` | 同じ。`predict` の `type: "bool"` を `"noul"` に直す |
| choice の答え | `{"type": "choice", "choice", "probabilities": {選択肢: 確率}, "confidence"}` | API ref、`ChoiceResponse` | 同じ 4 フィールド。`probabilities` は criteria と同じ順序 |
| score の答え | `{"type": "score", "score": 確率で重み付けした水準, "legend": {"0": 水準, …}, "probabilities": {"0": 確率, …}, "confidence"}` | API ref、`ScoreResponse`、`ScoreLegend` | 同じ 5 フィールド。`score` は `predict` の期待値。`legend` は**送られた水準をそのまま**（object の水準なら object）返す |
| `probabilities` | 合計 1 の浮動小数 | API ref | head の softmax（`predict` の値、小数第 4 位で丸め）。既定は未較正。`--temperatures` を渡すと較正後 |
| `confidence` | 0〜1、確率から導く。定義は API ref に無い | confidence の頁 | 下の §3 の定義で、返す `probabilities` から計算する。**`predict` の `confidence`（最大確率）とは別の値** |
| `usage` | `{"input_tokens", "output_tokens"}` | API ref、SDK `Usage` | `input_tokens` = backbone が読んだトークン数（joint encoding では state を質問の数だけ数える）。`output_tokens` は常に 0（生成しない） |
| sokudan 独自 | — | — | 最上位の `sokudan` に 1 つにまとめる: `state_format`、`state_tokens`、`state_truncated`、`backbone_passes`、`calibrated`、`order_marginalize`、`latency_ms`。`usage` は仕様の 2 フィールドだけにしてある |

## 3. confidence の定義

| 型 | 定義 | 出典 |
|---|---|---|
| choice | `(p_max − 1/K) / (1 − 1/K)`、[0, 1] に切る。K = 1 なら 1 | docs.typesafe.ai/confidence の説明用ウィジェットの式 `(K·peak − 1)/(K − 1)`。kev と Laya の README も Jev の定義としてこの式を挙げる |
| score | `max(0, 1 − E|level − mode| / D)`。`mode` は最も確率の高い水準、`D` は水準上の一様分布の中央からの平均距離（3 水準で 2/3） | kev の README（TypeSafe の参照アダプタ `system-one-adapter` 0.2.1 の式とある）。TypeSafe のドキュメント本文には式が無い |

- 公開されている答えで確かめました（`test_confidence_formula_reproduces_published_confidences`）。公開値の確率は小数第 2 位までなので、一致は 0.011 以内です。
  - API ref の choice: 確率 0.88 / 0.12 / 0.0 → 0.82（公開値 0.81）。score: 0.0 / 0.95 / 0.05 → 0.925（0.92）。
  - kev の例: choice 0.47 / 0.28 / 0.25 → 0.205（0.21）。score 0.00 / 0.56 / 0.44 → 0.34（0.34）。

## 4. エラー

| 状態 | 読み取った意味 | sokudan |
|---|---|---|
| 401 | キーが無い・不正 | 返さない（検査しない） |
| 422 | 本文の検証に失敗。本文に問題のフィールド | 同じ。FastAPI の `detail`（`loc` と `msg`）か、sokudan の parser の理由の文字列 |
| 429 | レート制限 | キューが満杯のとき（`--max-concurrency` + `--max-queue` を超えたとき）。`Retry-After: 1` |
| 529 | 過負荷 | 返さない（429 に寄せる） |
| — | — | 413: state が 100,000 文字を超える。503: モデルが無い |

## 5. 公開例のあいだの食い違い

| 項目 | TypeSafe のドキュメント | Lev | kev | Laya | sokudan の選択 |
|---|---|---|---|---|---|
| `model` | 必須 | in-process の例とサーバーの例には無い（SDK が既定値を付けるはず、未確認） | 例に `"kev-latest"` | curl の例に無い | 任意 |
| レスポンスの `model` | 解決後の版（`jev-1.13.0`） | 記載なし | 要求の名前をそのまま（`kev-latest`） | 記載なし | 自分の名前（`sokudan-ja-310m`） |
| choice の選択肢数 | 最大 255（最小は記載なし） | 数百（ルータ） | 1〜255 | トークン予算で決まり、126〜254 を超えると 422 | 1〜255 |
| score の水準数 | 2 以上、API は 10 まで | 2〜10 | **1〜255** | 記載なし | 2〜10 |
| score の `null` 水準 | SDK の型は許す | 記載なし | 記載なし | **422** | 422 |
| `instructions` | API ref は必須、SDK の型は任意・`null` 可 | 記載なし | 任意 | 記載なし | 必須 |
| `state: null` | SDK の型は許す | 記載なし | 記載なし | 記載なし | 422 |
| confidence | 式は本文に無い（ウィジェットは choice の式） | 記載なし（`levbench confidence` で統計量を調べる） | choice / score の式を明記 | **1 − 正規化エントロピー**で、Jev と違うと明記 | kev の式（§3） |
| `usage.output_tokens` | 例では 18〜34（0 ではない） | 0 | 答えを直列化したトークン数（例で 161） | 0 | 0 |
| 最上位の追加フィールド | 無し | 記載なし | `latency_ms` | `routing`（Router の場合） | `sokudan` |
| 認証 | 必須 | 記載なし | `KEV_API_KEY` を設定したときだけ必須 | `LAYA_API_KEY` を設定したときだけ必須、不正な形式は 401 | 検査しない |
| state が object のとき | 推奨（「多くのリクエストで object を」） | 例は `{"ticket": …}` | 「object と array はラベル付きのテキストに変換」 | 例は `{"body": …}` | 既存の `key: value` 行。**sokudan では文字列を推奨**（学習時の入力と違うため。README の注意） |

- Lev と Laya の README には、生の JSON のレスポンス例がありません（Lev は属性アクセス、Laya は「Jev と同じスキーマ」と書くだけ）。形の比較は、TypeSafe の API ref の 3 例と kev の 1 例に対して行いました。
- Lev のサーバーの例、kev の Python の例は SDK のオブジェクトで書かれています。フィクスチャでは同じフィールドの JSON に書き直し、そのことを `translated` に記録しました。

## 6. まだ無いもの

- **推論時の順序平均（`order_marginalize`）**: 設定だけあり、既定は off。on にすると起動を拒否します（黙って無視しない）。実装は、順序の実験の結果を見て決めます。
- `/v1/models`（kev にある）、`permute` / `separate`（kev にある）、`x-typesafe-request-id` ヘッダ: 実装していません。
