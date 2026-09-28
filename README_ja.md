# 即断 / sokudan

*[English README](README.md)*

**日本語 System One 意思決定モデル。**
日本語テキスト（state）と型付き質問（questions）を受け取り、
**テキストを一切生成せずに**単一フォワードパスで型付き回答と確率を返すエンコーダモデルです。

生成しないので、パースするものがなく、ハルシネーションする余地もありません。

- モデル: `sokudan-ja-310m`（314.6M / バックボーン [`sbintuitions/modernbert-ja-310m`](https://huggingface.co/sbintuitions/modernbert-ja-310m), MIT）
- ライセンス: Apache-2.0
- 開発環境: RTX 5090 (Blackwell, sm_120) 1枚

> **Status: v0.3.0。** 重みは v0.2（2026-09-27）と同じで、v0.1（2026-09-20〜21 の 2 日スプリント）と同じ設定で学習した 8 本の重みを平均した model soup です。v0.2.1 では、`bool` の温度較正を既定で on にし、`/v1/systemone` 互換サーバー（`sokudan serve`）を加えました。v0.3.0 では、Apple Silicon で MLX で動くようにし、プラットフォームごとに入る依存を分けました（[CHANGELOG](https://github.com/hiroki-abe-58/sokudan/blob/main/CHANGELOG.md)）。
> 公開先: [`GeneLab/sokudan-ja-310m`](https://huggingface.co/GeneLab/sokudan-ja-310m)（v0.1 は revision `v0.1`）
> **この README の数値はすべて本機で実行したコードの出力です。** 推定値はありません。
> 未測定のものは「測定していない」と書きます。

## v0.2 — v0.1 の 8 シードの model soup

v0.2 は、v0.1 と同じ設定で学習した 8 本（seed 0〜7）の重みを、そのまま平均したモデルです。
アーキテクチャ、学習データ、推論コードは v0.1 と同じで、推論は 1 本分のコストのままです。
どの組み合わせを出すかは事前に決めた規則で選び（[`docs/release_candidate.md`](docs/release_candidate.md)）、
`bench_ja` はリリース規則を commit してから 1 回だけ測りました。

`bench_ja` 300 件（較正前。v0.1 は 3 シード平均 ± SD、**v0.2 は soup 1 体の 1 回の値**）:

| | choice acc | score RPS↓ | score acc | bool acc | bool AUROC |
|---|---|---|---|---|---|
| **v0.2** | **0.880** | **0.075** | **0.817** | 0.780 | **0.844** |
| v0.1 | 0.847 ± 0.009 | 0.090 ± 0.023 | 0.763 ± 0.088 | 0.788 ± 0.010 | 0.789 ± 0.043 |

- **bool acc は v0.1 より 0.008 低い値です。**
  - `bool` は true を過少予測します（mean P(true) 0.133、gold の陽性率 0.297）。
  - 閾値は、利用者の事前確率に合わせて決めてください。
- 英語（`bench_en` 290 件、日本語のみで学習）: choice 0.872、score RPS 0.114、bool acc 0.690（多数決 0.683 とほぼ同じ）。
- **soup の作り方**:
  - 8 本のチェックポイントの浮動小数のテンソルを、すべて float32 で平均します（ヘッドを含む）。
  - `scripts/make_soup.py` が同じ規則で平均し、メンバーと soup の SHA-256 を照合します。
  - メンバーのチェックポイントは配布していません。学習し直しても同じ重みにはなりません（スコアラの初期値がシードで決まらないこと、GPU の非決定性）。
- **v0.1 を使い続ける場合**: `sokudan.load("GeneLab/sokudan-ja-310m@v0.1")`。
- 全指標は [`docs/benchmarks.md`](docs/benchmarks.md) §10 とモデルカードにあります。
- 位置感度を含む限界は、下の Limits にあります。

## `bench_ja` 300 件、3 シード平均 ± 標準偏差（v0.1）

| 対象 | choice acc | score RPS↓ | bool acc | bool AUROC |
|---|---|---|---|---|
| **sokudan-ja-310m v0.1** | **0.847 ± 0.009** | **0.090 ± 0.023** | **0.788 ± 0.010** | **0.789 ± 0.043** |
| `laya-multilingual` (ja) | 0.747 | 0.232 | 0.543 | 0.523 |
| 多数決クラス | 0.380 | 0.197 | 0.703 | — |
| ランダム | 0.253 | 0.201 | 0.513 | — |

全指標・較正前後・シード別の生値は [`docs/benchmarks.md`](docs/benchmarks.md)。
**`score` のシード分散は大きく**、v0.1 として配布している重み（seed 0、revision `v0.1`）の実測は acc 0.663 / RPS 0.117 です。

---

## なぜ作ったか — 先に測った結果

作る前に、既存の System One 系モデルが**日本語で**どう振る舞うかを実測しました
（300件の日本語業務文 `bench_ja`、全文同梱、[`docs/baseline_ja.md`](docs/baseline_ja.md)）。

### `bench_ja` / `bench_en` のライセンスと使い方

`data/bench_ja.jsonl` と `data/bench_en.jsonl` は **CC BY 4.0** で配布します（コードの Apache-2.0 とは別です）。

> **評価用途を想定しており、学習データとしての利用は控えてください。**
> これはお願いであり、ライセンス上の制限ではありません。
> このセットは未知スキーマへの汎化を測るための held-out テストセットで、
> 一度学習に使われるとその役目を果たせなくなります。

> **訂正（2026-09-21）。** `bench_ja` の **bool 質問（「送信者は解約・契約終了を示唆しているか」）は
> 学習データに一度も現れていません**。ただし当日の `bench_ja` リーク検査は不完全で、
> **`解約` が train+val の質問文 718 行、`解除` が 868 行に含まれていました**。
> いずれも `choice` 質問の選択肢説明（`criteria` の値）と選択肢ラベルとしての出現です
> （`survey_freetext` の「申込や解約などの手順」、`contract_clause` の選択肢「解除」）。
> 当時の検査が `criteria` の**キーしか読んでおらず値を検査していなかった**こと、および
> `解除` が語彙リストに入っていなかったことが原因です。
> カタログと検査は `scripts/check_catalog_leak.py` で修正済み。
> **2026-09-20 の s0 / s1 の数値は訂正しません**——bool 質問は未出現であり、
> 語彙が入っていてもなお `bool` が転移しなかったという事実は変わらないためです。

| 対象 | choice acc | score RPS↓ | bool acc | bool AUROC |
|---|---|---|---|---|
| `laya-multilingual` (ja) | 0.747 | 0.232 | 0.543 | **0.523** |
| 多数決クラス | 0.380 | **0.197** | **0.703** | — |
| ランダム | 0.253 | 0.201 | 0.513 | — |

読み取れたこと:

- **`choice`（多クラス選択）は日本語でも動く。** 多数決の約2.0倍。ここは空白地帯ではありません
- **`score`（順序尺度）は多数決以下。** 原因を切り分けたところ、
  スキーマだけ変えた5条件すべてで**提示順の第1選択肢が300件中0〜1件しか選ばれない**
  位置バイアスでした。同じ「急がない」が、先頭だと0件、末尾だと250件
- **`bool` は AUROC 0.523** で、順位付けができていません。閾値のズレではありません

**そこで `sokudan` は `score` と `bool` を狙います。** `choice` は報告しますが目標にしません。

---

## Quickstart

```bash
pip install sokudan
```

- 開発版（`main` ブランチ）: `pip install git+https://github.com/hiroki-abe-58/sokudan.git`。手元で開発するときは、clone して `pip install -e .` です。
- Colab のノートブック（無料の CPU ランタイム。3 型、較正の on/off、`sokudan serve` を curl で叩くところまで）: [Open in Colab](https://colab.research.google.com/github/hiroki-abe-58/sokudan/blob/main/notebooks/sokudan_quickstart.ipynb)（[`notebooks/sokudan_quickstart.ipynb`](notebooks/sokudan_quickstart.ipynb)）

### インストールされるもの（v0.3.0）

`pip install sokudan` で入る配列ライブラリは、プラットフォームで変わります。

| プラットフォーム | 入るもの | `sokudan.load()` の実行先 |
|---|---|---|
| Apple Silicon、macOS 14 以降 | MLX（`mlx>=0.32.2,<0.33`）。**torch は入らない** | MLX、float16 |
| Linux、Windows、Intel Mac、macOS 13 の Apple Silicon | torch | torch（cuda → mps → cpu） |

- `pip install "sokudan[torch]"` は、どのプラットフォームでも torch を足します（Apple Silicon で `backend="torch"` を使うとき、学習・評価のスクリプトを動かすとき）。
- `pip install "sokudan[mlx]"` は MLX を明示する extra です（入るプラットフォームは上と同じ）。`pip install "sokudan[serve]"` はサーバーを足します。

### バックエンドと dtype（v0.3.0）

```python
agent = sokudan.load("GeneLab/sokudan-ja-310m")                         # backend="auto"
agent = sokudan.load("GeneLab/sokudan-ja-310m", backend="mlx", dtype="float32")
agent = sokudan.load("GeneLab/sokudan-ja-310m", backend="torch", device="cpu")
print(agent.backend)                                                     # 使われているバックエンド
```

- `backend="auto"`（既定）は MLX（Apple Silicon で mlx が入っているとき）→ torch の mps → cuda → cpu の順に試します。各候補は短いリクエストを 1 回答えてから使われ、失敗すると warning を出して次に進みます。
- `backend` や `device` を明示したときはフォールバックしません。入っていないライブラリを指定すると、`pip install "sokudan[torch]"` などを案内する ImportError になります。
- MLX の既定は float16 です（ヘッドは float32）。`bench_ja` では、MLX の float32 と float16 は、4 指標とも小数第 3 位まで torch と同じでした。
- 詳細と実測: [`docs/mlx_ja.md`](docs/mlx_ja.md)

```python
import sokudan

agent = sokudan.load("GeneLab/sokudan-ja-310m")     # または手元の runs/.../model.pt
result = agent.predict(
    "先月の請求で同じ金額が二回引き落とされています。至急ご確認ください。",   # state は文字列で渡す
    {
        "department": {"type": "choice",
                       "instructions": "この問い合わせはどの部署が担当すべきか",
                       "criteria": {"請求": "支払い・返金", "技術": "不具合・障害",
                                    "営業": "料金・新規契約", "その他": "上記以外"}},
        "urgency":    {"type": "score",
                       "instructions": "この依頼の緊急度は",
                       "criteria": ["急がない", "早めに", "業務が止まっている"]},
        "churn":      {"type": "noul",
                       "instructions": "解約を示唆しているか"},
    },
)
print(result["answers"]["department"]["choice"])
```

> **state は文字列で渡してください。** dict（例: `{"body": ...}`）で渡すと `key: value` の行に整形されます
> （`body: 先月の…`）。これは学習と評価で使った入力（本文そのまま）と違うので、出力が変わります。
> held-out の 1 事例では、P(true) が 0.318（文字列）から 0.145（`{"body": ...}`）に動きました。

質問タイプは `choice` / `score` / `bool`（`noul` はエイリアス）。
スキーマはリクエストごとに自由で、再学習は要りません。

### 確率の出所（ここが差別化点）

返ってくる確率は **head のロジットを直接読んだもの**です。
モデルに「どれくらい自信があるか」を自己申告させた値ではありません。
**v0.2.1 以降の `load` は、重みに同梱した `calibration.json` の `bool` の温度（1 つ）だけを既定で当てます。** `choice` と `score` の確率は較正していない生の値です。

```python
agent = sokudan.load("GeneLab/sokudan-ja-310m")                     # bool だけ較正（既定）
agent = sokudan.load("GeneLab/sokudan-ja-310m", temperatures=None)  # 較正なし（v0.2 と同じ）
```

- 応答の `calibrated` は、温度を当てた答えがあるかを表します。`calibrated_answers` は、その質問 ID の一覧です。
- `bool` の較正で、`bench_ja` の bool ECE は 0.181 → 0.105 になりました。bool acc と AUROC は変わりません（[`docs/calibration.md`](docs/calibration.md)）。
- `score` と `choice` を較正しないのは、val で fit した `score` の温度が `bench_ja` の score RPS を悪化させたためです（0.075 → 0.132）。

他のシードの重みは revision で取れます:

```python
agent = sokudan.load("GeneLab/sokudan-ja-310m@seed1")
```

---

## 設計の要点

詳細は [`docs/architecture.md`](docs/architecture.md)。要点だけ:

### state と question を 1 系列にする（joint encoding）

```
[CLS] 指示 [SEP] 選択肢+マーカー [SEP] state [SEP]
                              |
                  [backbone 25層]   ... 質問ごとに1回（state も毎回入る）
                              |
                    marker positions -> scorer
                              |
          choice/bool -> 質問内 softmax    score -> 動的Kの cumulative link
```

当初は state と question を**別系列**にし、cross-attention head で繋いでいました。
`modernbert-ja-310m` の `local_attention: 128` / `global_attn_every_n_layers: 3`
（`config.json` の実値）から、「連結すると質問が state を見られない」と考えたためです。

**同一条件で両方を学習して、逆の結果が出たので撤回しました。**

| | 別系列 + head | **joint** |
|---|---|---|
| held-out AUROC（未知スキーマ×未知文書） | 0.506 | **0.872** |
| うち意図推論（非明示） | 0.486 | **0.786** |
| val bool AUROC | 0.529 | **0.984** |
| 定常 epoch 時間 | 464 秒 | **330 秒** |

実データの平均系列長は **160 トークン**（state 134 + 質問 26）で、local の窓 128 とほぼ同じでした。
懸念した「位置 3000 の質問ブロック」はこの分布では起きません。
一方、別系列は質問と state の相互作用を head の 2 層だけに通すのに対し、joint は 25 層すべてを使えます。

詳細と撤回の記録は [`docs/architecture.md`](docs/architecture.md) §1.1 / §1.2。
質問は 1 問ずつ別のリクエスト行として処理されるので、**順序不変性は引き続き構造的に保証されます。**

### `score` は動的 K の cumulative link

K はリクエスト時に決まるので、固定 K の CORAL は使えません。

```
b_k = softplus(w · h_k) ≥ 0
P(y > k) = sigmoid(base_k + a - Σ_{j≤k} b_j)      ← 累積和が単調なので CDF は構造的に単調
p_k = P(y > k-1) - P(y > k)
```

`base_k` は「どの K でも初期分布が一様」になる閉形式の項で、
学習される補正はその上に乗る単調な項です。初期状態でどの K でも `1/K` を 5e-3 以内で出します。
損失は **RPS**（CE は「隣に外す」と「両端に外す」を同じ罰にするため不適）。

### encoding は単一実装

`sokudan/encoding/` が state / question をテンソルにする唯一の場所で、
train / eval / serve はすべてここを import します。
学習時と推論時でトークナイズが分岐した瞬間にこのプロジェクトは死ぬので、
マーカー位置のラウンドトリップをテストで固定しています。

---

## ベンチマーク

- [`docs/baseline_ja.md`](docs/baseline_ja.md) — 既存モデルの日本語実測（`bench_ja` 300件）
- [`docs/benchmarks.md`](docs/benchmarks.md) — **`sokudan` v0.1 の全指標**（3 シード、較正前後、state 差し替え、位置バイアス）
- [`docs/gate_a.md`](docs/gate_a.md) — 環境とスループットの実測
- [`docs/licenses.md`](docs/licenses.md) — 採用したもののライセンス一次確認

すべて再現コマンドと生データ（JSON）付きです。

---

## Limits（正直に）

**v0.2 について**（数値は [`docs/benchmarks.md`](docs/benchmarks.md) §10、[`docs/release_candidate.md`](docs/release_candidate.md) §5〜§6）:

- **v0.2 の `bench_ja` の値は、soup 1 体を 1 回だけ実行した値です。** ばらつきは付いていません。
- **4 段階の `score` で、第 1 選択肢がほとんど選ばれません。**
  - `bench_ja` の位置検査の条件 E で、v0.2 は第 1 選択肢を 300 件中 **5 件**しか選びませんでした（v0.1 の 3 シードは 58 / 16 / 62 件）。acc は 0.347 です。
  - 3 段階の条件 A〜D では、第 1 選択肢が 78〜99 件選ばれています。
- **held-out の状態でも、第 1 スロットはやや不利です**（Laya の presentation_checks と同じ定義、30 状態）。
  - 全選択肢同一の対照: score −0.249、choice −0.280（0 が中立）。
  - 全順列の第 1 スロット率: score 0.239、choice 0.289（順序に依存しなければ 1/3）。
- **bool acc は v0.1 より低く（0.780 と 0.788）、true を過少予測します。** 閾値は利用者の事前確率に合わせて決めてください。
- **state を dict で渡すと、出力が変わります**（上の Quickstart の注意）。

**v0.1 から続くもの**:

- **3 シードの平均 ± 標準偏差です**（seed 0 / 1 / 2）。ただし **`score` のシード分散は大きく**、
  acc は 0.663 / 0.800 / 0.827（SD 0.088）と振れます。平均 0.763 は**どのシードの実測値でもありません**。
  **v0.1 として配布している重み（revision `v0.1`）は seed 0** なので、その実測は score acc 0.663 / RPS 0.117 です。
- **`bool` は true を過少予測します。** mean P(true) 0.125 に対し gold の陽性率は 0.297 です。
  AUROC 0.789 なので**順位付けは機能**していますが、**閾値の位置がずれています**。
  argmax をそのまま使うと true を取りこぼします。**温度スケーリングでは補正されません**
  （較正後も 0.170）。利用者側の事前確率に合わせて閾値を決めてください。
- **選択肢が 4 段階の `score` で精度が落ちます。** 位置バイアス検査の E 条件（4段階）は
  acc **0.427 ± 0.072** で、3 段階の A〜D（0.697〜0.788）から明確に落ちます。
  **ただし原因は K ではありません**: held-out を K 別に分解すると K=4 は学習ビュー最多
  （24.4%）で acc も K=3 より高く、K に対して単調でもありません
  （[`docs/benchmarks.md`](docs/benchmarks.md) §5 の追試）。
  残る説明はその条件固有の選択肢の並びですが、**未検証の仮説です。**
  いずれにせよ K≥4 の性能を K=3 から外挿しないでください。
- **較正は `bool` だけです（v0.2.1 以降の既定）。** `score` と `choice` は生の確率です。
  v0.2 で val に fit した `score` の温度は、`bench_ja` の score RPS を 0.075 → 0.132 に悪化させました（v0.1 でも 0.090 → 0.149）。
  `bool` の温度（held-out で交差評価して fit）は、`bench_ja` の bool ECE を 0.181 → 0.105、`bench_en` を 0.249 → 0.151 に下げました（[`docs/calibration.md`](docs/calibration.md)）。
- **評価は `bench_ja` の3スキーマのみ**（部署ルーティング4択 / 緊急度3段階 / 解約示唆）、
  300件・1ドメインです。他のタスクでの性能は測定していません。
- **`choice` で Laya を上回ることは目標にしていません**（事前実測で既に実用水準だったため）。
  報告はしますが、そこを改善する設計はしていません。
- **学習データは合成データのみ**（**21ドメイン**、ラベル条件付き生成）。規模は3段で書きます:

  | 段 | 数 |
  |---|---|
  | 生成した文書 | **4,833 文書** |
  | ラベル付き (文書, 質問) ペア | **31,243 ペア** |
  | スキーマ拡張後のビュー | **79,552 ビュー** |

  **「ビュー」を「例」と読み替えないでください。** 同じ文書・同じラベルを
  スキーマ表記だけ変えて複数回見せたものを含みます。独立な事例数は「ペア」の段です。
  生成者は単一モデル（`qwen3:30b-a3b-instruct-2507-q4_K_M`）なので、
  その語彙・言い回しの癖が学習データ全体に乗っています。
  JGLUE などの実データは、ライセンスの一次確認に時間を要するため採用していません。
- **`bench_ja` と同じドメインの文書が学習データに含まれます**（事業者への問い合わせフォームの
  自由記述）。`bench_ja` が測るのは**未知スキーマ**への汎化であり、未知ドメインへの汎化では
  ありません。`bench_ja` の 3 スキーマ（部署ルーティング / 緊急度 / 解約示唆）は
  学習に一度も出していません。
- **`bench_ja` も合成データです。** 生成に使ったモデルと LLM-as-classifier ベースラインは
  同一モデルなので、そのベースラインは公平な比較対象ではなく上限の目安です。
- **レイテンシは質問数に比例します。** v0.1 は joint encoding で、**質問ごとに state を
  再エンコード**します。N 問のリクエストは backbone を N 回通ります。
  「state は 1 リクエストにつき 1 回」という当初の主張は撤回しました
  （[`docs/architecture.md`](docs/architecture.md) §1.2）。質問側エンコードのキャッシュも
  joint では効きません。
- **state が長くなると未知スキーマの精度が下がります。** backbone の `local_attention` は 128 です。
  held-out の AUROC はトークン数で **0–200: 0.904 / 200–400: 0.882 / 400–600: 0.866 /
  600–: 0.730** と単調に低下します（[`docs/benchmarks.md`](docs/benchmarks.md) §6 の追試）。
  学習済み属性は 600 トークンまでほぼ平坦（0.997）です。
  **600 トークン超は n=33 で標準誤差が約 0.09**あり、そこでの急落は断定できません。
  学習データの state は平均 134 トークン・p95 237 トークンです。
- **位置に依存する質問は苦手です。** held-out の位置依存属性
  「文末が問いかけで終わっているか」は **AUROC 0.688** にとどまり、
  意味を問う属性（意図推論 0.786、明示的要求 0.995）より明確に低く出ました。
- **エンコーディングは `[CLS] 指示 [SEP] 選択肢+マーカー [SEP] state [SEP]` の joint 方式**で、
  これは Laya と同じ配置です。state と質問を別系列にする方式も実装して A/B しましたが、
  未知スキーマへの意図推論が転移しませんでした（held-out AUROC 0.506 対 0.872）。
- **既定で較正されるのは `bool` だけです。** `score` と `choice` の確率は較正されていません。
- **未実装**: act/escalate ヘッド（学習信号を定義できないので作らない）、
  RLCD (Stage 3)、ONNX/TensorRT。
- **FlashAttention-2 は使っていません。** この機械ではビルドできませんでした
  （CUDA Toolkit 13.1 と torch の 12.8 の不一致）。`sdpa` + 長さバケット化で動かしています。
- **head 内に RoPE を入れていません。** backbone からコピーした attention 重みは
  学習時に隣にあった位置信号なしで動いています。初期化としては妥当ですが、
  アブレーションは未実施です。

## `/v1/systemone` 互換サーバー

```bash
pip install "sokudan[serve]"
sokudan serve --port 8000
```

開発版: `pip install "sokudan[serve] @ git+https://github.com/hiroki-abe-58/sokudan.git"`

`POST /v1/systemone` は、TypeSafe の公開 API リファレンスと同じ形のリクエストを受け、同じ形の答えを返します。
その形式で書かれたクライアントは、base URL を `http://127.0.0.1:8000` に替えるだけで使えます。
`GET /health` は、読み込んだモデル、較正の有無、JSON の state をどう描画するかを返します。

- 公開ドキュメントと、公開実装（Lev、kev、Laya）の README にある例だけを読んで作りました。TypeSafe の SDK もサービスも使っていません。
- **state は文字列で送ってください**（上の Quickstart の注意と同じ理由です）。
- 手順: [`docs/serving.md`](docs/serving.md)。各フィールドの扱いと、資料どうしの食い違い: [`docs/systemone_wire_format.md`](docs/systemone_wire_format.md)。

## TypeSafe Jev について

TypeSafe の利用規約（Master Customer Agreement 2.3(b)）が、同サービスおよびその出力を類似製品の開発に用いることを禁じているため、本プロジェクトでは Jev を実測していません。
**このリポジトリには Jev を呼ぶコードが存在しません。** 上の互換サーバーも、公開ドキュメントに書かれた形式に合わせたもので、Jev の出力は使っていません。本プロジェクトは TypeSafe と提携していません。

---

## 開発

```bash
uv venv --python 3.11
uv sync --extra dev --extra bench
cp .env.example .env            # ローカルLLM等のキーはすべて環境変数
uv run pytest
uv run ruff check .
```

- v0.3.0 から、データ生成・学習・評価だけが使う依存（`datasets`、`fugashi`、`unidic-lite`、`matplotlib`）は extra `train` にあります。`dev` extra は `sokudan[train]` を含むので、`uv sync --extra dev` で従来どおり入ります。`dev` を使わずにスクリプトだけ動かすときは `uv sync --extra train`（pip なら `pip install "sokudan[train]"`）です。
- Apple Silicon（macOS 14 以降）では基本の依存に torch が入りません。torch を使うテストや学習には `--extra torch` も付けてください。

再現手順（`bench_ja` の生成からベースライン実測まで）は
[`docs/baseline_ja.md`](docs/baseline_ja.md) の冒頭にあります。

コード中の docstring にある `SOKUDAN_SPEC.md §…` は
[`SOKUDAN_SPEC.md`](SOKUDAN_SPEC.md) への参照です。
**これは 2026-09-20 のスプリント開始時点の設計仕様であり、現在の実装とは一致しません**
（§6.1 / §6.2 の separate encoding は撤回済み）。
何をどう決めて、どこで間違えたかを追えるように残してあります。
現在の正本は [`docs/architecture.md`](docs/architecture.md) です。

## ライセンス

Apache-2.0。バックボーン `sbintuitions/modernbert-ja-310m` は MIT（`NOTICE` を参照）。
