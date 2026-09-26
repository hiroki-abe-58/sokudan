# 即断 / sokudan

*[English README](README_en.md)*

**日本語 System One 意思決定モデル。**
日本語テキスト（state）と型付き質問（questions）を受け取り、
**テキストを一切生成せずに**単一フォワードパスで型付き回答と確率を返すエンコーダモデルです。

生成しないので、パースするものがなく、ハルシネーションする余地もありません。

- モデル: `sokudan-ja-310m`（314.6M / バックボーン [`sbintuitions/modernbert-ja-310m`](https://huggingface.co/sbintuitions/modernbert-ja-310m), MIT）
- ライセンス: Apache-2.0
- 開発環境: RTX 5090 (Blackwell, sm_120) 1枚

> **Status: v0.1、2026-09-20〜21 の 2 日スプリントで作ったものです。**
> 公開先: [`GeneLab/sokudan-ja-310m`](https://huggingface.co/GeneLab/sokudan-ja-310m)
> **この README の数値はすべて本機で実行したコードの出力です。** 推定値はありません。
> 未測定のものは「測定していない」と書きます。

## `bench_ja` 300 件、3 シード平均 ± 標準偏差

| 対象 | choice acc | score RPS↓ | bool acc | bool AUROC |
|---|---|---|---|---|
| **sokudan-ja-310m** | **0.847 ± 0.009** | **0.090 ± 0.023** | **0.788 ± 0.010** | **0.789 ± 0.043** |
| `laya-multilingual` (ja) | 0.747 | 0.232 | 0.543 | 0.523 |
| 多数決クラス | 0.380 | 0.197 | 0.703 | — |
| ランダム | 0.253 | 0.201 | 0.513 | — |

全指標・較正前後・シード別の生値は [`docs/benchmarks.md`](docs/benchmarks.md)。
**`score` のシード分散は大きく**、配布している重み（seed 0）の実測は acc 0.663 / RPS 0.117 です。

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
pip install -e .
```

```python
import sokudan

agent = sokudan.load("GeneLab/sokudan-ja-310m")     # または手元の runs/.../model.pt
result = agent.predict(
    {"body": "先月の請求で同じ金額が二回引き落とされています。至急ご確認ください。"},
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

質問タイプは `choice` / `score` / `bool`（`noul` はエイリアス）。
スキーマはリクエストごとに自由で、再学習は要りません。

### 確率の出所（ここが差別化点）

返ってくる確率は **head のロジットを直接読んだもの**です。
モデルに「どれくらい自信があるか」を自己申告させた値ではありません。
較正済みの確率がほしい場合は Stage 2 の温度を渡してください:

```python
agent = sokudan.load("GeneLab/sokudan-ja-310m",
                     temperatures="temperatures.json")
```

**温度を渡さない場合、確率は較正されていません。** 下の Limits を読んでください。
**ただし同梱の温度は `score` の RPS を悪化させます**（0.090 → 0.149）。
`choice` と `bool` だけに使うか、自前の検証セットで再フィットしてください。

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

- **3 シードの平均 ± 標準偏差です**（seed 0 / 1 / 2）。ただし **`score` のシード分散は大きく**、
  acc は 0.663 / 0.800 / 0.827（SD 0.088）と振れます。平均 0.763 は**どのシードの実測値でもありません**。
  **HF で配布している重みは seed 0** なので、その実測は score acc 0.663 / RPS 0.117 です。
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
- **温度較正は `score` を悪化させます。** choice ECE 0.147→0.092、bool ECE 0.202→0.129 と
  改善する一方、**score RPS は 0.090→0.149 と悪化**します。既定は未較正です。
  `temperatures.json` は同梱しますが、**自前の検証セットで再フィットすることを推奨します**。
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
- **温度を渡さない限り確率は較正されていません。**
- **未実装**: act/escalate ヘッド（学習信号を定義できないので作らない）、
  RLCD (Stage 3)、ONNX/TensorRT。
- **FlashAttention-2 は使っていません。** この機械ではビルドできませんでした
  （CUDA Toolkit 13.1 と torch の 12.8 の不一致）。`sdpa` + 長さバケット化で動かしています。
- **head 内に RoPE を入れていません。** backbone からコピーした attention 重みは
  学習時に隣にあった位置信号なしで動いています。初期化としては妥当ですが、
  アブレーションは未実施です。

## TypeSafe Jev について

TypeSafe の利用規約（Master Customer Agreement 2.3(b)）が、同サービスおよびその出力を類似製品の開発に用いることを禁じているため、本プロジェクトでは Jev を実測していません。
**このリポジトリには Jev を呼ぶコードが存在しません。**

---

## 開発

```bash
uv venv --python 3.11
uv sync --extra dev --extra bench
cp .env.example .env            # ローカルLLM等のキーはすべて環境変数
uv run pytest                   # 538 tests
uv run ruff check .
```

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
