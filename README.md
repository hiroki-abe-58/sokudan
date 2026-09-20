# 即断 / sokudan

本リポジトリは (1) 日本語 System One モデル `sokudan` のコード、(2) 評価セット `bench_ja`、
(3) Laya の日本語実測 [`docs/baseline_ja.md`](docs/baseline_ja.md) を含みます。
**学習済みモデル v0.1 は準備中です。**

**日本語 System One 意思決定モデル。**
日本語テキスト（state）と型付き質問（questions）を受け取り、
**テキストを一切生成せずに**単一フォワードパスで型付き回答と確率を返すエンコーダモデルです。

生成しないので、パースするものがなく、ハルシネーションする余地もありません。

- モデル: `sokudan-ja-310m`（338M / バックボーン [`sbintuitions/modernbert-ja-310m`](https://huggingface.co/sbintuitions/modernbert-ja-310m), MIT）
- ライセンス: Apache-2.0
- 開発環境: RTX 5090 (Blackwell, sm_120) 1枚

> **Status: v0.1、2026-09-20 の1日スプリントで作ったものです。**
> **この README の数値はすべて本機で実行したコードの出力です。** 推定値はありません。
> 未測定のものは「測定していない」と書きます。誇張すると死にます（SOKUDAN_SPEC.md §1）。

---

## なぜ作ったか — 先に測った結果

作る前に、既存の System One 系モデルが**日本語で**どう振る舞うかを実測しました
（300件の日本語業務文 `bench_ja`、全文同梱、[`docs/baseline_ja.md`](docs/baseline_ja.md)）。

### `bench_ja` のライセンスと使い方

`data/bench_ja.jsonl` は **CC BY 4.0** で配布します（コードの Apache-2.0 とは別です）。

> **評価専用です。学習データに混ぜないでください。**
> このセットは未知スキーマへの汎化を測るための held-out テストセットで、
> 一度学習に使われるとその役目を果たせなくなります。

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

agent = sokudan.load("runs/s0/model.pt")            # 学習済みチェックポイント
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
agent = sokudan.load("runs/s0/model.pt", temperatures="runs/s0/temperatures.json")
```

**温度を渡さない場合、確率は較正されていません。** 下の Limits を読んでください。

---

## 設計の要点

詳細は [`docs/architecture.md`](docs/architecture.md)。要点だけ:

### state と question を別系列にする

```
state    --[backbone]--> H_state  (1, L_s, 768)   ... 1リクエストにつき1回だけ
question --[backbone]--> H_q      (N, L_q, 768)   ... N問をバッチ次元で1回
                              |
                  [decision head × 2層: self-attn + cross-attn into H_state]
                              |
          choice/bool -> 質問内 softmax    score -> 動的Kの cumulative link
```

`modernbert-ja-310m` は `local_attention: 128` / `global_attn_every_n_layers: 3`
（`config.json` で確認した実値）。25層のうち global は3層に1層だけなので、
state の後ろに質問を連結すると**質問が state をほとんど見られません**。
分離すればどちらも短い系列のまま、事前学習と同じ条件で動きます。

副作用として、**質問は互いに別の行にいるので順序不変性が構造的に保証されます。**
マスクで「見えないことにする」のではなく、存在しません。

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
- `sokudan` 自体のベンチマークは v0.1 公開時に追加します
- [`docs/gate_a.md`](docs/gate_a.md) — 環境とスループットの実測
- [`docs/licenses.md`](docs/licenses.md) — 採用したもののライセンス一次確認

すべて再現コマンドと生データ（JSON）付きです。

---

## Limits（正直に）

- **単一シードです。** SOKUDAN_SPEC.md §9 は3シード以上の平均±標準偏差を要求していますが、
  1日スプリントのため1シードに落としました。**この数値に分散は付いていません。**
  シードを変えたときにどれだけ動くかは測定していません。
- **評価は `bench_ja` の3スキーマのみ**（部署ルーティング4択 / 緊急度3段階 / 解約示唆）、
  300件・1ドメインです。他のタスクでの性能は測定していません。
- **`choice` で Laya を上回ることは目標にしていません**（事前実測で既に実用水準だったため）。
  報告はしますが、そこを改善する設計はしていません。
- **学習データは合成データのみ**（14ドメイン、ラベル条件付き生成）。規模は3段で書きます:

  | 段 | 数 |
  |---|---|
  | 生成した文書 | TBD 文書 |
  | ラベル付き (文書, 質問) ペア | TBD ペア |
  | スキーマ拡張後のビュー | TBD ビュー |

  **「ビュー」を「例」と読み替えないでください。** 同じ文書・同じラベルを
  スキーマ表記だけ変えて複数回見せたものを含みます。独立な事例数は「ペア」の段です。
  生成者は単一モデル（`qwen3:30b-a3b-instruct-2507-q4_K_M`）なので、
  その語彙・言い回しの癖が学習データ全体に乗っています。
  JGLUE などの実データは、ライセンスの一次確認に時間を要するため当日は採用していません。
- **`bench_ja` も合成データです。** 生成に使ったモデルと LLM-as-classifier ベースラインは
  同一モデルなので、そのベースラインは公平な比較対象ではなく上限の目安です。
- **温度を渡さない限り確率は較正されていません。**
- **未実装**: act/escalate ヘッド（学習信号を定義できないので作らない）、
  RLCD (Stage 3)、ONNX/TensorRT、HF Space デモ、HTTP API。
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
uv run pytest                   # 465 tests
uv run ruff check .
```

再現手順（`bench_ja` の生成からベースライン実測まで）は
[`docs/baseline_ja.md`](docs/baseline_ja.md) の冒頭にあります。

コード中の docstring にある `SOKUDAN_SPEC.md §…` は内部の設計仕様書への参照です。
**その仕様書はこのリポジトリには含めていません**（節番号だけが残っています）。

## ライセンス

Apache-2.0。バックボーン `sbintuitions/modernbert-ja-310m` は MIT（`NOTICE` を参照）。
