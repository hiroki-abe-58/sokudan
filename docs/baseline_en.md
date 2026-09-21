# English baseline — `bench_en`

> **この文書の数値はすべて本機で実行したコードの出力です。** 推定値はありません。
> 未測定の項目は明示的に「測定していない」と書きます。

実測日: **2026-09-22**
評価セット: `data/bench_en.jsonl` 290 件（`bench_ja` と同じ 3 スキーマ・同じラベル重み、英語）
生データ: `runs/baseline_en/results.json`、`runs/position_bias_en_*.json`、
`runs/position_bias_ja_laya_english.json`

再現:
```bash
uv run python scripts/build_bench_en.py --n 300
uv run python scripts/run_baseline_en.py --bench data/bench_en.jsonl --out runs/baseline_en
for m in laya-multilingual laya; do
  uv run python scripts/probe_position_bias.py --model "laya:convaiinnovations/$m" \
      --lang en --bench data/bench_en.jsonl --out "runs/position_bias_en_$m.json"
done
```

---

## 0. この文書が答える問い

[`docs/baseline_ja.md`](baseline_ja.md) §6.2 は、`laya-multilingual` が日本語で
**5 つのスキーマ条件すべてで提示順の第 1 選択肢を 300 件中 0〜1 件しか選ばない**ことを
測定しました。これを作者に報告するなら、最初に来る問いは
**「それは日本語の問題か、モデルの問題か」**です。

`bench_en` は言語だけを変えた対照です。ラベルの重みは `bench_ja` から**インポート**しており
（別々に書けば必ずずれるため）、ランダム化の軸・生成モデル・棄却の契約・blind 3 値検算は
すべて同じです。位置バイアスの 5 条件も記号と `reverse` / `gold_map` の意味を揃えてあります。

## 1. 結論 — **日本語固有ではなく、`laya-multilingual` 固有です**

第 1 選択肢が argmax になった件数（提示順そのまま、remap なし）:

| モデル | 日本語（`bench_ja` 300 件） | **英語（`bench_en` 290 件）** |
|---|---|---|
| **`laya-multilingual`** | **0, 0, 1, 1, 0** | **0, 0, 0, 0, 0** |
| `laya`（英語版） | 13, 56, 8, 1, 110 | 65, 74, 0, 5, 4 |

**`laya-multilingual` は英語でも 5 条件すべてで 0/290 です。** 英語は学習言語なので、
「入力が読めていない」では説明できません。

`laya`（英語版）は英語の条件 A / B で 22.4% / 25.5% と**正常に第 1 選択肢を選びます**。
つまりこれは `score` というタスクの性質でも、評価ハーネスの性質でもありません。

## 2. `laya-multilingual`、英語、5 条件

| 条件 | 提示した選択肢 | 提示順の argmax 件数 | **第1選択肢** | acc |
|---|---|---|---|---|
| A original | Not urgent / Soon / Work is blocked | [0, 32, 258] | **0** | 0.269 |
| B reversed | Work is blocked / Soon / Not urgent | [0, 5, 285] | **0** | 0.310 |
| C reworded | Low / Medium / High | [0, 255, 35] | **0** | 0.445 |
| D reworded, reversed | High / Medium / Low | [0, 8, 282] | **0** | 0.303 |
| E four levels | Not urgent at all / Not urgent / Soon / Work is blocked | [0, 3, 20, 267] | **0** | 0.200 |

決定的なのは A と B の対比です。**`Not urgent` は先頭に置かれると 0 件、末尾に置かれると
285 件**選ばれます。同じ 290 件・同じ問い・同じ語で、置き場所だけが違います。
日本語の「急がない」が先頭 0 件・末尾 250 件だったのと同じ形です。

語の意味でも、順序の向きでも、段階数（K=3/4）でもなく、**位置**で決まっています。

## 3. `laya`（英語版）、5 条件 — 部分的にしか起きない

| 条件 | 提示順の argmax 件数 | 第1選択肢 | 選択率 | acc |
|---|---|---|---|---|
| A original | [65, 96, 129] | 65 | 22.4% | 0.583 |
| B reversed | [74, 4, 212] | 74 | 25.5% | 0.421 |
| C reworded | [0, 286, 4] | **0** | **0.0%** | 0.493 |
| D reworded, reversed | [5, 285, 0] | 5 | 1.7% | 0.497 |
| E four levels | [4, 178, 19, 89] | 4 | 1.4% | 0.466 |

英語版モデルには「死んだスロット」は**一貫しては**存在せず、特定のスキーマ表記
（Low/Medium/High 系）でのみ起きます。日本語に同じモデルを入れた場合も部分的でした
（4.3% / 18.7% / 2.7% / 0.3% / 36.7%）。

## 4. 精度・較正（`bench_en` 290 件）

| 対象 | choice acc | choice ECE↓ | score RPS↓ | score acc | bool acc | bool AUROC | bool ECE↓ |
|---|---|---|---|---|---|---|---|
| `laya-multilingual` (en) | 0.817 | 0.080 | **0.340** | 0.266 | 0.645 | **0.355** | 0.305 |
| `laya` (english model) | **0.883** | 0.182 | **0.149** | **0.590** | 0.672 | 0.494 | 0.216 |
| 多数決クラス | 0.331 | 0.669 | 0.257 | 0.486 | **0.683** | 0.500 | 0.317 |
| ランダム | 0.272 | 0.021 | 0.197 | 0.307 | 0.548 | 0.546 | 0.195 |

読み取れること:

- **`choice` は英語でも両モデルとも動きます**（0.817 / 0.883、多数決 0.331 の 2.5 倍以上）
- **`laya-multilingual` の `score` は英語でもランダム以下です**（RPS 0.340 対ランダム 0.197、
  acc 0.266 対ランダム 0.307）。§2 の位置バイアスと整合します
- **`laya-multilingual` の `bool` AUROC は 0.355** で、**0.5 を下回っています**——
  順位付けが偶然より悪い、つまり反転しています。日本語での 0.523 より悪い値です
- `laya`（英語版）は `score` で RPS 0.149 / acc 0.590 と、英語では実用水準です。
  **multilingual 版との差はここに集中しています**

> **`bench_en` も合成データです。** 生成に使ったモデルは
> `qwen3:30b-a3b-instruct-2507-q4_K_M` 単一で、その語彙・言い回しの癖が全体に乗っています。

## 5. 検算（blind、temperature 0）

生成条件をそのままゴールドとし、同じモデルにゴールドを見せずに読み直させた一致率:

| 項目 | 一致率 |
|---|---|
| department | **90.3%** |
| churn | **93.5%** |
| urgency | **61.7%**（うち「unclear」3.5%） |
| 3 項目すべて一致 | 51.7% |

`urgency` の一致率が低いのは 3 段階の緊急度が本質的に曖昧だからで、**この数字は捨てずに
記録しています**。`bench_ja` には検算工程がないため直接の比較はできません。

**検算者は生成者と同一モデルです。** したがってこれは「生成器が自分の指示に従ったか」の
測定であって、「ラベルが真か」の測定ではありません。

棄却の内訳と実現したラベル分布は `data/bench_en.manifest.json` にあります
（290/300 採用、Billing 90 / Technical 96 / Sales 56 / Other 48、churn 陽性率 31.7%）。

## 6. ハーネスは原因ではありません

公式 `laya` パッケージの README どおりの最小コード（`laya.load` と `agent.predict` の 2 行、
間にこちらのラッパーを挟まない）で `bench_ja` を再実行し、保存済みの確率と突合しました。
**floored の最大 \|Δ\| = 0.0**（raw で 2.2e-16、float の丸め）。
`runs/diagnostics/laya_harness_check.json`。

## 7. TypeSafe Jev について

TypeSafe の利用規約（Master Customer Agreement 2.3(b)）が、同サービスおよびその出力を
類似製品の開発に用いることを禁じているため、本プロジェクトでは Jev を実測していません。
**このリポジトリには Jev を呼ぶコードが存在しません。**
