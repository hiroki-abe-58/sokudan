# lev（interfaze-ai/lev）と sokudan v0.2 の同一ハーネス比較

記事から参照するための文書です。数字は、`docs/baseline_lev/` に写した結果 JSON 10 本（`C:\Users\hirok\jev-oss\lev-eval\results\` の原本と同じもの）と、`C:\Users\hirok\jev-oss\REPORT.md` の Lev の節（2026-09-27 21:17–22:25 JST の実測）だけです。未測定のものは書いていません。

## 1. 計測条件

- **マシン**: RTX 5090 の 1 台。lev と sokudan を同じマシンで、同じ計測コードで測りました。
- **計測コード**: `jev-oss/lev-eval/`（lev と sokudan のリポジトリの外に置いたもの）。
  - `lev_bench.py`: `bench_ja` / `bench_en` の 3 型。質問は sokudan の `bench_questions()` と同じ（department は 4 択の choice、urgency は 3 水準の score で低い順、churn は noul）。3 問を 1 件ごとに 1 回の `system_one` で聞きます。
  - `lev_position_probe.py`: score の位置感度。定義は Laya の `presentation_checks.py` と同じです。
  - `sokudan_bench.py`: sokudan を同じハーネス（`lev_bench.run_bench`）で測るための包み。モデルの呼び出しだけを `sokudan.load("GeneLab/sokudan-ja-310m").predict` に差し替えています。
- **採点**: sokudan の `score_baseline` と同じ定義です。
  - 確率は 5e-5 で下限を付けて正規化します。
  - bool は [1 − p, p] の argmax（引き分けは false）です。
  - AUROC は平均順位、RPS は K − 1 で正規化します。
- **モデルと環境**:

| | lev | sokudan v0.2 |
|---|---|---|
| チェックポイント | `interfaze-ai/lev`（HF main f8ef711、Apache-2.0、ベースは Qwen/Qwen3.5-4B + LoRA） | `GeneLab/sokudan-ja-310m`（タグ v0.2 = fcc67be を別の作業ツリーに入れたもの） |
| パッケージ | lev 0.1.1 | sokudan 0.2.0 |
| torch | 2.8.0+cu128 | 2.11.0+cu128 |
| 精度 | bf16（lev の既定） | fp32（`predict` の経路） |
| 較正 | **あり**（同梱の `calibration.json`。score T = 2.80、noul T = 2.33、choice T = 1.61〜1.79）と**なし**（読み込み後に空の `CalibrationProfile`、T = 1）の両方 | なし（v0.2 の公開値と同じ条件。温度を渡していない） |

- **bench のファイル**: pin のハッシュ（`08ed6d1d…` / `dcf62c36…`）と一致する LF 版を使いました（checkout の CRLF 版はハッシュが違うため）。
- lev は、モデルカードに「English only」と書かれたモデルです。`bench_ja` の数字は、日本語への転移の測定です。

## 2. 3 型の精度（同一ハーネス）

**bench_en（290 件）**:

| モデル | choice acc | score RPS↓ | score acc | bool acc | bool AUROC | mean P(true)（正例率 0.317） |
|---|---|---|---|---|---|---|
| lev（較正あり） | **0.917** | 0.150 | 0.583 | **0.938** | **0.978** | 0.303 |
| lev（較正なし） | 0.917 | 0.189 | 0.583 | 0.938 | 0.979 | 0.285 |
| sokudan v0.2（較正なし） | 0.872 | **0.114** | **0.652** | 0.690 | 0.621 | 0.134 |

**bench_ja（300 件）**:

| モデル | choice acc | score RPS↓ | score acc | bool acc | bool AUROC | mean P(true)（正例率 0.297） |
|---|---|---|---|---|---|---|
| lev（較正あり、English only） | **0.887** | 0.138 | 0.643 | **0.817** | **0.889** | 0.159 |
| lev（較正なし） | 0.887 | 0.172 | 0.643 | 0.817 | 0.891 | 0.120 |
| sokudan v0.2（較正なし） | 0.880 | **0.075** | **0.817** | 0.780 | 0.844 | 0.133 |

- accuracy は温度に依りません。
- lev の noul は、0〜8 の評点の分布から読みます。そのため温度は単調変換にならず、AUROC がわずかに動きます（en 0.978 → 0.979、ja 0.889 → 0.891）。
- sokudan の同一ハーネスの値は、公開値（`docs/benchmarks.md` §10）と一致しました。違いは、bench_ja の bool AUROC の +0.0002（0.844081 対 0.843868）だけです。`predict` が確率を小数 4 桁に丸めるために、引き分けが増えたためと考えられます（REPORT.md）。
- **sokudan v0.2.1 では bool の較正が既定で on になります。** 上の sokudan の行は v0.2 の較正なしの値です。v0.2.1 の bool 較正は、bool acc と AUROC を変えません（`docs/calibration.md` §9）。

## 3. score の位置感度

定義は Laya の `presentation_checks.py` と同じです。
- **全選択肢同一の対照**: K 個の選択肢がすべて同じ文字列の質問で、slot 0 の対数確率 − 全スロットの平均。
- **全順列の第 1 スロット率**: 3 水準の全 6 順列で、第 1 スロットが選ばれる割合。

| bench（状態数） | モデル | 対照: 較正あり | 対照: 較正なし | 第 1 スロット率 | スロット別 argmax | 順序に依らない状態 |
|---|---|---|---|---|---|---|
| bench_en（290） | lev | −0.332（loo −0.336〜−0.330） | **−0.931**（loo −0.942〜−0.924） | 0.270（1,740 判定） | 469 / 604 / 667 | 212 / 290 |
| bench_ja（300） | lev | −0.518（loo −0.522〜−0.515） | **−1.453**（loo −1.465〜−1.445） | 0.240（1,800 判定） | 432 / 635 / 733 | 188 / 300 |
| 参考: sokudan の held-out 30 状態（日本語。bench ではない） | sokudan v0.2 | — | −0.249 | 0.239 | 43 / 76 / 61 | — |
| 参考: Laya の回帰テストの閾値 | — | — | ≥ −0.20 | ≥ 0.15 | — | — |

- 第 1 スロット率は、較正の有無で変わりません（argmax は温度で動かない）。
- lev は score を並べ替えの平均にかけません（`DecisionEngine._orders`: ordered scales retain the low-to-high ordering）。
- sokudan の行は、測った状態が違います（sokudan の held-out 30 状態、`docs/release_candidate.md` §6）。同じ状態での比較ではありません。

**設定別の対照（較正なし）**:

| bench | 設定 | K = 3 | K = 4 | K = 5 |
|---|---|---|---|---|
| en | moderate | −0.332 | −0.664 | −0.899 |
| en | a request | −0.907 | −1.293 | −1.491 |
| ja | 中程度 | −0.529 | −0.837 | −1.098 |
| ja | 依頼 | −1.650 | −2.143 | −2.462 |

## 4. 速度と VRAM（RTX 5090、1 件ずつ）

| モデル | bench_ja ms / 件 | bench_en ms / 件 | 問数 / 件 | ピーク VRAM allocated / reserved |
|---|---|---|---|---|
| sokudan v0.2（314.6M、`predict`、fp32） | **23.6** | **28.4** | 3 | 1.26〜1.29 / 1.41〜1.50 GiB |
| lev（Qwen3.5-4B + LoRA、bf16、較正あり） | 152.1 | 160.9 | 3 | 9.18〜9.21 / 14.24〜15.34 GiB |
| lev（較正なし） | 158.3 | 164.8 | 3 | 9.18〜9.21 / 14.24〜15.34 GiB |
| lev の位置の検査（較正あり） | 399.3 ms / 状態 | 414.7 ms / 状態 | 12 | 11.28〜11.29 / 26.31〜28.69 GiB |

- 1 件あたりの時間は、読み込みの後から全件の終わりまでの wall time を件数で割ったものです（最初の呼び出しのウォームアップを含む）。
- reserved は PyTorch のキャッシュアロケータが確保した量です。実際に使った量の目安は allocated のほうです。
- sokudan 単体の計測（`docs/latency.md`、ウォームアップ 20 件を除いた中央値）では、bench_ja の 3 問 × 1 件が 22.8 ms（p95 27.5 ms）でした。

## 5. 再現のコマンド

作業場所は `C:\Users\hirok\jev-oss\lev-eval\` です。venv は `jev-oss\.venvs\lev`（lev）と `jev-oss\.venvs\sokudan-v02`（sokudan v0.2）です。

```
# lev（較正あり / なし）
python lev_bench.py --lang ja --checkpoint interfaze-ai/lev --bench <bench_ja.jsonl> --out results/lev_bench_ja.json
python lev_bench.py --lang en --checkpoint interfaze-ai/lev --bench <bench_en.jsonl> --out results/lev_bench_en.json
python lev_bench.py --lang ja --checkpoint interfaze-ai/lev --bench <bench_ja.jsonl> --out results/lev_bench_ja_uncal.json --uncalibrated
python lev_bench.py --lang en --checkpoint interfaze-ai/lev --bench <bench_en.jsonl> --out results/lev_bench_en_uncal.json --uncalibrated

# score の位置感度（較正あり / なし）
python lev_position_probe.py --lang en --checkpoint interfaze-ai/lev --bench <bench_en.jsonl> --out results/lev_bench_en_position.json
python lev_position_probe.py --lang ja --checkpoint interfaze-ai/lev --bench <bench_ja.jsonl> --out results/lev_bench_ja_position.json
python lev_position_probe.py --lang en ... --out results/lev_bench_en_position_uncal.json --uncalibrated
python lev_position_probe.py --lang ja ... --out results/lev_bench_ja_position_uncal.json --uncalibrated

# sokudan v0.2（同じハーネス、較正なし）
python sokudan_bench.py --lang ja --checkpoint GeneLab/sokudan-ja-310m --bench <bench_ja.jsonl> --out results/sokudan_v02_bench_ja.json
python sokudan_bench.py --lang en --checkpoint GeneLab/sokudan-ja-310m --bench <bench_en.jsonl> --out results/sokudan_v02_bench_en.json
```

- `<bench_*.jsonl>` は、pin のハッシュと一致する LF 版です。
- `bench_ja` / `bench_en` は CC BY 4.0 で、評価用です（作者は学習データに使わないよう求めている。ライセンスの条件ではなく依頼）。

## 6. ファイル（`docs/baseline_lev/`）

| ファイル | 中身 |
|---|---|
| `lev_bench_en.json` / `lev_bench_ja.json` | lev、較正あり、3 型 |
| `lev_bench_en_uncal.json` / `lev_bench_ja_uncal.json` | lev、較正なし、3 型 |
| `lev_bench_en_position.json` / `lev_bench_ja_position.json` | lev、較正あり、score の位置感度 |
| `lev_bench_en_position_uncal.json` / `lev_bench_ja_position_uncal.json` | lev、較正なし、score の位置感度 |
| `sokudan_v02_bench_en.json` / `sokudan_v02_bench_ja.json` | sokudan v0.2、同じハーネス、較正なし |

- どれも 0.7〜1.6 KB で、原本をそのまま写しました（要約は作っていない）。
