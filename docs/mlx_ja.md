# MLX バックエンド（Apple Silicon）

English: [`mlx.md`](mlx.md)。数値はすべて下記の 1 台での実測です。測っていないものは「測定していない」と書きます。

`sokudan.load` は、Apple Silicon では PyTorch の代わりに [MLX](https://github.com/ml-explore/mlx) でモデルを動かせます。重みは同じで、forward を移植したものです。`predict` の応答の形は torch 版と同じです。

## インストール

```bash
pip install sokudan
```

- v0.3.0 から、Apple Silicon の macOS 14 以降（Darwin 23 以降）では、基本のインストールで `mlx>=0.32.2,<0.33` が入り、**torch は入りません**。それ以外では torch が入り、MLX は入りません（マーカーは `tests/test_packaging.py` で固定）。`pip install "sokudan[torch]"` はどこでも torch を足します。`"sokudan[mlx]"` は同じマーカーで MLX を明示する extra です。
- このバックエンドを動かした MLX は 0.32.2 だけです。
- 下記の機械で、0.3.0 の wheel を新しい Python 3.11 環境に入れた結果: 68 パッケージ、`mlx` と `mlx-metal` 0.32.2、torch なし、`site-packages` 934 MB（大きいもの: unidic_lite 249 MB、mlx 159 MB、pyarrow 127 MB、transformers 114 MB）。`sokudan.load()` は MLX float16 を選び、Quickstart の `predict` が通り、torch は import されませんでした。`[torch]` 付きでは 74 パッケージ、1.6 GB で、`load()` は MLX を、`backend="torch"` は mps を選びました。
- `import sokudan` は torch を import しません。torch が入っている環境では、MLX でロードしても torch が import されることがあります（tokenizer と backbone の config に使う transformers が、入っている torch を import するため）。

## 使い方

```python
import sokudan

agent = sokudan.load("GeneLab/sokudan-ja-310m")          # backend="auto"
print(agent.backend)                                      # <MLXBackend mlx gpu float16>
agent = sokudan.load("GeneLab/sokudan-ja-310m", backend="mlx", dtype="float32")
agent = sokudan.load("GeneLab/sokudan-ja-310m", backend="torch")   # PyTorch のモデル
```

### `backend` / `device` / `dtype`

| 引数 | 値 | 挙動 |
|---|---|---|
| `backend` | `"auto"`（既定） | 次の順に試す: MLX（`mlx` が import でき、Apple Silicon のとき）→ torch `mps` → torch `cuda` → torch `cpu` |
| | `"mlx"` / `"torch"` | そのバックエンドだけ |
| `device` | `"auto"`（既定） | torch バックエンドでは `cuda` → `mps` → `cpu` |
| | `"cpu"` / `"cuda"` / `"mps"` | torch のデバイスとしてそのまま使う。`backend="auto"` でも、そのデバイスの torch になる |
| `dtype` | `None`（既定） | バックエンドの既定。MLX は `float16`、torch は `float32` |
| | `"float16"` / `"float32"` | MLX の backbone の精度。torch は `float32` のみ |

- **自己診断とフォールバック。** 各候補をロードしたあと、短いリクエストを 1 回答えさせます（choice・score・bool を 1 問ずつ）。`backend="auto"` かつ `device="auto"` のとき、ここで例外が出たら（確率が有限でない場合を含む）`RuntimeWarning` を出して次の候補に進みます。どれもロードできなければ、全候補の失敗を並べた `RuntimeError` を出します。`backend` か `device` を明示したときはフォールバックせず、その例外をそのまま出します。
- 使われているバックエンドは `agent.backend` で見えます（`agent.backend.name`、`agent.device`、`agent.dtype`）。
- `dtype` は backbone にだけ効きます。ヘッド（マーカーの scorer と順序ヘッド、5 テンソル）は常に float32 で計算します。

### MLX バックエンドが扱うもの

- joint エンコーディング（公開モデル）の、`model.safetensors` と `config.json` があるディレクトリ（Hub の repo id か手元のディレクトリ）。`.pt` と separate エンコーディングは torch が必要です。`backend="auto"` なら warning を出して torch に落ちます。
- 重みは公開している `model.safetensors` を `mx.load` で読みます。backbone の 152 テンソルは `backbone.model.` を剥がし、ヘッドの 5 テンソルは名前そのままです（変換表は `tests/test_mlx_backend.py` で固定）。
- 実装していないもの: 量子化した backbone（dtype の表を参照）、`mx.compile`、エンコード済みプロンプトのキャッシュ。

### `sokudan serve`

`sokudan serve --backend auto|mlx|torch --dtype ...` は、両方を `sokudan.load` に渡します。既定は `auto` なので、Apple Silicon で MLX が入っていれば MLX（float16）で動き、`/health` の `device` は `"gpu"` です。`--device cpu` / `mps` / `cuda` を付けると、そのデバイスの torch で動きます。起動ログに選ばれたものが出ます（例: `sokudan serve: loaded; backend=mlx device=gpu dtype=float16 calibrated=True`）。`sokudan[serve]` をクリーンに入れた環境で、Quickstart のリクエストへの `/v1/systemone` の答えは、確率・choice・score・noul とも `predict()` と同じでした（差 0）。

## torch との一致

一致セット: 320 state / 630 問。README・Space のデモ・テストの state から合成しました（`scripts/mlx/parity_set.py`、seed 20260928。bool 218、choice 180（K 2〜8）、score 232（K 3〜7）、1 state あたり 1〜3 問、joint 長 18〜1024 トークン、うち 4 問は切り詰め）。基準は torch の CPU、float32。比べるのはヘッドの生の確率（温度の前）で、全選択肢について問ごとに比べます（`scripts/mlx/dump_probs.py`、`scripts/mlx/compare_probs.py`）。

dtype のゲート（型ごと）: argmax 一致率 ≥ 99.5%、平均絶対差 ≤ 2e-3、最大絶対差 ≤ 5e-2。float32 は、全問の argmax 一致かつ最大絶対差 ≤ 1e-3 を条件にしました。

| dtype | 型 | argmax 一致 | 平均絶対差 | 最大絶対差 | ゲート |
|---|---|---|---|---|---|
| float32 | choice | 180/180（100.00%） | 3.20e-07 | 9.12e-06 | 通過 |
| float32 | score | 232/232（100.00%） | 2.07e-07 | 2.62e-06 | 通過 |
| float32 | bool | 218/218（100.00%） | 4.89e-07 | 8.11e-06 | 通過 |
| float16 | choice | 180/180（100.00%） | 2.92e-04 | 5.62e-03 | 通過 |
| float16 | score | 231/232（99.57%） | 1.81e-04 | 5.32e-03 | 通過 |
| float16 | bool | 218/218（100.00%） | 4.18e-04 | 8.22e-03 | 通過 |
| 8bit・group 64 | choice | 179/180（99.44%） | 2.17e-03 | 5.79e-02 | 不通過（argmax・平均・最大） |
| 8bit・group 64 | score | 230/232（99.14%） | 1.63e-03 | 1.48e-02 | 不通過（argmax） |
| 8bit・group 64 | bool | 218/218（100.00%） | 2.19e-03 | 2.74e-02 | 不通過（平均） |
| 8bit・group 32 | choice | 179/180（99.44%） | 1.97e-03 | 5.07e-02 | 不通過（argmax・最大） |
| 8bit・group 32 | score | 231/232（99.57%） | 1.06e-03 | 1.19e-02 | 通過 |
| 8bit・group 32 | bool | 218/218（100.00%） | 1.57e-03 | 1.90e-02 | 通過 |
| 4bit・group 64 | choice | 156/180（86.67%） | 3.73e-02 | 6.78e-01 | 不通過（argmax・平均・最大） |
| 4bit・group 64 | score | 206/232（88.79%） | 2.04e-02 | 3.00e-01 | 不通過（argmax・平均・最大） |
| 4bit・group 64 | bool | 206/218（94.50%） | 5.18e-02 | 6.08e-01 | 不通過（argmax・平均・最大） |
| 4bit・group 32 | choice | 162/180（90.00%） | 2.92e-02 | 5.86e-01 | 不通過（argmax・平均・最大） |
| 4bit・group 32 | score | 208/232（89.66%） | 1.88e-02 | 2.89e-01 | 不通過（argmax・平均・最大） |
| 4bit・group 32 | bool | 209/218（95.87%） | 3.19e-02 | 5.29e-01 | 不通過（argmax・平均・最大） |

- float16 はゲートを満たしたので既定にしました。argmax が食い違った score の 1 問は、torch 側でほぼ同率です（0.4110 と 0.4103。MLX float16 は 0.4105 と 0.4106）。
- 8bit と 4bit は、float16 の backbone に `nn.quantize` をかけたもの（全 Linear とトークン埋め込み）です。どれもゲートを満たさないので、`dtype` の選択肢にしていません。
- float16 の backbone には、単発の大きな誤差があります。430 トークンの入力（Quickstart の state の繰り返し、3 問）で、torch との hidden state の最大差は 4.71 です。場所はマーカー以外の位置のチャネル 578 で、torch の値は -19.5 でした。マーカー位置での差は最大 0.035（平均 8.1e-4）で、答えの確率の差は最大 2.2e-4、argmax は変わりません。

### `bench_ja`

`bench_ja`（300 件。部署の choice 4 択、緊急度の score 3 段階、解約示唆の bool）を、2026-09-28 に commit `b1fe755` で、構成ごとに 1 回だけ実行しました。手順は既存のもの（`scripts/run_baseline_ja.py --skip-laya --skip-llm --sokudan-checkpoint <snapshot 5f91a0d> --sokudan-backend ... --sokudan-device ... --sokudan-dtype ...`）で、学習用 collator の 1 質問 16 件ずつのバッチ、較正なし、ほかの行と同じ指標モジュールです。torch 2.14.0 の CPU、mlx 0.32.2。

| 構成 | choice acc | score RPS | bool acc | bool AUROC |
|---|---|---|---|---|
| 公開値 v0.2（torch、2026-09-27） | 0.880 | 0.075 | 0.780 | 0.844 |
| torch cpu float32 | 0.880000 | 0.074509 | 0.780000 | 0.843868 |
| MLX float32 | 0.880000 | 0.074510 | 0.780000 | 0.843868 |
| MLX float16 | 0.880000 | 0.074512 | 0.780000 | 0.844454 |
| MLX float16 − torch | 0 | +0.000002 | 0 | +0.000586 |

- choice・score の argmax と、bool の 0.5 のどちら側かが torch と食い違った問題は、MLX のどちらの構成でも 0 件でした。torch との確率の最大差は、MLX float32 で 5.0e-06（choice）、1.9e-06（score）、1.1e-05（bool の P(true)）、MLX float16 で 6.7e-03、1.9e-03、4.9e-03 です。
- 実行前に固定した規則: acc か AUROC が torch より 0.01 を超えて低い、または RPS が 0.005 を超えて高い場合だけ、既定を float32 に変える。どちらも起きなかったので、既定は float16 のままです。

## 速度とメモリ

**条件。** Apple M1 Max（P コア 8 + E コア 2）、64 GB、macOS 15.6.1。Python 3.11.16、torch 2.14.0（8 スレッド）、mlx 0.32.2。2026-09-28 17:26〜17:36。機械は空いていませんでした。1 分値が 4 を切るのを 15 分待っても 19.29 のままで、各プロセス開始時は 14.2〜19.3 でした（Adobe Illustrator と macOS のストレージ系サービスが動いていた）。電源の状態は記録していません。以下の数値はすべてこの負荷の下での値です。

**方法**（`scripts/mlx/bench.py`）。`agent.predict` のエンドツーエンド（エンコード・forward・答えの組み立て、既定の較正）。構成とラウンドごとに別プロセスで、各セルはウォームアップ 3 回のあと 30 回計測。3 ラウンドで、4 構成の順番はラウンドごとにずらしました。入力: `short` = Quickstart の state（state 13 トークン）、`long` = 同じ文の繰り返し（state 390 トークン。joint 長は q1 / q3 で平均 430 / 414）、`1k` = 繰り返しで state 949 トークン（989 / 973）。q1 = department の choice（4 択）、q3 = q1 + urgency（score 3 段階）+ refund（bool）で backbone 3 行。

ラウンドごとの中央値 ms（r1 / r2 / r3）、括弧内は p95:

| 入力 / 質問 | torch cpu | torch mps | MLX float32 | MLX float16 |
|---|---|---|---|---|
| short / q1 | 88.7 / 85.6 / 85.2（99.0 / 89.6 / 92.9） | 20.3 / 20.6 / 20.2（21.0 / 22.2 / 20.8） | 11.3 / 11.3 / 11.2（11.7 / 16.5 / 11.6） | 10.4 / 10.3 / 10.2（13.3 / 10.7 / 10.6） |
| short / q3 | 147.9 / 147.3 / 147.5（160.7 / 154.6 / 157.4） | 27.6 / 28.7 / 28.0（28.5 / 29.6 / 28.7） | 20.1 / 20.0 / 20.4（20.6 / 20.5 / 22.1） | 17.8 / 17.8 / 17.8（18.3 / 18.1 / 18.2） |
| long / q1 | 271.6 / 258.2 / 268.5（295.3 / 277.2 / 287.3） | 47.5 / 48.6 / 47.7（49.2 / 51.0 / 49.0） | 46.6 / 46.6 / 46.8（47.2 / 47.1 / 47.3） | 36.7 / 37.2 / 36.4（37.7 / 37.9 / 37.7） |
| long / q3 | 666.2 / 645.8 / 671.9（689.2 / 660.0 / 716.3） | 119.3 / 133.4 / 119.0（168.2 / 164.8 / 171.3） | 165.2 / 171.9 / 161.3（210.6 / 211.8 / 208.2） | 95.9 / 95.7 / 94.2（116.2 / 114.9 / 118.7） |
| 1k / q1 | 591.0 / 575.0 / 580.0（664.5 / 599.2 / 607.5） | 129.8 / 132.0 / 130.4（140.5 / 137.7 / 135.4） | 137.7 / 137.1 / 136.4（174.6 / 141.6 / 144.0） | 96.8 / 96.8 / 96.4（104.2 / 101.5 / 100.3） |
| 1k / q3 | 1486.8 / 1484.0 / 1527.7（1567.9 / 1549.1 / 1583.6） | 403.3 / 405.0 / 411.2（455.1 / 432.8 / 440.1） | 414.5 / 401.9 / 405.3（469.5 / 452.9 / 475.3） | 279.4 / 272.9 / 276.2（294.2 / 284.1 / 293.1） |

- `long / q3` では、torch mps と MLX float16 は計測の最初の 20 回ほどが中央値付近で、その後に遅くなります（125〜185 ms、97〜121 ms）。MLX float32 は最初が 112〜120 ms で、その後は 128〜241 ms の間でばらつきます。原因は調べていません。

メモリ（MiB。プロセスごと、ラウンド 1 / 2 / 3）。ピーク RSS はロードを含みます。MLX の `mx.get_peak_memory()` はロード直後に 1 回、`mx.reset_peak_memory()` のあと全計測の終わりに 1 回読みました（3 ラウンドで同じ値）:

| | ピーク RSS | `mx` ロード後ピーク | `mx` ロード後 active | `mx` predict 中ピーク |
|---|---|---|---|---|
| torch cpu | 2740 / 2743 / 2738 | - | - | - |
| torch mps | 2742 / 2745 / 2744 | - | - | - |
| MLX float32 | 1545 / 1563 / 1557 | 1380 | 1200 | 3390 |
| MLX float16 | 1565 / 1554 / 1566 | 1800 | 600 | 1700 |

- ロード時間（`sokudan.load`、重みはキャッシュ済み、自己診断を含む）: torch cpu 3.9〜5.1 秒、torch mps 4.7〜5.1 秒、MLX 2.2〜2.8 秒。
- 計測に使った 2 つの環境の `site-packages`（`du -sh`）: torch 側 1.5 G（torch 2.14.0、transformers、sokudan の依存に加え pytest・ruff・fastapi・uvicorn）、MLX 側 717 M（mlx 0.32.2、torch なしの transformers に加え、下見で入れた laya-mlx・mlx-embeddings・mlx-vlm・mlx-audio と pytest）。最小構成の `sokudan[mlx]` のサイズは測定していません。

## エンコーダの出典

`sokudan/backends/mlx/modernbert.py` は、laya-mlx 0.2.0（https://github.com/mizorewww/laya-mlx 、PyPI の wheel。ファイルの sha256 は `24ef50d4…`）の `laya_mlx/model.py` 13〜154 行（`EncoderConfig` から `ModernBert` まで）を無改変で取り込んだものです。ライセンスは Apache License 2.0（`NOTICE` を参照）。attention・RoPE・layer norm はもともと `mx.fast` を使っているので、置き換えた箇所はありません。laya-mlx の decision head、プロンプト組み立て、agent は持ち込んでいません。sokudan のヘッドは `sokudan/backends/mlx/model.py` です。
