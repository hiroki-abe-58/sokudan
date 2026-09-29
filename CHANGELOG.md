# Changelog

数値はすべて本機で実行したコードの出力です。未測定のものは「測定していない」と書きます。

## v0.3.0 — 2026-09-29 / パッケージ 0.3.0

**モデルの重みは v0.2 のまま（変更なし）です**（`GeneLab/sokudan-ja-310m` の `model.safetensors` と `calibration.json` は v0.2.1 と同じ）。MLX の数値は M1 Max（64 GB、macOS 15.6.1）での実測です（`docs/mlx.md`）。

### アップグレード時の注意

- **開発環境（リポジトリの clone）では `uv sync --extra dev` で同期してください。** `dev` は、学習・評価用の依存（extra `train`）を含みます。
- **extra なしで `uv sync` を実行すると、学習用の依存が消えます。** Windows での実測（uv 0.11.29、`uv sync --extra dev --locked` で作った 77 パッケージの環境）で、extra なしの `uv sync --locked` は 36 パッケージ（`datasets`、`fugashi`、`matplotlib`、`unidic-lite`、`pyarrow`、`pandas`、`pytest`、`ruff` ほか）をアンインストールし、41 パッケージになりました。
- Apple Silicon（macOS 14 以降）では、torch が基本の依存から外れます。torch を使うときは `sokudan[torch]`（開発環境では `--extra torch`）を入れてください。

### Python 3.11〜3.13 に対応

- `requires-python = ">=3.11,<3.14"`（0.2.1 までは 3.11 のみ）。
- `uv.lock` は、3.12 / 3.13 の wheel が加わっただけで、既存の版は変わっていません（追加されたパッケージは `audioop-lts` 0.2.2 だけ）。
- Windows（研究ブランチ、PyPI の CPU 版 torch 2.14.0、fugashi 1.5.2 の cp313 wheel、unidic-lite 1.0.8）で、3.12.13 と 3.13.14 の新しい環境でテスト全件を回しました（3.11 は 764 passed / 7 skipped。3.12 / 3.13 で落ちたのは環境のテスト 2 件で、Python の版のテストは直した）。
- Colab の Python 3.13.15 でノートブックの全セルが通ったことは、チャット側の記録にあります（このリポジトリでは測っていない）。
- Colab ノートブック（`notebooks/sokudan_quickstart.ipynb`）: Python の版で分けていた `--ignore-requires-python` を外し、`sokudan[serve]>=0.3.0` を入れます。

### インストールされるものが変わります

- **Apple Silicon の macOS 14 以降（Darwin 23 以降）では、`pip install sokudan` で MLX（`mlx>=0.32.2,<0.33`）が入り、torch は入りません。** それ以外（Linux、Windows、Intel Mac、macOS 13 の Apple Silicon）では従来どおり torch が入ります。
- extras: `sokudan[torch]`（どこでも torch を足す）、`sokudan[mlx]`（上と同じ条件で MLX）、`sokudan[serve]`。
- `backend="torch"` / `"mlx"` を指定して、そのライブラリが入っていないときは、`pip install "sokudan[torch]"` / `"sokudan[mlx]"` を案内する ImportError になります。
- Apple Silicon で torch を使う既存のコード（学習・評価のスクリプト、`backend="torch"`）は `sokudan[torch]` が必要です。
- **データ生成・学習・評価だけが使う依存（`datasets`、`fugashi`、`unidic-lite`、`matplotlib`）は extra `train` に移しました。** `load` + `predict`（MLX・torch）と `sokudan serve` のどれでも import されないことを、`sys.modules` と `python -X importtime` で確かめています。スクリプトを動かすときは `pip install "sokudan[train]"`（`dev` extra は `train` を含む）。移動の前後で、630 問の合成セットの確率は torch cpu・MLX float16 とも同一（差 0）でした。
- 0.3.0 の wheel をクリーンな Python 3.11 環境（Apple Silicon）に入れると、38 パッケージ（torch なし、mlx 0.32.2）、`site-packages` 393 MB でした（`train` を分ける前は 66 パッケージ、934 MB）。`[torch]` 付きでは 44 パッケージ、1.1 GB です。

### 既知の事項

- **torch が入っている環境では、MLX でロードしても torch が import されます。** tokenizer と backbone の config に使う transformers が、入っている torch を import するためです（`import sokudan` 自体は torch を import しません）。
- **`uv.lock` の torch は、`tool.uv.sources` の cu128 index（torch 2.11.0+cu128）のままです。** lock 内の torch の wheel は manylinux x86_64 / aarch64 と win_amd64 だけで、macOS 用はありません。source に `sys_platform != 'darwin'` を付けて lock し直すと、macOS 用の torch 2.14.0（wheel は `macosx_14_0_arm64` のみ）が加わる一方、linux / win の torch エントリにも `resolution-markers` の 5 行が加わったので、変更は入れていません。

### MLX バックエンド（Apple Silicon）

- **`sokudan.load(..., backend="auto" | "mlx" | "torch", dtype=...)`。** 既定の `auto` は MLX（`mlx` が import でき、Apple Silicon のとき）→ torch `mps` → `cuda` → `cpu` の順に試します。各候補は短いリクエストを 1 回答えてから使われ、失敗すると warning を出して次に進みます。選ばれたものは `agent.backend` で見えます。
- MLX の既定は float16（backbone のみ。ヘッドは float32）。`dtype="float32"` も選べます。8bit / 4bit の量子化は一致のゲートを満たさなかったので、選択肢にしていません。動かした MLX は 0.32.2 だけです。
- **`bench_ja`（構成ごとに 1 回、commit `b1fe755`）:**

  | | choice acc | score RPS | bool acc | bool AUROC |
  |---|---|---|---|---|
  | torch cpu float32 | 0.880 | 0.0745 | 0.780 | 0.8439 |
  | MLX float32 | 0.880 | 0.0745 | 0.780 | 0.8439 |
  | MLX float16 | 0.880 | 0.0745 | 0.780 | 0.8445 |

  予測（argmax、bool の 0.5 のどちら側か）が torch と食い違った問題は、MLX のどちらでも 0 件です。
- torch の CPU（float32）との一致（320 state / 630 問の合成セット、生の確率）:
  - float32: argmax 630/630、最大絶対差 9.1e-06
  - float16: argmax 629/630（外れた 1 問は torch 側で 0.4110 と 0.4103 のほぼ同率）、最大絶対差 8.2e-03
- `predict` 1 回の中央値（3 ラウンド、1 分値 14〜19 の負荷の下）: Quickstart の state・1 問で torch cpu 85〜89 ms / torch mps 20〜21 ms / MLX float16 10.2〜10.4 ms。state 949 トークン・3 問で 1484〜1528 / 403〜411 / 273〜279 ms（全セルは `docs/mlx.md`）。ピーク RSS は torch 約 2.7 GiB、MLX 約 1.5 GiB。
- ModernBERT のエンコーダは laya-mlx 0.2.0 の実装を無改変で取り込みました（Apache-2.0、`NOTICE`）。

### 較正: choice は raw のまま

- choice の温度（val で fit した choice/4 = 2.318）を、保存済みの bench の出力に当てて見直しました（事前登録、`docs/calibration.md` §11〜§12）。
  - ECE（15 等分位ビン）は、`bench_ja` で 0.088 → 0.066 と下がりましたが、`bench_en` では 0.091 → 0.228 と上がりました。
  - 「両方で改善」の条件を満たさないので、**既定で較正するのは `bool` だけのまま**です。
- `bench_ja` では、300 件中 224 件が P(top1) ≥ 0.99 で、その 6.25% が誤答でした。

### ドキュメント

X の公開スレッドでの報告を受けて、測り直した数字と運用の目安を足しました。

- **`docs/choice_guidance.md`**（英語）: choice の運用の目安を 5 点にまとめました。どれも測った数字つきです。
  1. 説明文を書く。
  2. 包括的な選択肢（その他）は、argmax ではなく P(その他) に閾値を置き、先頭と最後を避ける。
  3. 位置の影響。
  4. 確認用途では上位 2 つを見せる。
  5. confidence を数値で出す前に、分布を確かめる。
- **Limits**: 包括的な選択肢の項を足しました（`bench_ja` の「その他」の recall 0.289、検査での選択率、AUROC 0.94、near-miss の対）。
- **`docs/benchmarks.md` §11**: 部署ルーティングのクラス別の成績（混同行列、Top-2）、全 24 順序でのスロット別選択率、説明文の有無。
- **`docs/probe_catch_all.md`**: 「その他」の検査（著者 1 人の 90 件）。スクリプトは `scripts/probe_catch_all.py`。
- **`docs/catch_all_prereg.md`**: 次の学習の候補（包括的な選択肢が正解の例を加える）の事前登録。実行しておらず、パッケージ 0.3.0 には含まれません。
- **Used by**: README / README_ja に、会計アプリの勘定科目の候補提示（PoC）での利用を、X の公開スレッドへのリンクつきで載せました。

### その他

- **`sokudan.load`（torch）が、transformers の読み込みの報告（「UNEXPECTED: head.*」）を出さなくなりました。**
  - 報告にあったのは、事前学習の backbone に付いている masked-LM の head のキーです。
  - その直後に、学習済みの重みを strict な `load_state_dict` で全部読み込むので、利用者に意味のない表示でした。
  - `TorchBackend.load` が backbone を作る間だけ、transformers のログを ERROR にします。学習の経路（`Backbone.load`）では報告を残します。MLX の経路では、この報告は出ません。
  - 重みが変わらないことは `tests/test_load_report.py` で、630 問の合成セットの確率が変わらないこと（torch cpu・MLX float16、差 0）は実行して確かめています。
- **`sokudan serve --backend auto|mlx|torch --dtype ...`。** 起動ログに、選ばれた backend・device・dtype を出します。
- **`scripts/run_baseline_ja.py --sokudan-backend / --sokudan-device / --sokudan-dtype`。** sokudan の行は `sokudan.load` 経由になり、`.pt` のほか safetensors のディレクトリや Hub id も読めます。問題ごとの確率を `item_probs.npz` に保存します。
- `uv.lock` を更新しました（mlx 0.32.2 を追加。以前の lock は sokudan 0.2.0 のままでした）。
- **`device` の既定を `"auto"` に**（`load` と `sokudan serve --device`）。torch では `cuda` → `mps` → `cpu` の順に選びます。明示した `cpu` / `cuda` / `mps` は従来どおりです。M1 Max で `mps` と `cpu` の確率の差は最大 4.5e-06（630 問、argmax は全問一致）。
- **`import sokudan` は torch を import しなくなりました。** forward は `sokudan/backends/`（`torch_backend.py`、`mlx/`）に分け、`predict.py` はエンコード・バッチ組み立て・温度・答えの組み立てだけを持ちます。torch の出力は分離の前後で同一です（630 問、差 0）。

## v0.2.1 — 2026-09-28 / パッケージ 0.2.1

**モデルの重みは v0.2 と同じです**（`GeneLab/sokudan-ja-310m` の `model.safetensors` は変えていない）。

### 較正: `bool` を既定で on

- **`sokudan.load` は、重みに同梱した `calibration.json` の `bool` の温度（bool/2 = 2.070）を既定で当てます。** `temperatures=None` で無効にでき、その場合は v0.2 と同じ生の確率です。
  - 温度は、held-out（frozen 9,850 行）の 2 分割交差で判定し、全行で fit したものです（`docs/calibration.md` §1〜§5）。
  - bench は、v0.2 の出力（1 回だけ推論し直し、記録の値と差 0.0 で一致）に後から当てて判定しました（§6〜§9）。

  | | bool ECE 前 → 後 | \|平均 P(true) − 正例率\| 前 → 後 |
  |---|---|---|
  | `bench_ja` | 0.181 → 0.105 | 0.164 → 0.098 |
  | `bench_en` | 0.249 → 0.151 | 0.183 → 0.096 |

  - bool acc（閾値 0.5）と AUROC は、構成上変わりません。
- **`score` と `choice` は較正しません。** val で fit した score の温度（score/3 = 4.50）が、`bench_ja` の score RPS を 0.0745 → 0.1322 に悪化させたためです（`docs/calibration.md` §7）。
- **応答に `calibrated`（温度を当てた答えがあるか）と `calibrated_answers`（その質問 ID）を足しました。**
- `calibration.json` は Hub の `main` に置き、リポジトリにも同じもの（`assets/calibration.json`）を入れています。

### `sokudan serve`: `/v1/systemone` 互換のサーバー

- `sokudan/serve/systemone.py`、`sokudan/serve/wire.py`。TypeSafe の公開 API リファレンスの形で受けて返します。手順は `docs/serving.md`、各フィールドの扱いは `docs/systemone_wire_format.md` です。
  - noul の答えの `type` は `"noul"`、`confidence` はワイヤ形式の定義（choice は `(p_max − 1/K)/(1 − 1/K)`）です。
  - 較正は `load` と同じ既定です。`--temperatures none`（または `SOKUDAN_TEMPERATURES=none`）で無効になります。`/health` は、読み込んだモデルの温度を返します。各応答の `sokudan.calibrated` / `calibrated_answers` は、その応答で温度を当てた答えを表します。
  - 推論時の順序平均の設定（`--order-marginalize`）は削除しました。順序の実験（推論時の平均、学習時の perm-KL、データ側の並べ替え）は、どれも見送りました（モデルカードの Limits）。

### デモ: Colab ノートブック（Space の代わり）

- **`notebooks/sokudan_quickstart.ipynb`**（[Open in Colab](https://colab.research.google.com/github/hiroki-abe-58/sokudan/blob/main/notebooks/sokudan_quickstart.ipynb)）: PyPI からの install → 3 型の `predict`（README の例）→ bool 較正の on/off での P(true) の違い → ノートブックの中で `sokudan serve` を起動して curl で 1 回叩く、まで。無料の CPU ランタイムで動く構成です。
  - sokudan 0.2.1 は Python 3.11 だけを宣言しているので、3.11 以外（Colab のランタイム）では `--ignore-requires-python` を付けて入れます。
- **Hugging Face Space は、無料枠では作成できません（PRO が必要）。Colab で代替しました。** `GeneLab/sokudan-demo` の作成は、HF に 402 で拒否されました（2026-09-28。「Gradio の Space を無料の cpu-basic で置くには PRO が必要」という応答。`docs/release_v0.2.1.md`）。`spaces/demo/` のコードはリポジトリに残しています。
- 以前の `space/`（v0.1 用）は削除しました。

### PyPI

- **PyPI: `sokudan` 0.2.1**（https://pypi.org/project/sokudan/）。`pip install sokudan`、サーバーは `pip install "sokudan[serve]"` です。README と `docs/serving.md` の install 行もこれにしました（`git+` の行は開発版として残す）。
- パッケージ名 `sokudan`、版 0.2.1 として配布できるように `pyproject.toml` を整えました（依存の上限、`huggingface-hub` と `safetensors` の明記、extras `serve` と `demo`、sdist の対象の絞り込み。`docs/release_pypi.md`）。

### ドキュメント

- README: 正本を英語の `README.md` にし、日本語の全文を `README_ja.md` にしました。`README_en.md` は削除しました。README のリンクは絶対 URL です。
- `docs/baseline_lev.md`: lev（interfaze-ai/lev）と sokudan v0.2 を、同じマシン・同じハーネスで比べた表（3 型、score の位置感度、速度と VRAM）。
- モデルカード（v0.2.1）: 較正の節、位置感度の「試した対策と結果」（3 系統とも不採用）、別ドメインのデータを足した試験（specialist 0.703、held-out は低下）。
- `docs/calibration.md`、`docs/latency.md`（v0.2 の 3 問 × 1 件: 中央値 22.8 ms、RTX 5090）。`docs/benchmarks.md` §10 の見出しは「v0.2（S8_old: v0.1 seed 0〜7 の soup）」にしました。

## v0.2 — 2026-09-27（モデル）/ パッケージ 0.2.0

### モデル（Hugging Face `GeneLab/sokudan-ja-310m` の `main`）

- **v0.2 は model soup です。** v0.1 と同じ設定で学習した 8 本（seed 0〜7）の重みを、すべて float32 で単純平均しました（ヘッドを含む）。
  - アーキテクチャ、学習データ、推論コードは v0.1 と同じで、推論のコストも 1 本分のままです。
- **選定**: 候補は 4 つの soup（8 本、別コードの 8 本、16 本、24 本）で、事前に commit した規則で選びました（`docs/release_candidate.md`）。
  - 規則: held-out の M1m が最大で、ガードレールが非劣化であること。
- **`bench_ja` は、リリース規則を commit してから 1 回だけ**測りました。規則は満たしています。
- v0.1 の重みは、revision `v0.1` として残しています（`sokudan.load("GeneLab/sokudan-ja-310m@v0.1")`）。

### v0.1 からの数値の変化

`bench_ja` 300 件、較正前。v0.1 は 3 シード平均 ± SD、v0.2 は soup 1 体の 1 回の値です。

| | choice acc | score RPS↓ | score acc | bool acc | bool AUROC |
|---|---|---|---|---|---|
| v0.1 | 0.847 ± 0.009 | 0.090 ± 0.023 | 0.763 ± 0.088 | 0.788 ± 0.010 | 0.789 ± 0.043 |
| **v0.2** | **0.880** | **0.075** | **0.817** | 0.780 | **0.844** |
| 差 | +0.033 | −0.016 | +0.053 | **−0.008** | +0.055 |

held-out（未知スキーマ、合成。比較相手は v0.1 と同じ設定の 16 本の平均 ± SD。v0.2 の値は選定にも使った値）:

| | M1m | M2（400〜799 トークン） | M5（`implies_declining`） |
|---|---|---|---|
| 16 本 | 0.8239 ± 0.0142 | 0.8304 ± 0.0341 | 0.7657 ± 0.0356 |
| **v0.2** | **0.8644** | **0.8590** | **0.8346** |

`bench_en` 290 件（v0.2 のみ。v0.1 は測定していない）: choice 0.872、score RPS 0.114、score acc 0.652、bool acc 0.690、bool AUROC 0.621。

### 既知の弱点

- **4 段階の `score` で、第 1 選択肢がほとんど選ばれません**（`bench_ja` の位置検査の条件 E で 300 件中 5 件。v0.1 の 3 シードは 58 / 16 / 62 件）。
- held-out の状態でも、第 1 スロットはやや不利です。
  - 全選択肢同一の対照: score −0.249、choice −0.280。
  - 全順列の第 1 スロット率: score 0.239、choice 0.289（中立なら 0 と 1/3）。
- **bool acc が v0.1 より 0.008 低く、true を過少予測します**（mean P(true) 0.133、gold の陽性率 0.297）。
- `bench_en` の bool acc（0.690）は、多数決（0.683）とほぼ同じです。
- 長い state と、位置に依存する表層の属性（`ends_with_question` 0.782）では弱いままです。
- `predict` に state を dict で渡すと `key: value` の行に整形され、学習時と違う入力になります。**文字列で渡してください。**

### パッケージとコード（0.1.1 → 0.2.0）

- `sokudan.load` が、学習時の `input_order`（`question_first` / `state_first` / `sandwich`）と `local_attention` を checkpoint から復元するようになりました。
- 温度のフィットが scipy を使わなくなりました（golden-section 探索）。コアの依存だけで `calibrate` が動きます。
- `sokudan probe-position` コマンドを追加しました。位置バイアスの検査で、`--shuffle` は選択肢を事例ごとに並べ替えます。
- 学習ループ: トークナイズ結果の例ごとのキャッシュ、バッチ組み立ての先読みスレッド、loss の同期の間引き。
  - 学習と評価の数値は変わりません。`tests/test_perf_identity.py` で、変更前の記録と比べています。
- `scripts/eval_local_attention.py --save-probs`: 1 回の推論で、評価の値と事例ごとの予測を両方出します。
- `scripts/make_soup.py`（soup の再構成と SHA-256 の照合）、`scripts/export_v02.py`（HF 形式の書き出しと読み込みの確認）を追加しました。
- ヘッドの初期化を 2026-09-26 にシードで決まるようにしましたが、事前登録した退行チェック（16 対 16、p = 0.047）の結果、2026-09-27 に元に戻しました（`docs/regression_check.md`）。
- pytest は、既定で CUDA を隠します。GPU テストは `@pytest.mark.gpu` を付け、`SOKUDAN_TEST_GPU=1` のときだけ走ります。
- 研究の手順は `docs/research_protocol.md` にあります（事前登録、8 本以上の分布での比較、基準分布は 16 本以上）。

## v0.1.1 — 2026-09-22

- `pip install sokudan` の直後の `predict()` が、scipy の ModuleNotFoundError で落ちる問題を直しました（scipy の import を遅延）。

## v0.1 — 2026-09-21

- 最初の公開です。joint encoding、意図条件付きの合成コーパス、`bench_ja` のゲートを 3 条件とも通過（3 シード平均）。
