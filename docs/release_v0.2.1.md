# v0.2.1 の公開記録（2026-09-28）

数字と状態は、すべてこの日に実行・取得したものです。PyPI には上げていません。

## 1. 公開したもの

| 対象 | 場所 | 識別子 |
|---|---|---|
| GitHub `main` | https://github.com/hiroki-abe-58/sokudan | `eef4a4f`（`2401960` からの fast-forward） |
| GitHub タグ | https://github.com/hiroki-abe-58/sokudan/tree/v0.2.1 | `v0.2.1`（注釈付き）→ `eef4a4f` |
| GitHub Release | https://github.com/hiroki-abe-58/sokudan/releases/tag/v0.2.1 | 本文は `CHANGELOG.md` の v0.2.1 の項（公開時点のもの） |
| HF モデル `main` | https://huggingface.co/GeneLab/sokudan-ja-310m | `bb09c2045809c33baa0e97fba95cb09ae099870c`（親 `6cfe9396…` = v0.2 の公開時の main） |
| HF タグ | https://huggingface.co/GeneLab/sokudan-ja-310m/tree/v0.2.1 | `v0.2.1` → `bb09c20` |
| HF Space | `GeneLab/sokudan-demo` | **作成できませんでした**（§4） |
| PyPI | — | 上げていません（§6） |

## 2. 組み立て（`release/v0.2.1`、作業ツリー `sokudan-release`）

1. `origin/main`（`2401960`）から `release/v0.2.1` を切り、`feat/systemone-server` をマージしました（`ad4e6a5`）。**競合はありませんでした。**
2. 研究ブランチ（`main-with-model-results`）から、`docs/public_release_procedure.md` の手順でファイル単位に持ち込みました（`7ebfc67`）。
   - 較正の既定 on: `sokudan/predict.py`、`sokudan/serve/app.py`、`tests/test_calibration_load.py`、`tests/fixtures/predict_golden.json` とその生成スクリプト。
   - `sokudan/order_marginalize.py` と `tests/test_order_marginalize.py`（`predict` の `order_marginalize` 引数、既定 off）。
   - `docs/calibration.md`、`docs/latency.md`、`docs/baseline_lev.md` と `docs/baseline_lev/`（10 本）、`docs/model_card_v0.2.md`（v0.2.1）、`docs/benchmarks.md`（§10 の見出し）。
   - 測定スクリプト 4 本（`calibration_heldout.py`、`bench_calibration.py`、`bench_calibration_bool.py`、`latency_v02.py`）。
   - `assets/calibration.json`: 研究側の `runs/release_candidate/calibration.json` と同じバイト列（SHA-256 `02b789759a9c81a4f778607331ce05c53c280f875fda9ab74e3ebe19b71d6e7c`、`bool/2` = 2.07006759103101 だけ）。
3. 食い違いの解消（同じ commit）:
   - `sokudan serve --temperatures` の既定を `load()` と同じにしました（同梱の bool 較正を既定で on、`none` / `off` と `SOKUDAN_TEMPERATURES=none` で無効）。`/health` の `calibrated` と `calibration.temperatures` は、読み込んだモデルの温度から作ります。各応答の `sokudan.calibrated` / `calibrated_answers` は、その応答の `predict` の値です。
   - `--order-marginalize` と `docs/serving.md` の該当行を削除しました。
   - `README.md`（英語、正本）: Limits の較正の行を差し替え、相対リンクをすべて絶対 URL にしました。`README_en.md` と旧 `space/` を削除しました。`README_ja.md` へのリンクは先頭に残しています。
   - `pyproject.toml` と `sokudan/__init__.py` を 0.2.1 に、`CHANGELOG.md` に v0.2.1 の項を足しました。
   - モデルカードの草稿の HTML コメント 2 つを削除し、研究側にしかない `docs/multidomain.md` への参照を「研究側の記録」と書き換えました。
4. `serve --help` の既定モデルの説明（「Hub main = v0.2」）を直しました（`eef4a4f`）。

## 3. 公開前の確認

- **テスト全件**（`SOKUDAN_TEST_GPU=1`、`sokudan-release/.venv`）: **755 passed, 25 skipped**。スキップは、公開の作業ツリーにない研究用のデータや記録（`data/*.jsonl`、`runs/perf/pre/*`、教師の出力、`runs/calibration/*.npz`）によるものです。研究ブランチの 767 件との差は、そのスキップです。
- **ruff**: `ruff check sokudan scripts tests spaces` は clean。
- **wheel**: `uv build` の wheel を、新しい venv（Python 3.11）に `[serve]` 付きで入れ、`import sokudan`（0.2.1）と `sokudan serve --help` が動くことを確認しました。

## 4. HF Space

2026-09-28 15:35:13 に `create_repo("GeneLab/sokudan-demo", repo_type="space", space_sdk="gradio", space_hardware="cpu-basic")` を実行し、次のエラーで拒否されました（そのまま）:

```
HfHubHTTPError Client error '402 Payment Required' for url 'https://huggingface.co/api/repos/create' (Request ID: Root=1-6aba0aa2-1ad5c12f73fd47b31938a884;6ec80de3-9f86-4a01-a426-0c75075808c1)
For more information check: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/402

Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription. Subscribe at https://huggingface.co/pro
```

- 指示どおり、記録して次に進みました。Static Space など、ほかの形では作っていません。
- `spaces/demo/` のコード（`sokudan @ git+…@v0.2.1` を pin）は、リポジトリにそのまま入っています。作成できる状態になれば、そのまま push できます。
- `CHANGELOG.md` の Space の行は、公開後の commit でこの結果に合わせて直しました。GitHub Release の本文も同じ内容に更新しました。

## 5. 公開後の確認

**未認証の HTTP**（2026-09-28 15:35:28、`curl -L`、認証ヘッダなし）:

| URL | HTTP | バイト | 確認した内容 |
|---|---|---|---|
| `https://raw.githubusercontent.com/hiroki-abe-58/sokudan/v0.2.1/README.md` | 200 | 8,432 | Limits に「`bool` is calibrated by default … ECE 0.181 → 0.105 … (0.075 → 0.132)」 |
| `https://raw.githubusercontent.com/hiroki-abe-58/sokudan/main/docs/serving.md` | 200 | 9,366 | `order-marginalize` の出現 0 回 |
| `https://github.com/hiroki-abe-58/sokudan` | 200 | 323,703 | |
| `https://github.com/hiroki-abe-58/sokudan/releases/tag/v0.2.1` | 200 | 208,242 | |
| `https://huggingface.co/GeneLab/sokudan-ja-310m/raw/v0.2.1/README.md` | 200 | 19,869 | 見出し「sokudan-ja-310m v0.2.1」、Limits の「試した対策と結果（3 系統とも採用していません）」と「別のドメインのデータを足したとき」 |
| `https://huggingface.co/GeneLab/sokudan-ja-310m/resolve/main/calibration.json` | 200 | 253 | SHA-256 `02b78975…6e7c`（リポジトリの `assets/calibration.json` と一致） |
| `https://huggingface.co/GeneLab/sokudan-ja-310m/resolve/v0.2.1/calibration.json` | 200 | 253 | |
| `https://huggingface.co/GeneLab/sokudan-ja-310m` | 200 | 203,472 | |

**HF のファイル**（タグ `v0.2.1`）: `model.safetensors` は 1,258,474,752 バイト、LFS の SHA-256 `8750a833b3faa537a39709b8377b82a096286f040d78bb1915b82d9d101b5965` で、v0.2 と同じです（重みは変えていない）。v0.1 の頃からある `temperatures.json`（7,039 バイト）には触れていません。`load()` の既定は `calibration.json` だけを読みます。HF の既存のタグ `v0.2` は `942c80c…` を指したままです。

**タグからのインストールと CPU での serve**（15:36:01 – 15:38:24）:

- 新しい一時 venv（Python 3.11）に `uv pip install "sokudan[serve] @ git+https://github.com/hiroki-abe-58/sokudan.git@v0.2.1"` で入れました（指示の `pip install` の代わりに `uv pip`）。入った sokudan は 0.2.1、torch は `2.11.0+cu128` でした（uv のキャッシュから。この機械で読み込める版）。
- `CUDA_VISIBLE_DEVICES="" sokudan serve --device cpu --port 8765` で起動し、`/health` は `"device":"cpu","sokudan_version":"0.2.1","calibrated":true` と `calibration.temperatures = {"bool/2": 2.07006759103101}` を返しました。
- `docs/serving.md` の例（請求の二重引き落とし、3 問）を `curl` で送った応答:

| 質問 | 型 | 答え |
|---|---|---|
| department | choice | `請求`（請求 0.9975、技術 0.0004、営業 0.0013、その他 0.0008、confidence 0.9967） |
| urgency | score | 1.0899（0.2075 / 0.4951 / 0.2974、confidence 0.2426） |
| churn | noul | 0.0903 |

  - `sokudan` の拡張は `"calibrated":true,"calibrated_answers":["churn"]`、`backbone_passes` 3、`latency_ms` 245.96 でした（CPU、1 回だけ。速度の測定ではありません）。
- サーバーのプロセス（python と sokudan.exe）を止め、`/health` に応答がないことを確認してから、一時 venv を削除しました（バックグラウンドのジョブは、止めたため終了コード 127 で終わっています）。

## 6. PyPI（上げていない）

- `sokudan-release` で `dist/` を作り直しました（`uv build`）。`twine check` は両方 PASSED です。
  - `sokudan-0.2.1-py3-none-any.whl` SHA-256 `6357173d58adc61aa40ea82747a3d90ad0d79fb940a1d1496dee66d8fc38cc4d`
  - `sokudan-0.2.1.tar.gz` SHA-256 `0a7c5291fddce378dd92060407d16273d3401f6d8f2ce0e0d9997a113fd44aa5`
- 2026-09-28 15:39:58 の時点で、`https://pypi.org/pypi/sokudan/json` と `sokudan-ja` は、どちらも 404 でした。
- 実行するのは Hiroki さんです（`docs/release_pypi.md` §4.2、トークンは環境変数にだけ置く）:

```powershell
cd C:\Users\hirok\sokudan-release
$env:TWINE_USERNAME = "__token__"; $env:TWINE_PASSWORD = Read-Host -AsSecureString "PyPI token" | ConvertFrom-SecureString -AsPlainText
uvx twine upload dist\*; Remove-Item Env:TWINE_PASSWORD
```

## 7. 規則にない判断

1. **`main` へは fast-forward で入れました。** `release/v0.2.1` は `origin/main` から切ってあり、`feat/systemone-server` のマージ commit（`ad4e6a5`）がすでに履歴にあるため、もう 1 つマージ commit を作る必要がありませんでした。
2. **HF の `calibration.json` と `README.md` は 1 つの commit で上げ、`parent_commit` に v0.2 の公開時の main（`6cfe939`）を指定しました。** その間にほかの変更が入っていれば失敗するようにするためです。
3. **HF の `temperatures.json`（v0.1 の頃のもの）は消していません。** 指示にないためです。`load()` の既定では読まれません。
4. **Space は 402 のあと、ほかの形（Static Space、別の hardware）を試していません。**
5. **この記録と、CHANGELOG の Space の行の修正は、タグの後の commit として `main` に入れました。** タグ `v0.2.1` の中身（`eef4a4f`）は変えていません。GitHub Release の本文は、直した CHANGELOG の項に差し替えました。

## 8. 仕上げ: PyPI の install 行と Colab ノートブック（2026-09-28 15:48 –）

- **PyPI の確認**（15:48:35）: `https://pypi.org/pypi/sokudan/json` の版は 0.2.1（公開された版は 0.2.1 だけ）でした。wheel と sdist の SHA-256 は §6 の `dist/` と一致しています（`6357173d…`、`0a7c5291…`）。
- **install 行**: `README.md`、`README_ja.md`、`docs/serving.md`、モデルカードを `pip install sokudan`（サーバーは `pip install "sokudan[serve]"`）にしました。`git+` の行は「開発版」として残しています。README に PyPI のバッジを足しました。
- **`notebooks/sokudan_quickstart.ipynb`**: PyPI からの install → 3 型の `predict` → bool 較正の on/off → ノートブックの中で `sokudan serve` を起動して `curl` で 1 回、まで。README の Quickstart の直後に「Open in Colab」のバッジを置きました。
- **ローカルでの実行**（`python -m nbconvert --execute`、`CUDA_VISIBLE_DEVICES=""`、新しい venv に PyPI から入れる）: Python 3.12.13（16:01:20 – 16:02:46）と 3.11.9（16:02:49 – 16:04:10）の両方で、コードセル 8 つがエラーなしで通りました。torch は PyPI の `2.14.0+cpu`、`device: cpu` です。

| | calibrated | raw |
|---|---|---|
| 請求の二重引き落とし（README の例） | 0.0903 | 0.0083 |
| 他社のほうが安いので今月いっぱいで契約を終わりに | 0.9447 | 0.9972 |
| 新しいプランの料金表を送って | 0.0982 | 0.0100 |

  - `curl` の応答は department `請求`、urgency 1.0899、churn 0.0903、`calibrated_answers` `["churn"]` でした。
- **Colab 自体では実行していません。** Colab のランタイムの Python の版も、この環境からは確認していません。

**規則にない判断**:

1. **Python 3.11 以外では `--ignore-requires-python` を付けて入れます。** sokudan 0.2.1 は `requires-python = ">=3.11,<3.12"` を宣言していて、そのままでは新しい Python（Colab）で `pip install sokudan` が通らないためです。3.12 で通ることは、上のローカルの実行で確かめました。宣言を広げるのは次の版の判断として残しています（0.2.1 は上げ直せない）。
2. ローカルの実行は、パスの短い一時ディレクトリで行いました。作業用のディレクトリでは、torch の同梱ライセンスのパスが Windows の長さの上限を超え、pip が `WinError 206` で失敗したためです（ノートブックの問題ではない）。`jupyter` のランチャー exe はアプリケーション制御に止められたので、`python -m nbconvert` で実行しました。
3. Colab の中の `curl` の応答はファイル（`resp.json`）に書き、Python で UTF-8 として読みます。Windows のローカル実行で、`!` の出力の文字コードに左右されないようにするためです。

**HF のモデルカード**: `README.md` だけを main に上げました（`5f91a0d962b45df1794cfadff008a4da88a67a53`、親 `bb09c20`）。重みは `8750a833…` のままです。タグ `v0.2.1` は `bb09c20` のまま動かしていません（タグの時点のカードには PyPI の行がありません）。

- 訂正（§5 の追記）: `list_repo_refs` の `942c80c…`（v0.2）と `ad117df…`（v0.2.1）は、注釈付きタグのオブジェクトの ID でした。`model_info(revision=…)` で解決すると、v0.2 → `6cfe939`、v0.2.1 → `bb09c20` です。

**未認証の HTTP**（16:05:30、`curl -L`）: 次のすべてが 200 でした。

| URL | バイト | 確認した内容 |
|---|---|---|
| `https://raw.githubusercontent.com/hiroki-abe-58/sokudan/main/README.md` | 9,101 | `pip install sokudan` の行、PyPI のバッジ、Colab のバッジの URL |
| `https://raw.githubusercontent.com/hiroki-abe-58/sokudan/main/README_ja.md` | 24,093 | |
| `https://raw.githubusercontent.com/hiroki-abe-58/sokudan/main/docs/serving.md` | 9,446 | `pip install "sokudan[serve]"` と開発版の行 |
| `https://raw.githubusercontent.com/hiroki-abe-58/sokudan/main/CHANGELOG.md` | 10,138 | |
| `https://github.com/hiroki-abe-58/sokudan/blob/main/notebooks/sokudan_quickstart.ipynb` | 255,410 | |
| `https://raw.githubusercontent.com/hiroki-abe-58/sokudan/main/notebooks/sokudan_quickstart.ipynb` | 8,826 | JSON として読め、セル 14（nbformat 4） |
| `https://colab.research.google.com/github/hiroki-abe-58/sokudan/blob/main/notebooks/sokudan_quickstart.ipynb` | 97,320 | Colab のアプリの枠。ノートブックが Colab で開けることまでは、この取得では確認していない |
| `https://colab.research.google.com/assets/colab-badge.svg` | 2,369 | |
| `https://img.shields.io/pypi/v/sokudan` | 1,275 | バッジの文字列に `v0.2.1` |
| `https://pypi.org/project/sokudan/` | 3,038 | 本文は小さい（JS のページの可能性）。版は JSON API で確認済み |
| `https://huggingface.co/GeneLab/sokudan-ja-310m/raw/main/README.md` | 20,151 | `pip install sokudan` と Colab のリンク |
