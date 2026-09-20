# Gate A — FlashAttention-2 判定と注意機構スループット実測

> SOKUDAN_SPEC.md §4.1 / §14.2（ブロック 0:00–1:00）
> **この文書の数値はすべて本機で実行した `scripts/gate_a_attention.py` の出力。推定値はない。**

実測日時: **2026-09-20T13:16:11+0900**
生データ: `docs/data/gate_a.json`（`runs/` は .gitignore 対象なのでコミット用に複製）

再現コマンド:
```bash
uv run python scripts/gate_a_attention.py --json docs/data/gate_a.json
```

---

## 1. 判定

**Gate A = FALLBACK（FA2 は使えない）→ `attn_implementation="sdpa"` + 長さバケット化 padding で進む。**

§4.1 が想定していた「sm_120 向け wheel が無い」ではなく、**別の理由**で失敗した。正確に記録する。

### flash-attn のビルド失敗（実測）

`uv pip install flash-attn --no-build-isolation` を試行。2 段階で失敗した。

1 回目 — ビルド依存の宣言漏れ:
```
ModuleNotFoundError: No module named 'psutil'
hint: flash-attn@2.8.3.post1 depends on psutil, but doesn't declare it as a build dependency
```

`psutil` を入れて再試行。2 回目 — **CUDA バージョン不一致**:
```
File ".venv/Lib/site-packages/torch/utils/cpp_extension.py", line 545, in _check_cuda_version
    raise RuntimeError(CUDA_MISMATCH_MESSAGE, cuda_str_version, torch.version.cuda)
RuntimeError: ('The detected CUDA version (%s) mismatches the version
that was used to compilePyTorch (%s). Please make sure to use the same
CUDA versions.', '13.1', '12.8')
```

原因は明確で、sm_120 とは無関係:

| | 実測値 |
|---|---|
| システムの CUDA Toolkit (`nvcc --version`) | **13.1** (V13.1.115, `CUDA_PATH=...\CUDA\v13.1`) |
| PyTorch がビルドされた CUDA | **12.8** (`torch.version.cuda`) |

flash-attn は `torch.utils.cpp_extension` 経由で CUDA 拡張をビルドするため、
両者の一致を要求する。**sm_120 のカーネルが存在しないのではなく、
この機械の toolkit で torch 拡張をコンパイルできない。**

### 採らなかった選択肢と理由

- **CUDA 12.8 Toolkit を別途入れて再ビルド** — 技術的には通る見込みがあるが、
  インストールと flash-attn のフルビルドは §14.2 のブロック 1（1 時間）に収まらない。
  §14.2「速度の最適化は当日の仕事ではない」に従い、後日の課題とする。
- **CUDA 13.x ビルドの torch に合わせる** — `arch_list` の確認からやり直しになる。
  現在の `2.11.0+cu128` は sm_120 で正常動作しており（下記）、崩す理由がない。

**後日の TODO**: CUDA 12.8 Toolkit を入れて flash-attn のビルドを再試行し、
FA2 と sdpa の差を同じスクリプトで測り直す。

---

## 2. 測定環境（実測）

| 項目 | 値 |
|---|---|
| Platform | Windows-10-10.0.26200-SP0 (Windows 11 Pro 26200) |
| Python | 3.11.9 |
| PyTorch | **2.11.0+cu128** (CUDA build 12.8) |
| transformers | 5.17.0 |
| GPU | NVIDIA GeForce RTX 5090 |
| `get_device_capability()` | **(12, 0)** = sm_120 |
| `get_arch_list()` | `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']` |
| Driver / CUDA runtime | 595.95 / 13.2 |
| Backbone | `sbintuitions/modernbert-ja-310m` |

### 注意: 既存のグローバル環境は使えなかった

この機械のグローバル Python 3.11 には `torch 2.11.0.dev20260119+cu126` が入っていたが、
**sm_120 のカーネルを含まない**ため実行時に落ちる。`import` と `is_available()` は通るので、
見落としやすい。`tests/test_env.py::test_bf16_matmul_actually_runs_on_device` は
この失敗を再現・検出するために置いてある。

```
arch_list = ['sm_50','sm_60','sm_61','sm_70','sm_75','sm_80','sm_86','sm_90']   # sm_120 が無い
capability = (12, 0)
matmul bf16 -> AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

プロジェクトは `.venv`（uv, cu128）で完結しており、グローバル環境は変更していない。

---

## 3. スループット実測（sdpa フォールバックの実力）

ModernBERT-Ja-310M、bf16、batch_size 32、warmup 3 回 + 計測 10 回、padded 入力。

| impl | seq_len | ms/mean | ms/p50 | ms/p95 | seq/s | tok/s | peak VRAM (MiB) |
|---|---|---|---|---|---|---|---|
| **sdpa** | 128 | 20.96 | 20.95 | 23.02 | 1526.6 | 195410.4 | 745.3 |
| **sdpa** | 256 | 39.71 | 39.79 | 41.56 | 805.9 | 206298.0 | 872.7 |
| **sdpa** | 512 | 79.63 | 79.70 | 81.73 | 401.9 | 205755.3 | 1131.0 |
| **sdpa** | 1024 | 173.95 | 173.33 | 179.81 | 184.0 | 188373.7 | 1659.5 |
| eager | 128 | 23.61 | 23.62 | 24.50 | 1355.6 | 173514.7 | 757.6 |
| eager | 256 | 51.78 | 51.57 | 55.67 | 618.0 | 158200.8 | 958.7 |
| eager | 512 | 131.79 | 131.51 | 135.83 | 242.8 | 124318.2 | 1787.0 |
| eager | 1024 | 378.76 | 378.03 | 384.21 | 84.5 | 86514.7 | 4907.5 |

sdpa / eager の比（実測）:

| seq_len | sdpa は eager の何倍速いか |
|---|---|
| 128 | 1.13x |
| 256 | 1.30x |
| 512 | 1.66x |
| 1024 | **2.18x** |

`flash_attention_2` は上表に無い。**測れていないものは書かない。**

### §4.1 の中止条件の判定

> 「想定より 3 倍以上遅いなら、バックボーンを `modernbert-ja-130m` に落とす選択肢を含めて報告する」

**該当しない。310m を継続する。** 根拠:

- FA2 との直接比較はできない（ビルドできないため）。したがって「FA2 比 3 倍」は本日は判定不能。
- 代わりに、当日の学習規模で間に合うかを絶対値で見る。§7.4 の当日データ規模は 2〜3 万件。
  sokudan は state 側と question 側を別系列でエンコードする（§6.2）ので、
  1 サンプルあたり概ね seq=512 級 1 本 + seq=256 級 N 本に相当する。
  **seq=512 で 401.9 seq/s、seq=256 で 805.9 seq/s** という実測値は、
  3 万件 1 epoch のフォワードが分オーダーで終わる水準であり、
  §14.2 ブロック 7:00–8:00（学習 1 時間）に収まる。
- VRAM も seq=1024 / bs=32 で 1659.5 MiB と、32GB に対して十分な余裕がある。

したがって sdpa で先へ進む。**ただし上記は forward のみの実測**であり、
backward・optimizer を含む実学習スループットは Phase 4 で別途測る。

---

## 4. 設計への反映

- `attn_implementation="sdpa"` を既定にする。unpadding 経路は使わない。
- padded 入力なので、**長さでバケット化したバッチ**を作る（§6.2）。
- `torch.compile` は当日は切る（§6.2 の指示どおり。可変長 + 可変選択肢数で再コンパイルが起きる）。
- seq=128 と 256 の差が小さい（20.96ms vs 39.71ms、ほぼ線形）ことは、
  question 側を 256 トークン以内に収める §5.2 の方針と矛盾しない。

---

## 5. 付随して確認したこと

### バックボーンの日本語 fill-mask（§4.1）

```
mask_token = '<mask>' | id = 5
日本の首都は<mask>です。       -> 東京都 0.348 / 東京 0.288 / 東京都千代田区 0.098 / 、 0.009 / 東京都中央区 0.009
この問い合わせは<mask>部門が担当します。 -> 管理 0.117 / 広報 0.076 / マーケティング 0.066 / 総務 0.043 / 法務 0.037
```

日本語で正常に動作。`mask_token_id` はハードコードせずトークナイザから読む（§5.2）。

### `config.json` の実値（§6.1 の設計根拠の裏取り）

一次情報 https://huggingface.co/sbintuitions/modernbert-ja-310m/raw/main/config.json より:

| キー | 実値 |
|---|---|
| `local_attention` | **128** |
| `global_attn_every_n_layers` | **3** |
| `hidden_size` | 768 |
| `num_hidden_layers` | 25 |
| `num_attention_heads` | 12 |
| `max_position_embeddings` | 8192 |
| `vocab_size` | 102400 |

§6.1 が v1 設計を破棄した根拠（「3層に1層だけが global、残りは 128 トークン窓」）は
**この実値で裏が取れている**。

### 名前空間の空き（§0）

`scripts/check_namespaces.py` の出力（2026-09-20T13:11:48+0900）:

| 対象 | HTTP | 解釈 |
|---|---|---|
| PyPI `sokudan` (JSON API) | 404 | 公開されていない＝空き |
| PyPI `sokudan` (simple index) | 404 | 同上 |
| GitHub `hiroki-abe-58/sokudan` | 404 | 公開リポジトリとしては存在しない |
| HF `hiroki-abe-58/sokudan-ja-310m` | 401 | **未認証リクエストは拒否。空きかどうかは判定できない** |
| HF 公開検索 `sokudan` | — | 該当モデルなし |

**HF の 401 は「空き」を意味しない。** 押さえるなら `HF_TOKEN` 付きで確認するか、
実際に作成して確かめる必要がある。ここでは推測しない。
