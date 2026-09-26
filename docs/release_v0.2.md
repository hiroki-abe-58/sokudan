# v0.2 の公開記録（2026-09-27）

v0.2 = S8_old（v0.1 と同じ設定で学習した seed 0〜7 の 8 本の単純平均、model soup）。選定と bench は `docs/release_candidate.md`、数値は `docs/benchmarks.md` §10 とモデルカードにあります。

## 1. 公開したもの

### GitHub（https://github.com/hiroki-abe-58/sokudan）

| 項目 | 値 |
|---|---|
| `main` | `fcc67be21bdc7d73fa7e5665ef9eeab300b34689`（`788306c` からの fast-forward、2026-09-27 03:48 push） |
| タグ `v0.2` | 注釈付きタグ `4e4f5a7`。指す commit は `fcc67be` |
| タグのページ | https://github.com/hiroki-abe-58/sokudan/releases/tag/v0.2 |

**内容**:
- コード、テスト、スクリプト、docs。
- README / README_en の v0.2 節（soup の作り方、`bench_ja` の v0.1 対 v0.2 の表、位置感度を含む Limits、state を文字列で渡す例と dict の注意）。
- `CHANGELOG.md`、パッケージ 0.2.0。

**公開 `main` の作り方**:
- 開発用のブランチ（非公開）と公開 `main` には共通の祖先がありません。開発用の履歴には生成データ（学習コーパスのバックアップなど）も含まれています。
- そのため、開発用のブランチを公開 `main` にマージすることはせず、公開 `main`（`788306c`）の上に、公開するファイルを 1 つずつ取り出して 1 commit にしました。
- 公開 `main` にだけあった変更は、そのまま残しています（`bench_ja` / `bench_en` のライセンス表記の統一、`SOKUDAN_SPEC.md` の履歴文書の注記、v0.1 のモデルカード `docs/model_card.md`、`docs/licenses.md`）。

**含めていないもの**:
- チェックポイント（`runs/`）。
- `data/` の生成物・キャッシュ・検証出力（公開済みの `bench_ja` / `bench_en` とそのマニフェストだけが公開のまま）。
- 教師ターゲット。
- 非公開と明記した内部メモ（Day 1〜3 の記録、初期の夜間レポート 4 本、v0.2 / v0.3 の試行記録、公開手順メモ、ライセンスの読み取りメモ、ラベル上限の記録）。
- README の草稿。

**そのほか**:
- 公開した docs と scripts から、ローカルの絶対パスを除きました。
- 生成データがない checkout でも、テストが失敗ではなくスキップになるようにしました（`tests/conftest.py`）。

### Hugging Face（https://huggingface.co/GeneLab/sokudan-ja-310m）

| ref | commit | `model.safetensors` の SHA-256 | 内容 |
|---|---|---|---|
| `main` | `6cfe9396c8d75492823aeb55cb81237d873ba955` | `8750a833b3faa537a39709b8377b82a096286f040d78bb1915b82d9d101b5965` | v0.2 の重み、config、tokenizer、`temperatures.json`、モデルカード |
| タグ `v0.2` | → `6cfe939` | 同上 | |
| ブランチ `v0.1` | `8c617aeb5d278631bc0c579df73b936592a142a9` | `5f940cb3614be21a11ce445a95b07585113468bea7cb1a4ec03eb42cc4d53f8c` | 以前の `main`（`10d80cd`）の上に、README の 1 段落（state を文字列で渡す注意）だけを追記 |
| ブランチ `seed1` / `seed2` | `705d18f` / `1b2f6c8` | （触っていない） | v0.1 の seed 1 / 2 |

- 時刻: `v0.1` の作成 03:48:38、`main` への upload 03:51:46、タグと `v0.1` の README 03:52:02。

## 2. 確認

### 2.1 未認証の HTTP（2026-09-27 03:52〜03:53）

| 対象 | 結果 |
|---|---|
| GitHub `README.md` / `README_en.md` / `docs/research_protocol.md` / `CHANGELOG.md`（raw） | すべて 200。README の Status は v0.2、protocol に §10 あり |
| GitHub で公開していないはずのファイル（内部メモ、生成データのバックアップなど） | 404 |
| GitHub のタグのページ `v0.2` | 200 |
| HF のモデルカード（`main`） | 200、見出しは「sokudan-ja-310m v0.2」 |
| HF `resolve/main/model.safetensors` | X-Linked-Size 1,258,474,752、X-Linked-ETag `8750a833…` |
| HF `resolve/v0.2/model.safetensors` | 同じサイズ、`8750a833…` |
| HF `resolve/v0.1/model.safetensors` | 同じサイズ、`5f940cb3…`（以前の `main` と一致） |
| HF の `v0.1` の README | 200、追記の段落が 1 つ |

### 2.2 公開後のインストールと読み込み（2026-09-27 03:53〜03:55、新しい一時 venv、確認後に削除）

- **手順**:
  1. GitHub の `main`（`fcc67be`）から `pip install git+…` しました。
  2. HF から、トークンなし・空のキャッシュで重みを取得しました。
  3. held-out の 5 事例で、`predict`（state は文字列）の P(true) を、公開前に保存した値と比べました。

| 読み込み | デバイス | 保存済みの予測との一致 |
|---|---|---|
| `GeneLab/sokudan-ja-310m`（main = v0.2） | CPU | 5 事例とも完全一致 |
| 同上 | CUDA | 5 事例とも完全一致 |
| `GeneLab/sokudan-ja-310m@v0.1` | CUDA | 5 事例とも完全一致 |
| 同上 | CPU | 4 事例が一致。1 事例は 0.7931 と 0.7930（`predict` は小数第 4 位で丸める） |

- 保存済みの予測は、実際には CUDA で計算されていました。
  - 環境変数で GPU を隠したつもりでしたが、隠れていませんでした。その記録の「CPU」という注記は誤りで、この確認で分かりました。
  - v0.1 の CPU での 1e-4 の差は、デバイスの違いによるものです。

## 3. 公開の途中で分かったこと

- **`safetensors` の書き出しのバイト列が、実行ごとに変わることがあります。**
  - ヘッダの `__metadata__`（2 つのキー）の順序が、実行ごとに入れ替わります（safetensors 0.8.0）。テンソルは同一です。
  - 公開前の再書き出しで、SHA-256 が確認済みの `8750a833…` と違う値（`1ee7372a…`）になりました。
  - 同じテンソル・同じメタデータで書き出し直し、`8750a833…` と一致したバイト列を upload しました（テンソルが `model.pt` と一致することも再確認）。
- **`predict` に dict の state を渡すと、出力が変わります。** v0.2 のカードと README の使用例は文字列に直しました。v0.1 の README には、注意の段落を追記しました。
