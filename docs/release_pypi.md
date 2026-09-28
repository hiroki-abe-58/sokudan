# PyPI に出す手順（下準備。まだ公開していない）

## 1. 名前

2026-09-27 に確認しました:

| 名前 | `https://pypi.org/pypi/<名前>/json` | `https://pypi.org/simple/<名前>/` |
|---|---|---|
| `sokudan` | 404 | 404 |
| `sokudan-ja` | 404 | 404 |

- **`sokudan` は登録されていません。** 第 1 候補は `sokudan`、取れなかったときの候補は `sokudan-ja` です。
- PyPI に予約の仕組みはありません。名前は最初のアップロードで決まります。404 でも、PyPI 側の禁止リストや、既存の名前と紛らわしいという判定で拒まれることがあります。そのときは `sokudan-ja` にし、`pyproject.toml` の `name` を替えます（import 名の `sokudan` は替えなくてよい）。

## 2. 公開の前に決めること

1. **版番号。** `pyproject.toml` は `0.2.0` のままです。ただし GitHub のタグ `v0.2`（パッケージ 0.2.0）には `sokudan serve` がありません。同じ `0.2.0` で中身の違うものを出さないよう、**`0.2.1`（または `0.3.0`）に上げてから**出すことを勧めます。上げるときは `pyproject.toml` と `sokudan/__init__.py` の `__version__` の両方を替え、`CHANGELOG.md` に節を足します。
2. **README。** PyPI のページには `README.md`（英語）が出ます。README の相対リンク（`docs/serving.md` など）は PyPI 上では切れます。英語 README の草稿を確定するときに、絶対 URL（`https://github.com/hiroki-abe-58/sokudan/blob/main/...`）に替えるかを決めてください。
3. **Python の版。** `requires-python = ">=3.11,<3.12"` です。3.12 以降では試していないので、そのままにしてあります。PyPI では 3.12 の利用者が入れられません。
4. **torch。** PyPI の torch は、Linux では CUDA 版、Windows と macOS では CPU 版が入ります。CPU だけで使う Linux の利用者には `pip install torch --index-url https://download.pytorch.org/whl/cpu` を先に案内します。

## 3. `pyproject.toml` で整えたこと

- ライセンスを SPDX の式（`license = "Apache-2.0"`）と `license-files = ["LICENSE", "NOTICE"]` にしました（hatchling 1.27 以上）。
- 依存に上限を付けました（今の `uv.lock` の版の次のメジャー版の手前まで）。
- `sokudan.predict` が直接 import している `huggingface-hub` と `safetensors` を、依存に明記しました。これまでは transformers 経由で入っていただけでした。
- extras: `serve`（fastapi、uvicorn）、`demo`（gradio）。`dev` と `bench` はそのまま。
- sdist に入れるものを、パッケージ、テスト、README（英・日）、CHANGELOG、LICENSE、NOTICE、`pyproject.toml` に絞りました（研究用の `scripts/`、`data/`、`docs/` の画像、`uv.lock` は入れない）。
- `uv.lock` を更新しました（gradio とその依存の追加だけで、既存の版は変わっていません）。

**確認したこと**（2026-09-27、`sokudan-public/.venv`、公開はしていない）:
- `python -m build` で `sokudan-0.2.0-py3-none-any.whl` と `sokudan-0.2.0.tar.gz` ができる。`twine check` は両方 PASSED。
- 新しい一時 venv に wheel を `[serve]` 付きで入れると、宣言した範囲で依存が解決し、wheel から入ったパッケージで `tests/test_systemone_server.py`、`tests/test_serve.py`、`tests/test_schema.py` の 115 件が通る。`python -m sokudan.cli serve --help` が動く。
  - torch は、この機械で読み込める `2.11.0+cu128` を先に入れてから試しました。PyPI の既定の torch では試していません。

## 4. 公開の手順（トークンは Hiroki さんが持つ）

以下は PowerShell の例です。トークンはコマンドラインに直接書かず、そのセッションの環境変数にだけ置きます。

```powershell
cd C:\Users\hirok\sokudan-public
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue
.venv\Scripts\python.exe -m build
.venv\Scripts\python.exe -m twine check dist\*
```

### 4.1 まず TestPyPI

```powershell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = Read-Host -AsSecureString "TestPyPI token" | ConvertFrom-SecureString -AsPlainText
.venv\Scripts\python.exe -m twine upload --repository testpypi dist\*
Remove-Item Env:TWINE_PASSWORD
```

新しい venv で入れて確かめます（依存は本物の PyPI から取る）:

```powershell
uv venv --python 3.11 $env:TEMP\sokudan-testpypi
uv pip install --python $env:TEMP\sokudan-testpypi\Scripts\python.exe `
  --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ `
  --index-strategy unsafe-best-match "sokudan[serve]"
& $env:TEMP\sokudan-testpypi\Scripts\python.exe -c "import sokudan; print(sokudan.__version__)"
& $env:TEMP\sokudan-testpypi\Scripts\python.exe -m sokudan.cli serve --help
```

README の Quickstart（`sokudan.load("GeneLab/sokudan-ja-310m")` の 6 行）を実行して、`請求` が出ることも確認します。

### 4.2 PyPI

```powershell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = Read-Host -AsSecureString "PyPI token" | ConvertFrom-SecureString -AsPlainText
.venv\Scripts\python.exe -m twine upload dist\*
Remove-Item Env:TWINE_PASSWORD
```

- トークンは、最初の 1 回は「アカウント全体」の範囲でしか作れません。公開後に、プロジェクト `sokudan` に限ったトークンに作り直してください。
- 一度上げた版番号は、消しても同じ番号では二度と上げられません。

### 4.3 公開のあと

1. 未認証で `https://pypi.org/project/sokudan/` が開けること、README が表示されることを確かめる。
2. 新しい venv で `pip install sokudan` と Quickstart を実行する。
3. README（英・日）とモデルカードの `pip install git+…` を `pip install sokudan` に替える。`spaces/demo/requirements.txt` もパッケージ名での指定に替えられる。
4. GitHub にタグを打ち、`docs/release_v0.2.md` と同じ形で公開記録を残す。

## 5. 今後の選択肢

- GitHub Actions の Trusted Publishing（OIDC）にすると、トークンを手元に置かずに済みます。タグの push で公開する流れにできます。
