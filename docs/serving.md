# sokudan を `/v1/systemone` 互換サーバーとして動かす

`sokudan serve` は、TypeSafe の `/v1/systemone` と同じワイヤ形式で答える HTTP サーバーです。
`/v1/systemone` を話すクライアントなら、base URL を差し替えるだけで sokudan に向けられます。
形式の詳細と、各フィールドを sokudan がどう扱うかは [`systemone_wire_format.md`](systemone_wire_format.md) にあります。

## 起動

```bash
pip install "sokudan[serve] @ git+https://github.com/hiroki-abe-58/sokudan.git"
sokudan serve                                   # GeneLab/sokudan-ja-310m（HF の main = v0.2）を 127.0.0.1:8000 で
```

| オプション | 既定 | 意味 |
|---|---|---|
| `--model` | `GeneLab/sokudan-ja-310m` | Hub の repo id（`@revision` 可。v0.1 は `GeneLab/sokudan-ja-310m@v0.1`）、`model.safetensors` のあるディレクトリ、または `model.pt` |
| `--host` | `127.0.0.1` | 他の機械から受けるなら `0.0.0.0` |
| `--port` | `8000` | |
| `--device` | CUDA があれば `cuda` | `cpu` で CPU 推論 |
| `--temperatures` | なし（未較正） | `temperatures.json`。**同梱の温度は score の RPS を悪化させます**（README の Limits） |
| `--max-concurrency` / `--max-queue` | 1 / 32 | 同時に走らせる推論の数と、待たせる数。超えると 429 |
| `--order-marginalize` | off | 推論時の順序平均。**未実装で、指定すると起動を拒否します** |

モデルは起動時に 1 回だけ読み込みます。初回は Hub から重みを取得します。

## 確認

```bash
curl -s localhost:8000/health
```

`/health` は、読み込んだモデル、較正の有無、`order_marginalize`、state の描画のしかた、confidence の定義、上限を返します。モデルが無いときは 503 です。

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "先月の請求で同じ金額が二回引き落とされています。至急ご確認ください。",
  "questions": {
    "department": {"type": "choice", "instructions": "この問い合わせはどの部署が担当すべきか",
                   "criteria": {"請求": "支払い・返金", "技術": "不具合・障害",
                                "営業": "料金・新規契約", "その他": null}},
    "urgency": {"type": "score", "instructions": "この依頼の緊急度は",
                "criteria": ["急がない", "早めに", "業務が止まっている"]},
    "churn": {"type": "noul", "instructions": "解約を示唆しているか"}
  }
}'
```

返る形（`…` は数値）:

```jsonc
{
  "model": "sokudan-ja-310m",
  "answers": {
    "department": {"type": "choice", "choice": "請求",       // 最も確率の高い選択肢
                   "probabilities": {"請求": …, "技術": …, "営業": …, "その他": …},
                   "confidence": …},
    "urgency": {"type": "score", "score": …,                 // 水準番号の期待値（0〜2）
                "legend": {"0": "急がない", "1": "早めに", "2": "業務が止まっている"},
                "probabilities": {"0": …, "1": …, "2": …},
                "confidence": …},
    "churn": {"type": "noul", "noul": …}                     // P(はい)
  },
  "usage": {"input_tokens": …, "output_tokens": 0},
  "sokudan": {"state_format": "text", "state_tokens": …, "state_truncated": false,
              "backbone_passes": 3, "calibrated": false, "order_marginalize": false,
              "latency_ms": …}
}
```

- `model` は要求に書いても無視され、答えたモデルの名前が返ります。
- `usage.output_tokens` は常に 0 です。テキストを生成しないためです。
- `sokudan` の下は sokudan 独自の情報です。`backbone_passes` が質問数と同じなのは、v0.2 が質問ごとに state をエンコードし直す（joint encoding）ためで、**レイテンシは質問数に比例します**。
- **state は文字列で送ってください。** object を送ると `key: value` の行に描画され、学習時の入力と違うので出力が変わります（README の注意）。どちらで読んだかは `sokudan.state_format` に出ます。

## base URL を差し替える

`/v1/systemone` を話すクライアントの base URL を `http://127.0.0.1:8000` にします。
API キーを必須にするクライアントには、任意の文字列を渡してください。**このサーバーは `Authorization` を検査しません**（ローカル用）。外に出すときは、認証をする前段を置いてください。

素の HTTP（Python、`httpx`）:

```python
import httpx

BASE_URL = "http://127.0.0.1:8000"          # 以前は別のサービスの URL だったところ
response = httpx.post(f"{BASE_URL}/v1/systemone", json={
    "state": "ログインが数日前から断続的に切れます。業務が止まっています。",
    "questions": {"urgent": {"type": "noul", "instructions": "至急の対応が必要か"}},
}, headers={"Authorization": "Bearer local"}, timeout=60)
print(response.json()["answers"]["urgent"]["noul"])
```

- Lev と kev の README は、TypeSafe の Python SDK の `TypeSafeClient(base_url=..., api_key="local")` で自分のサーバーに向ける例を載せています。**sokudan ではその SDK を使った確認はしていません**。確認したのは素の HTTP と、各 README の公開例のリクエストです（`tests/test_systemone_server.py`）。
- 既存のクライアントから移すときに違うところ:
  - **英語の業務文は想定外です。** 学習は日本語のみで、`bench_en` の bool acc は多数決とほぼ同じです（README）。
  - `score` は 2〜10 水準、`choice` は 1〜255 選択肢。水準に `null` は使えません（422）。
  - `confidence` は choice が `(p_max − 1/K)/(1 − 1/K)`、score が `max(0, 1 − E|level − mode|/D)` です（[`systemone_wire_format.md`](systemone_wire_format.md) §3）。**閾値は sokudan の出力で決め直してください。** 確率は既定で未較正です。
  - `noul` の `criteria`（`true` / `false` の説明）は受け付けますが、精度への影響は測っていません。

## エラー

| 状態 | いつ |
|---|---|
| 422 | 質問が読めない（型が不明、criteria の形が違う、水準の数、空の instructions、`state: null` など）。本文の `detail` に理由 |
| 413 | state が 100,000 文字を超える |
| 429 | 推論待ちがいっぱい。`Retry-After: 1` |
| 503 | モデルが読み込まれていない |

## Windows での注意

- **PowerShell の `curl` は `Invoke-WebRequest` の別名です。** 上の例は `curl.exe` と書くか、Git Bash で実行してください。
- **日本語の本文はファイルから送るのが確実です。** コマンドラインに直接書くと、コードページによっては文字化けします。UTF-8 で保存した JSON を `curl.exe ... --data-binary "@req.json"` で送ってください。PowerShell の `Invoke-RestMethod` なら `-ContentType 'application/json; charset=utf-8'` を付け、本文は `[System.Text.Encoding]::UTF8.GetBytes($json)` で渡します。
- **Hub のキャッシュで、symlink の警告が出ます。** 開発者モードが無効だと symlink を作れず、ファイルを複製して保存します。動作には影響しません。`HF_HUB_DISABLE_SYMLINKS_WARNING=1` で警告を消せます。キャッシュの場所は `HF_HOME` で変えられます。
- **`--host 0.0.0.0` にすると、初回に Windows Defender ファイアウォールの許可を求められます。** ローカルだけで使うなら既定の `127.0.0.1` のままにしてください。
- **アプリケーション制御（Smart App Control など）が torch の DLL を止めることがあります。** 作者の環境では、新しく入れた torch のビルドで `shm.dll` の読み込みが「アプリケーション制御ポリシーによってブロック」されました。同じ機械で読み込めている版の torch に揃えると動きました。
- 止めるときは Ctrl+C です。`--reload` は用意していません（モデルを読み直すため）。

## 以前のサーバー（`sokudan.serve.app`）との関係

`sokudan/serve/app.py` は、環境変数（`SOKUDAN_CHECKPOINT` など）で設定する以前のサーバーで、sokudan 独自の形（noul の答えの `type` が `"bool"`、confidence が最大確率、sokudan 独自の usage）で答えます。
`sokudan serve` が動かすのは、この文書の互換サーバー（`sokudan/serve/systemone.py`）です。どちらも同じ `Agent.predict` を呼びます。
