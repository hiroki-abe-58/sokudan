# README の書き換え（草稿）: 差分の要約

ブランチ `feat/systemone-server`。公開はしていません。

## ファイルの入れ替え

| ファイル | 前 | 後 |
|---|---|---|
| `README.md` | 日本語の README（341 行、正本） | **英語の README（新規に書いたもの、97 行）** |
| `README_ja.md` | 無し | 前の `README.md` そのもの + 下の 3 点の変更 |
| `README_en.md` | 前の `README.md` の英訳（「日本語版が正本」と明記） | `README.md` と `README_ja.md` への案内だけ（外部からのリンクを切らないため） |

`README_ja.md` で変えたのは次の 3 点だけです:
1. 先頭の英語版へのリンクを `README_en.md` から `README.md` に。
2. 「`/v1/systemone` 互換サーバー」の節を「TypeSafe Jev について」の前に追加。Jev の節に「互換サーバーは公開ドキュメントの形式に合わせたもので、Jev の出力は使っていない。TypeSafe とは提携していない」を追記。
3. 開発の節の `uv run pytest  # 538 tests` から件数を外した（件数が古く、今は 721 passed / 28 skipped）。

## 英語 README の構成

1 行の説明 → バッジ（license、HF、v0.2）→ Why（生成なし・日本語ネイティブ・質問ごとに 1 パス）→ 30 秒の Quickstart（`pip install git+…`、6 行の Python、期待される出力）→ `bench_ja` の表（v0.2 対 `laya-multilingual`）→ v0.2 の作り方（8 シードの soup、事前登録、再現）→ Limits（位置感度を含む）→ `/v1/systemone` 互換サーバー → 日本語 README などへのリンク。

- **Quickstart の出力は実行して確かめました。** 2026-09-27、HF の `main`（v0.2）、CPU、`sokudan-public/.venv`。出力は `請求`。
- **数値の出所**: すべて公開中の README（日本語・英語）と HF のモデルカード（`docs/model_card_v0.2.md`）にある値です。

| README の値 | 出所 |
|---|---|
| v0.2: choice 0.880、RPS 0.075、score acc 0.817、bool acc 0.780、AUROC 0.844 | README の v0.2 節、モデルカード |
| laya-multilingual: 0.747 / 0.232 / 0.543 / 0.523、多数決 0.380 / 0.197 / 0.703、ランダム 0.253 / 0.201 / 0.513 | README の表 |
| リリース規則（AUROC +0.01、RPS −0.005、acc −0.01 以内）、bool acc −0.008 | モデルカード |
| `8750a833…5b965` | モデルカードの SHA-256 表 |
| 位置: 300 件中 5 件、acc 0.347、第 1 スロット率 0.239 / 0.289 | README の Limits、モデルカード |
| mean P(true) 0.133、陽性率 0.297 | README、モデルカード |
| 較正後の RPS 0.132 | モデルカード（`val_v2` でフィットした温度） |
| local attention 128、state 平均 134・p95 237 | モデルカード |
| `bench_en` bool acc 0.690、多数決 0.683 | README、モデルカード |
| 21 ドメイン、4,833 文書、31,243 ペア | README |
| 314.6M | README |

- `laya-multilingual` の score acc（0.443）は `docs/baseline_ja.md` にしか無いので、表から score acc の列を外しました（v0.2 の 0.817 は本文に書いた）。

## 英語 README から落としたもの（`README_ja.md` にだけ残る）

前の `README_en.md` にあって、新しい英語 README に無いもの:

- 「なぜ作ったか」の事前実測の節と、`bench_ja` の 2026-09-21 の訂正（リーク検査の不備）。
- v0.1 の 3 シードの表と、シード別の注記。
- 設計の節（joint encoding の撤回の経緯と表、動的 K の cumulative link、encoding の単一実装）。
- 較正の使い方、seed の revision の読み込み方。
- Limits の v0.1 由来の項目の多く（4 段階の score の追試、温度較正、同一ドメインの学習データ、FlashAttention-2、head の RoPE など）。英語 README では 7 項目に絞り、全部は `README_ja.md` とモデルカードへ案内しています。

**英語だけを読む人には、設計の節と Limits の全文が英語で読めなくなります。** 残すなら、前の `README_en.md` の該当部分を `docs/design_en.md` のような別の文書に移すのが手です（今回はしていません）。

## 確認してほしい点

1. **どちらを正本にするか。** 前は「日本語版が正本、英語は翻訳」でした。新しい英語 README はそれを書いていません。英語 README は短い要約で、日本語版の方が詳しい、という関係です。
2. **互換サーバーの書き方。** 見出しは「`/v1/systemone`-compatible server」にし、「Jev 互換」という言い方は避けました。「公開ドキュメントと公開実装の README の例から作った」「TypeSafe とは提携していない」「このリポジトリは TypeSafe のサービスを呼ばない」の 3 点を書いています。利用規約（MCA 2.3(b)）との関係で、この節を公開してよいかは、公開の前に判断してください。
3. **Why の 3 点目**は「single pass」ではなく「One forward pass per question」と書きました。v0.2 は質問ごとに state をエンコードし直すので、「1 リクエストで 1 パス」とは書けません。
4. **`pip install git+…`** のままにしてあります。PyPI に出したら `pip install sokudan`（または別名）に替えます（`docs/release_pypi.md`）。
