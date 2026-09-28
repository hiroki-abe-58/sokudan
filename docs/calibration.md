# v0.2 の温度較正（held-out の 2 分割交差、事前登録）

T0 = 2026-09-27 23:16:01（night_19）。手順は `docs/research_protocol.md` に従います。**§1〜§3 は実行の前に書いて commit しています。以後動かしません。** 結果は §4 以降に追記します。

- 背景: val で fit した温度（`runs/release_candidate/temperatures.json`、`scripts/calibrate.py`）は、すでにあります。今回は fit と評価を分け、held-out で交差評価してから出荷するかを決めます。
- 読み込み時に任意で当てる仕組み（`sokudan.load(checkpoint, temperatures=...)`、既定は off）は、すでにあります。出荷するときは、同じ形式のファイルを置きます。

## 1. 方法（`scripts/calibration_heldout.py`）

- **対象**: v0.2（S8_old）の事例ごとの確率です。保存済みの `runs/ens/probs_k8_0_7.npz` を使い、推論はし直しません。
  - frozen の予測がモデルの出力と最大差 0.0 で一致することは、night_15 で確認済みです（`docs/order_marginalization.md` §4、night_18 §8）。
- **温度**: （質問型 × 選択肢数）ごとに 1 つです。NLL 最小化で fit し（`sokudan.calibration.temperature.fit_temperature`）、`softmax(log p / T)` で当てます（既存の定義と同じ）。
- **データ**:
  - **bool**: frozen held-out 9,850 行（保存は P(true) なので、[1 − P(true), P(true)] として扱う）。
  - **choice と score**: held-out には choice と score の行がありません。そのため `val_v2`（choice 1,976 行、score 2,320 行）を使います。**指示（held-out）からの逸脱で、理由はこのとおりです。**
- **fit と評価の分離**: 各集合を `doc_id` の SHA-256 の先頭バイトの偶奇で A と B に分けます。
  - A で fit して B に当て、B で fit して A に当てます（2 分割交差）。
  - 全行を、自分が入っていない半分で fit した温度で評価します。
  - fit する半分で 25 行未満のバケツは、T = 1 のままにします。
- **出荷用の温度**: 全行で fit します（25 行未満のバケツは T = 1）。

## 2. 指標（交差評価、T = 1 と同じ行で比べる）

| 指標 | 定義 |
|---|---|
| bool ECE | P(true) の ECE。[0, 1] を 10 等分し、Σ（n_b / N）× \|平均 P(true) − 正例率\| |
| 平均 P(true) − 正例率 | frozen held-out |
| bool acc（閾値 0.5） | P(true) > 0.5 を true とする |
| bool AUROC | 変わらないことの確認 |
| choice ECE | top-label の ECE（10 分割、`sokudan.calibration.metrics.ece`） |
| score RPS | 行の平均 |
| argmax | 変わらないことの確認（choice と score） |

- **事前に書いておくこと**: 温度（T > 0）は順位と argmax を変えません。2 択の P(true) > 0.5 も変えません。
  - そのため、bool acc（0.5）と AUROC は構成上変わりません。平均 P(true) と正例率のずれ（過少予測）も、閾値の位置が動かないので温度では直りません（`docs/model_card_v0.2.md` の Limits と同じ）。
  - 温度が変えるのは、確率の鋭さ（ECE、NLL、RPS）です。

## 3. 出荷の条件（機械的に適用）

- **交差評価で bool ECE が T = 1 より小さく**、かつ **bool acc（0.5）の差が −0.005 以上**。
- **満たせば**: 全行で fit した温度を `runs/release_candidate/calibration.json` に置きます（`load(temperatures=...)` の形式）。
  - bool/2 は必ず入れます。
  - choice と score の温度は、交差評価でその型の指標（choice ECE、score RPS）が悪化しなかった型だけ入れます。
  - 既定は off のままです（`load` の既定は None）。読み込みのテストを付けます。
- **満たさなければ**: 出荷しません。
- **bench**: 保存済みの v0.2 の bench の事例ごとの出力に、全行 fit の温度を当てます。
  - 確認した保存内容: `bench_ja` は、choice の事例ごとの確率（`choice_probs.npz`）だけがあります。bool と score の事例ごとの出力はなく、`bench_en` は集計値だけです。**生ロジットはどちらにもありません。**
  - 指示どおり確率から逆算しないので、**bench の ECE と bool acc は TBD** とします。

## 記録: 実行が commit より先に 1 回走った（23:27:26）

- §1〜§3 の文面は、実行の前に書き終えていました。しかし、書き出しと commit を 1 つのコマンドでつないでいたため、次のことが起きました。
  - スクリプトの ruff が失敗し、書き出しと commit が行われませんでした。
  - 改行で区切っていた次の行のスクリプトの実行は、そのまま走りました（23:27:26）。
  - つまり、**事前登録の commit より先に、結果を 1 度見ています。**
- 対応:
  - §1〜§3 は、実行の前に書いた文面のまま、1 文字も変えずに commit しました（この節だけを足した）。
  - そのとき出たファイル（`runs/calibration/results.json`、`runs/release_candidate/calibration.json`）は、消しました。
  - スクリプトは、ruff の指摘 2 件だけを直しました（使わない変数の名前、クロージャの引数の束縛）。動作は変わりません。
  - 本番の出力は、この commit の後に実行し直したものです（§4）。
- 判定の規則は結果を見る前に決まっていたので、判定は規則どおりに行います。ただし、順序が逆になったことは、ここに記録として残します。

---

# 結果（2026-09-27 23:28:33、`runs/calibration/results.json`）

## 4. 交差評価（T = 1 と同じ行で比較）

**bool**（frozen held-out 9,850 行。A 4,824 行 / B 5,026 行）:
- 半分ごとの温度: A で fit 2.083、B で fit 2.058。

| 指標 | T = 1 | 交差 fit | 差 |
|---|---|---|---|
| bool ECE（P(true)、10 分割） | 0.1218 | **0.0616** | −0.0602 |
| 平均 P(true) − 正例率 | −0.0657 | −0.0381 | +0.0276 |
| bool acc（0.5） | 0.8163 | 0.8163 | 0.0000（構成上） |
| AUROC（2 つの半分をまとめたもの） | 0.909833 | 0.909797 | −0.000035 |

- **AUROC の差について**: 2 つの半分に別の温度を当てたため、まとめた値だけがわずかに動きました。半分ごとの AUROC は変わっていません（A 0.907798 → 0.907798、B 0.911852 → 0.911852）。閾値 0.5 の判定も全行で変わりません。
- **§2 の書き方の訂正**: 事前登録では、平均 P(true) と正例率のずれは温度で直らないと書きました。
  - 閾値 0.5 の位置が動かないことは正しいです。
  - しかし T > 1 は確率を 0.5 の側へ寄せるので、平均 P(true) − 正例率は −0.066 → −0.038 と縮みました（実測）。
  - 閾値の位置の問題は、温度では直りません。

**choice**（`val_v2` 1,976 行）: top-label ECE（10 分割）は 0.1770 → **0.0516** です。argmax は全行で変わりません。

**score**（`val_v2` 2,320 行）: RPS は 0.1586 → **0.1498** です。argmax は全行で変わりません。

- 1 回目の実行（23:27:26、commit 前。記録節を参照）と、本番の実行の数字は同じでした。

## 5. 判定（§3 を機械的に適用）

| 条件 | 結果 | |
|---|---|---|
| 交差評価で bool ECE が T = 1 より小さい | 0.0616 < 0.1218 | 満たす |
| bool acc（0.5）の差 ≥ −0.005 | 0.0000 | 満たす |

**判定: 出荷の条件を満たします。**
- 全行で fit した温度を `runs/release_candidate/calibration.json` に置きました（ローカルのみ、公開していない）。
  - bool/2 = 2.070。
  - choice と score は、交差評価で悪化しなかったので入れました（choice/2〜9、score/2〜7。値は val で fit した既存の `temperatures.json` と、有効数字の範囲で同じです）。
- **読み込み**: `sokudan.load(checkpoint, temperatures="runs/release_candidate/calibration.json")` で当てられます。
  - 既定（`temperatures=None`）は off のままです。
  - ファイルの読み取りを `sokudan.predict.read_temperatures` に切り出しました（`load` の中身を移しただけで、動作は同じ）。
  - `tests/test_calibration_load.py`（3 件）で確認しています。形式の読み取り、出荷したファイルの読み取り、既定の off で出力が記録と一致すること、温度を当てると分布だけが変わり argmax と bool の 0.5 の側が変わらないこと、の 4 点です。
- **bench**: **TBD** です（事例ごとの bool と score の出力も、生ロジットも保存されていないため。§3）。

---

# night_20: bench の較正 ECE（§6 は実行の前に書いて commit）

## 6. 手順（チャット側の指示、2026-09-28 night_20）

- **bench の再推論は「記録の補完」として 1 回だけ**です（チャット側の許可）。v0.2（S8_old）で `bench_ja` と `bench_en` を推論し直し、事例ごとの生ロジット（`marker_logits`）と確率を保存します（`runs/calibration/bench_*.npz`）。
  - 推論の経路は、記録を作った経路そのものです（`SokudanBaseline.run`。`bench_en` は `scripts/run_bench_en_sokudan.py` と同じく 3 問を差し替える）。`run_model` を包んで、出力を取っておくだけにします（`scripts/bench_calibration.py`）。
- **記録との一致の確認**: `score_baseline` で採点し、保存済みの `results.json` の値と比べます。
  - 比べる値: choice acc、score RPS、score acc、bool acc、bool AUROC。許容は 1e-4 です。
  - **1 つでも一致しなければ、止めて報告します**（較正の計算はしない）。
- **較正**: `runs/release_candidate/calibration.json` の温度（choice/4、score/3、bool/2）を、確率に後処理で当てます（`apply_temperature`。`predict` と同じ当て方）。
- **指標**（較正の前後。確率の下限処理なしの生の確率で計算。記録の採点は下限 5e-5 を入れるので、「前」の値は記録と小数の末尾で違うことがある）:

| 指標 | 定義 |
|---|---|
| bool ECE | P(true)、10 分割（§2 と同じ） |
| choice ECE | top-label、10 分割 |
| score RPS | 行の平均 |
| 平均 P(true) − 正例率 | |
| argmax | 変わらないことの確認 |

  - 記録の `results.json` の ECE（別の定義）も、参考として並べます。
- **既定 on の条件**（機械的に適用）: `bench_ja` の bool ECE が改善し、かつ score RPS の悪化が +0.005 未満であること。
  - **満たせば**: 「v0.2.1 で既定 on（`temperatures=None` で無効化できる）」。
  - **満たさなければ**: 「opt-in」。
  - `bench_en` は記述だけで、判定には使いません。
- **事前に書いておくこと**:
  - `calibration.json` の score/3 の温度は 4.50 です。val で fit した既存の温度（4.50）とほぼ同じです。
  - その既存の温度を当てた記録（`bench_ja` の「sokudan-ja-310m + 温度較正」の行）では、score RPS が 0.0745 → 0.1322 に悪化していました。今回の判定も、score RPS で落ちる可能性があります（予測）。
- **記録**: 1 つ前の commit（`f688499`）はスクリプトだけで、この §6 は入っていませんでした（ruff の失敗で書き出しが止まったため）。実行はしていません。

## 7. 結果（2026-09-28 02:39:59 – 02:40:21、`runs/calibration/bench.json`）

- **記録との一致**: `bench_ja` と `bench_en` の両方で、5 つの値（choice acc、score RPS、score acc、bool acc、bool AUROC）の差がすべて 0.0 でした（許容 1e-4）。
  - 生ロジットと確率は `runs/calibration/bench_ja.npz` と `bench_en.npz` にあります（ローカル）。
- **使った温度**: choice/4 = 2.318、score/3 = 4.504、bool/2 = 2.070。

| 指標 | bench_ja 前 | bench_ja 後 | bench_en 前 | bench_en 後 |
|---|---|---|---|---|
| bool ECE（P(true)、10 分割） | 0.1812 | **0.1049** | 0.2490 | 0.1514 |
| 平均 P(true) − 正例率 | −0.1639 | −0.0983 | −0.1832 | −0.0959 |
| bool acc（0.5） | 0.7800 | 0.7800 | 0.6897 | 0.6897 |
| choice ECE（top-label、10 分割） | 0.0932 | 0.0653 | 0.0728 | **0.2281** |
| score RPS↓ | 0.0745 | **0.1322** | 0.1140 | 0.1616 |

- argmax は、両方の bench のすべての型で変わりませんでした。
- 記録の定義での bool ECE（較正前）は、bench_ja 0.1808、bench_en 0.2414 でした。

**判定（§6 を機械的に適用）: 「opt-in」**
- `bench_ja` の bool ECE は改善しました（0.1812 → 0.1049）。
- しかし score RPS が +0.0577 悪化しました（0.0745 → 0.1322。条件は +0.005 未満）。
- v0.2.1 では、`calibration.json` を同梱し、既定は off（`temperatures=` で指定したときだけ当てる）とします。

**判定には使わない記述**:
- score RPS の悪化は、score/3 の温度 4.50 によるもので、事前の予測どおりです（val で fit した温度での記録と同じ値 0.1322）。
- `bench_en` では、choice ECE も 0.073 → 0.228 と悪化しました。
- bool だけに温度を当てる形なら、bool ECE の改善だけが残る計算になります。ただし、この形は事前登録していないので測っていません（チャット側の判断事項）。

---

# night_21: bool だけの較正の再判定（§8 は実行の前に書いて commit）

## 8. 事前登録（チャット側の指示、2026-09-28 night_21）

- **出荷する温度は bool（type = bool、K = 2）だけです。** score と choice の温度は出荷しません。
- **推論はし直しません。** §7 で保存した bench の出力（`runs/calibration/bench_ja.npz`、`bench_en.npz` の生ロジットと確率）に、`calibration.json` の bool/2 の温度（2.070）だけを当てます（`scripts/bench_calibration_bool.py`）。
  - 当て方は、生ロジットの `softmax(logits / T)` です。保存済みの確率に `apply_temperature`（`predict` と同じ）を当てた値との最大差も記録します。
- **判定**（機械的に適用）: `bench_ja` と `bench_en` の**両方**で、次の 2 つを満たすこと。
  - bool ECE（P(true)、10 分割）が改善する。
  - |平均 P(true) − 正例率| が縮む。
  - **満たせば**: 「v0.2.1 で bool 較正を既定 on（無効化できる）」。
  - **満たさなければ**: 「opt-in」。
- **構成上変わらないことの確認**: bool acc（閾値 0.5）、bool AUROC、choice と score のすべての値（当てていない）。
- 判定の後、`runs/release_candidate/calibration.json` を bool だけの形に作り直します。元のファイルは `calibration_full.json` として残します。

## 9. 結果（2026-09-28 10:40、`runs/calibration/bench_bool.json`）

- 当てた温度は bool/2 = 2.070 です。生ロジットで当てた値と、保存済みの確率に `apply_temperature` を当てた値の最大差は、bench_ja で 2.4e-8、bench_en で 1.9e-8 でした。

| 指標 | bench_ja 前 | bench_ja 後 | bench_en 前 | bench_en 後 |
|---|---|---|---|---|
| bool ECE（P(true)、10 分割） | 0.1812 | **0.1049** | 0.2490 | **0.1514** |
| \|平均 P(true) − 正例率\| | 0.1639 | **0.0983** | 0.1832 | **0.0959** |
| bool acc（0.5） | 0.7800 | 0.7800 | 0.6897 | 0.6897 |
| bool AUROC | 0.8439 | 0.8439 | 0.6207 | 0.6207 |

- 閾値 0.5 の判定は、全件で変わりませんでした。choice と score には当てていないので、値は §7 の「前」のままです。

**判定（§8 を機械的に適用）: 「v0.2.1 で bool 較正を既定 on（無効化できる）」**。bench_ja と bench_en の両方で、bool ECE が改善し、|平均 P(true) − 正例率| が縮みました。

- `runs/release_candidate/calibration.json` を、bool/2 だけの形に作り直しました（SHA-256 `02b78975…`）。元のファイルは `calibration_full.json`（SHA-256 `effa6e2b…`）として残しています。どちらもローカルです。
- 読み込みのテスト（`tests/test_calibration_load.py`）は、作り直したファイルでも pass しました。
- **コードの既定は変えていません**（`load(temperatures=None)` は off のまま）。v0.2.1 で既定 on にするとき、同梱のしかた（例: Hub の `temperatures.json` として既定で読む）は、チャット側の判断です。

---

# v0.2.1: bool 較正の既定 on（2026-09-28 night_22）

## 10. 実装と応答のキー

**読み込み**（`sokudan.predict.load`、`resolve_temperatures`）:

| `temperatures` の値 | 当てるもの |
|---|---|
| 既定（`DEFAULT_CALIBRATION = "calibration.json"`） | チェックポイントの横（`.pt` のあるディレクトリ、safetensors のディレクトリ、または Hub から落とした場所）にある `calibration.json` の **bool の温度だけ**。ファイルがなければ較正なし |
| `None` | 較正なし（v0.2 と同じ生の確率） |
| ファイルの名前、パス、dict | そのファイルまたは dict の温度を、**型を問わずすべて**当てる（明示したときだけ） |

- 出荷する `calibration.json` は bool/2 だけです（§9）。
  - 既定では、仮にファイルに score や choice の温度が入っていても当てません（bool だけに絞る）。
  - 同梱は、Hub のリポジトリの `calibration.json` として置く想定です（Hub からの読み込みは `*.json` を落とすので、そのまま読まれる）。
- **測定用のスクリプトは `temperatures=None` を明示しました**（`typed_decisions_ja_eval.py`、`latency_v02.py`、`multidomain_eval.py`、`export_v02.py`）。v0.2 の生の確率で測った記録と、同じ定義を保つためです。
- **サーバー**（このリポジトリの `sokudan/serve/app.py`）:
  - `SOKUDAN_TEMPERATURES` が未設定なら、既定と同じ動きをします。
  - `none` または `off` なら較正なし、パスならそのファイルです。

**応答のキー**（`Agent.predict` の戻り値の最上位。sokudan-web の `feat/systemone-server` から拾う名前）:

| キー | 型 | 意味 |
|---|---|---|
| `calibrated` | bool | この応答の答えのうち、1 つでも温度を当てたものがあれば true。bool 較正だけの既定では、bool の質問を含む応答で true、choice と score だけの応答では false |
| `calibrated_answers` | list[str] | 温度を当てた質問 ID（`questions` のキー）の一覧。既定では bool の質問だけ |

- このリポジトリの `/v1/systemone` は、`Agent.predict` の `calibrated` と `calibrated_answers` をそのまま返します。
  - `calibrated` が false のときは、これまでどおり `calibration_note` を付けます。
  - `/ready` の `calibrated` は、読み込んだモデルに温度が 1 つでもあるかを表します。

**テスト**（`tests/test_calibration_load.py`）:
- 既定は、横の `calibration.json` の bool だけを取ります。
- ファイルがなければ較正なしで、`None` は off です。
- 明示したファイルや dict は、すべて当てます。
- 出荷したファイルの既定は、bool/2 だけです。
- **保存済みの bench の出力**（`bench_ja`、`bench_en`）で、既定 on の温度を当てても、bool の閾値 0.5 の判定と AUROC は off と一致し、ECE は下がります。
- 応答の `calibrated` / `calibrated_answers` は、bool の質問を含むときだけ true / `["churn"]` です。
- フラグを入れる前の記録（`predict_golden.json`）との一致は、新しい 2 つのキーを除いて比べ、温度なしの `Agent` では `calibrated: false` であることを確かめています。
