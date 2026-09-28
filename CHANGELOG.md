# Changelog

数値はすべて本機で実行したコードの出力です。未測定のものは「測定していない」と書きます。

## 未公開（ブランチ `feat/systemone-server`）

モデルの重みは変えていません。

- `sokudan serve`: `/v1/systemone` 互換のサーバー（`sokudan/serve/systemone.py`、`sokudan/serve/wire.py`）。TypeSafe の公開 API リファレンスの形で受けて返します。手順は `docs/serving.md`、各フィールドの扱いは `docs/systemone_wire_format.md`。
  - noul の答えの `type` は `"noul"`、`confidence` はワイヤ形式の定義（choice は `(p_max − 1/K)/(1 − 1/K)`）。どちらも `Agent.predict` の返り値（`"bool"`、最大確率）とは違います。`predict` は変えていません。
  - 推論時の順序平均（`--order-marginalize`）は未実装で、指定すると起動を拒否します。
- README: `README.md` を英語に、日本語は `README_ja.md` に（`docs/readme_rewrite_notes.md`）。
- `spaces/demo/`: v0.2 用の Gradio デモ（CPU）。state を文字列で渡します。
- `pyproject.toml`: 依存の上限、`huggingface-hub` と `safetensors` の明記、extras `demo`、sdist の対象の絞り込み（`docs/release_pypi.md`）。

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
