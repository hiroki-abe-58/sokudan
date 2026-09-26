---
language:
- ja
license: apache-2.0
library_name: transformers
pipeline_tag: text-classification
tags:
- japanese
- system-one
- calibrated-decisions
- classification
- routing
- ordinal
- model-soup
base_model: sbintuitions/modernbert-ja-310m
---

# sokudan-ja-310m v0.2

**日本語 System One 意思決定モデル。**
日本語テキスト（state）と型付き質問（`choice` / `score` / `bool`）を受け取り、テキストを生成せずに、1 回のフォワードパスで型付きの回答と確率を返します。

> **v0.2（2026-09-27）は、v0.1 と同じ設定で学習した 8 本（seed 0〜7）の重みを単純平均した model soup です。**
> アーキテクチャ、学習データ、推論コードは v0.1 と同じで、推論のコストも 1 本分のままです。
> `bench_ja` は、事前に commit したリリース規則のもとで 1 回だけ測り、規則を満たしました。
> **この文書の数値は、すべて本機で実行したコードの出力です。** 未測定のものは「測定していない」と書きます。

**v0.1 を使い続ける場合**は、revision `v0.1` を指定してください（v0.1 のときの `main` と同じ重みです）:

```python
import sokudan
agent = sokudan.load("GeneLab/sokudan-ja-310m@v0.1")          # v0.1
# または huggingface_hub で:
# snapshot_download("GeneLab/sokudan-ja-310m", revision="v0.1")
```

- バックボーン: [`sbintuitions/modernbert-ja-310m`](https://huggingface.co/sbintuitions/modernbert-ja-310m)（MIT）
- 総パラメータ: 314,614,274（backbone 314,611,968 + scorer 768 + ordinal head 1,538。v0.1 と同じ）
- ライセンス: Apache-2.0
- コード: https://github.com/hiroki-abe-58/sokudan

## 使い方

```bash
pip install git+https://github.com/hiroki-abe-58/sokudan.git
```

```python
import sokudan

agent = sokudan.load("GeneLab/sokudan-ja-310m")   # v0.2
result = agent.predict(
    "先月の請求で同じ金額が二回引き落とされています。至急ご確認ください。",   # state は文字列で渡す
    {
        "department": {"type": "choice",
                       "instructions": "この問い合わせはどの部署が担当すべきか",
                       "criteria": {"請求": "支払い・返金", "技術": "不具合・障害",
                                    "営業": "料金・新規契約", "その他": "上記以外"}},
        "urgency":    {"type": "score",
                       "instructions": "この依頼の緊急度は",
                       "criteria": ["急がない", "早めに", "業務が止まっている"]},
        "churn":      {"type": "noul", "instructions": "解約を示唆しているか"},
    },
)
```

> **state は文字列で渡してください。**
> - dict（例: `{"body": ...}`）で渡すと、`key: value` の行に整形されます（`body: 先月の請求で…`）。
> - これは学習と評価で使った入力（本文そのまま）と違う入力で、出力が変わります。
> - held-out の 1 事例では、P(true) が 0.318（文字列）から 0.145（`{"body": ...}`）に動きました（1 事例の観測で、系統的には測っていません）。

## 評価結果

### `bench_ja` 300 件（較正前）

v0.1 は 3 シード（seed 0〜2）の平均 ± SD です。**v0.2 は soup 1 体を 1 回だけ実行した値で、ばらつきは付いていません。**

| | choice acc | choice ECE↓ | score RPS↓ | score acc | score MAE↓ | bool acc | bool ECE↓ | bool AUROC | bool mean P(true) |
|---|---|---|---|---|---|---|---|---|---|
| **v0.2** | **0.880** | 0.099 | **0.075** | **0.817** | 0.210 | 0.780 | 0.181 | **0.844** | 0.133 |
| v0.1 | 0.847 ± 0.009 | 0.147 ± 0.003 | 0.090 ± 0.023 | 0.763 ± 0.088 | 0.258 ± 0.086 | 0.788 ± 0.010 | 0.202 ± 0.013 | 0.789 ± 0.043 | 0.125 ± 0.012 |
| 多数決クラス | 0.380 | 0.000 | 0.197 | 0.460 | 0.540 | 0.703 | 0.000 | 0.500 | 0.297 |
| ランダム | 0.253 | 0.003 | 0.201 | 0.403 | 0.777 | 0.513 | 0.013 | 0.489 | 0.500 |

- **v0.2 の bool acc は、v0.1 の平均より 0.0078 低い値です**（ほかの 4 指標は上回った）。
- **リリース規則**（`bench_ja` の前に commit）: bool AUROC +0.01 以上、score RPS −0.005 以上の改善、choice / bool / score acc がいずれも −0.01 以内。
  - 結果は AUROC +0.055、RPS −0.016、choice +0.033、bool acc −0.008、score acc +0.053 で、すべて満たしました。
- v0.1 の配布重み（seed 0）の単体の値は、choice 0.843、RPS 0.117、score acc 0.663、bool acc 0.793、AUROC 0.837 でした。
- 温度較正後（`temperatures.json`、`val_v2` でフィット）: choice ECE 0.075、score RPS 0.132、bool ECE 0.119。
  - v0.1 と同じく、較正は score の RPS を悪化させます。**既定は未較正です。**

### `bench_en` 290 件（記述。v0.2 は日本語のみで学習。soup 1 体の 1 回の値）

| | choice acc | score RPS↓ | score acc | bool acc | bool AUROC |
|---|---|---|---|---|---|
| v0.2 | 0.872 | 0.114 | 0.652 | 0.690 | 0.621 |
| v0.1 | 測定していない | 測定していない | 測定していない | 測定していない | 測定していない |
| 多数決クラス | 0.331 | 0.257 | 0.486 | 0.683 | 0.500 |

- bool acc 0.690 は、多数決（0.683）とほぼ同じです。英語で `bool` を使わないでください。

### held-out（未知スキーマ、合成）

v0.1 と同じ設定の 16 本（seed 0〜15。2026-09-21〜27 に学習）の平均 ± SD と比べています。

| 指標 | 16 本の平均 ± SD（最小〜最大） | **v0.2** |
|---|---|---|
| M1m（held-out 属性の AUROC の平均） | 0.8239 ± 0.0142（0.8000〜0.8493） | **0.8644** |
| M1（held-out bool AUROC、まとめて） | 0.8763 ± 0.0148 | **0.9098** |
| M2（state 400〜799 トークン帯の AUROC） | 0.8304 ± 0.0341（0.7628〜0.8885） | **0.8590** |
| M5（`implies_declining` の AUROC） | 0.7657 ± 0.0356 | **0.8346** |
| val choice acc | 0.6975 ± 0.0160 | 0.7176 |
| val score RPS↓ | 0.1775 ± 0.0060 | 0.1586 |
| val score acc | 0.5228 ± 0.0114 | 0.5491 |
| val bool acc | 0.9320 ± 0.0026 | 0.9331 |
| val bool ECE↓（10 分割） | 0.0625 ± 0.0030 | 0.0502 |
| held-out bool ECE↓ | 0.1932 ± 0.0236 | 0.1184 |

| 属性 | 16 本 | v0.2 |
|---|---|---|
| `ends_with_question` | 0.7025 ± 0.0559 | 0.7824 |
| `implies_declining` | 0.7657 ± 0.0356 | 0.8346 |
| `implies_escalation` | 0.7975 ± 0.0228 | 0.8355 |
| `implies_running_out_of_patience` | 0.8629 ± 0.0192 | 0.8741 |
| `requests_owner_change` | 0.9909 ± 0.0045 | 0.9955 |

- **v0.2 の held-out の値は、候補の選定にも使った値です。** 候補は 4 つの soup で、規則は held-out の M1m が最大であることと、ガードレールが非劣化であること。
  - 選定と評価が同じ集合なので、上振れを含みえます。独立な確認は `bench_ja` の 1 回だけです。

## soup の作り方と再現

- **構成**: v0.1 と同じコマンドで学習した seed 0〜7 の 8 本です。
  - 浮動小数のテンソルを、すべて float32 で平均して元の dtype に戻します。整数のテンソルと設定は seed 0 のものです。
  - ヘッドも含めて、全パラメータを平均しています。
- **再現**: `scripts/make_soup.py` が、8 本のチェックポイント（`runs/v01_seed{0..7}/model.pt`）から同じ規則で平均します。
  - メンバーと soup の SHA-256 を照合します。
  - 作者の環境では、soup の `model.pt` の SHA-256 まで一致しました。
- **メンバーのチェックポイントは配布していません。** また、同じコマンドで学習し直しても、同じ重みにはなりません。理由は 2 つあります。
  1. スコアラの初期値がシードで決まらない。
  2. GPU の非決定性。同じコード・同じシードの 2 回の学習で、M5 が 0.720 と 0.826 まで動いた例があります。
  - 下の SHA-256 は、作者が配布物を検証するための記録です。

| メンバー（`model.pt`） | SHA-256 |
|---|---|
| seed 0 | `bfd6506656ae34e4e98143ffdda8304c412acadfb5f4c5a8b68a3c72ae7f053e` |
| seed 1 | `82758cf6d14518f274f7b2030cb0c5f78110227504d0991f8990488b120b1dde` |
| seed 2 | `34b9c3a33b1689433e98605c6c1fd6659a687df3a98b3480ded682929db48e0e` |
| seed 3 | `da84566b06b70cf678276a8e9ef69238234e7a13f66ae15c89a8903283b2b957` |
| seed 4 | `dee1fbadbdacb494a9ef098584607469253c07e573ce7bcbc59e648a5d6d1f6a` |
| seed 5 | `b0b5e1987e0626fb31c74637a70d90a229833be9ef8db4a914b8e492676bee01` |
| seed 6 | `ff0ae8e48c866b799af2b558e8205503f5383d7b36dfcaac4a353c3775934312` |
| seed 7 | `d50e2044602d8b070d150b2b7f8d131d2dd9213e3045d35ee25b915c752f9bf2` |
| **soup（`model.pt`）** | `c301449145c97f9e317fb6eec716df17275fceef32e8a4baac2c9071d93504fe` |
| **この repo の `model.safetensors`** | `8750a833b3faa537a39709b8377b82a096286f040d78bb1915b82d9d101b5965` |

## Limits（正直に）

### 位置と提示のしかたへの感度

- **4 段階の `score` で、第 1 選択肢がほとんど選ばれません。**
  - 対象は `bench_ja` の位置検査の条件 E（全く急がない / 急がない / 早めに / 業務が止まっている）です。
  - v0.2 は第 1 選択肢を 300 件中 **5 件**しか選ばず、acc は 0.347 でした（v0.1 の 3 シードは 58 / 16 / 62 件、acc 0.427 ± 0.072）。
  - 3 段階の条件 A〜D では、第 1 選択肢が 78〜99 件選ばれました。gold で第 1 選択肢が正解の件数は 77〜85 件です。
  - **K ≥ 4 の順序尺度の性能を、K = 3 から外挿しないでください。**
- **held-out の状態でも、第 1 スロットはやや不利です。**
  - 定義は Laya の presentation_checks と同じ 2 つで、held-out の 30 状態で測りました。

| 検査 | v0.2 | v0.1 seed 0 | 順序に依存しない場合 |
|---|---|---|---|
| score の全選択肢同一の対照（slot 0 の中心化 log 確率） | −0.249 | −0.400 | 0 |
| score の全順列の第 1 スロット率 | 0.239 | 0.283 | 1/3 |
| choice の全選択肢同一の対照 | −0.280 | −0.523 | 0 |
| choice の全順列の第 1 スロット率 | 0.289 | 0.317 | 1/3 |

  - score の対照には、ordinal の分布の形（端と中央のスロットの差）も入ります。

### `bool` の閾値と P(true) の過少予測

- **`bool` は true を過少予測します。**
  - `bench_ja` での mean P(true) は 0.133 で、gold の陽性率は 0.297 です。
  - AUROC 0.844 なので順位付けは機能していますが、閾値の位置がずれています。
- v0.2 の bool acc（0.780）は v0.1 の平均（0.788）より低く、閾値 0.5 のまま使うと取りこぼしが増えます。
- **利用者の事前確率に合わせて、閾値を決めてください。** 温度較正は順位と argmax を変えないので、これを直しません。

### 長文

- state が長くなると、未知スキーマの精度が下がります。
  - backbone の `local_attention` は 128 です。学習データの state は平均 134 トークン、p95 237 トークンです。
  - 400〜799 トークン帯の AUROC（M2）は、v0.2 で 0.859 でした。v0.1 の設定の 16 本では 0.763〜0.889 と、学習ごとに大きく動きました。

### 位置に依存する表層の属性

- 「文末が問いかけで終わっているか」（`ends_with_question`）の AUROC は、v0.2 で 0.782 です。意味を問う属性（0.835〜0.996）より低い値です。
- 構造に依存する属性（`uses_bullet_points`）は、v0.1 のどのシードでも 0.85 に届いていません。v0.2 では測っていません。

### そのほか

- 学習データは合成データのみです。生成者は単一モデル（qwen3:30b-a3b-instruct-2507）です。
- 評価セットの `bench_ja` / `bench_en` も合成データで、ドメインは業務の問い合わせ文だけです。
- `bool` は、学習で見ていない言い回しの質問では弱いです。自分のデータで検証してから使ってください。
- レイテンシは質問数に比例します（質問ごとに state をエンコードし直す）。v0.2 のレイテンシは測っていません（構造は v0.1 と同じ）。
- 温度を渡さない限り、確率は較正されていません。
- 人間の最終判断を置き換える用途（採用、与信、懲戒、医療、法務の決定）には使わないでください。

## `bench_ja` / `bench_en` のライセンスと使い方

`bench_ja` と `bench_en`（GitHub の `data/bench_ja.jsonl`、`data/bench_en.jsonl`）は **CC BY 4.0** で配布しています（コードとこのモデルの Apache-2.0 とは別です）。

> **評価用途を想定しており、学習データとしての利用は控えてください。**
> これはお願いであり、ライセンス上の制限ではありません。
> このセットは未知スキーマへの汎化を測るための held-out テストセットで、
> 一度学習に使われるとその役目を果たせなくなります。

## TypeSafe Jev について

TypeSafe の利用規約（Master Customer Agreement 2.3(b)）が、同サービスとその出力を類似製品の開発に使うことを禁じています。そのため、本プロジェクトでは Jev を実測していません。**このリポジトリには Jev を呼ぶコードがありません。**

## 引用

```bibtex
@misc{modernbert-ja,
  title  = {{ModernBERT-Ja}},
  author = {SB Intuitions},
  year   = {2025},
  url    = {https://huggingface.co/sbintuitions/modernbert-ja-310m}
}
```
