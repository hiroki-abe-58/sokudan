# ライセンス台帳

**ルール (SOKUDAN_SPEC.md §1-2): ライセンスを推測しない。**
一次情報の URL と該当条文の引用を必ず添える。確認できないものは「保留」とし、理由を書く。

凡例:
- ✅ **採用可** — 一次情報を確認し、商用利用・再配布が可能
- ⚠️ **評価専用** — 学習には使わず、再配布もしない条件でのみ利用
- ⛔ **不採用** — 条件が合わない
- ⏸ **保留（未確認）** — 一次情報をまだ当たっていない。**この状態のものは使わない**

最終更新: 2026-09-20

---

## 1. バックボーンモデル

### `sbintuitions/modernbert-ja-310m` — ✅ 採用可

| | |
|---|---|
| 一次情報 | https://huggingface.co/sbintuitions/modernbert-ja-310m/blob/main/LICENSE |
| モデルカード | https://huggingface.co/sbintuitions/modernbert-ja-310m （`license: mit`, `## License` 節） |
| 確認日 | 2026-09-20（`raw/main/LICENSE` を取得して確認） |
| ライセンス | MIT License, Copyright (c) 2025 SB Intuitions |

該当条文の引用:

> MIT License
>
> Copyright (c) 2025 SB Intuitions
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.

**判断**: MIT は Apache-2.0 との組み合わせ配布に支障なし。著作権表示の保持義務があるため、
リポジトリの `NOTICE` と HF モデルカードに SB Intuitions の表示を残す。

参考（同日 `config.json` を取得して確認した実値。§6.1 の設計根拠）:

| キー | 実値 |
|---|---|
| `local_attention` | 128 |
| `global_attn_every_n_layers` | 3 |
| `hidden_size` | 768 |
| `num_hidden_layers` | 25 |
| `num_attention_heads` | 12 |
| `max_position_embeddings` | 8192 |
| `vocab_size` | 102400 |

---

## 2. 学習データ候補（§7.3）

**現時点ではいずれも未確認。ブロック 5:30–7:00 で一次情報を当たるまで使わない。**

| 候補 | 用途 | 状態 | 備考 |
|---|---|---|---|
| JGLUE JNLI | `choice` / `bool` | ⏸ 保留（未確認） | サブタスクごとにライセンスが異なる。個別確認 |
| JGLUE JCommonsenseQA | `choice`（5択） | ⏸ 保留（未確認） | 同上 |
| JGLUE JSTS | `score`（0〜5 の6段階） | ⏸ 保留（未確認） | 同上 |
| JGLUE MARC-ja | 感情 | ⏸ 保留（未確認） | 元の Amazon レビューコーパスが提供終了の可能性。再配布可否を厳格に |
| livedoor ニュースコーパス | トピック分類 | ⏸ 保留（未確認） | **CC BY-ND 系の疑い。ND なら派生データセットの再配布は不可** |
| WRIME | `score`（感情強度） | ⏸ 保留（未確認） | 利用条件を一次情報で確認 |
| ラベル条件付き合成データ | 全プリミティブ | ✅ 採用可 | 生成モデル `qwen3:30b-a3b-instruct-2507-q4_K_M` = Apache-2.0。下記 §5 |

ND 条項に該当するものが出た場合は「学習には使わず、再配布しない評価専用 (⚠️)」に
回す選択肢を検討し、その判断理由をここに記録する。

---

## 3. 比較対象モデル（§4.2 のベースライン）

| 対象 | 用途 | 状態 | 備考 |
|---|---|---|---|
| `convaiinnovations/laya-multilingual` | 日本語ベースライン実測 | ✅ 採用可 | Apache-2.0（モデルカード YAML `license: apache-2.0`、`commercial-use` タグ）。2026-09-20 確認 |
| `convaiinnovations/laya` | 英語版に日本語を入力 | ✅ 採用可 | 同上 |
| TypeSafe Jev | 日本語ベースライン実測 | ⛔ **不採用（実測しない）** | TypeSafe の利用規約（Master Customer Agreement 2.3(b)）が、同サービスおよびその出力を類似製品の開発に用いることを禁じているため、本プロジェクトでは Jev を実測していない。 **このリポジトリには Jev を呼ぶコードが存在しない。** |

**注意**: ベンチマーク目的でのモデル出力の利用と、その数値の公開は別問題。
数値を公開する前に各モデルカード／利用規約のベンチマーク公開条項を確認すること。

---

## 4. TypeSafe Jev を実測しない理由（2026-09-20 判断）

**判断**: 呼ばない。ベースライン一覧から外す。

**根拠**: TypeSafe の利用規約（Master Customer Agreement 2.3(b)）が、同サービスおよびその出力を類似製品の開発に用いることを禁じているため、本プロジェクトでは Jev を実測していない。

**影響**:
- §14.1 の成果物 A は **Laya のみで成立させる**（`laya-multilingual` (ja) /
  `laya` (en モデルに日本語入力) / 多数決 / ランダム / ローカル LLM-as-classifier）。
- `docs/baseline_ja.md` の Jev 欄には理由のみを書き、**数値は載せない**。

**公開ドキュメントの扱い**: TypeSafe の公開ドキュメント（Model jaggedness など）は
公開情報として参照してよい。ただし `sokudan` の設計根拠として引用する場合は
必ず「公開ドキュメントより」と出典を明記すること。API を叩いて得た値との区別を曖昧にしない。

---

## 5. 生成モデル（合成データの出所）— 2026-09-20 確認

### `qwen3:30b-a3b-instruct-2507-q4_K_M`（ollama 経由 / 上流 `Qwen/Qwen3-30B-A3B-Instruct-2507`）— ✅ 採用可

| | |
|---|---|
| 上流の一次情報 | https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507/blob/main/LICENSE |
| モデルカードのメタデータ | `"license":"apache-2.0"` |
| 手元の配布物 | `ollama show --license qwen3:30b-a3b-instruct-2507-q4_K_M` → Apache License 2.0 の標準全文 |
| 確認方法 | LICENSE 全文を取得し、`output` / `derivative` / `distill` を全行検索 |

**確認できたこと**: 取得した LICENSE は **Apache-2.0 の標準全文のみ**で、
「モデル出力の利用制限」「生成物を用いた別モデルの学習禁止」に相当する追加条項は**含まれていない**。
Apache-2.0 の "Derivative Works" への言及はライセンス本文の定義節であり、出力への制限ではない。

**判断**: この生成モデルの出力（`bench_ja` および §7.4(b) の合成学習データ）は、
学習利用・再配布ともに可能と判断する。

**注意**: ollama の配布物と上流 HF リポジトリが同一の重みであることは、
量子化の性質上ハッシュ一致では確認していない。ollama 側が同梱する LICENSE が
Apache-2.0 であることを一次情報として採用している。

### 本プロジェクトが生成したデータ

`data/bench_ja.jsonl` は上記モデルの出力であり、Apache-2.0 の下で再配布可能と判断する。
ただし **`bench_ja` は評価専用**であり、学習には一切使わない（SOKUDAN_SPEC.md §4.2）。
これはライセンス上の制約ではなく、評価の健全性のための自己制約。
