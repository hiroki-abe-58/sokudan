# 即断 (Sokudan) — 日本語 System One 意思決定モデル / 設計仕様 v3

> ## ⚠ これは履歴文書です
>
> 2026-09-20 のスプリント開始時点の設計仕様であり、**現在の実装とは一致しません。**
> 公開しているのは、何をどう決めて、どこで間違えたかを追えるようにするためです。
>
> **v0.1 で撤回した主要な設計判断:**
> - §6.1 / §6.2 の **separate encoding**（state と question を別系列にし cross-attention head で繋ぐ）
>   は A/B で負けたため撤回し、**joint encoding** を採用しました。
>   理由と実測値は [`docs/architecture.md`](docs/architecture.md) §1.1。
> - 「**state は 1 リクエストにつき 1 回だけエンコードする**」という §6.2 の主張も撤回しました
>   （同 §1.2）。joint は質問ごとに state を再エンコードします。
>
> **TypeSafe Jev の実測は行いません。** TypeSafe の利用規約
> （Master Customer Agreement 2.3(b)）が、同サービスおよびその出力を類似製品の開発に
> 用いることを禁じているためです。本文中の該当箇所は
> `[9/21 削除: …]` に置き換えてあります。**それ以外は原文のままです。**
>
> 現在の正本は [`docs/architecture.md`](docs/architecture.md) と
> [`docs/benchmarks.md`](docs/benchmarks.md) です。

---

> リポジトリ: `git@github.com:hiroki-abe-58/sokudan.git`
> リポジトリルートに置き、Claude Code に **§15 のキックオフ指示**を渡して使う。
> 各 Phase の終わりで必ず止まり、人間のレビューを受けること。
>
> **v3 は「2026-09-20 の1日で公開可能な成果物を出す」ためのスプリント版。**
> §14 に当日のタイムボックスと切り捨てリストがある。§1〜§12 は設計の正本で、
> §14 が「今日はそのうちどこまでやるか」を決める。矛盾したら §14 が勝つ。

---

## 0. プロジェクトの一行要約

日本語テキスト（state）と型付き質問（questions）を受け取り、**テキストを一切生成せずに**
単一フォワードパスで型付き回答と較正済み確率を返すエンコーダモデル。
TypeSafe Jev / Laya の日本語版にあたるが、**未知スキーマに対するゼロショット汎化**と
**順序尺度 (`score`) の精度**の2点で元家を上回ることを目標とする。

- モデル名: `sokudan-ja-310m`
- パッケージ名: `sokudan`（**Phase 0 で PyPI / GitHub / HF の名前空間の空きを確認すること**）
- ライセンス: Apache-2.0（バックボーンが MIT なので互換）
- ターゲット GPU: RTX 5090 (Blackwell, sm_120, 32GB GDDR7) 1枚

---

## 1. 絶対に守るルール（Claude Code 向け）

1. **数字を捏造しない。** ベンチマーク値・レイテンシ・精度は、必ず自分で実行したコードの
   出力のみを記載する。未測定なら `TBD` と書く。「推定」「おそらく」を成果物に書かない。
2. **ライセンスを推測しない。** データセット・モデルを採用する前に、一次情報の LICENSE /
   利用規約 URL を `docs/licenses.md` に URL と該当条文の引用付きで記録する。
   商用利用・再配布の可否が確認できないものは採用せず「保留」に理由付きで記載する。
3. **encoding ロジックは単一の実装のみ。** 学習時と推論時で state/question をテンソルに
   する処理が分岐した瞬間にこのプロジェクトは死ぬ。`sokudan/encoding/` に唯一の実装を置き、
   train / eval / serve はすべてそれを import する。重複実装は見つけ次第統合。
4. **Phase をまたいで勝手に進まない。** 各 Phase 完了時に、やったこと・測定値・
   次の判断材料をまとめて報告し、指示を待つ。
5. **Gate と書かれた条件を満たさないまま次へ進まない。** 満たせない場合は報告して止まる。
6. 依存は `pyproject.toml` に集約。requirements.txt は作らない。
7. テストは実装と同じコミットで書く。`encoding` / `calibration` / `head` は
   テストなしでのマージ禁止。

---

## 2. 設計原則

- **DRY**: トークナイズ、確率の正規化、メトリクス計算は一箇所に。
- **SOLID**:
  - 単一責任: `model` は forward だけ、`calibration` は温度だけ。
  - 開放閉鎖: 質問タイプの追加が既存コードの改変なしにできること。Registry パターンで登録。
  - 依存性逆転: 教師 LLM、較正器、データソースはすべて `typing.Protocol` で抽象化し、
    上位モジュールは具象クラスを import しない。
- **高凝集・低結合**: `train` が `serve` を import していたら設計ミス。
  共有するのは `schema` / `encoding` / `model` のみ。
- **関心の分離**: 「何を聞くか(schema)」「どうテンソルにするか(encoding)」
  「どう推論するか(model)」「どう較正するか(calibration)」を混ぜない。

---

## 3. ディレクトリ構成

```
sokudan/
  schema/          # pydantic v2。質問・回答・リクエスト・レスポンス
    question.py    # ChoiceQuestion / ScoreQuestion / BoolQuestion + Registry
    response.py
  encoding/        # 唯一のトークナイズ実装
    state.py       # state -> input_ids
    question.py    # question -> input_ids + marker_positions（キャッシュ可能）
  model/
    backbone.py    # ModernBERT-Ja のラッパ
    head.py        # cross-attention decision head
    ordinal.py     # score 用の cumulative-link パラメータ化
    sokudan.py     # 合成
  calibration/
    temperature.py
    metrics.py     # ECE / Brier / NLL / RPS / reliability diagram
  data/
    teacher.py     # Teacher Protocol + 実装
    schema_aug.py  # スキーマランダム化（最重要）
    builders/
  train/
    stage1_distill.py
    stage2_rlcd.py
    loop.py
  eval/
    runner.py
    baselines.py
    report.py
  serve/
    api.py
    export.py
tests/
docs/
  licenses.md
  architecture.md
  benchmarks.md
```

---

## 4. Phase 0 — 環境と、やめる判断のための計測

**この Phase の目的は環境構築ではなく「作る価値があるか」の確認。**

### 4.1 環境

- Python 3.11 / uv。RTX 5090 は Blackwell (sm_120) なので PyTorch は CUDA 12.8 以降。
  `torch.cuda.get_device_capability() == (12, 0)` を確認するスモークテストを書く。
- transformers >= 4.48.0。`sbintuitions/modernbert-ja-310m` の fill-mask が日本語で動くこと。
- **Gate A — Flash Attention 2**: ModernBERT の unpadding 経路は flash-attn に依存する。
  sm_120 向けの wheel が無い可能性が高い。ビルドを試し、
  - 成功 → FA2 を使う
  - 失敗 → `attn_implementation="sdpa"` にフォールバックし、
    **padded sdpa での実測スループットを記録してから先へ進む**。
    ここで想定より 3 倍以上遅いなら、バックボーンを `modernbert-ja-130m` に
    落とす選択肢を含めて報告する。

### 4.2 ベースライン計測（作る前にやる）— `bench_ja`

日本語の評価セット `bench_ja` を 300 件作る。**ラベル条件付き生成**を使う:
ローカル LLM（vLLM / Ollama、手持ちのもの）に「請求部門宛の、やや苛立った、
解約をほのめかす問い合わせメールを書け」のように**正解ラベルを先に指定して**生成させる。
生成条件そのものが正解ラベルになるので、アノテーション不要でゴールドが手に入る。
`choice`（部署ルーティング 4択）/ `score`（緊急度 3段階）/ `bool`（解約示唆）を各 1 問ずつ付ける。

これで以下を**全部同じ 300 件、同じ質問**で実測する:

1. `convaiinnovations/laya-multilingual`（精度・ECE・レイテンシ）
2. `convaiinnovations/laya`（英語版に日本語をそのまま入れた場合）
3. [9/21 削除: TypeSafe MCA 2.3(b) により Jev の実測は行わない方針に変更]
4. 多数決クラス、ランダム
5. LLM-as-classifier（生成に使ったローカル LLM 自身）

**Gate B**: Laya-multilingual が日本語でまともに動いてしまった場合
（[9/21 削除: TypeSafe MCA 2.3(b) により Jev の実測は行わない方針に変更]）、
このプロジェクトの前提（日本語が空白地帯）が崩れる。その事実を報告して止まること。
逆に壊れていることが確認できたら、この計測結果そのものが最初の記事になる。
`docs/baseline_ja.md` に再現コマンド・reliability diagram の PNG 付きで残す。
`bench_ja` はそのまま Phase 5 の「未知スキーマ」テストセットの一部にもなる
（**学習には絶対に混ぜない**）。

---

## 5. Phase 1 — schema と encoding

### 5.1 質問プリミティブ

```python
# choice: 排他的な選択肢
{"type": "choice",
 "instructions": "この問い合わせはどの部署が担当すべきか",
 "criteria": {"請求": "支払い・返金・請求書", "技術": "不具合・障害", "営業": "料金・新規契約"}}

# score: 順序尺度。単なる多クラスではない（§6.3 参照）
{"type": "score",
 "instructions": "この依頼の緊急度は",
 "criteria": ["急がない", "早めに", "業務が止まっている"]}

# bool: P(true) の較正済み確率。Jev の noul に相当（エイリアス noul も受ける）
{"type": "bool",
 "instructions": "解約を示唆しているか"}
```

- Registry に型名 → ハンドラを登録。新しい型の追加で既存コードを触らない。
- `choice` の選択肢数は 2〜20 を想定。それ以上は「粗い choice → 細かい choice」の
  2段構成を推奨する旨を docstring に明記。

### 5.2 トークナイズ

**state 側**（`encoding/state.py`）:
```
[CLS] {state} [SEP]
```

**question 側**（`encoding/question.py`）— state を含めない。ここが v1 との最大の違い:
```
[CLS] {instructions} [SEP]
  {option_1_label}: {option_1_desc} <mask>
  {option_2_label}: {option_2_desc} <mask>
  ...
[SEP]
```

- マーカーは **MLM 事前学習で使われた `<mask>` トークンをそのまま流用**する。
  新しい special token を足して埋め込みをリサイズすると、事前学習の事前分布を
  捨てることになる。`tokenizer.mask_token_id` を使い、ハードコードしない。
- question 側は state を含まないので通常 256 トークン以内に収まる。
  **これは ModernBERT の local attention 窓 (128) と相性が良い**（§6.1 参照）。
- question 側のエンコード結果は**スキーマが同じなら完全に再利用可能**。
  `hash(question)` をキーにキャッシュする設計にしておく（本番のレイテンシに直結）。
- `marker_positions: list[int]` を返す。学習と推論でズレたら全部壊れるので
  ラウンドトリップのテストを必ず書く。

**完了条件**: schema バリデーション、encoding ラウンドトリップ、マーカー位置、
境界ケース（選択肢1個、state が 8192 超、絵文字、半角カナ、空文字）のテストが通る。

---

## 6. Phase 2 — モデル

### 6.1 なぜ v1 の「ブロック対角マスクで全質問を1本に詰める」設計を捨てるのか

**ModernBERT は全層が global attention ではない。** `modernbert-ja-310m` の config は
`local_attention: 128`、`global_attn_every_n_layers: 3`。つまり **3層に1層だけが global**、
残りは 128 トークンのスライディングウィンドウしか見ない。

したがって、state を先頭に置いて質問ブロックを後ろに連結すると:

- 位置 3000 にある質問ブロックは、local 層では state を**まったく見られない**
- 情報が 3層に 1回しか流れず、深刻なボトルネックになる
- さらに悪いことに、**質問が系列のどこに落ちるかで挙動が変わる**。
  質問数が増えると後ろの質問ほど不利になり、v1 で主張した「質問順序に対する不変性」は
  そもそも成立しない

v1 のこの設計はそのまま実装すると静かに劣化する。破棄する。

### 6.2 採用する設計 — 共有バックボーン + cross-attention head

```
state    --[backbone]--> H_state  (L_s × 768)   ... 1リクエストにつき1回だけ
question --[backbone]--> H_q      (L_q × 768)   ... N問をバッチ次元で1回
                              |
                    [decision head: 2〜4層]
                      self-attn over H_q
                      cross-attn into H_state   ... H_state は全質問にブロードキャスト
                              |
                    marker positions を抜き出す
                              |
                 scorer (Linear 768->1) -> 質問内 softmax
```

- backbone は state と question で**重みを共有**する（別モデルを2個持たない）。
- 各系列が短いので local/global の交替が事前学習時と同じ条件で働く。マスク改造は一切不要。
- state のエンコードは質問数に関係なく 1 回。Laya は質問ごとに state を再エンコードして
  いるので、**ここが構造的な優位**。ただし主張する前に §9 で実測すること。
- 質問側は完全にキャッシュ可能。同じスキーマを繰り返し使う本番では `H_q` を使い回せる。
- 順序不変性は構造として保証される（質問は互いに見えない）。

**head の初期化（1時間で収束させるための要点）**:
- self-attn 層は backbone の**最後の 2 層の重みをコピー**して初期化する。ランダム初期化だと
  マーカー位置の表現が事前学習の `<mask>` 事前分布から切れて、収束に無駄な時間がかかる。
- cross-attn の出力射影は**ゼロ初期化**する（ControlNet / adapter と同じ発想）。
  学習開始時点では head が「question 側の表現をそのまま通す」恒等写像として振る舞い、
  state の情報は勾配に従って徐々に混ざる。学習が安定する。
- scorer (Linear 768→1) は小さな分散で初期化し、初期の softmax がほぼ一様になるようにする。
  初期から自信満々だと proper scoring rule の勾配が暴れる。

**`torch.compile` の注意**: `L_s`, `L_q`, 選択肢数が毎バッチ変わるので、素の compile は
再コンパイル地獄になる。`dynamic=True` にするか、**当日は compile を切って**、
長さでバケット化した padding で対応する。速度の最適化は当日の仕事ではない。

**リスク**: 質問と state の相互作用が head の数層だけに閉じ込められる。
head を 2層 / 4層で比較するアブレーションを必ず実施し、`docs/architecture.md` に記録する。
足りなければ head を増やすか、cross-attention を backbone の後半層にも差し込む案を検討。

### 6.3 `score` は多クラス softmax にしない

Laya の最弱プリミティブは `score`（SST-5 で 0.372）。ここは素直に取りに行く。

- **cumulative link 方式、ただし K はリクエスト時に決まる**ことに注意。固定 K の CORAL を
  そのまま持ってくると死ぬ。動的 K で単調性を保証する定式化:
  - 各順序ラベル `k` のマーカー位置から `b_k = softplus(w · h_k)` を読む（非負）
  - question 全体のプール表現から `a = v · h_pool` を読む
  - `P(y > k) = sigmoid(a - Σ_{j≤k} b_j)`。累積和が単調増加なので CDF は構造的に単調
  - `p_k = P(y > k-1) - P(y > k)` で各段階の確率に戻す
  これなら K=2 でも K=7 でも同じ head で動き、選択肢の追加が既存の重みを壊さない。
- 学習損失は **RPS (ranked probability score)** を主とする。
  CE は「隣に外す」と「両端に外す」を同じ罰にするので順序尺度には不適。
- 返り値は期待値 `Σ i * p_i` と分布の両方。期待値だけ返さない。

### 6.4 act / escalate ヘッド

v1 ではヘッドだけ置いて学習信号を書いていなかった。**信号を定義できないなら v1 では作らない。**

作る場合の定義: 「このモデル自身の argmax が正解と一致するか」を予測する二値ヘッド。
confidence（回答分布の最大値）とは別物で、confidence が高くても系統的に間違える
領域を検出するのが目的。ラベル生成には**交差適合が必須**——
学習データを k-fold に分け、各 fold について他 fold で学習したモデルの
予測正誤をラベルにする。自分の学習データ上の正誤をそのまま使うと、
過学習した「常に正解」ラベルになって無意味になる。

この手間を払わないなら、**v1 では素直に切って、confidence 閾値だけで運用する**。

### 6.5 学習設定

- backbone フルファインチューン（LoRA ではない）。310M なら 5090 で VRAM は余裕。
- bf16 + `torch.compile`。勾配チェックポイントは不要なはず（実測で判断）。
- 乱数シードを固定し、**最低 3 シードで回して分散を報告する**。単一シードの数字は出さない。

**完了条件**: ランダム初期化で forward が通り、`sum(p) == 1`、
質問数・選択肢数を変えても形状が正しく、`score` の CDF が単調。

---

## 7. Phase 3 — データ（本当の勝負どころ）

### 7.1 教師の確率をそのまま信じない

**v1 の最大の穴**: 「教師 LLM から soft label を取る」と書いたが、
LLM が自己申告する確率は一般に過信気味で、それを KL で蒸留すると
**教師の較正ズレをそっくり継承する**。以下のいずれかで確率を作ること:

1. **ゴールドラベル優先**。既存データセットの正解ラベルがあるならそれを使う。
2. **サンプリング頻度**: 教師を temperature > 0 で N 回（N=8〜16）サンプリングし、
   経験頻度を soft target にする。自己申告より遥かにマシ。
3. **ロジット直読**: ローカル vLLM を教師にする場合、選択肢先頭トークンの
   logprob から分布を作る。最も筋が良い。

`Teacher` Protocol は「分布を返す」契約にし、上の 1〜3 を実装差し替えで選べるようにする。

```python
class Teacher(Protocol):
    async def label(self, state: str, question: Question) -> Distribution: ...
```

レート制限・リトライ・キャッシュ（state+question のハッシュキー）は共通デコレータで。
**API 教師を使う場合、Phase 3 着手前にコスト見積もりを出して報告すること。**

### 7.2 スキーマランダム化 — 最重要

Laya のゼロショットが多数決ベースラインを下回った原因はここだと仮説を立てる。
学習時に以下をランダム化し、「ラベル ID を覚える」のではなく
「スキーマを読んで判断する」ことを強制する:

- 選択肢の**順序**をシャッフル
- 選択肢の**表記ゆれ**（「請求」「課金」「支払い関連」）
- 選択肢の**数**を増減（ダミー選択肢、「その他」「該当なし」の有無）
- `instructions` の言い換え（敬体/常体、長短、丁寧語）
- 同じ state に複数の異なるスキーマを生成
- 無関係なディストラクタ選択肢の混入

`schema_aug.py` に集約し、変換が可逆（正解ラベルが追従する）ことをテストする。
**アブレーション必須**: ランダム化あり/なしで未知スキーマ汎化がどれだけ変わるかを測り、
`docs/benchmarks.md` に載せる。これ自体が記事のコアになる。

### 7.3 データソースとライセンス（要注意）

**採用前に必ず一次情報を当たること。以下は「調べるべき対象」であって「使える」ではない。**

| 候補 | 用途 | 既知のリスク |
|---|---|---|
| JGLUE (JNLI, JCommonsenseQA 等) | 含意・常識 | サブタスクごとにライセンスが異なる。個別確認 |
| JGLUE MARC-ja | 感情 | 元の Amazon レビューコーパスが提供終了している可能性。**再配布可否を厳格に確認** |
| livedoor ニュースコーパス | トピック分類 | **CC BY-ND 系の疑いあり。ND なら派生データセットの再配布は不可**。最優先で確認 |
| WRIME | `score` の感情強度 | 利用条件を一次情報で確認 |
| 合成データ（LLM生成） | 問い合わせ/モデレーション/議事録 | 生成に使ったモデルの利用規約（出力の利用制限）を確認 |

ND 条項に引っかかるものが出てきた場合、**学習には使わず、再配布しない評価専用**に
回すという選択肢も検討し、その判断を `docs/licenses.md` に記録する。
迷ったら合成データの比率を上げる。

### 7.4 当日用データ — ゴールドラベルが「タダで手に入る」2系統だけを使う

API 教師は当日は使わない（コストと時間の不確実性）。以下の 2 系統で、
すべてゴールドラベル付きの学習データを作る:

**(a) JGLUE のプリミティブ対応表**（ライセンスは `docs/licenses.md` で一次確認してから）

| JGLUE タスク | sokudan プリミティブ | 変換 |
|---|---|---|
| JNLI | `choice`（含意/矛盾/中立の3択） と `bool`（「前提は仮説を含意するか」） | state = 前提+仮説、スキーマランダム化で表記ゆれを作る |
| JCommonsenseQA | `choice`（5択） | ディストラクタが最初から入っている。スキーマ読解の練習に最適 |
| JSTS | `score`（類似度 0〜5 の 6 段階） | 順序尺度のゴールド。`score` head の主戦場 |

**(b) ラベル条件付き合成データ**（§4.2 と同じ手法、規模を拡大）

ローカル LLM で「業務っぽい」スキーマ群を生成する。生成条件＝正解なので教師不要:
部署ルーティング、緊急度、解約示唆、フィッシング判定、モデレーション（誹謗中傷/宣伝/正常）、
議事録の決定事項有無、レビューの星、など **10〜15 種類のスキーマ**。
`bench_ja` のスキーマと**意図的に別の言い回し・別の選択肢構成**にして、
未知スキーマ汎化を正直に測れるようにする。

当日の規模目標: (a)+(b) で **2〜3 万件**。310M / 5090 なら 1 epoch が数分で終わる規模。
10 万件は後日。

**完了条件**: 学習 10万件 / 検証 1万件 / テスト 1万件。
テストセットは**学習に一度も出現していないスキーマ**のみで構成すること。
これがゼロショット汎化の唯一の正直な測り方。

---

## 8. Phase 4 — 学習

### Stage 1: soft-label 蒸留
- 損失: 教師分布への KL（`score` は RPS）
- 1 epoch、bf16、AdamW、cosine schedule、warmup 5%
- 310M / 10万件なら 5090 で 1時間以内に終わる想定。大きく超えるなら実装を疑う

### Stage 2: 温度較正
- **(質問タイプ, 選択肢数) の組ごとに温度を1つ**フィット。検証セットで NLL 最小化
- **この時点の ECE を必ず記録する。** これが次の Stage の判断基準になる

### Stage 3: RLCD（**条件付き・省略可**）
- 方策が分布を報告 → ロジットに平均ゼロのガウスノイズを加えて探索
- 報酬は strictly proper scoring rule（log + spherical、`score` には RPS）
- 更新は group-mean baseline 付き REINFORCE（GRPO 風）
- 正直な確率を報告したときのみ期待報酬が最大になる性質を、
  おもちゃの分布で数値的に検証する単体テストを先に書く

**Gate C**: Stage 3 は「Stage 1 + 温度較正」を有意に上回った場合のみ採用する。
soft-label 蒸留で既に分布を学習しており、較正の大半は温度スケーリングで説明できる可能性が高い。
アブレーションで差が出ないなら**実装を捨てて README にその事実を書く**。
「やってみたが効かなかった」は立派な知見で、そう書ける方が信用される。

**完了条件**: 各 Stage 前後の全メトリクスを記録した `runs/<id>/report.md` が出る。

---

## 9. Phase 5 — 評価

### メトリクス
accuracy / macro-F1 / **ECE**（reliability diagram を PNG 出力）/ Brier / NLL /
`score` には RPS と MAE / レイテンシ p50・p95（1問・10問・50問）/ questions per sec。
**3シード以上、平均±標準偏差で報告する。**

### ベースライン（これが無いと数字に意味がない）
1. ランダム
2. 多数決クラス ← **下回ったら即座に報告して止まる**
3. 素の分類器（ModernBERT-Ja + 固定ラベル分類ヘッド、スキーマ汎化なし）＝ 特化時の上限目安
4. LLM-as-classifier（教師モデル本体）
5. `laya-multilingual` の日本語実測（Phase 0 で取った値）

### 主張してよいこと / いけないこと
- 「state を1回だけエンコードするので Laya より速い」→ **自分で両方測ってから言う**
- 「Jev より高精度」→ Jev API を自分で叩いていないなら言わない。
  第三者が公開した数値との比較なら、その旨を明記する
- 未知スキーマで 2 を上回らないなら、「特化用のベースです」以上のことは主張しない

---

## 10. Phase 6 — 配信

- FastAPI で `POST /v1/systemone`。**リクエスト/レスポンス形状は Jev / LocalJev と
  ワイヤ互換にする**。形状は推測せず、公開 SDK / ドキュメント / LocalJev の実装から
  実際のフィールド名（`nouls` など）を確認して合わせること。
  既存の LangChain 連携・LiteLLM パススルーがそのまま刺さるのが最大の普及レバレッジ。
- ただし README に「確率の出所は head のロジット直読であり、
  LocalJev のようなモデル自己申告ではない」ことを明記する。ここが差別化点。
- 質問側エンコードのキャッシュを有効にする（§5.2）。キャッシュ有無のレイテンシ差を実測。
- `/ready` `/healthz`、バッチ上限、キュー溢れ時の 429 / 529。
- ONNX エクスポート → TensorRT。5090 で 1問 10ms 切りを目標に、**実測値のみ**載せる。
- `pip install sokudan` → 3行で動く Quickstart。

---

## 11. 成果物チェックリスト

- [ ] GitHub リポジトリ（Apache-2.0、README 日本語/英語併記）
- [ ] Hugging Face モデル（モデルカードに **Limits を正直に**書く。ここを盛ると死ぬ）
- [ ] HF Space デモ（日本語の問い合わせメールを貼ると即判定）
- [ ] `docs/baseline_ja.md`（Phase 0 の Laya 日本語実測）
- [ ] `docs/licenses.md`
- [ ] `docs/benchmarks.md`（全部自前実測・再現コマンド・シード分散付き）
- [ ] `docs/architecture.md`（cross-attention head とローカル注意の制約の図解）

---

## 12. 参照

- Laya モデルカード: https://huggingface.co/convaiinnovations/laya
- Laya 実装: https://github.com/NandhaKishorM/laya
- LocalJev（ワイヤ互換ブリッジの参考実装）: https://github.com/githubnext/localjev
- TypeSafe System One 概念: https://docs.typesafe.ai/concepts/system-one
- ModernBERT-Ja: https://huggingface.co/sbintuitions/modernbert-ja-310m
- ModernBERT 論文: https://arxiv.org/abs/2412.13663

---

## 13. v1 からの変更点

| # | 変更 | 理由 |
|---|---|---|
| 1 | **ブロック対角マスクによる質問パッキングを破棄**、共有バックボーン + cross-attention head に変更 | ModernBERT は `local_attention: 128` / `global_attn_every_n_layers: 3`。遠くに置いた質問ブロックは local 層から state が見えず、静かに劣化する。順序不変性も成立しない |
| 2 | `score` を多クラス softmax から **cumulative link + RPS 損失**に変更 | Laya の最弱プリミティブ。CE は順序尺度に不適 |
| 3 | 教師の soft label を**サンプリング頻度 / ロジット直読**に変更 | LLM の自己申告確率は過信気味で、蒸留すると較正ズレを継承する |
| 4 | RLCD を必須 Stage から**アブレーション条件付き**に降格 | 蒸留＋温度較正で足りる可能性が高い。効かないなら捨てて書く方が誠実 |
| 5 | act/escalate ヘッドに**交差適合による学習信号を定義**、定義できないなら削除 | v1 はヘッドだけ置いて学習方法を書いていなかった |
| 6 | Laya 日本語ベースライン計測を Phase 5 → **Phase 0 に前倒し**（Gate B） | 作る前にやめる判断ができる。かつ最初の記事になる |
| 7 | FA2 ビルド可否を Phase 0 の **Gate A** に昇格 | sm_120 の wheel が無い可能性が高く、速度前提が崩れる |
| 8 | 質問側エンコードの**キャッシュ**を明示 | state と質問を分離した副産物。本番レイテンシに直結 |
| 9 | ライセンスリスク（ND 条項、提供終了コーパス）を具体名で列挙 | 「確認せよ」だけでは踏む |
| 10 | 全数値に**3シード以上の分散**を要求 | 単一シードの差は簡単に有意でなくなる |

### v2 → v3

| # | 変更 | 理由 |
|---|---|---|
| 11 | [9/21 削除: TypeSafe MCA 2.3(b) により Jev の実測は行わない方針に変更] | — |
| 12 | `bench_ja` を**ラベル条件付き生成**で作る手順を明記 | 生成条件＝正解。アノテーションもAPI教師も不要 |
| 13 | head の**初期化戦略**（backbone 末尾層コピー、cross-attn ゼロ初期化、scorer 小分散） | 1 時間で収束させるため。ランダム初期化は無駄に時間を食う |
| 14 | `score` head を**動的 K** で定式化（累積 softplus） | 固定 K の CORAL はリクエスト時に K が決まる設計と両立しない |
| 15 | `torch.compile` の再コンパイル問題を明記、当日は切る | 可変長 + 可変選択肢数で素の compile は破綻する |
| 16 | 当日データを **JGLUE 3 タスク + ラベル条件付き合成**に限定、2〜3 万件 | 全部ゴールド。API 教師の不確実性を排除 |
| 17 | §14 スプリント計画と切り捨て順、§15 キックオフ指示を追加 | 今日中に公開するため |

---

## 14. 2026-09-20 スプリント計画（今日やること）

### 14.1 今日の「どや顔」の定義

順に難易度が上がる。**A は確実に取る。B を取りにいく。C は明日以降。**

| | 成果物 | 何がすごいか | 必要な Phase |
|---|---|---|---|
| **A** | `docs/baseline_ja.md`: Laya を**日本語で同一条件で実測**（[9/21 削除: TypeSafe MCA 2.3(b) により Jev の実測は行わない方針に変更]） | 日本語での実測比較は誰も出していない | Phase 0 のみ |
| **B** | `sokudan-ja-310m` v0.1: 未知スキーマの `bench_ja` で **Laya-multilingual を上回る**チェックポイント + HF 公開 | 日本語 System One の最初の実装。かつ cross-attn 設計で state を1回しかエンコードしない | Phase 1〜5（縮小版） |
| **C** | Jev ワイヤ互換 API + HF Space デモ | 既存 SDK がそのまま刺さる | Phase 6 |

**A が取れなかった場合（Laya が日本語で普通に動いた場合）は、B の意義が変わる。**
その時点で立ち止まって方針を再検討する。作り始めない。

### 14.2 タイムボックス（合計 ~10 時間想定、各ブロックは硬い締切）

| 時間 | やること | 終了時に存在するもの |
|---|---|---|
| 0:00–1:00 | リポジトリ足場、uv、torch/sm_120 確認、**Gate A**（FA2 ビルド試行、失敗なら sdpa で実測して先へ）、`sokudan` 名前空間の空き確認 | `pyproject.toml`、スモークテスト通過、Gate A の結果メモ |
| 1:00–3:00 | `bench_ja` 300 件をラベル条件付き生成 → Laya ×2、多数決、ランダム、ローカル LLM を実測（[9/21 削除: TypeSafe MCA 2.3(b) により Jev の実測は行わない方針に変更]）。reliability diagram 出力 | **`docs/baseline_ja.md`（この時点で公開可能）** |
| 3:00–5:30 | `schema` / `encoding` / `model`（backbone ラッパ、cross-attn head、動的 K の score head）/ テスト。act-escalate ヘッドは**作らない** | forward が通り、テスト緑 |
| 5:30–7:00 | JGLUE 3 タスク + 合成 10〜15 スキーマの builder、`schema_aug`、2〜3 万件生成 | `data/train.jsonl` / `val.jsonl`、`docs/licenses.md` |
| 7:00–8:00 | Stage 1 蒸留（ゴールドなので実質 CE + RPS）を 3 シード、温度較正 | `runs/*/report.md` ×3 |
| 8:00–9:00 | `bench_ja` で最終評価。ベースライン全部と並べる。スキーマランダム化あり/なしのアブレーション（時間があれば） | `docs/benchmarks.md` |
| 9:00–10:00 | HF にモデル push（Limits を正直に）、README、`pip install -e .` で 3 行 Quickstart が動くこと | **公開可能な状態** |

### 14.3 今日は切るもの（後日）

- RLCD（Stage 3）— Gate C のアブレーション自体を後日
- act / escalate ヘッド
- TensorRT / ONNX
- HF Space デモ
- Jev ワイヤ互換 API（forward が動けば FastAPI は 1 時間だが、優先度は B の後）
- mypy strict
- head 2層 / 4層のアブレーション（当日は 2 層固定）
- 10 万件データ、API 教師

### 14.4 遅れたときの切り捨て順

1. 3 シード → 1 シード（ただし README に「単一シード」と明記）
2. 動的 K の score head → 素の softmax（RPS で評価だけはする）
3. スキーマランダム化のアブレーション → なし（ランダム化そのものは切らない）
4. JGLUE → 合成データのみ（ライセンス確認に時間が溶けたらここ）

**どれだけ遅れても A（`docs/baseline_ja.md`）は削らない。**
