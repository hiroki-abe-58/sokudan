# Sokudan (即断) — a Japanese System One decision model

**Give it Japanese text (`state`) and typed questions (`questions`); it returns typed
answers and probabilities in a single forward pass, without generating any text.**

Because nothing is generated, there is nothing to parse and nowhere for a
hallucination to live.

- Model: `sokudan-ja-310m` (314.6M; backbone [`sbintuitions/modernbert-ja-310m`](https://huggingface.co/sbintuitions/modernbert-ja-310m), MIT)
- Licence: Apache-2.0
- Built on a single RTX 5090 (Blackwell, sm_120)

> **Status: v0.1, built in a two-day sprint on 2026-09-20/21.**
> Published at [`GeneLab/sokudan-ja-310m`](https://huggingface.co/GeneLab/sokudan-ja-310m).
> **Every number in this README is the output of code run on that machine.** None are
> estimates. Anything unmeasured is written as "not measured".

*This is a translation of [`README.md`](README.md). The Japanese version is
authoritative; if the two disagree, the Japanese one is right.*

## `bench_ja`, 300 items, mean ± standard deviation over 3 seeds

| model | choice acc | score RPS↓ | bool acc | bool AUROC |
|---|---|---|---|---|
| **sokudan-ja-310m** | **0.847 ± 0.009** | **0.090 ± 0.023** | **0.788 ± 0.010** | **0.789 ± 0.043** |
| `laya-multilingual` (ja) | 0.747 | 0.232 | 0.543 | 0.523 |
| majority class | 0.380 | 0.197 | 0.703 | — |
| random | 0.253 | 0.201 | 0.513 | — |

Every metric, calibrated and uncalibrated, plus the per-seed values, is in
[`docs/benchmarks.md`](docs/benchmarks.md). **`score` varies a lot across seeds**: the
published weights are seed 0, whose own figures are acc 0.663 / RPS 0.117.

---

## Why this exists — the measurement that came first

Before building anything, we measured how existing System One models behave **in
Japanese** (300 Japanese business messages, `bench_ja`, shipped in full, see
[`docs/baseline_ja.md`](docs/baseline_ja.md)).

### `bench_ja` licence and intended use

`data/bench_ja.jsonl` is distributed under **CC BY 4.0** (separate from the code's
Apache-2.0).

> **Evaluation only. Do not mix it into training data.**
> It is a held-out test set for measuring generalisation to unseen schemas, and it
> stops serving that purpose the moment it is trained on.

> **Correction (2026-09-21).** `bench_ja`'s boolean question ("does the sender suggest
> cancellation or contract termination?") **never appeared in the training data.**
> However, the leak check used on day 1 was incomplete: **解約 ("cancellation") appeared
> in 718 training/validation question strings and 解除 ("termination") in 868**, both as
> `choice` option descriptions (the *values* of `criteria`) and as an option label. The
> check read only the `criteria` *keys*, and 解除 was not in its term list. The catalogue
> and the check are fixed in `scripts/check_catalog_leak.py`. **The day-1 numbers are
> not restated**: the bench boolean itself never appeared, and `bool` failed to transfer
> even with that vocabulary present.

| model | choice acc | score RPS↓ | bool acc | bool AUROC |
|---|---|---|---|---|
| `laya-multilingual` (ja) | 0.747 | 0.232 | 0.543 | **0.523** |
| majority class | 0.380 | **0.197** | **0.703** | — |
| random | 0.253 | 0.201 | 0.513 | — |

What that showed:

- **`choice` (multi-class selection) works in Japanese**, at roughly 2.0x the majority
  baseline. This is not an empty space.
- **`score` (ordinal) is below the majority baseline.** Isolating the cause: across
  five conditions that changed only the schema, **the first presented option was
  chosen 0-1 times out of 300 in every one of them**. The same label 「急がない」 ("not
  urgent") drew 0 selections in first position and 250 in last.
- **`bool` sits at AUROC 0.523** — the ranking does not work. This is not a threshold
  problem.

**So `sokudan` targets `score` and `bool`.** `choice` is reported but not a goal.

---

## Quickstart

```bash
pip install -e .
```

```python
import sokudan

agent = sokudan.load("GeneLab/sokudan-ja-310m")     # or a local runs/.../model.pt
result = agent.predict(
    {"body": "先月の請求で同じ金額が二回引き落とされています。至急ご確認ください。"},
    {
        "department": {"type": "choice",
                       "instructions": "この問い合わせはどの部署が担当すべきか",
                       "criteria": {"請求": "支払い・返金", "技術": "不具合・障害",
                                    "営業": "料金・新規契約", "その他": "上記以外"}},
        "urgency":    {"type": "score",
                       "instructions": "この依頼の緊急度は",
                       "criteria": ["急がない", "早めに", "業務が止まっている"]},
        "churn":      {"type": "noul",
                       "instructions": "解約を示唆しているか"},
    },
)
print(result["answers"]["department"]["choice"])
```

Question types are `choice` / `score` / `bool` (`noul` is an alias). The schema is
free per request; no retraining is needed.

**The model is trained on Japanese only.** Its behaviour on other languages is not
measured.

### Where the probabilities come from

They are read **directly off the head's logits**. They are not a number the model was
asked to report about its own confidence. For calibrated probabilities, pass the
stage-2 temperatures:

```python
agent = sokudan.load("GeneLab/sokudan-ja-310m",
                     temperatures="temperatures.json")
```

**Without temperatures the probabilities are not calibrated** — see Limits.
**But the shipped temperatures make `score` RPS worse** (0.090 → 0.149). Use them for
`choice` and `bool` only, or refit on your own validation set.

The other seeds are available as revisions:

```python
agent = sokudan.load("GeneLab/sokudan-ja-310m@seed1")
```

---

## Design

Full detail in [`docs/architecture.md`](docs/architecture.md).

### State and question in one sequence (joint encoding)

```
[CLS] instructions [SEP] options+markers [SEP] state [SEP]
                              |
                  [backbone, 25 layers]   ... once per question (the state goes in each time)
                              |
                    marker positions -> scorer
                              |
          choice/bool -> softmax within the question    score -> dynamic-K cumulative link
```

The original design put **state and question in separate sequences** joined by a
cross-attention head. The reasoning was `modernbert-ja-310m`'s `local_attention: 128`
and `global_attn_every_n_layers: 3` (values read from `config.json`): concatenate
them, and the question cannot see the state.

**Both were trained under identical conditions and the result was the opposite, so
the judgement was withdrawn.**

| | separate + head | **joint** |
|---|---|---|
| held-out AUROC (unseen schema × unseen document) | 0.506 | **0.872** |
| of which implicit intent | 0.486 | **0.786** |
| val bool AUROC | 0.529 | **0.984** |
| steady-state epoch time | 464 s | **330 s** |

The measured mean sequence length in this data is **160 tokens** (state 134 +
question 26), about the width of the local window, so the "question block at position
3000" the argument assumed never occurs. Meanwhile the separate arm pushes all
question-state interaction through 2 head layers, where joint gets all 25.

The withdrawal is recorded in [`docs/architecture.md`](docs/architecture.md) §1.1 and
§1.2. Questions are still processed as separate rows, so **order invariance remains
structural**.

### `score` uses a dynamic-K cumulative link

K is fixed at request time, so fixed-K CORAL does not apply.

```
b_k = softplus(w · h_k) ≥ 0
P(y > k) = sigmoid(base_k + a - Σ_{j≤k} b_j)      ← monotone by construction
p_k = P(y > k-1) - P(y > k)
```

`base_k` is a closed-form term that makes the initial distribution uniform for any K.
The loss is **RPS** (cross-entropy charges the same price for missing by one level as
for missing by three).

### One encoding implementation

`sokudan/encoding/` is the only place a state or a question becomes token ids, and
train / eval / serve all import it. The moment tokenisation diverges between training
and inference this project is dead, so the marker-position round trip is pinned by
tests.

---

## Benchmarks

- [`docs/benchmarks.md`](docs/benchmarks.md) — **all v0.1 metrics** (3 seeds, calibrated
  and not, state ablation, position bias)
- [`docs/baseline_ja.md`](docs/baseline_ja.md) — existing models measured in Japanese
- [`docs/gate_a.md`](docs/gate_a.md) — environment and throughput
- [`docs/licenses.md`](docs/licenses.md) — first-hand licence checks

All with reproduction commands and raw JSON.

---

## Limits (honestly)

- **These are means ± standard deviations over 3 seeds** (0 / 1 / 2). But **`score`
  varies a lot across seeds**: acc is 0.663 / 0.800 / 0.827 (SD 0.088). The mean 0.763
  **is not any seed's measured value**. **The published weights are seed 0**, whose
  figures are score acc 0.663 / RPS 0.117.
- **`bool` under-predicts true.** Mean P(true) is 0.125 against a gold positive rate of
  0.297. AUROC 0.789 means **the ranking works**, but **the threshold sits in the wrong
  place** — taking the argmax directly will miss true cases. **Temperature scaling does
  not fix this** (0.170 after calibration). Choose a threshold against your own prior.
- **Accuracy drops on 4-level `score`.** Condition E of the position-bias probe
  (4 levels) scores acc 0.427, clearly below the 3-level conditions A-D (0.697-0.788).
  Do not extrapolate K≥4 from K=3.
- **Calibration makes `score` worse.** choice ECE 0.147→0.092 and bool ECE 0.202→0.129
  improve, but **score RPS goes 0.090→0.149**. The default is uncalibrated.
  `temperatures.json` ships with the model, but **refit it on your own validation set**.
- **Evaluation covers only `bench_ja`'s 3 schemas** (4-way department routing /
  3-level urgency / churn suggestion), 300 items in one domain. Performance on other
  tasks is not measured.
- **Beating Laya on `choice` was never a goal** (the prior measurement already showed
  it usable). It is reported, not optimised for.
- **Training data is synthetic only** (**21 domains**, label-conditioned generation):

  | tier | count |
  |---|---|
  | documents generated | **4,833** |
  | labelled (document, question) pairs | **31,243** |
  | views after schema expansion | **79,552** |

  **Do not read "views" as "examples."** The same document with the same label appears
  several times under different schema surface forms. The independent-case count is
  the "pairs" row. The generator is a single model
  (`qwen3:30b-a3b-instruct-2507-q4_K_M`), so its vocabulary and phrasing habits run
  through the whole corpus.
- **Documents from `bench_ja`'s own domain are present in training** (free text from a
  business enquiry form). `bench_ja` measures generalisation to an **unseen schema**,
  not to an unseen domain. Its three schemas never appear in training.
- **`bench_ja` is synthetic too.** It was generated by the same model used for the
  LLM-as-classifier baseline, so that baseline is an upper-bound reference rather than
  a fair comparison.
- **Latency scales with the number of questions.** v0.1 is joint encoding, so **the
  state is re-encoded per question** and an N-question request runs the backbone N
  times. The original claim that the state is encoded once per request is withdrawn
  ([`docs/architecture.md`](docs/architecture.md) §1.2). Question-side caching does not
  apply either.
- **Position-dependent questions and long states are not validated.** The backbone's
  `local_attention` is 128. The held-out position-dependent attribute "does it end on
  a question?" reached only **AUROC 0.688**, clearly below the meaning-based attributes
  (implicit intent 0.786, explicit request 0.995). Training states average 134 tokens
  (p95 237), and **performance above 300 tokens is not measured**.
- **Probabilities are not calibrated unless you pass temperatures.**
- **Not built**: act/escalate heads (no definable training signal), RLCD (stage 3),
  ONNX/TensorRT.
- **FlashAttention-2 is not used.** It would not build on this machine (CUDA Toolkit
  13.1 against torch's 12.8). It runs on `sdpa` with length bucketing.
- **No RoPE inside the head.** Attention weights copied from the backbone run without
  the positional signal they were trained beside. Reasonable as an initialisation, but
  the ablation has not been run.

## On TypeSafe Jev

TypeSafe's terms (Master Customer Agreement 2.3(b)) prohibit using the service and its
outputs to develop a competing product, so **this project has not measured Jev.**
**There is no code in this repository that calls Jev.**

---

## Development

```bash
uv venv --python 3.11
uv sync --extra dev --extra bench
cp .env.example .env
uv run pytest
uv run ruff check .
```

Reproduction steps (from generating `bench_ja` to measuring the baselines) are at the
top of [`docs/baseline_ja.md`](docs/baseline_ja.md).

The `SOKUDAN_SPEC.md §…` citations in docstrings refer to
[`SOKUDAN_SPEC.md`](SOKUDAN_SPEC.md). **That is the design as it stood at the start of
the sprint on 2026-09-20 and does not match the current implementation** (§6.1 / §6.2,
separate encoding, is withdrawn). It is kept so the decisions — and the mistakes — can
be traced. [`docs/architecture.md`](docs/architecture.md) is authoritative now.

## Licence

Apache-2.0. The backbone `sbintuitions/modernbert-ja-310m` is MIT (see `NOTICE`).
