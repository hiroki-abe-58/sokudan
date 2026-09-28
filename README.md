# sokudan (即断)

**A Japanese System One decision model: send text and typed questions, get typed answers with probabilities, without generating a single token.**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/hiroki-abe-58/sokudan/blob/main/LICENSE)
[![Hugging Face: GeneLab/sokudan-ja-310m](https://img.shields.io/badge/%F0%9F%A4%97%20model-GeneLab%2Fsokudan--ja--310m-yellow)](https://huggingface.co/GeneLab/sokudan-ja-310m)
[![Release: v0.2.1](https://img.shields.io/badge/release-v0.2.1-green)](https://github.com/hiroki-abe-58/sokudan/releases/tag/v0.2.1)

*日本語: [README_ja.md](https://github.com/hiroki-abe-58/sokudan/blob/main/README_ja.md)*

## Why

- **No generation.** Answers are read off a decision head's logits. There is no text to parse, no JSON to validate, and no way to answer outside the options you sent.
- **Japanese-native.** The backbone is [`sbintuitions/modernbert-ja-310m`](https://huggingface.co/sbintuitions/modernbert-ja-310m), and the model is trained and evaluated on Japanese. On Japanese business messages it beats `laya-multilingual` on every metric below.
- **One forward pass per question.** No decoding loop and no retries. 314.6M parameters, so it also runs on a CPU.

## Quickstart (30 seconds)

Python 3.11.

```bash
pip install git+https://github.com/hiroki-abe-58/sokudan.git
```

```python
import sokudan

agent = sokudan.load("GeneLab/sokudan-ja-310m")
questions = {"department": {"type": "choice", "instructions": "この問い合わせはどの部署が担当すべきか",
                            "criteria": {"請求": "支払い・返金", "技術": "不具合・障害", "営業": "料金・新規契約", "その他": "上記以外"}}}
result = agent.predict("先月の請求で同じ金額が二回引き落とされています。至急ご確認ください。", questions)
print(result["answers"]["department"]["choice"])
```

Expected output:

```text
請求
```

`result["answers"]["department"]["probabilities"]` holds the full distribution over the four options.
Question types are `choice`, `score` (an ordinal scale) and `noul` (P(yes); `bool` is an alias). The schema is free per request; no retraining.

**Pass the state as a string.** A dict is rendered as `key: value` lines, which is not the input the model was trained on, and the output changes.

**Calibration (v0.2.1).** `load` applies one temperature to `noul`/`bool` answers by default (the `calibration.json` shipped with the weights); `choice` and `score` probabilities are raw. `sokudan.load(..., temperatures=None)` turns it off. Each result says which answers were calibrated (`calibrated`, `calibrated_answers`).

## `bench_ja`: v0.2 against `laya-multilingual`

300 Japanese business messages, three unseen schemas (4-way department routing, 3-level urgency, churn suggestion). Uncalibrated. Each model row is a single run.

| | choice acc | score RPS↓ | bool acc | bool AUROC |
|---|---|---|---|---|
| **sokudan-ja-310m v0.2** | **0.880** | **0.075** | **0.780** | **0.844** |
| `laya-multilingual` (ja) | 0.747 | 0.232 | 0.543 | 0.523 |
| majority class | 0.380 | 0.197 | 0.703 | — |
| random | 0.253 | 0.201 | 0.513 | — |

- `bench_ja` ships in this repo (`data/bench_ja.jsonl`, CC BY 4.0). Please use it for evaluation, not training.
- v0.2's score accuracy is 0.817. Every metric, v0.1's three-seed figures and `bench_en` are in the [model card](https://huggingface.co/GeneLab/sokudan-ja-310m) and [`docs/benchmarks.md`](https://github.com/hiroki-abe-58/sokudan/blob/main/docs/benchmarks.md).

## How v0.2 was made

- **A model soup of eight seeds.** Seeds 0-7, each trained exactly as v0.1 was, averaged tensor by tensor in float32 (heads included). Architecture, data and inference code are v0.1's; inference costs one model.
- **Chosen by a rule fixed in advance.** Four candidate soups; the rule (largest held-out M1m, no guardrail regressions) was committed before any `bench_ja` run ([`docs/release_candidate.md`](https://github.com/hiroki-abe-58/sokudan/blob/main/docs/release_candidate.md), [`docs/research_protocol.md`](https://github.com/hiroki-abe-58/sokudan/blob/main/docs/research_protocol.md)).
- **`bench_ja` was measured once**, after the release rule was committed: bool AUROC +0.01 or better, score RPS −0.005 or better, and choice / bool / score accuracy within −0.01 of v0.1. All were met; bool accuracy is the one that went down (−0.008).
- **Reproducing it.** `scripts/make_soup.py` rebuilds the soup from the eight checkpoints and checks every SHA-256. The published `model.safetensors` is `8750a833…5b965` (full hashes in the model card). The member checkpoints are not distributed, and retraining does not reproduce them bit for bit.
- v0.1 remains available: `sokudan.load("GeneLab/sokudan-ja-310m@v0.1")`.

## Limits

- **Position sensitivity.** On a four-level `score` (condition E of the position probe) v0.2 chose the first option for 5 of 300 items, accuracy 0.347. Do not extrapolate from three levels to four or more. On held-out states the first slot is still slightly disfavoured: first-slot rate over all orders 0.239 (score) and 0.289 (choice), against 1/3 if order did not matter. Three kinds of fix were tried (averaging over orders at inference, a permutation-KL term in training, shuffling option order in the training data); each moved some position checks but none kept held-out accuracy non-inferior, so none shipped (model card, Limits).
- **`bool` under-predicts true.** Mean P(true) 0.133 against a gold rate of 0.297 (0.198 with the default bool calibration). The ranking works (AUROC 0.844); set the threshold from your own prior. Calibration does not move the 0.5 threshold.
- **`bool` is calibrated by default** (one temperature, ECE 0.181 → 0.105 on `bench_ja`); **`score` and `choice` are left raw** because the score temperature fitted on validation made bench RPS worse (0.075 → 0.132). Details: [`docs/calibration.md`](https://github.com/hiroki-abe-58/sokudan/blob/main/docs/calibration.md).
- **Latency grows with the number of questions.** The state is re-encoded for every question.
- **Long states lose accuracy.** The backbone's local attention window is 128 tokens; training states average 134 tokens (p95 237).
- **Japanese only.** On `bench_en`, bool accuracy is 0.690, level with the majority class (0.683).
- **Synthetic data.** Training data (21 domains, 4,833 documents, 31,243 labelled pairs) and `bench_ja` are both generated by one LLM. Not measured on other tasks. Do not use it to replace a human decision about hiring, credit, discipline, medicine or law.

The full list is in [README_ja.md](https://github.com/hiroki-abe-58/sokudan/blob/main/README_ja.md#limits正直に) and the model card.

## `/v1/systemone`-compatible server

```bash
pip install "sokudan[serve] @ git+https://github.com/hiroki-abe-58/sokudan.git"
sokudan serve --port 8000
```

`POST /v1/systemone` takes the same request and returns the same answer shapes as TypeSafe's public API reference, so a client written for that format can point its base URL at `http://127.0.0.1:8000`. `GET /health` reports the loaded model, calibration, and how a JSON state is rendered.
Built from public documentation and the examples in open implementations' READMEs; not affiliated with or endorsed by TypeSafe AI, and this repository never calls their service.
Guide: [`docs/serving.md`](https://github.com/hiroki-abe-58/sokudan/blob/main/docs/serving.md). Field-by-field table: [`docs/systemone_wire_format.md`](https://github.com/hiroki-abe-58/sokudan/blob/main/docs/systemone_wire_format.md).

## More

- [README_ja.md](https://github.com/hiroki-abe-58/sokudan/blob/main/README_ja.md): the Japanese README, with the design notes (joint encoding, the dynamic-K cumulative link) and every limit.
- [`docs/architecture.md`](https://github.com/hiroki-abe-58/sokudan/blob/main/docs/architecture.md), [`docs/baseline_ja.md`](https://github.com/hiroki-abe-58/sokudan/blob/main/docs/baseline_ja.md), [`docs/baseline_lev.md`](https://github.com/hiroki-abe-58/sokudan/blob/main/docs/baseline_lev.md) (against lev on the same machine and harness), [`CHANGELOG.md`](https://github.com/hiroki-abe-58/sokudan/blob/main/CHANGELOG.md).
- Development: `uv sync --extra dev`, `uv run pytest`, `uv run ruff check .`

Apache-2.0. The backbone `sbintuitions/modernbert-ja-310m` is MIT (see `NOTICE`).
