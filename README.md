# sokudan (即断)

**A Japanese System One decision model: send text and typed questions, get typed answers with probabilities, without generating a single token.**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/hiroki-abe-58/sokudan/blob/main/LICENSE)
[![Hugging Face: GeneLab/sokudan-ja-310m](https://img.shields.io/badge/%F0%9F%A4%97%20model-GeneLab%2Fsokudan--ja--310m-yellow)](https://huggingface.co/GeneLab/sokudan-ja-310m)
[![Release: v0.3.0](https://img.shields.io/badge/release-v0.3.0-green)](https://github.com/hiroki-abe-58/sokudan/releases/tag/v0.3.0)
[![PyPI: sokudan](https://img.shields.io/pypi/v/sokudan)](https://pypi.org/project/sokudan/)

*日本語: [README_ja.md](https://github.com/hiroki-abe-58/sokudan/blob/main/README_ja.md)*

## Why

- **No generation.** Answers are read off a decision head's logits. There is no text to parse, no JSON to validate, and no way to answer outside the options you sent.
- **Japanese-native.** The backbone is [`sbintuitions/modernbert-ja-310m`](https://huggingface.co/sbintuitions/modernbert-ja-310m), and the model is trained and evaluated on Japanese. On Japanese business messages it beats `laya-multilingual` on every metric below.
- **One forward pass per question.** No decoding loop and no retries. 314.6M parameters, so it also runs on a CPU.

## Quickstart (30 seconds)

Python 3.11.

```bash
pip install sokudan
```

Development version (the `main` branch): `pip install git+https://github.com/hiroki-abe-58/sokudan.git`.

What `pip install sokudan` brings depends on the platform (v0.3.0):

| platform | array library installed | `sokudan.load()` runs on |
|---|---|---|
| Apple silicon, macOS 14 or later | MLX (`mlx>=0.32.2,<0.33`); **no torch** | MLX, float16 |
| Linux, Windows, Intel Mac, Apple silicon on macOS 13 | torch | torch: cuda, then mps, then cpu |

- On an M1 Max, one 5-option choice takes about 13 ms with MLX, 1.4–1.7x faster than torch on MPS; see [`docs/mlx.md`](https://github.com/hiroki-abe-58/sokudan/blob/main/docs/mlx.md#speed-on-an-idle-machine-2026-10-01).
- `pip install "sokudan[torch]"` adds torch on any platform (for `backend="torch"` on Apple silicon, and for the training and evaluation scripts).
- To use a GPU on Windows / Linux, install a CUDA build of torch first (for example `pip install torch --index-url https://download.pytorch.org/whl/cu128`), then install sokudan.
- `pip install "sokudan[mlx]"` names MLX explicitly (same platforms as above). `pip install "sokudan[serve]"` adds the server.

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

**Backends (v0.3.0).** `sokudan.load(..., backend="auto" | "mlx" | "torch", dtype=None | "float16" | "float32")`. `auto` tries MLX (Apple silicon with `mlx` installed), then torch on mps, cuda and cpu; each candidate answers one short self-check request, and a failure is a warning followed by the next candidate. An explicit `backend` or `device` does not fall back. `agent.backend` says which one is in use. MLX runs float16 by default (the heads stay float32); on `bench_ja`, MLX float32 and float16 give the same four metrics as torch to three decimals. Details and measurements: [`docs/mlx.md`](https://github.com/hiroki-abe-58/sokudan/blob/main/docs/mlx.md).

**Calibration (since v0.2.1).** `load` applies one temperature to `noul`/`bool` answers by default (the `calibration.json` shipped with the weights); `choice` and `score` probabilities are raw. `sokudan.load(..., temperatures=None)` turns it off. Each result says which answers were calibrated (`calibrated`, `calibrated_answers`).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hiroki-abe-58/sokudan/blob/main/notebooks/sokudan_quickstart.ipynb) The same steps in a notebook on a free CPU runtime: the three question types, calibration on and off, and `sokudan serve` called with `curl` ([`notebooks/sokudan_quickstart.ipynb`](https://github.com/hiroki-abe-58/sokudan/blob/main/notebooks/sokudan_quickstart.ipynb)).

## `bench_ja`: v0.2 against `laya-multilingual`

300 Japanese business messages, three unseen schemas (4-way department routing, 3-level urgency, churn suggestion). Uncalibrated. Each model row is a single run.

| | choice acc | score RPS↓ | bool acc | bool AUROC |
|---|---|---|---|---|
| **sokudan-ja-310m v0.2** | **0.880** | **0.075** | **0.780** | **0.844** |
| `laya-multilingual` (ja) | 0.747 | 0.232 | 0.543 | 0.523 |
| majority class | 0.380 | 0.197 | 0.703 | — |
| random | 0.253 | 0.201 | 0.513 | — |

- `bench_ja` and `bench_en` ship in this repo (`data/bench_ja.jsonl`, `data/bench_en.jsonl`) under CC BY 4.0, separate from the code's Apache-2.0. Please use them for evaluation, not training (a request, not a licence restriction).
- v0.2's score accuracy is 0.817. Every metric, v0.1's three-seed figures and `bench_en` are in the [model card](https://huggingface.co/GeneLab/sokudan-ja-310m) and [`docs/benchmarks.md`](https://github.com/hiroki-abe-58/sokudan/blob/main/docs/benchmarks.md).

## How v0.2 was made

- **A model soup of eight seeds.** Seeds 0-7, each trained exactly as v0.1 was, averaged tensor by tensor in float32 (heads included). Architecture, data and inference code are v0.1's; inference costs one model.
- **Chosen by a rule fixed in advance.** Four candidate soups; the rule (largest held-out M1m, no guardrail regressions) was committed before any `bench_ja` run ([`docs/release_candidate.md`](https://github.com/hiroki-abe-58/sokudan/blob/main/docs/release_candidate.md), [`docs/research_protocol.md`](https://github.com/hiroki-abe-58/sokudan/blob/main/docs/research_protocol.md)).
- **`bench_ja` was measured once**, after the release rule was committed: bool AUROC +0.01 or better, score RPS −0.005 or better, and choice / bool / score accuracy within −0.01 of v0.1. All were met; bool accuracy is the one that went down (−0.008).
- **Reproducing it.** `scripts/make_soup.py` rebuilds the soup from the eight checkpoints and checks every SHA-256. The published `model.safetensors` is `8750a833…5b965` (full hashes in the model card). The member checkpoints are not distributed, and retraining does not reproduce them bit for bit.
- v0.1 remains available: `sokudan.load("GeneLab/sokudan-ja-310m@v0.1")`.

## Limits

- **Position sensitivity.** On a four-level `score` (condition E of the position probe) v0.2 chose the first option for 5 of 300 items, accuracy 0.347. Do not extrapolate from three levels to four or more. On held-out states the first slot is still slightly disfavoured: first-slot rate over all orders 0.239 (score) and 0.289 (choice), against 1/3 if order did not matter. Three kinds of fix were tried (averaging over orders at inference, a permutation-KL term in training, shuffling option order in the training data); each moved some position checks but none kept held-out accuracy non-inferior, so none shipped (model card, Limits). A catch-all option ("その他", "Other") is chosen less when it sits first or last; ordinary options hardly depend on position on `bench_ja` (all 24 orders of the four departments: each slot chosen 0.243–0.256 of the time).
- **Catch-all options are under-chosen.** The argmax rarely picks "その他 / Other" even when it is right:
  - On `bench_ja`, "その他" recall is 0.289 (11 of 38; `bench_en` 0.354). Misses go mostly to 技術 / Technical, while precision is 0.846.
  - In a small probe (90 expense descriptions by one author, 10 accounting categories + "その他: 上記以外"), it was chosen for 0.295 of the states that fit no category with random option orders, and 0.155 with "その他" last.
  - P(その他) still ranks those states well (AUROC 0.94). Read P(catch-all) and set your own threshold; do not put the catch-all first or last. Guide: [`docs/choice_guidance.md`](https://github.com/hiroki-abe-58/sokudan/blob/main/docs/choice_guidance.md).
  - Near-miss pairs that people also split (消耗品費 / 事務用品費; 会議費 / 交際費 for meals with clients) are not separated either.
- **`bool` under-predicts true.** Mean P(true) 0.133 against a gold rate of 0.297 (0.198 with the default bool calibration). The ranking works (AUROC 0.844); set the threshold from your own prior. Calibration does not move the 0.5 threshold.
- **`bool` is calibrated by default** (one temperature, ECE 0.181 → 0.105 on `bench_ja`); **`score` and `choice` are left raw** because the score temperature fitted on validation made bench RPS worse (0.075 → 0.132). Details: [`docs/calibration.md`](https://github.com/hiroki-abe-58/sokudan/blob/main/docs/calibration.md). `choice` was re-checked for v0.3.0 and stays raw: the validation temperature lowered `choice` ECE on `bench_ja` (0.088 → 0.066) but raised it on `bench_en` (0.091 → 0.228).
- **Latency grows with the number of questions.** The state is re-encoded for every question.
- **Long states lose accuracy.** The backbone's local attention window is 128 tokens; training states average 134 tokens (p95 237).
- **Japanese only.** On `bench_en`, bool accuracy is 0.690, level with the majority class (0.683).
- **Synthetic data.** Training data (21 domains, 4,833 documents, 31,243 labelled pairs) and `bench_ja` are both generated by one LLM. Not measured on other tasks. Do not use it to replace a human decision about hiring, credit, discipline, medicine or law.

The full list is in [README_ja.md](https://github.com/hiroki-abe-58/sokudan/blob/main/README_ja.md#limits正直に) and the model card.

## `/v1/systemone`-compatible server

```bash
pip install "sokudan[serve]"
sokudan serve --port 8000
```

Development version: `pip install "sokudan[serve] @ git+https://github.com/hiroki-abe-58/sokudan.git"`.

`POST /v1/systemone` takes the same request and returns the same answer shapes as TypeSafe's public API reference, so a client written for that format can point its base URL at `http://127.0.0.1:8000`. `GET /health` reports the loaded model, calibration, and how a JSON state is rendered.
Built from public documentation and the examples in open implementations' READMEs; not affiliated with or endorsed by TypeSafe AI, and this repository never calls their service.
Guide: [`docs/serving.md`](https://github.com/hiroki-abe-58/sokudan/blob/main/docs/serving.md). Field-by-field table: [`docs/systemone_wire_format.md`](https://github.com/hiroki-abe-58/sokudan/blob/main/docs/systemone_wire_format.md).

## JavaScript, Homebrew and containers

Version 0.3.0 is available on [npm](https://www.npmjs.com/package/sokudan)
as a TypeScript HTTP SDK and CLI launcher, through our Homebrew tap, and as a
[Linux amd64 CPU container](https://github.com/hiroki-abe-58/sokudan/pkgs/container/sokudan). See the
[distribution guide](https://github.com/hiroki-abe-58/sokudan/blob/main/docs/distribution.md)
for setup, publication status and release steps.

## Used by

- Expense account suggestion in an accounting app (PoC): given a purchase description, sokudan ranks 1–3 candidate accounts for a human to confirm, running on an M1 Max. [Thread on X](https://x.com/t28k2/status/2104322335671206306)

## More

- [README_ja.md](https://github.com/hiroki-abe-58/sokudan/blob/main/README_ja.md): the Japanese README, with the design notes (joint encoding, the dynamic-K cumulative link) and every limit.
- [`docs/architecture.md`](https://github.com/hiroki-abe-58/sokudan/blob/main/docs/architecture.md), [`docs/baseline_ja.md`](https://github.com/hiroki-abe-58/sokudan/blob/main/docs/baseline_ja.md), [`docs/baseline_lev.md`](https://github.com/hiroki-abe-58/sokudan/blob/main/docs/baseline_lev.md) (against lev on the same machine and harness), [`CHANGELOG.md`](https://github.com/hiroki-abe-58/sokudan/blob/main/CHANGELOG.md).
- Development: `uv sync --extra dev`, `uv run pytest`, `uv run ruff check .`. From v0.3.0 the dependencies only data building, training and evaluation use (`datasets`, `fugashi`, `unidic-lite`, `matplotlib`) are in the `train` extra; `dev` includes `sokudan[train]`, so `uv sync --extra dev` still installs them (`uv sync --extra train` / `pip install "sokudan[train]"` without the dev tools). On Apple silicon with macOS 14+, add `--extra torch` for the tests and training that need torch.

Apache-2.0. The backbone `sbintuitions/modernbert-ja-310m` is MIT (see `NOTICE`).
