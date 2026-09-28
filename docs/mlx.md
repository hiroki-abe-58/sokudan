# MLX backend (Apple silicon)

Japanese: [`mlx_ja.md`](mlx_ja.md). Every number here was measured on one machine (below);
anything not measured says so.

`sokudan.load` can run the model with [MLX](https://github.com/ml-explore/mlx) on Apple
silicon instead of PyTorch. The answers come from the same weights through a port of the
same forward pass; `predict` returns the same response shape.

## Install

```bash
pip install sokudan
```

- From v0.3.0, on Apple silicon with macOS 14 or later (Darwin 23+), the base install
  brings `mlx>=0.32.2,<0.33` and **not** torch; everywhere else it brings torch and not
  MLX (the markers are fixed by `tests/test_packaging.py`). `pip install "sokudan[torch]"`
  adds torch anywhere; `"sokudan[mlx]"` names MLX explicitly, with the same marker.
- MLX 0.32.2 is the only version this backend has been run with.
- A clean install of the 0.3.0 wheel into a new Python 3.11 environment on the machine
  below: 38 packages, `mlx` and `mlx-metal` 0.32.2, no torch, `site-packages` 393 MB (the
  largest distributions: mlx-metal 155 MB, transformers 103 MB, numpy 33 MB). `sokudan.load()`
  chose MLX float16 and the Quickstart `predict` ran; torch was not imported. With
  `[torch]`: 44 packages, 1.1 GB; `load()` still chose MLX and `backend="torch"` chose mps.
- `datasets`, `fugashi`, `unidic-lite` and `matplotlib` are in the `train` extra: none of
  them is imported by load + predict (MLX or torch) or by `sokudan serve`. Before they
  moved, the same install was 66 packages and 934 MB.
- `import sokudan` does not import torch. When torch is installed, loading on MLX can
  still import it: transformers (used for the tokenizer and the backbone config) imports
  an installed torch.

## Use

```python
import sokudan

agent = sokudan.load("GeneLab/sokudan-ja-310m")          # backend="auto"
print(agent.backend)                                      # <MLXBackend mlx gpu float16>
agent = sokudan.load("GeneLab/sokudan-ja-310m", backend="mlx", dtype="float32")
agent = sokudan.load("GeneLab/sokudan-ja-310m", backend="torch")   # the PyTorch model
```

### `backend`, `device`, `dtype`

| argument | values | behaviour |
|---|---|---|
| `backend` | `"auto"` (default) | tries, in order: MLX (when `mlx` imports and the machine is Apple silicon), torch on `mps`, torch on `cuda`, torch on `cpu` |
| | `"mlx"` / `"torch"` | that backend only |
| `device` | `"auto"` (default) | with the torch backend: `cuda`, then `mps`, then `cpu` |
| | `"cpu"` / `"cuda"` / `"mps"` | a torch device, used as given; with `backend="auto"` this selects torch on that device |
| `dtype` | `None` (default) | the backend's default: MLX `float16`, torch `float32` |
| | `"float16"` / `"float32"` | MLX backbone precision. torch accepts `float32` only |

- **Self-check and fallback.** Each candidate is loaded and then answers one short request
  (a choice, a score and a bool question). With `backend="auto"` and `device="auto"`, an
  exception there -- including a non-finite probability -- is reported as a
  `RuntimeWarning` and the next candidate is tried; if none loads, `load` raises
  `RuntimeError` listing every failure. An explicit `backend` or `device` does not fall
  back: the exception is raised.
- `agent.backend` is the backend in use (`agent.backend.name`, `agent.device`,
  `agent.dtype`).
- `dtype` applies to the backbone. The heads (the marker scorer and the ordinal head, 5
  tensors) always run in float32.

### What the MLX backend runs

- The joint encoding (the published model) from a directory holding `model.safetensors`
  and `config.json` -- a Hub repo id or a local directory. A `.pt` file and the separate
  encoding need torch; with `backend="auto"` they fall through to torch with a warning.
- The weights are the published `model.safetensors`, read with `mx.load`: the 152 backbone
  tensors lose their `backbone.model.` prefix, the 5 head tensors keep their names
  (`tests/test_mlx_backend.py` fixes the table).
- Not implemented: quantized backbones (see the dtype table), `mx.compile`, a cache of
  encoded prompts.

### `sokudan serve`

`sokudan serve --backend auto|mlx|torch --dtype ...` passes both to `sokudan.load`
(default `auto`, so on Apple silicon with MLX installed it serves MLX float16; `/health`
reports `"device": "gpu"`). `--device cpu` / `mps` / `cuda` serves torch on that device.
The startup log names the choice, e.g.
`sokudan serve: loaded; backend=mlx device=gpu dtype=float16 calibrated=True`. In a clean
`sokudan[serve]` install, the `/v1/systemone` answer to the Quickstart request had the
same probabilities, choice, score and noul as `predict()` (difference 0).

## Agreement with torch

Parity set: 320 states / 630 questions, synthesised from the states in the README, the
Space demo and the tests (`scripts/mlx/parity_set.py`, seed 20260928; bool 218, choice 180
with K 2-8, score 232 with K 3-7; 1-3 questions per state; joint length 18-1024 tokens, 4
truncated). The reference is torch on the CPU in float32. Probabilities are the head's
raw ones (before temperatures), compared per question over every option
(`scripts/mlx/dump_probs.py`, `scripts/mlx/compare_probs.py`).

Gate for a dtype, per question type: argmax agreement >= 99.5%, mean absolute difference
<= 2e-3, max absolute difference <= 5e-2. float32 was held to argmax agreement on every
question and max abs <= 1e-3.

| dtype | type | argmax agree | mean abs | max abs | gate |
|---|---|---|---|---|---|
| float32 | choice | 180/180 (100.00%) | 3.20e-07 | 9.12e-06 | pass |
| float32 | score | 232/232 (100.00%) | 2.07e-07 | 2.62e-06 | pass |
| float32 | bool | 218/218 (100.00%) | 4.89e-07 | 8.11e-06 | pass |
| float16 | choice | 180/180 (100.00%) | 2.92e-04 | 5.62e-03 | pass |
| float16 | score | 231/232 (99.57%) | 1.81e-04 | 5.32e-03 | pass |
| float16 | bool | 218/218 (100.00%) | 4.18e-04 | 8.22e-03 | pass |
| 8-bit, group 64 | choice | 179/180 (99.44%) | 2.17e-03 | 5.79e-02 | fail (argmax, mean, max) |
| 8-bit, group 64 | score | 230/232 (99.14%) | 1.63e-03 | 1.48e-02 | fail (argmax) |
| 8-bit, group 64 | bool | 218/218 (100.00%) | 2.19e-03 | 2.74e-02 | fail (mean) |
| 8-bit, group 32 | choice | 179/180 (99.44%) | 1.97e-03 | 5.07e-02 | fail (argmax, max) |
| 8-bit, group 32 | score | 231/232 (99.57%) | 1.06e-03 | 1.19e-02 | pass |
| 8-bit, group 32 | bool | 218/218 (100.00%) | 1.57e-03 | 1.90e-02 | pass |
| 4-bit, group 64 | choice | 156/180 (86.67%) | 3.73e-02 | 6.78e-01 | fail (argmax, mean, max) |
| 4-bit, group 64 | score | 206/232 (88.79%) | 2.04e-02 | 3.00e-01 | fail (argmax, mean, max) |
| 4-bit, group 64 | bool | 206/218 (94.50%) | 5.18e-02 | 6.08e-01 | fail (argmax, mean, max) |
| 4-bit, group 32 | choice | 162/180 (90.00%) | 2.92e-02 | 5.86e-01 | fail (argmax, mean, max) |
| 4-bit, group 32 | score | 208/232 (89.66%) | 1.88e-02 | 2.89e-01 | fail (argmax, mean, max) |
| 4-bit, group 32 | bool | 209/218 (95.87%) | 3.19e-02 | 5.29e-01 | fail (argmax, mean, max) |

- float16 meets the gate and is the default. The one score question whose argmax differs
  is a near tie in torch (0.4110 vs 0.4103; MLX float16 0.4105 vs 0.4106).
- 8-bit and 4-bit are `nn.quantize` over the float16 backbone (every Linear and the token
  embedding). None meets the gate, so none is offered as a `dtype`.
- The float16 backbone has large single-value errors: on a 430-token input (the
  Quickstart state repeated, 3 questions) the largest hidden-state difference from torch
  is 4.71, at a non-marker position, in channel 578, where the torch value is -19.5. At
  the marker positions the difference is at most 0.035 (mean 8.1e-4), and the answers'
  probabilities differ by at most 2.2e-4 (argmax unchanged).

### `bench_ja`

`bench_ja` (300 items; 4-way department choice, 3-level urgency score, churn bool), run
once per configuration on 2026-09-28 at commit `b1fe755` with the existing procedure
(`scripts/run_baseline_ja.py --skip-laya --skip-llm --sokudan-checkpoint <snapshot 5f91a0d>
--sokudan-backend ... --sokudan-device ... --sokudan-dtype ...`): the training collator's
batches of 16 items per question, uncalibrated, the same metrics module as every other
row. torch 2.14.0 on the CPU, mlx 0.32.2.

| configuration | choice acc | score RPS | bool acc | bool AUROC |
|---|---|---|---|---|
| published v0.2 (torch, 2026-09-27) | 0.880 | 0.075 | 0.780 | 0.844 |
| torch cpu float32 | 0.880000 | 0.074509 | 0.780000 | 0.843868 |
| MLX float32 | 0.880000 | 0.074510 | 0.780000 | 0.843868 |
| MLX float16 | 0.880000 | 0.074512 | 0.780000 | 0.844454 |
| MLX float16 − torch | 0 | +0.000002 | 0 | +0.000586 |

- No item's choice or score argmax, and no bool answer's side of 0.5, differs between
  torch and either MLX configuration. Largest probability differences from torch: MLX
  float32 5.0e-06 (choice), 1.9e-06 (score), 1.1e-05 (bool P(true)); MLX float16 6.7e-03,
  1.9e-03, 4.9e-03.
- Rule fixed before the run: float16 stays the default unless an accuracy or the AUROC is
  more than 0.01 below torch, or RPS more than 0.005 above. Neither happened.

## Speed and memory

**Conditions.** Apple M1 Max (8 performance + 2 efficiency cores), 64 GB, macOS 15.6.1;
Python 3.11.16, torch 2.14.0 (8 threads), mlx 0.32.2; 2026-09-28 17:26-17:36. The machine
was not idle: the 1-minute load average was still 19.29 after the 15-minute wait for it to
fall below 4, and 14.2-19.3 at the start of every process (Adobe Illustrator and macOS
storage services were running). The power state was not recorded. Every number below was
measured under that load.

**Method** (`scripts/mlx/bench.py`). `agent.predict` end to end (encoding, forward, answers;
default calibration), one process per configuration and round: 3 warm-up calls, then 30
timed calls per cell; 3 rounds, the four configurations in rotated order each round.
Inputs: `short` = the Quickstart state (13 state tokens); `long` = the same sentence
repeated (390 state tokens; joint sequences of 430 / 414 tokens on average for q1 / q3);
`1k` = repeated to 949 state tokens (989 / 973). q1 = the department choice (4 options);
q3 = q1 + urgency (score, 3 levels) + refund (bool), i.e. 3 backbone rows.

Median ms per round (r1 / r2 / r3), p95 in parentheses:

| input / questions | torch cpu | torch mps | MLX float32 | MLX float16 |
|---|---|---|---|---|
| short / q1 | 88.7 / 85.6 / 85.2 (99.0 / 89.6 / 92.9) | 20.3 / 20.6 / 20.2 (21.0 / 22.2 / 20.8) | 11.3 / 11.3 / 11.2 (11.7 / 16.5 / 11.6) | 10.4 / 10.3 / 10.2 (13.3 / 10.7 / 10.6) |
| short / q3 | 147.9 / 147.3 / 147.5 (160.7 / 154.6 / 157.4) | 27.6 / 28.7 / 28.0 (28.5 / 29.6 / 28.7) | 20.1 / 20.0 / 20.4 (20.6 / 20.5 / 22.1) | 17.8 / 17.8 / 17.8 (18.3 / 18.1 / 18.2) |
| long / q1 | 271.6 / 258.2 / 268.5 (295.3 / 277.2 / 287.3) | 47.5 / 48.6 / 47.7 (49.2 / 51.0 / 49.0) | 46.6 / 46.6 / 46.8 (47.2 / 47.1 / 47.3) | 36.7 / 37.2 / 36.4 (37.7 / 37.9 / 37.7) |
| long / q3 | 666.2 / 645.8 / 671.9 (689.2 / 660.0 / 716.3) | 119.3 / 133.4 / 119.0 (168.2 / 164.8 / 171.3) | 165.2 / 171.9 / 161.3 (210.6 / 211.8 / 208.2) | 95.9 / 95.7 / 94.2 (116.2 / 114.9 / 118.7) |
| 1k / q1 | 591.0 / 575.0 / 580.0 (664.5 / 599.2 / 607.5) | 129.8 / 132.0 / 130.4 (140.5 / 137.7 / 135.4) | 137.7 / 137.1 / 136.4 (174.6 / 141.6 / 144.0) | 96.8 / 96.8 / 96.4 (104.2 / 101.5 / 100.3) |
| 1k / q3 | 1486.8 / 1484.0 / 1527.7 (1567.9 / 1549.1 / 1583.6) | 403.3 / 405.0 / 411.2 (455.1 / 432.8 / 440.1) | 414.5 / 401.9 / 405.3 (469.5 / 452.9 / 475.3) | 279.4 / 272.9 / 276.2 (294.2 / 284.1 / 293.1) |

- In `long / q3`, torch mps and MLX float16 run at about their median for the first ~20
  timed calls and then slow down (to 125-185 ms and 97-121 ms); MLX float32 starts at
  112-120 ms and then varies between 128 and 241 ms. The cause was not investigated.

Memory (MiB; per process, rounds 1 / 2 / 3). Peak RSS includes loading. MLX's
`mx.get_peak_memory()` is read once after loading and, after `mx.reset_peak_memory()`,
once after all the timed calls (identical in the three rounds):

| | peak RSS | `mx` peak after load | `mx` active after load | `mx` peak during predict |
|---|---|---|---|---|
| torch cpu | 2740 / 2743 / 2738 | - | - | - |
| torch mps | 2742 / 2745 / 2744 | - | - | - |
| MLX float32 | 1545 / 1563 / 1557 | 1380 | 1200 | 3390 |
| MLX float16 | 1565 / 1554 / 1566 | 1800 | 600 | 1700 |

- Load time (`sokudan.load`, weights already cached, including the self-check): torch cpu
  3.9-5.1 s, torch mps 4.7-5.1 s, MLX 2.2-2.8 s.
- `site-packages` (`du -sh`) of the two measurement environments: 1.5 G for the torch one
  (torch 2.14.0, transformers, sokudan's dependencies, plus pytest, ruff, fastapi and
  uvicorn), 717 M for the MLX one (mlx 0.32.2, transformers without torch, plus laya-mlx,
  mlx-embeddings, mlx-vlm, mlx-audio and pytest from the earlier survey). The size of a
  minimal `sokudan[mlx]` install was not measured.

## Source of the encoder

`sokudan/backends/mlx/modernbert.py` is `laya_mlx/model.py` lines 13-154
(`EncoderConfig` through `ModernBert`) from laya-mlx 0.2.0
(https://github.com/mizorewww/laya-mlx, PyPI wheel; file sha256 `24ef50d4…`), unmodified,
under the Apache License 2.0 (see `NOTICE`). It already uses `mx.fast` for attention,
RoPE and layer norm, so nothing was replaced. laya-mlx's decision head, prompt building
and agent are not included; sokudan's heads are `sokudan/backends/mlx/model.py`.
