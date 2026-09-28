# Using `choice` questions well

Five practical notes for `choice` questions, written after a report in a public X thread from a user who put sokudan in front of an accounting-category suggestion step. Every number below was measured on this repository's data, with the v0.2.1 weights (the same as v0.2); `choice` probabilities are raw (not calibrated) in v0.2.1. Sources:

- `bench_ja` / `bench_en`: [`docs/benchmarks.md`](benchmarks.md) §11. This is department routing: 4 options, the last being "その他 / Other: none of the above".
- The catch-all probe: [`docs/probe_catch_all.md`](probe_catch_all.md). It covers 90 expense descriptions against ten accounting categories plus "その他: 上記以外". The set was written by one author and is small; the numbers there describe that set only.
- The calibration re-check: [`docs/calibration.md`](calibration.md) §11–§12.

None of these is fixed automatically at inference time. The right thresholds depend on your data and on what an error costs you.

## 1. Give every option a description

On `bench_ja` department routing, emptying the descriptions (labels only: 請求 / 技術 / 営業 / その他) dropped the result sharply:

| | acc | top-2 | median P(top1) |
|---|---|---|---|
| With descriptions | 0.880 | 0.963 | 0.997 |
| Labels only | **0.537** | 0.747 | 0.546 |

With the catch-all probe's accounting categories, the labels-only condition barely moved: 0.852 → 0.843 on the 80 states that have a category. Those labels (旅費交通費, 水道光熱費, ...) already say what they cover. Short names such as department names do not. A one-line description is cheap, and it was the largest single effect we measured.

## 2. Catch-all options ("その他", "Other", "none of the above")

The model rarely **picks** a catch-all by argmax, even when it is right:

- `bench_ja`: "その他" recall is **0.289** (11 of 38). `bench_en`: 0.354 (17 of 48). Misses go mostly to 技術 / Technical.
  - When it does pick "その他", it is usually right: precision 0.846 / 0.810.
- Catch-all probe, 10 categories + "その他: 上記以外", on the 10 states that fit no category:
  - chosen **0.295** of the time with random option orders;
  - chosen **0.155** with "その他" fixed last;
  - chosen **0.055** with "その他" last and no descriptions on any option.
- Missed states go to the listed category that looks closest. In the probe: rent → utilities (20 of 20), revenue stamps → communication (20 of 20).

The probability still **ranks** well. On the probe, P(その他) separates "the answer is その他" from the rest with AUROC 0.94 (10 categories) and 0.91 (5 categories). So:

**Read P(catch-all) and apply your own threshold instead of the argmax:**

```python
answer = agent.predict(state, questions)["answers"]["account"]
p_other = answer["probabilities"]["その他"]
label = "その他" if p_other >= THRESHOLD else answer["choice"]
# choose THRESHOLD on a few hundred of your own labelled states:
# recall on catch-all states vs. how often other states are pulled into it
```

The same threshold behaves very differently across data sets, so it has to be chosen on your own labelled examples. The table below is post hoc and in-sample, with the catch-all option last:

| Set | P ≥ 0.05: recall / false-positive rate | P ≥ 0.1 | P ≥ 0.2 |
|---|---|---|---|
| `bench_ja` (38 catch-all items) | 0.500 / 0.008 | 0.395 / 0.008 | 0.368 / 0.008 |
| `bench_en` (48) | 0.812 / 0.293 | 0.771 / 0.207 | 0.479 / 0.112 |
| Probe, 10 categories + その他 | 0.810 / 0.109 | 0.630 / 0.026 | 0.160 / 0.003 |
| Probe, 5 categories + その他 | 0.892 / 0.215 | 0.771 / 0.101 | 0.565 / 0.036 |

**Do not put the catch-all first or last.** In the probe with 5 categories + その他, P(その他) on the states whose answer was その他 was:

- 0.218 in the first slot;
- 0.213 in the last slot;
- 0.37–0.42 in the middle slots.

On `bench_ja` the same pattern is small: recall by the catch-all's slot is 0.303 / 0.325 / 0.346 / 0.307.

**Give the catch-all a description as well.** In the probe, with the catch-all last, it was chosen 0.155 of the time with "上記以外" and 0.055 without any descriptions.

## 3. Option position: small for ordinary options, visible for catch-alls and longer scales

- **Ordinary options:** position hardly matters on our benchmarks.
  - `bench_ja` department routing asked in all 24 orders: each slot's option was chosen at rates of 0.243 / 0.251 / 0.256 / 0.249, against 0.25 if order did not matter.
  - The answer changed across the 24 orders for 8.7% of the items.
  - Held-out states with 3 options in all 6 orders: 0.289 / 0.328 / 0.383. The first slot is chosen least, the last most.
- **Catch-all options:** position does matter; they are chosen less at either end (section 2).
- **Ordinal `score` questions:** position matters with 4 or more levels. On a 4-level `score`, the first level was chosen for 5 of 300 items ([`docs/benchmarks.md`](benchmarks.md) §10.3). Do not extrapolate from 3 levels.

## 4. For review workflows, show the top two

When a person confirms the suggestion, show the two most probable options, not one:

| | top-1 | top-2 |
|---|---|---|
| `bench_ja`, all | 0.880 | **0.963** |
| `bench_ja`, "その他" items | 0.289 | 0.763 |
| `bench_en`, all | 0.872 | **0.983** |
| `bench_en`, "Other" items | 0.354 | 0.917 |

## 5. Check the probabilities before showing confidence as a number

`choice` probabilities are raw in v0.2.1 and stay raw in v0.3.0:

- One temperature (fitted on validation data) lowered the `choice` ECE on `bench_ja`: 0.088 → 0.066, with 15 equal-mass bins.
- The same temperature raised it on `bench_en`: 0.091 → 0.228, because `bench_en` was already close to calibrated.
- The pre-registered rule required an improvement on both, so `choice` is not calibrated by default.

What that means for a UI:

- On `bench_ja`, 224 of 300 answers had P(top1) ≥ 0.99, and **14 of those (6.25%) were wrong**. A displayed "99%" is not 99%.
- The `/v1/systemone` `confidence` for `choice` is (p_max − 1/K)/(1 − 1/K). Its median on `bench_ja` is 0.996.

Before you show a confidence value, look at the distribution of P(top1) and at the error rate per band on your own labelled data. The ranking is still useful for ordering a review queue.

## The same pattern as `bool`

`bool` shows the same shape as the catch-all: the ranking works, the threshold is off.

- On `bench_ja`, the mean P(true) is 0.133 against a true rate of 0.297, or 0.198 with the default bool calibration. The AUROC is 0.844.
- The 0.5 threshold under-calls "true" in the same way the argmax under-calls a catch-all.
- In both cases, set the threshold from your own prior or labelled data (README, Limits).

See also [`docs/probe_catch_all.md`](probe_catch_all.md) for the full probe: its conditions, per-state results and near-miss confusions (消耗品費 / 事務用品費, 会議費 / 交際費).
