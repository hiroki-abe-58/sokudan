"""Baselines for `bench_ja` (SOKUDAN_SPEC.md §4.2, §9).

Every baseline answers the **same 300 items with the same three questions** and
returns probability vectors in the same shape, so one metrics implementation
(`sokudan.calibration.metrics`) scores all of them.

A measurement detail that matters for honesty: the Laya API returns probabilities
rounded to 4 decimal places, so a confident miss can carry a literal `0.0` for the
gold class and send NLL to infinity. `floor_and_renormalise` applies the *same*
floor to every baseline, including sokudan later, and the floor is recorded in the
output. This is a limit of the measurement, not a property of any model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from sokudan.eval.bench_ja import DEPARTMENTS, URGENCY_LEVELS, BenchItem, bench_questions

# The Laya API rounds to 4 dp, so the smallest representable non-zero value is 1e-4.
# Half of that is the least arbitrary floor available.
PROB_FLOOR = 5e-5


def floor_and_renormalise(probs: np.ndarray, floor: float = PROB_FLOOR) -> np.ndarray:
    """Clip probabilities away from exact zero, then renormalise rows to sum to 1."""
    out = np.clip(np.asarray(probs, dtype=np.float64), floor, None)
    return out / out.sum(axis=1, keepdims=True)


@dataclass
class BaselineOutput:
    """Probabilities for each of the three bench_ja questions, plus timing."""

    name: str
    choice_probs: np.ndarray          # (N, 4) over DEPARTMENTS order
    score_probs: np.ndarray           # (N, 3) over URGENCY_LEVELS order
    bool_p_true: np.ndarray           # (N,)  P(churn == True)
    per_item_latency_s: list[float] = field(default_factory=list)
    parse_failures: int = 0
    parse_attempts: int = 0
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def parse_failure_rate(self) -> float | None:
        if self.parse_attempts == 0:
            return None
        return self.parse_failures / self.parse_attempts


class Baseline(Protocol):
    name: str

    def run(self, items: list[BenchItem]) -> BaselineOutput: ...


# --------------------------------------------------------------------------
# Trivial floors (§9 baselines 1 and 2)
# --------------------------------------------------------------------------

class RandomBaseline:
    """Uninformative but honest: ~uniform probabilities, uniformly random argmax.

    A plain `np.full(..., 1/K)` would be wrong to score. `argmax` breaks exact ties
    toward index 0, so a perfectly uniform baseline would "predict" the first option
    every time and score the base rate of that option -- indistinguishable from the
    majority-class baseline, which is not what a random baseline means.

    Adding a tiny per-item jitter (1e-6, far below the 4-decimal resolution any model
    here reports) makes the argmax uniformly random while leaving the probabilities
    uniform for calibration purposes. Accuracy lands near 1/K and ECE near 0, which
    is the honest description of a model that knows nothing and says so.
    """

    name = "ランダム"
    JITTER = 1e-6

    def __init__(self, seed: int = 20260920) -> None:
        self._seed = seed

    def _uniform_with_random_argmax(self, n: int, k: int, rng: np.random.Generator) -> np.ndarray:
        probs = np.full((n, k), 1.0 / k) + self.JITTER * rng.random((n, k))
        return probs / probs.sum(axis=1, keepdims=True)

    def run(self, items: list[BenchItem]) -> BaselineOutput:
        n = len(items)
        rng = np.random.default_rng(self._seed)
        boolean = self._uniform_with_random_argmax(n, 2, rng)
        return BaselineOutput(
            name=self.name,
            choice_probs=self._uniform_with_random_argmax(n, len(DEPARTMENTS), rng),
            score_probs=self._uniform_with_random_argmax(n, len(URGENCY_LEVELS), rng),
            bool_p_true=boolean[:, 1],
            notes={
                "description": "uniform probabilities; argmax randomised by a 1e-6 jitter",
                "seed": self._seed,
                "why_jitter": "argmax breaks exact ties to index 0, which would make a "
                              "uniform baseline score the first option's base rate",
            },
        )


class MajorityClassBaseline:
    """Predict the empirical class prior of `bench_ja` itself.

    The prior is read off the evaluation set, so this baseline sees labels no real
    model gets. That is deliberate -- it makes it a *hard* floor. Anything that
    cannot beat it has learned nothing (§9: 下回ったら即座に報告して止まる).
    """

    name = "多数決クラス"

    def run(self, items: list[BenchItem]) -> BaselineOutput:
        n = len(items)
        dept_keys = list(DEPARTMENTS)
        dept_counts = np.array(
            [sum(1 for i in items if i.department == k) for k in dept_keys], dtype=np.float64
        )
        urg_counts = np.array(
            [sum(1 for i in items if i.urgency == k) for k in range(len(URGENCY_LEVELS))],
            dtype=np.float64,
        )
        churn_rate = float(sum(i.churn for i in items)) / n

        return BaselineOutput(
            name=self.name,
            choice_probs=np.tile(dept_counts / dept_counts.sum(), (n, 1)),
            score_probs=np.tile(urg_counts / urg_counts.sum(), (n, 1)),
            bool_p_true=np.full(n, churn_rate),
            notes={
                "description": "empirical prior of bench_ja, computed on the eval set itself",
                "prior_department": dict(zip(dept_keys, (dept_counts / n).round(4), strict=True)),
                "prior_urgency": list((urg_counts / n).round(4)),
                "prior_churn_true": round(churn_rate, 4),
            },
        )


# --------------------------------------------------------------------------
# Laya (§9 baseline 5)
# --------------------------------------------------------------------------

class LayaBaseline:
    """A Laya checkpoint answering the bench_ja questions verbatim."""

    def __init__(self, model_id: str, label: str, device: str | None = None) -> None:
        self.model_id = model_id
        self.name = label
        self._device = device
        self._agent: Any | None = None

    def _load(self) -> Any:
        if self._agent is None:
            import laya  # imported lazily: only the bench extra needs it

            self._agent = laya.load(self.model_id, device=self._device)
        return self._agent

    def run(self, items: list[BenchItem]) -> BaselineOutput:
        agent = self._load()
        questions = bench_questions()
        dept_keys = list(DEPARTMENTS)

        choice_rows, score_rows, bool_vals, latencies = [], [], [], []
        malformed = 0

        for item in items:
            started = time.perf_counter()
            result = agent.predict({"body": item.state}, questions)
            latencies.append(time.perf_counter() - started)

            answers = result.get("answers", {})

            dept = answers.get("department", {}).get("probabilities", {})
            row = np.array([float(dept.get(k, 0.0)) for k in dept_keys])
            if row.sum() <= 0:
                malformed += 1
                row = np.full(len(dept_keys), 1.0 / len(dept_keys))
            choice_rows.append(row / row.sum())

            urg = answers.get("urgency", {}).get("probabilities", {})
            # Laya keys score probabilities by level index, as strings.
            srow = np.array([float(urg.get(str(k), 0.0)) for k in range(len(URGENCY_LEVELS))])
            if srow.sum() <= 0:
                malformed += 1
                srow = np.full(len(URGENCY_LEVELS), 1.0 / len(URGENCY_LEVELS))
            score_rows.append(srow / srow.sum())

            noul = answers.get("churn", {}).get("noul", None)
            if noul is None:
                malformed += 1
                noul = 0.5
            bool_vals.append(float(noul))

        return BaselineOutput(
            name=self.name,
            choice_probs=np.vstack(choice_rows),
            score_probs=np.vstack(score_rows),
            bool_p_true=np.array(bool_vals),
            per_item_latency_s=latencies,
            parse_failures=malformed,
            parse_attempts=len(items) * 3,
            notes={
                "model_id": self.model_id,
                "description": "structured output; no text parsing required",
                "probability_resolution": "API returns 4-decimal probabilities",
            },
        )


# --------------------------------------------------------------------------
# LLM-as-classifier (§9 baseline 4)
# --------------------------------------------------------------------------

_CHOICE_PROMPT = """次の問い合わせ文を読み、担当すべき部署を1つ選んでください。

# 問い合わせ文
{state}

# 選択肢
{options}

選択肢の語を**1語だけ**出力してください。説明や記号は一切書かないでください。"""

_SCORE_PROMPT = """次の問い合わせ文を読み、依頼の緊急度を1つ選んでください。

# 問い合わせ文
{state}

# 選択肢
{options}

選択肢の語を**1語だけ**出力してください。説明や記号は一切書かないでください。"""

_BOOL_PROMPT = """次の問い合わせ文を読み、送信者が契約の終了・他社への乗り換えを
示唆しているかを判定してください。

# 問い合わせ文
{state}

「はい」か「いいえ」の**どちらか1語だけ**を出力してください。説明や記号は一切書かないでください。"""


def _parse_one_word(text: str, options: list[str]) -> int | None:
    """Match the reply against the option list. Returns None when nothing matches."""
    cleaned = text.strip().strip("。.、,「」\"'*` \n\t")
    for index, option in enumerate(options):
        if cleaned == option:
            return index
    # A model that ignores "one word only" often still contains exactly one option.
    hits = [i for i, option in enumerate(options) if option in cleaned]
    if len(hits) == 1:
        return hits[0]
    return None


class LocalLLMClassifierBaseline:
    """The generator model used as a zero-shot classifier, at temperature 0.

    This is the baseline that has to *parse*. The parse failure rate is recorded
    because "nothing to parse" is one of the claims the encoder design makes, and
    a claim needs a measured counterpart.

    Note for the write-up: this model wrote `bench_ja`, so it is scoring its own
    generation distribution. It is the one baseline with an unfair advantage.
    """

    name = "ローカル LLM-as-classifier"

    def __init__(self, concurrency: int = 4) -> None:
        self._concurrency = concurrency

    def run(self, items: list[BenchItem]) -> BaselineOutput:
        import asyncio

        return asyncio.run(self._run_async(items))

    async def _run_async(self, items: list[BenchItem]) -> BaselineOutput:
        import asyncio

        from sokudan.data.local_llm import LocalLLM, LocalLLMError

        dept_keys = list(DEPARTMENTS)
        bool_options = ["いいえ", "はい"]
        failures = 0
        latencies: list[float] = []

        choice_idx: list[int | None] = [None] * len(items)
        score_idx: list[int | None] = [None] * len(items)
        bool_idx: list[int | None] = [None] * len(items)

        async with LocalLLM(concurrency=self._concurrency) as llm:

            async def ask(prompt: str, options: list[str]) -> tuple[int | None, float]:
                try:
                    result = await llm.chat(
                        [{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=16,
                    )
                except LocalLLMError:
                    return None, 0.0
                return _parse_one_word(result.text, options), result.latency_s

            async def one(index: int, item: BenchItem) -> None:
                dept_opts = "\n".join(f"- {k}: {v}" for k, v in DEPARTMENTS.items())
                urg_opts = "\n".join(f"- {name}" for name in URGENCY_LEVELS)
                results = await asyncio.gather(
                    ask(_CHOICE_PROMPT.format(state=item.state, options=dept_opts), dept_keys),
                    ask(_SCORE_PROMPT.format(state=item.state, options=urg_opts), URGENCY_LEVELS),
                    ask(_BOOL_PROMPT.format(state=item.state), bool_options),
                )
                choice_idx[index], score_idx[index], bool_idx[index] = (r[0] for r in results)
                latencies.append(sum(r[1] for r in results))

            await asyncio.gather(*(one(i, item) for i, item in enumerate(items)))

        def to_onehot(indices: list[int | None], k: int) -> np.ndarray:
            nonlocal failures
            rows = np.zeros((len(indices), k), dtype=np.float64)
            for i, idx in enumerate(indices):
                if idx is None:
                    failures += 1
                    rows[i, :] = 1.0 / k  # unparseable -> uninformative, not a free guess
                else:
                    rows[i, idx] = 1.0
            return rows

        choice_probs = to_onehot(choice_idx, len(dept_keys))
        score_probs = to_onehot(score_idx, len(URGENCY_LEVELS))
        bool_probs = to_onehot(bool_idx, 2)

        return BaselineOutput(
            name=self.name,
            choice_probs=choice_probs,
            score_probs=score_probs,
            bool_p_true=bool_probs[:, 1],
            per_item_latency_s=latencies,
            parse_failures=failures,
            parse_attempts=len(items) * 3,
            notes={
                "temperature": 0.0,
                "description": "one-word answer, parsed against the option list",
                "hard_labels": (
                    "emits a label, not a distribution, so top-1 confidence is 1.0 by "
                    "construction and its ECE is bounded below by (1 - accuracy). "
                    "Unparseable replies are scored as the uniform distribution."
                ),
                "caveat": "this model generated bench_ja, so it scores its own distribution",
            },
        )
