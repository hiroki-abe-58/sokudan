"""Does Laya's slot-0 suppression survive a change of option *prefix*?

    uv run python scripts/probe_option_rendering.py --out runs/option_rendering.json

`laya#131`: AlKor13 reports that dropping the `level N: ` prefix from `render_options`
makes the slot-0 suppression disappear **in the raw logits**, and asks for a labelled
set to say whether accuracy moves with it. `docs/baseline_ja.md` §6.2 and
`docs/baseline_en.md` have the labelled sets, so the question is answerable here.

Three renderings of the same `score` options, everything else identical:

    A  level 0: 急がない      the shipped `render_options`
    C  急がない               option text only, no prefix
    D  zero: 急がない         English word ordinals, matching AlKor13's condition

The Japanese run also uses English word ordinals in D rather than 零/一/二, so that D
differs from A in the *form* of the ordinal and not in its language.

Rather than rebuild the token sequence, `render_options` is patched for the duration of
a condition. `build_sequence` calls it as a module global, so every other step -- the
`question: instructions [SEP] [MASK] opt [MASK] opt [SEP] state` layout, the 48-token
option cap, the head budget, truncation -- is the code that shipped, byte for byte.
Copying `build_sequence` to take pre-rendered options is how the copy drifts and the
probe stops measuring the model it claims to.

Probabilities go through the same temperature bucket `Agent.system_one` applies, so the
accuracy and RPS here are comparable to §6.2. The raw logits are reported beside them
untouched, because that is the quantity AlKor13 measured and the one a softmax hides:
a constant offset on every slot vanishes under softmax, and a constant offset on *one*
slot does not.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

WORD_ORDINALS = ["zero", "one", "two", "three", "four", "five", "six", "seven",
                 "eight", "nine", "ten"]


def render_a(criteria: list[str]) -> list[str]:
    """What the package ships."""
    return [f"level {i}: {c}" for i, c in enumerate(criteria)]


def render_c(criteria: list[str]) -> list[str]:
    """No prefix at all."""
    return list(criteria)


def render_d(criteria: list[str]) -> list[str]:
    """English word ordinals, in both languages -- AlKor13's condition."""
    return [f"{WORD_ORDINALS[i]}: {c}" for i, c in enumerate(criteria)]


CONDITIONS: list[tuple[str, Any]] = [
    ("A prefix (shipped)", render_a),
    ("C no prefix", render_c),
    ("D word ordinal", render_d),
]


@contextmanager
def rendered_as(renderer):
    """Patch `render_options` for `score` questions only, everywhere it is looked up."""
    from laya import agent as laya_agent
    from laya import common as laya_common

    original = laya_common.render_options

    def patched(q: dict) -> list[str]:
        if q.get("t") != "score":
            return original(q)
        return renderer(list(q["crit"]))

    targets = [laya_common, laya_agent]
    saved = [(m, getattr(m, "render_options", None)) for m in targets]
    try:
        for module in targets:
            if hasattr(module, "render_options"):
                module.render_options = patched
        yield
    finally:
        for module, value in saved:
            if value is not None:
                module.render_options = value


@dataclass
class ConditionResult:
    name: str
    rendered: list[str]
    n: int
    slot_argmax_counts: list[int]
    first_slot_count: int
    accuracy: float
    rps: float
    mean_logit_by_slot: list[float]
    mean_logit_spread: float
    per_item: list[dict[str, Any]] = field(default_factory=list)


def raw_logits(agent: Any, state: str, instructions: str,
               criteria: list[str]) -> tuple[np.ndarray, float]:
    """One `score` question through the shipped path, returning logits before softmax."""
    import torch
    from laya.common import QTYPES, build_sequence, collate_items, temp_bucket

    question = {"t": "score", "ins": instructions, "crit": list(criteria)}
    max_len = agent.cfg.get("max_len", 512)
    head_max_len = agent.cfg.get("head_max_len", 192)
    ids, markers = build_sequence(agent.tok, {"body": state}, question, max_len, head_max_len)
    if len(markers) != len(criteria):
        raise ValueError(f"{len(markers)} markers for {len(criteria)} options; "
                         "an option was truncated out of the head budget")

    batch = collate_items(
        [[{"ids": ids, "markers": markers, "qtype": QTYPES["score"]}]],
        agent.tok.pad_token_id,
    )
    with torch.no_grad():
        logits, _ = agent.model(
            batch["input_ids"].to(agent.device),
            batch["attention_mask"].to(agent.device),
            batch["marker_pos"].to(agent.device),
            batch["marker_mask"].to(agent.device),
            batch["qtype"].to(agent.device),
        )
    row = logits[0, :len(criteria)].float().cpu().numpy()
    scale = agent.temperature_by_options.get(
        temp_bucket(QTYPES["score"], len(criteria)),
        agent.temperature[QTYPES["score"]],
    )
    return row, max(1e-3, float(scale))


def run_condition(agent: Any, name: str, renderer, states: list[str], gold: np.ndarray,
                  instructions: str, criteria: list[str], *,
                  keep_per_item: bool = False) -> ConditionResult:
    from sokudan.calibration.metrics import rps as rps_metric

    logit_rows, prob_rows, per_item = [], [], []
    with rendered_as(renderer):
        rendered = renderer(list(criteria))
        for index, state in enumerate(states):
            row, scale = raw_logits(agent, state, instructions, criteria)
            logit_rows.append(row)
            scaled = row / scale
            probs = np.exp(scaled - scaled.max())
            probs = probs / probs.sum()
            prob_rows.append(probs)
            if keep_per_item:
                per_item.append({
                    "index": index, "argmax_slot": int(probs.argmax()),
                    "gold": int(gold[index]),
                    "logits": [round(float(v), 4) for v in row],
                })

    logits = np.vstack(logit_rows)
    probs = np.vstack(prob_rows)
    counts = np.bincount(probs.argmax(axis=1), minlength=len(criteria))
    means = logits.mean(axis=0)
    return ConditionResult(
        name=name,
        rendered=rendered,
        n=len(states),
        slot_argmax_counts=counts.tolist(),
        first_slot_count=int(counts[0]),
        accuracy=float((probs.argmax(axis=1) == gold).mean()),
        rps=float(rps_metric(probs, gold)),
        mean_logit_by_slot=[round(float(v), 4) for v in means],
        mean_logit_spread=round(float(means.max() - means.min()), 4),
        per_item=per_item,
    )
