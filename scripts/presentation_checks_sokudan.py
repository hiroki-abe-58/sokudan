"""Laya's two presentation checks, applied to a sokudan checkpoint on held-out states.

    uv run python scripts/presentation_checks_sokudan.py \
        --checkpoint runs/release_candidate/model.pt --name S8_old

Definitions follow `laya-pr/research/eval/presentation_checks.py` (#131), for `score` and
for `choice`:

- **identical-option control** (`*_slot0_identical`): a question whose K options all carry
  the same text, so the options differ only by position. Metric: the log-probability of
  slot 0 minus the mean over the K slots, averaged over every state and every (text, K)
  configuration. For `choice` (a softmax over marker scores) this is exactly Laya's
  centred raw logit. For `score`, sokudan's output is an ordinal (cumulative-link)
  distribution, so the centred log-probability also carries the ordinal head's shape
  (end slots against middle ones), not position alone; it is reported as measured.
  sokudan's schema refuses identical option names, so these questions are built with
  `model_construct` (validation skipped); rendering and the forward pass are unchanged.
- **first slot over all orders** (`*_first_slot_permuted`): three real options in all
  3! = 6 orders per state; each option sits in each slot twice per state, so an
  order-independent model picks slot 0 in exactly 1/3 of the decisions.

States: the first `--n-states` distinct states of the frozen held-out set
(`data/v2/heldout_val_v2.jsonl`, file order) -- not `bench_ja` / `bench_en`.
Inference is the evaluation path (JointCollator, `run_model`, bf16 autocast, batch 32).
Writes `runs/release_candidate/presentation_<name>.json`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

INSTRUCTIONS = {"score": "この依頼の緊急度は", "choice": "この問い合わせはどの部署が担当すべきか"}
IDENTICAL_TEXTS = {"score": ("中程度", "依頼"), "choice": ("該当する", "その他")}
IDENTICAL_KS = (3, 4, 5)
LEVELS = {"score": ("急がない", "早めに", "業務が止まっている"),
          "choice": ("請求", "技術", "営業")}
DESCRIPTIONS = {"請求": "支払い・返金", "技術": "不具合・障害", "営業": "料金・新規契約"}


class _Pairs(list):
    """`criteria` for a choice question with repeated labels (validation is bypassed)."""

    def items(self):
        return iter(self)


def question(kind: str, options: list[str]):
    from sokudan.schema.question import ChoiceQuestion, ScoreQuestion

    if kind == "score":
        return ScoreQuestion.model_construct(type="score", instructions=INSTRUCTIONS[kind],
                                             criteria=list(options))
    pairs = _Pairs((o, DESCRIPTIONS.get(o, "")) for o in options)
    return ChoiceQuestion.model_construct(type="choice", instructions=INSTRUCTIONS[kind],
                                          criteria=pairs)


def states(n: int) -> list[str]:
    from scripts.eval_heldout import load_rows

    seen: list[str] = []
    for r in load_rows(Path("data/v2/heldout_val_v2.jsonl")):
        if r["state"] not in seen:
            seen.append(r["state"])
        if len(seen) == n:
            break
    return seen


def probabilities(checkpoint: str, cases: list[tuple[str, object, str]]) -> list[np.ndarray]:
    from transformers import AutoTokenizer

    from scripts.eval_heldout import load_checkpoint
    from sokudan.config import BACKBONE_MODEL_ID
    from sokudan.train.dataset import Example
    from sokudan.train.loop import TrainConfig, build_collator, run_model

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    model, encoding, device = load_checkpoint(checkpoint)
    config = TrainConfig(device=device, batch_size=32, encoding=encoding,
                         input_order=model.input_order)
    collator = build_collator(tokenizer, config)
    examples = [Example(state=s, question=q, label=0, ordered=(kind == "score"),
                        doc_id=f"probe-{i}", domain="probe", attribute=kind, kind=kind)
                for i, (s, q, kind) in enumerate(cases)]
    out: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), config.batch_size):
            group = examples[start:start + config.batch_size]
            batch = collator(group).to(device)
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
                probs = run_model(model, batch).probs.float().cpu().numpy()
            for row in range(len(group)):
                n = int(batch.marker_mask[row].sum())
                out.append(probs[row, :n])
    return out


def leave_one_out(per_state: list[float]) -> tuple[float, float]:
    values = [float(np.mean(per_state[:i] + per_state[i + 1:])) for i in range(len(per_state))]
    return min(values), max(values)


def run(checkpoint: str, n_states: int) -> dict:
    texts = states(n_states)
    cases: list[tuple[str, object, str]] = []
    plan: list[tuple[str, str, int, object]] = []  # (check, kind, state index, detail)
    for kind in ("score", "choice"):
        for si, s in enumerate(texts):
            for text in IDENTICAL_TEXTS[kind]:
                for k in IDENTICAL_KS:
                    cases.append((s, question(kind, [text] * k), kind))
                    plan.append((f"{kind}_slot0_identical", kind, si, (text, k)))
            for order in itertools.permutations(range(3)):
                cases.append((s, question(kind, [LEVELS[kind][i] for i in order]), kind))
                plan.append((f"{kind}_first_slot_permuted", kind, si, order))
    probs = probabilities(checkpoint, cases)
    report: dict = {"states": len(texts), "checks": {}}
    for kind in ("score", "choice"):
        name = f"{kind}_slot0_identical"
        per_state: dict[int, list[float]] = {}
        per_config: dict[str, list[float]] = {}
        for (check, _, si, detail), p in zip(plan, probs, strict=True):
            if check != name:
                continue
            logp = np.log(np.clip(p, 1e-12, None))
            value = float(logp[0] - logp.mean())
            per_state.setdefault(si, []).append(value)
            per_config.setdefault(f"{detail[0]}/K={detail[1]}", []).append(value)
        state_means = [float(np.mean(v)) for v in per_state.values()]
        report["checks"][name] = {
            "metric": float(np.mean(state_means)),
            "leave_one_out": leave_one_out(state_means),
            "by_config": {c: float(np.mean(v)) for c, v in per_config.items()},
        }
        name = f"{kind}_first_slot_permuted"
        firsts, by_slot = [], [0, 0, 0]
        by_label = dict.fromkeys(LEVELS[kind], 0)
        answers: dict[int, set] = {}
        per_state_first: dict[int, list[int]] = {}
        for (check, _, si, order), p in zip(plan, probs, strict=True):
            if check != name:
                continue
            slot = int(np.argmax(p))
            firsts.append(slot == 0)
            per_state_first.setdefault(si, []).append(int(slot == 0))
            by_slot[slot] += 1
            label = LEVELS[kind][order[slot]]
            by_label[label] += 1
            answers.setdefault(si, set()).add(label)
        state_rates = [float(np.mean(v)) for v in per_state_first.values()]
        report["checks"][name] = {
            "metric": float(np.mean(firsts)),
            "leave_one_out": leave_one_out(state_rates),
            "argmax_by_slot": by_slot,
            "picks_by_label": by_label,
            "order_invariant_states": sum(1 for a in answers.values() if len(a) == 1),
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--n-states", type=int, default=30)
    args = parser.parse_args()
    report = {"checkpoint": args.checkpoint, **run(args.checkpoint, args.n_states)}
    out = Path(f"runs/release_candidate/presentation_{args.name}.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
