"""Performance A must not change a number (docs/perf_a.md).

The reference records come from the code *before* A (worktree at 32435eb, recorded
with `scripts/perf_record.py` into `runs/perf/pre/`); every test here compares the
current code against them, or against itself with the new paths switched off.

(a) GPU: the first 20 training steps in deterministic mode -- each step's batch and its
    loss, bit for bit.
(b) CPU: both epochs' batch sequences hash to the reference, and `prefetched` hands them
    out in exactly that order.
(c) CPU: the example cache and the collate thread produce the same tensors, and the
    same truncation count, as a plain collator on the calling thread.
(d) GPU: the external evaluation's per-row predictions (`eval_local_attention
    --save-probs`) equal the reference `soup_eval.collect` predictions bit for bit, and
    its evaluation output equals the reference output.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
PRE = REPO / "runs" / "perf" / "pre"


def _need(name: str) -> Path:
    path = PRE / name
    if not path.exists():
        pytest.skip(f"reference record {path} not recorded yet")
    return path


def _examples(name: str, n: int | None = None):
    from sokudan.train.dataset import load_examples

    path = REPO / "data" / name
    if not path.exists():
        pytest.skip(f"{path} not present")
    examples = load_examples(path)
    return examples if n is None else examples[:n]


# (b) ------------------------------------------------------------------------------

def test_b_batch_sequences_match_the_reference(tokenizer):
    from sokudan.train.dataset import length_bucketed_batches
    from sokudan.train.loop import prefetched

    reference = json.loads(_need("batches.json").read_text(encoding="utf-8"))
    examples = _examples("train_v2b.jsonl")
    index_of = {id(e): i for i, e in enumerate(examples)}
    rng = random.Random(0)
    for epoch in reference["epochs"]:
        groups = length_bucketed_batches(examples, 24, tokenizer, rng=rng)
        served = [[index_of[id(e)] for e in g]
                  for g, _ in prefetched(groups, lambda g: len(g))]
        assert len(served) == epoch["n_batches"]
        assert hashlib.sha256(json.dumps(served).encode()).hexdigest() == epoch["sha256"]


def test_b_prefetched_stops_cleanly_when_the_consumer_stops():
    from sokudan.train.loop import prefetched

    groups = [[i] for i in range(100)]
    seen = []
    for group, value in prefetched(groups, lambda g: g[0] * 2):
        seen.append((group[0], value))
        if len(seen) == 5:
            break
    assert seen == [(i, 2 * i) for i in range(5)]


def test_b_prefetched_raises_the_collators_error():
    from sokudan.train.loop import prefetched

    def collate(group):
        if group[0] == 3:
            raise ValueError("bad row")
        return group[0]

    with pytest.raises(ValueError, match="bad row"):
        list(prefetched([[i] for i in range(10)], collate))


# (c) ------------------------------------------------------------------------------

def _same_batch(a, b) -> None:
    for field in ("input_ids", "attention_mask", "marker_positions", "marker_mask",
                  "labels", "ordered"):
        assert torch.equal(getattr(a, field), getattr(b, field)), field
    assert a.input_order == b.input_order


def test_c_cache_and_thread_give_the_plain_tensors(tokenizer):
    from sokudan.train.dataset import JointCollator, length_bucketed_batches
    from sokudan.train.loop import prefetched

    examples = _examples("train_v2b.jsonl", 600) + _examples("val_v2.jsonl", 300)
    groups = length_bucketed_batches(examples, 24, tokenizer, rng=random.Random(5))
    plain = JointCollator(tokenizer, cache=False)
    cached = JointCollator(tokenizer)
    for _ in range(2):  # the second pass is served from the cache
        for group, batch in prefetched(groups, cached):
            _same_batch(batch, plain(group))
    assert cached.truncated == plain.truncated


def test_c_cache_does_not_serve_a_different_object_with_a_recycled_id(tokenizer):
    import dataclasses

    from sokudan.train.dataset import JointCollator

    first = _examples("val_v2.jsonl", 2)
    collator = JointCollator(tokenizer)
    collator([first[0]])
    other = dataclasses.replace(first[1])
    collator._cache[id(other)] = (first[0], collator._cache[id(first[0])][1])  # forged
    _same_batch(collator([other]), JointCollator(tokenizer, cache=False)([other]))


def test_c_flushed_losses_are_the_per_step_floats():
    from sokudan.train.loop import _flush_losses

    values = [torch.tensor(v, dtype=torch.float32) for v in (0.1, 1 / 3, 2.5e-7, 1234.5678)]
    running: list[float] = []
    pending = list(values)
    _flush_losses(pending, running)
    assert running == [float(v) for v in values] and pending == []


# (a), (d): GPU ---------------------------------------------------------------------

def _run(args: list[str], cwd: Path = REPO) -> None:
    env = {**os.environ, "PYTHONPATH": str(cwd)}
    code = subprocess.run([sys.executable, *args], cwd=cwd, env=env).returncode
    assert code == 0


@pytest.mark.gpu
def test_a_first_20_steps_match_the_reference(tmp_path):
    reference = json.loads(_need("steps.json").read_text(encoding="utf-8"))
    repeat = json.loads(_need("steps_repeat.json").read_text(encoding="utf-8"))
    # docs/perf_a.md §3: bit-exact if the reference reproduces itself bit for bit, otherwise
    # within the reference's own run-to-run spread (the batches must match exactly either way)
    spread = max(abs(float.fromhex(a["loss"]) - float.fromhex(b["loss"]))
                 for a, b in zip(reference["steps"], repeat["steps"], strict=True))
    _run(["scripts/perf_record.py", "steps", "--out", str(tmp_path / "steps.json")])
    current = json.loads((tmp_path / "steps.json").read_text(encoding="utf-8"))
    assert len(current["steps"]) == len(reference["steps"]) == 20
    for mine, ref in zip(current["steps"], reference["steps"], strict=True):
        assert mine["indices"] == ref["indices"]
        gap = abs(float.fromhex(mine["loss"]) - float.fromhex(ref["loss"]))
        assert gap <= spread, (mine["batch"], mine["loss"], ref["loss"], spread)


@pytest.mark.gpu
def test_d_external_evaluation_predictions_match_the_reference(tmp_path):
    ref_probs = np.load(_need("preds.npz"))
    ref_eval = json.loads(_need("eval.json").read_text(encoding="utf-8"))[0]
    out = tmp_path / "eval.json"
    _run(["-m", "scripts.eval_local_attention", "--checkpoints", "runs/v01_seed0/model.pt",
          "--out", str(out), "--save-probs"])
    mine = np.load(f"{out}.probs0.npz")
    for key in ("frozen", "long", "val", "val_n"):
        assert np.array_equal(mine[key], ref_probs[key], equal_nan=True), key
    entry = json.loads(out.read_text(encoding="utf-8"))[0]
    for key in ("seconds", "probs", "checkpoint"):
        entry.pop(key, None)
        ref_eval.pop(key, None)
    assert entry == ref_eval
