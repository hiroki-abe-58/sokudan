"""Head initialisation is NOT a function of `--seed` (docs/regression_check.md §10).

The 2026-09-26 change that seeded before building the model (f75e3f2) was reverted on
2026-09-27 after the pre-registered 16 vs 16 check. These tests record the restored
behaviour, so a future change to it has to be deliberate:

(a) Two fresh processes with the same seed build *different* scorers: the scorer's
    initial values come from an unseeded generator, as in every baseline run.
(b) The ordinal head is built from constants, so it is identical across processes.
(c) The data order for a seed does not depend on how the model was built: the sampler
    draws from its own `random.Random(seed)` after `train` seeds.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from sokudan.train.dataset import length_bucketed_batches, load_examples
from sokudan.train.loop import set_seed

ROOT = Path(__file__).resolve().parent.parent
PROBE = """
import argparse, hashlib, sys
import sokudan.config  # noqa
sys.path.insert(0, "scripts")
import train
args = argparse.Namespace(seed=int(sys.argv[1]), encoding="joint", no_ordinal=False,
                          local_attention=None, input_order="question_first", head_layers=2)
m = train.build_model(args)
for name in ("scorer.weight", "ordinal.cut.weight", "ordinal.cut.bias", "ordinal.location.bias"):
    p = dict(m.named_parameters())[name]
    print(name, hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest())
"""


def _heads(seed: int) -> dict[str, str]:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    out = subprocess.run([sys.executable, "-c", PROBE, str(seed)], cwd=ROOT, env=env,
                         capture_output=True, text=True, check=True).stdout
    return dict(line.split() for line in out.splitlines() if line.startswith(("scorer",
                                                                              "ordinal")))


@pytest.mark.slow
def test_a_b_scorer_init_does_not_follow_the_seed():
    first, again = _heads(8), _heads(8)
    assert first["scorer.weight"] != again["scorer.weight"]          # (a) unseeded
    for name in ("ordinal.cut.weight", "ordinal.cut.bias", "ordinal.location.bias"):
        assert first[name] == again[name]                            # (b) constants


def _order(examples, tokenizer, seed: int, *, seed_before_build: bool) -> list[list[str]]:
    """Two epochs of batch order, the way `train` draws it, after either build flow."""
    import torch

    if seed_before_build:
        set_seed(seed)            # the reverted f75e3f2 flow: seed, then build
    torch.nn.Linear(768, 1)       # construction consumes the torch generator either way
    set_seed(seed)                # `train` seeds before anything else
    rng = random.Random(seed)     # `train`'s sampler
    epochs = []
    for _ in range(2):
        batches = length_bucketed_batches(examples, 24, tokenizer, rng=rng)
        epochs.append(["|".join(e.doc_id + e.attribute for e in b) for b in batches])
    return epochs


def test_c_data_order_does_not_depend_on_how_the_model_was_built(tokenizer):
    examples = load_examples(ROOT / "data" / "val_v2.jsonl")[:2000]
    for seed in (8, 11):
        unseeded = _order(examples, tokenizer, seed, seed_before_build=False)
        seeded = _order(examples, tokenizer, seed, seed_before_build=True)
        assert unseeded == seeded
    assert _order(examples, tokenizer, 8, seed_before_build=False) != _order(
        examples, tokenizer, 9, seed_before_build=False)
