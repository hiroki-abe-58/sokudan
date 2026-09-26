"""Is batch 12 x accumulation 2 the same training step as batch 24? (docs/noise_floor.md §5)

A tiny ModernBERT with random weights, fp32 on CPU, no dropout, the same initial
weights and the same examples, trained through the real `sokudan.train.loop.train`.

(a) 24 examples, one optimizer step: batch 24 x 1 and batch 12 x 2 end with parameters
    within 1e-5 of each other.
(b) On the real training sets, the per-optimizer-step learning-rate sequence of the two
    settings has the same length and the same values.
(c) The accumulation-1 path gives the same parameters as the loop from before
    `--grad-accum` existed (commit 858a1f5's parent, ca414a3).
Plus the gradient-level checks behind them: two full micro-batches reproduce the
batch-24 gradient; unequal micro-batches do not (a property of averaging per
micro-batch, measured and printed); and the short last group is divided by its own size.
"""

from __future__ import annotations

import copy
import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from sokudan.train.dataset import JointCollator, length_bucketed_batches, load_examples
from sokudan.train.loop import TrainConfig, accum_group_size, cosine_schedule, run_model, train
from sokudan.train.stage1_distill import stage1_loss

ROOT = Path(__file__).resolve().parent.parent


def _examples(n_per_kind: int = 8):
    rows = load_examples(ROOT / "data" / "val_v2.jsonl")
    picked = []
    for kind in ("choice", "score", "bool"):
        picked += [e for e in rows if e.kind == kind][:n_per_kind]
    return picked


def _tiny_model(tokenizer):
    from transformers import AutoModel, ModernBertConfig

    from sokudan.model.backbone import Backbone, BackboneSpec
    from sokudan.model.joint import SokudanJointModel

    torch.manual_seed(0)
    config = ModernBertConfig(
        vocab_size=len(tokenizer), hidden_size=32, num_attention_heads=4,
        num_hidden_layers=2, intermediate_size=64, local_attention=128,
        global_attn_every_n_layers=2, max_position_embeddings=1024,
        pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id, cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        attention_dropout=0.0, embedding_dropout=0.0, mlp_dropout=0.0,
    )
    backbone = Backbone(AutoModel.from_config(config, attn_implementation="sdpa"),
                        BackboneSpec.from_config(config))
    torch.manual_seed(1)
    return SokudanJointModel(backbone)


def _config(batch_size: int, accum: int, cls=TrainConfig) -> TrainConfig:
    kwargs = dict(seed=0, epochs=1, batch_size=batch_size, device="cpu", amp_dtype="float32",
                  encoding="joint", log_every=10**6)
    if accum != 1:
        kwargs["grad_accum"] = accum
    return cls(**kwargs)


def _trained(train_fn, config, tokenizer, examples):
    model = _tiny_model(tokenizer)
    train_fn(model, tokenizer, examples, examples[:4], config)
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _max_diff(a: dict, b: dict) -> float:
    return max(float((a[k] - b[k]).abs().max()) for k in a)


def test_a_one_step_batch24_equals_batch12_accum2(tokenizer):
    examples = _examples()
    assert len(examples) == 24
    p24 = _trained(train, _config(24, 1), tokenizer, examples)
    p12 = _trained(train, _config(12, 2), tokenizer, examples)
    moved = _max_diff(p24, {k: v for k, v in _tiny_model(tokenizer).state_dict().items()})
    diff = _max_diff(p24, p12)
    print(f"(a) max |p24 - p12x2| = {diff:.3e} (parameters moved up to {moved:.3e})")
    assert moved > 0
    assert diff <= 1e-5


def test_c_accum1_matches_the_loop_before_grad_accum(tokenizer, tmp_path):
    source = subprocess.run(["git", "show", "ca414a3:sokudan/train/loop.py"], cwd=ROOT,
                            capture_output=True, text=True, encoding="utf-8",
                            check=True).stdout
    assert "grad_accum" not in source
    path = tmp_path / "old_loop.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("old_loop", path)
    old = importlib.util.module_from_spec(spec)
    sys.modules["old_loop"] = old
    spec.loader.exec_module(old)

    examples = _examples()
    new_params = _trained(train, _config(24, 1), tokenizer, examples)
    old_params = _trained(old.train, _config(24, 1, cls=old.TrainConfig), tokenizer, examples)
    diff = _max_diff(new_params, old_params)
    print(f"(c) max |new accum-1 - pre-858a1f5| = {diff:.3e}")
    assert diff <= 1e-7


def _lr_sequence(n_examples_lengths: list, batch_size: int, accum: int, epochs: int = 2,
                 tokenizer=None) -> list[float]:
    """What `train` feeds the optimizer, step by step (same formulas and batching)."""
    import random

    examples = n_examples_lengths
    steps_per_epoch = math.ceil(math.ceil(len(examples) / batch_size) / accum)
    total = steps_per_epoch * epochs
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=1.0)
    apply = cosine_schedule(optimizer, total, int(total * 0.05))
    rng = random.Random(0)
    lrs, step = [], 0
    for _ in range(epochs):
        batches = length_bucketed_batches(examples, batch_size, tokenizer, rng=rng)
        for index in range(len(batches)):
            if index % accum == 0:
                lrs.append(apply(step))
            if (index + 1) % accum == 0 or index + 1 == len(batches):
                step += 1
    assert step == total  # the schedule ends exactly where the loop does
    return lrs


@pytest.mark.slow
@pytest.mark.parametrize("path", ["data/train_v2b.jsonl", "data/l2x2/train_long.jsonl"])
def test_b_learning_rate_sequence_matches_on_real_data(path):
    file = ROOT / path
    if not file.exists():
        pytest.skip(f"{path} not present")
    examples = load_examples(file)
    a = _lr_sequence(examples, 24, 1)
    b = _lr_sequence(examples, 12, 2)
    print(f"(b) {path}: {len(examples)} examples, {len(a)} vs {len(b)} optimizer steps")
    assert len(a) == len(b)
    assert a == b


def _grad(model, tokenizer, groups: list[list], divide_by: list[float]) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    collator = JointCollator(tokenizer)
    for group, d in zip(groups, divide_by, strict=True):
        batch = collator(group)
        out = run_model(model, batch)
        loss = stage1_loss(out.probs.float(), batch.labels, batch.ordered, batch.marker_mask).total
        (loss / d).backward()
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


def test_two_full_micro_batches_reproduce_the_batch24_gradient(tokenizer):
    examples = _examples()
    model = _tiny_model(tokenizer)
    model.train()
    g24 = _grad(model, tokenizer, [examples], [1])
    g12 = _grad(model, tokenizer, [examples[:12], examples[12:]], [2, 2])
    rel = float((g24 - g12).norm() / g24.norm())
    print(f"12+12 vs 24: relative gradient difference {rel:.3e}")
    assert rel <= 1e-5


def test_unequal_micro_batches_are_not_the_batch_mean(tokenizer):
    """Averaging per micro-batch weights the rows of a short micro-batch more."""
    examples = _examples()[:14]
    model = _tiny_model(tokenizer)
    model.train()
    g14 = _grad(model, tokenizer, [examples], [1])
    g12_2 = _grad(model, tokenizer, [examples[:12], examples[12:]], [2, 2])
    rel = float((g14 - g12_2).norm() / g14.norm())
    print(f"12+2 (mean of means) vs 14 (one mean): relative gradient difference {rel:.3e}")
    assert rel > 1e-3


def test_short_last_group_is_divided_by_its_own_size():
    assert [accum_group_size(i, 3, 2) for i in range(3)] == [2, 2, 1]
    assert [accum_group_size(i, 5617, 2) for i in (5614, 5615, 5616)] == [2, 2, 1]
    assert all(accum_group_size(i, 7, 1) == 1 for i in range(7))


def test_lone_last_micro_batch_step_equals_an_unaccumulated_step(tokenizer):
    """12 examples at batch 12 x accum 2: one lone micro-batch, divided by 1 after the fix."""
    examples = _examples()[:12]
    lone = _trained(train, _config(12, 2), tokenizer, copy.copy(examples))
    plain = _trained(train, _config(12, 1), tokenizer, copy.copy(examples))
    assert _max_diff(lone, plain) <= 1e-7
