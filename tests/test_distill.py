"""Distillation (docs/distill.md, Phase 2).

(a) With the flag off, one optimizer step gives the same parameters as the loop before
    distillation existed (commit 915979c); with alpha = 1 the teacher changes nothing.
(b) KL(teacher || student) is 0 when they are equal, and matches the closed form for a
    Bernoulli pair.
(c) Teacher targets match the training file row for row and are refused otherwise; the
    training rows share no attribute, document or state with the held-out and
    evaluation sets.
(d) The teacher's members were each run in their own saved input order.
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from sokudan.train.distill import distill_kl, load_targets, row_sha256, teacher_batch
from sokudan.train.loop import train
from tests.test_grad_accum import _config, _examples, _max_diff, _tiny_model, _trained

ROOT = Path(__file__).resolve().parent.parent
TEACHER = ROOT / "data" / "cache" / "distill" / "teacher_E_mix16.npz"


def _old_loop(tmp_path: Path):
    source = subprocess.run(["git", "show", "915979c:sokudan/train/loop.py"], cwd=ROOT,
                            capture_output=True, text=True, encoding="utf-8",
                            check=True).stdout
    assert "distill_kl" not in source  # (stage1_distill is the existing label loss)
    path = tmp_path / "loop_before_distill.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("loop_before_distill", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["loop_before_distill"] = module
    spec.loader.exec_module(module)
    return module


def test_a_flag_off_matches_the_loop_before_distillation(tokenizer, tmp_path):
    old = _old_loop(tmp_path)
    examples = _examples()
    new_params = _trained(train, _config(24, 1), tokenizer, examples)
    old_params = _trained(old.train, _config(24, 1, cls=old.TrainConfig), tokenizer, examples)
    assert _max_diff(new_params, old_params) == 0.0


def test_a_alpha_one_leaves_the_step_unchanged(tokenizer):
    examples = _examples()
    widths = [len(e.question.labels) for e in examples]
    probs = np.full((len(examples), max(widths)), np.nan, dtype=np.float32)
    for i, w in enumerate(widths):
        probs[i, :w] = 1.0 / w
    teacher = (probs, np.array(widths))
    config = _config(24, 1)
    config.distill_alpha = 1.0
    model = _tiny_model(tokenizer)
    train(model, tokenizer, examples, examples[:4], config, teacher=teacher)
    with_teacher = {k: v.detach().clone() for k, v in model.state_dict().items()}
    plain = _trained(train, _config(24, 1), tokenizer, examples)
    assert _max_diff(with_teacher, plain) == 0.0


def test_a_teacher_moves_the_step_when_alpha_below_one(tokenizer):
    examples = _examples()
    widths = [len(e.question.labels) for e in examples]
    probs = np.full((len(examples), max(widths)), np.nan, dtype=np.float32)
    for i, w in enumerate(widths):
        probs[i, :w] = 1.0 / w
    model = _tiny_model(tokenizer)
    train(model, tokenizer, examples, examples[:4], _config(24, 1), teacher=(probs,
                                                                              np.array(widths)))
    distilled = {k: v.detach().clone() for k, v in model.state_dict().items()}
    plain = _trained(train, _config(24, 1), tokenizer, examples)
    assert _max_diff(distilled, plain) > 0


def test_b_kl_is_zero_for_identical_distributions():
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 0]])
    p = torch.tensor([[0.3, 0.7, 0.0, 0.0], [0.1, 0.2, 0.3, 0.4], [0.5, 0.25, 0.25, 0.0]])
    assert float(distill_kl(p, p.clone(), mask)) == pytest.approx(0.0, abs=1e-7)


def test_b_kl_matches_the_bernoulli_closed_form():
    mask = torch.tensor([[1, 1]])
    t, s = 0.8, 0.6
    teacher = torch.tensor([[1 - t, t]])
    student = torch.tensor([[1 - s, s]])
    expected = t * math.log(t / s) + (1 - t) * math.log((1 - t) / (1 - s))
    assert float(distill_kl(student, teacher, mask)) == pytest.approx(expected, rel=1e-5)
    assert float(distill_kl(student, teacher, mask)) > 0


def test_b_padding_does_not_count():
    mask = torch.tensor([[1, 1, 0]])
    student = torch.tensor([[0.4, 0.6, 0.0]])
    teacher = torch.tensor([[0.4, 0.6, 0.9]])  # junk in the padded slot
    assert float(distill_kl(student, teacher, mask)) == pytest.approx(0.0, abs=1e-7)


def test_c_targets_must_match_the_training_file(tmp_path):
    lines = ['{"a": 1}', '{"a": 2}', '{"a": 3}']
    train_file = tmp_path / "train.jsonl"
    train_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    good = tmp_path / "good.npz"
    np.savez(good, probs=np.zeros((3, 2)), n_options=np.array([2, 2, 2]),
             row_sha256=np.array([row_sha256(x) for x in lines]))
    probs, n = load_targets(good, train_file)
    assert probs.shape == (3, 2) and list(n) == [2, 2, 2]
    bad = tmp_path / "bad.npz"
    np.savez(bad, probs=np.zeros((3, 2)), n_options=np.array([2, 2, 2]),
             row_sha256=np.array([row_sha256(x) for x in reversed(lines)]))
    with pytest.raises(ValueError):
        load_targets(bad, train_file)


def test_c_teacher_batch_checks_option_counts():
    targets = np.array([[0.2, 0.8, np.nan], [0.1, 0.2, 0.7]], dtype=np.float32)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    t = teacher_batch(targets, np.array([2, 3]), [0, 1], mask)
    assert torch.allclose(t, torch.tensor([[0.2, 0.8, 0.0], [0.1, 0.2, 0.7]]))
    with pytest.raises(ValueError):
        teacher_batch(targets, np.array([3, 3]), [0, 1], mask)


@pytest.mark.slow
def test_c_training_rows_exclude_held_out_and_evaluation_data():
    from scripts.distill_teacher import TRAIN, check_training_rows

    lines = [x for x in TRAIN.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(check_training_rows(lines)) == len(lines)


@pytest.mark.slow
def test_c_d_saved_teacher_matches_training_rows_and_member_orders():
    if not TEACHER.exists():
        pytest.skip("teacher targets not built yet")
    probs, n = load_targets(TEACHER, ROOT / "data" / "train_v2b.jsonl")
    assert len(probs) == len(n) == 67394
    sums = np.nansum(probs, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-4)
    blob = np.load(TEACHER)
    for member, order in zip(blob["members"], blob["orders"], strict=True):
        saved = torch.load(ROOT / str(member), map_location="cpu", weights_only=False)["config"]
        assert saved.get("input_order", "question_first") == str(order), member
    assert sorted(str(o) for o in blob["orders"]) == ["question_first"] * 8 + ["state_first"] * 8
