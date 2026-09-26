"""Distillation from an ensemble teacher (docs/distill.md).

    loss = alpha * stage1_loss + (1 - alpha) * KL(teacher || student)

`alpha` is 0.5 and the temperature 1 (the teacher's and the student's probabilities are
used as they are); neither is tuned. The KL runs over each row's real options: the
Bernoulli pair for bool, the options for choice, the levels for score (the student's
ordinal rows already carry level probabilities in `SokudanOutput.probs`).

Off by default. `scripts/train.py --distill-targets PATH` turns it on; without it the
training loop is the same code, op for op, as before this module existed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

EPS = 1e-8


def distill_kl(student: Tensor, teacher: Tensor, marker_mask: Tensor) -> Tensor:
    """Mean over rows of KL(teacher || student), summed over each row's real options."""
    if student.shape != teacher.shape or student.shape != marker_mask.shape:
        raise ValueError(f"shapes differ: student {tuple(student.shape)}, "
                         f"teacher {tuple(teacher.shape)}, mask {tuple(marker_mask.shape)}")
    real = marker_mask.to(torch.bool)
    t = torch.where(real, teacher, torch.zeros_like(teacher))
    terms = t * (torch.log(t.clamp_min(EPS)) - torch.log(student.clamp_min(EPS)))
    terms = torch.where(real & (t > 0), terms, torch.zeros_like(terms))
    return terms.sum(dim=1).mean()


def row_sha256(line: str) -> str:
    return hashlib.sha256(line.strip().encode("utf-8")).hexdigest()


def load_targets(path: str | Path, train_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Teacher probabilities in training-file order, checked row by row against the file.

    Raises unless the target file has exactly one row per non-empty line of the training
    file, in the same order, with the same content hash.
    """
    blob = np.load(path)
    lines = [line for line in Path(train_path).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    hashes = [row_sha256(line) for line in lines]
    stored = [str(h) for h in blob["row_sha256"]]
    if stored != hashes:
        raise ValueError(f"{path} does not match {train_path} row for row "
                         f"({len(stored)} target rows, {len(hashes)} training rows)")
    return np.asarray(blob["probs"], dtype=np.float32), np.asarray(blob["n_options"])


def teacher_batch(targets: np.ndarray, n_options: np.ndarray, rows: list[int],
                  marker_mask: Tensor) -> Tensor:
    """`(B, M)` teacher probabilities for the batch's rows, aligned with `marker_mask`."""
    width = marker_mask.shape[1]
    real = marker_mask.sum(dim=1).cpu().numpy()
    if not (n_options[rows] == real).all():
        raise ValueError("teacher option counts do not match the batch")
    block = np.nan_to_num(targets[rows, :width], nan=0.0)
    return torch.from_numpy(block).to(marker_mask.device)
