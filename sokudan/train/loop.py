"""Stage 1 training loop (SOKUDAN_SPEC.md §8).

bf16 autocast, AdamW, cosine schedule with 5% warmup, one epoch. `torch.compile` is
off: §6.2 rules it out for the sprint because sequence lengths and option counts vary
every batch, so the default mode recompiles constantly. Length bucketing in
`dataset.py` covers the padding cost instead.

Nothing here writes a number it did not measure. `runs/<id>/report.md` records the
metrics before and after, the seed, and the wall clock.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from sokudan.calibration.metrics import (
    accuracy,
    auroc,
    brier,
    ece,
    macro_f1,
    nll,
    ordinal_mae,
    rps,
)
from sokudan.model.sokudan import SokudanModel, SokudanOutput
from sokudan.train.dataset import (
    Batch,
    Collator,
    Example,
    JointBatch,
    JointCollator,
    length_bucketed_batches,
)
from sokudan.train.stage1_distill import stage1_loss


@dataclass
class TrainConfig:
    seed: int = 0
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 2e-5
    head_learning_rate: float = 1e-4
    """The head is new; the backbone is not. A single LR serves one of them badly."""
    ordinal_learning_rate: float = 1e-3
    """The ordinal head is 1,538 parameters controlling the whole `score` output.

    It is a bottleneck, not a layer: `cut` and `location` are one projection each, and
    everything the model can say about an ordinal question passes through them. At the
    head's own rate the spacing barely moved in a 930-step epoch (see the comment on
    `OrdinalHead.__init__`), so it gets its own, larger rate.
    """
    weight_decay: float = 0.01
    warmup_fraction: float = 0.05
    max_grad_norm: float = 1.0
    max_state_tokens: int = 1024
    ordinal_weight: float = 1.0
    log_every: int = 50
    amp_dtype: str = "bfloat16"
    device: str = "cuda"
    encoding: str = "separate"
    """`separate` (cross-attention head) or `joint` (one sequence, no head).

    Set here rather than inferred from the model so that a checkpoint, its collator
    and its diagnostics cannot disagree about which arm produced a number.
    """
    max_joint_tokens: int = 1024


@dataclass
class EpochLog:
    epoch: int
    steps: int
    mean_loss: float
    seconds: float
    examples_per_second: float
    val: dict[str, Any] = field(default_factory=dict)


def set_seed(seed: int) -> None:
    import random as _random

    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model: SokudanModel, config: TrainConfig) -> torch.optim.Optimizer:
    """Separate learning rates for the pretrained backbone and the new head.

    The backbone carries a pretrained prior worth preserving; the head and the
    ordinal cut points start from a neutral initialisation and have to move much
    further in one epoch.
    """
    buckets: dict[tuple[str, bool], list] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("backbone."):
            group = "backbone"
        elif name.startswith("ordinal."):
            group = "ordinal"
        else:
            group = "head"
        # Biases and norms are conventionally exempt from weight decay. The ordinal
        # head's cut bias is one of them, and it is load-bearing -- decaying it would
        # pull the spacing back toward the frozen regime it was just moved out of.
        buckets.setdefault((group, parameter.ndim <= 1), []).append(parameter)

    rates = {
        "backbone": config.learning_rate,
        "head": config.head_learning_rate,
        "ordinal": config.ordinal_learning_rate,
    }
    groups = [
        {"params": params, "lr": rates[group],
         "weight_decay": 0.0 if is_flat else config.weight_decay}
        for (group, is_flat), params in buckets.items()
    ]
    return torch.optim.AdamW(groups, betas=(0.9, 0.98), eps=1e-8)


def cosine_schedule(optimizer: torch.optim.Optimizer, total_steps: int, warmup: int):
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total_steps - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    def apply(step: int) -> float:
        scale = lr_at(step)
        for group, base in zip(optimizer.param_groups, base_lrs, strict=True):
            group["lr"] = base * scale
        return scale

    return apply


def _amp_dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


def build_collator(
    tokenizer: PreTrainedTokenizerBase, config: TrainConfig
) -> Collator | JointCollator:
    """The one place a `TrainConfig` becomes a collator.

    `train`, `evaluate`, the gate and the diagnostics all call this, so an arm can
    never be trained with one encoding and scored with the other.
    """
    if config.encoding == "joint":
        return JointCollator(tokenizer, max_joint_tokens=config.max_joint_tokens)
    if config.encoding != "separate":
        raise ValueError(f"unknown encoding {config.encoding!r}")
    return Collator(tokenizer, max_state_tokens=config.max_state_tokens)


def run_model(model: nn.Module, batch: Batch | JointBatch) -> SokudanOutput:
    """One call site for both encodings.

    `train`, `evaluate` and the diagnostics all go through here, so the separate and
    joint arms cannot drift into being scored by slightly different code -- which is
    the §1-3 rule applied to the forward pass rather than to tokenisation. Dispatch
    is on the batch, not on the model, because the collator is what decides which
    arrangement the tensors are in.
    """
    if isinstance(batch, JointBatch):
        return model(
            batch.input_ids, batch.attention_mask,
            batch.marker_positions, batch.marker_mask, batch.ordered,
        )
    return model(
        batch.state_input_ids, batch.state_attention_mask,
        batch.question_input_ids, batch.question_attention_mask,
        batch.marker_positions, batch.marker_mask, batch.ordered,
    )


@torch.no_grad()
def evaluate(
    model: SokudanModel,
    examples: list[Example],
    collator: Collator,
    config: TrainConfig,
    *,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Metrics on a held-out split, split by primitive.

    Probabilities are collected at full precision. Metrics are computed by
    `sokudan.calibration.metrics` -- the same functions that scored the baselines, so
    the numbers are comparable with `docs/baseline_ja.md` rather than merely similar.
    """
    model.eval()
    device = torch.device(config.device)
    amp = _amp_dtype(config.amp_dtype)
    batch_size = batch_size or config.batch_size

    buckets: dict[str, dict[str, list]] = {
        kind: {"probs": [], "labels": []} for kind in ("choice", "score", "bool")
    }

    import random as _random

    batches = length_bucketed_batches(
        examples, batch_size, collator.tokenizer, rng=_random.Random(0), shuffle=False
    )
    for group in batches:
        batch = collator(group).to(device)
        with torch.autocast(device_type=device.type, dtype=amp, enabled=amp != torch.float32):
            out = run_model(model, batch)
        probs = out.probs.float().cpu().numpy()
        labels = batch.labels.cpu().numpy()
        for row, example in enumerate(group):
            kind = example.kind if example.kind in buckets else "choice"
            n_options = int(batch.marker_mask[row].sum())
            buckets[kind]["probs"].append(probs[row, :n_options])
            buckets[kind]["labels"].append(int(labels[row]))

    model.train()

    results: dict[str, Any] = {}
    for kind, data in buckets.items():
        if not data["probs"]:
            continue
        results[kind] = _metrics_for_group(kind, data["probs"], data["labels"])
    results["n_examples"] = len(examples)
    return results


def _metrics_for_group(kind: str, prob_rows: list, labels: list) -> dict[str, float]:
    """Group rows by option count so ragged K can still be scored as arrays."""
    by_k: dict[int, dict[str, list]] = {}
    for row, label in zip(prob_rows, labels, strict=True):
        entry = by_k.setdefault(len(row), {"probs": [], "labels": []})
        entry["probs"].append(row)
        entry["labels"].append(label)

    total = sum(len(v["labels"]) for v in by_k.values())
    aggregate: dict[str, float] = {}

    def add(name: str, value: float, weight: int) -> None:
        aggregate[name] = aggregate.get(name, 0.0) + value * weight / total

    all_probs, all_labels = [], []
    for _k, entry in sorted(by_k.items()):
        probs = np.vstack(entry["probs"])
        gold = np.asarray(entry["labels"])
        weight = len(gold)
        add("accuracy", accuracy(probs, gold), weight)
        add("macro_f1", macro_f1(probs, gold), weight)
        add("ece", ece(probs, gold), weight)
        add("brier", brier(probs, gold), weight)
        add("nll", nll(probs, gold), weight)
        add("mean_confidence", float(probs.max(axis=1).mean()), weight)
        if kind == "score":
            add("rps", rps(probs, gold), weight)
            add("mae_argmax", ordinal_mae(probs, gold), weight)
            add("mae_expectation", ordinal_mae(probs, gold, use_expectation=True), weight)
        if kind == "bool" and probs.shape[1] == 2:
            all_probs.append(probs[:, 1])
            all_labels.append(gold)

    if kind == "bool" and all_probs:
        p_true = np.concatenate(all_probs)
        gold = np.concatenate(all_labels)
        if 0 < gold.sum() < len(gold):
            aggregate["auroc"] = auroc(p_true, gold)
        aggregate["mean_p_true"] = float(p_true.mean())
        aggregate["gold_true_rate"] = float(gold.mean())

    aggregate["n"] = float(total)
    return aggregate


def train(
    model: SokudanModel,
    tokenizer: PreTrainedTokenizerBase,
    train_examples: list[Example],
    val_examples: list[Example],
    config: TrainConfig,
    *,
    run_dir: Path | None = None,
) -> dict[str, Any]:
    import random as _random

    set_seed(config.seed)
    device = torch.device(config.device)
    amp = _amp_dtype(config.amp_dtype)
    model.to(device)
    model.train()

    collator = build_collator(tokenizer, config)
    rng = _random.Random(config.seed)

    steps_per_epoch = math.ceil(len(train_examples) / config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    optimizer = build_optimizer(model, config)
    apply_lr = cosine_schedule(optimizer, total_steps, int(total_steps * config.warmup_fraction))

    print(f"train {len(train_examples)} examples, val {len(val_examples)}, "
          f"{total_steps} steps, seed {config.seed}", flush=True)

    before = evaluate(model, val_examples, collator, config)
    print(f"  before training: {json.dumps(_brief(before), ensure_ascii=False)}", flush=True)

    logs: list[EpochLog] = []
    step = 0
    for epoch in range(config.epochs):
        batches = length_bucketed_batches(
            train_examples, config.batch_size, tokenizer, rng=rng
        )
        started = time.time()
        running: list[float] = []

        for batch_index, group in enumerate(batches):
            batch: Batch = collator(group).to(device)
            apply_lr(step)

            with torch.autocast(device_type=device.type, dtype=amp,
                                enabled=amp != torch.float32):
                out = run_model(model, batch)
            # The loss runs in fp32: RPS and the log both lose too much resolution in
            # bf16, which has about three decimal digits of mantissa.
            breakdown = stage1_loss(
                out.probs.float(), batch.labels, batch.ordered,
                batch.marker_mask, ordinal_weight=config.ordinal_weight,
            )
            loss = breakdown.total

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()

            running.append(float(loss.detach()))
            step += 1
            if (batch_index + 1) % config.log_every == 0:
                window = running[-config.log_every:]
                print(f"  epoch {epoch} step {batch_index + 1}/{len(batches)} "
                      f"loss {sum(window) / len(window):.4f}", flush=True)

        elapsed = time.time() - started
        val = evaluate(model, val_examples, collator, config)
        logs.append(EpochLog(
            epoch=epoch,
            steps=len(batches),
            mean_loss=sum(running) / max(len(running), 1),
            seconds=round(elapsed, 1),
            examples_per_second=round(len(train_examples) / max(elapsed, 1e-9), 1),
            val=val,
        ))
        print(f"  epoch {epoch} done in {elapsed:.0f}s  "
              f"val {json.dumps(_brief(val), ensure_ascii=False)}", flush=True)

    result = {
        "config": asdict(config),
        "n_train": len(train_examples),
        "n_val": len(val_examples),
        "before": before,
        "epochs": [asdict(log) for log in logs],
        "after": logs[-1].val if logs else before,
        # The joint arm has no question cache by construction: its encoding depends
        # on the state, so every pair is unique and there is nothing to memoise.
        "question_cache": (
            {"hits": collator.cache.hits, "misses": collator.cache.misses}
            if isinstance(collator, Collator)
            else {"hits": 0, "misses": 0, "not_applicable": "joint encoding"}
        ),
        "joint_truncated": getattr(collator, "truncated", 0),
    }

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result


def _brief(metrics: dict[str, Any]) -> dict[str, Any]:
    """The few numbers worth printing every epoch."""
    out: dict[str, Any] = {}
    for kind in ("choice", "score", "bool"):
        entry = metrics.get(kind)
        if not entry:
            continue
        if kind == "score":
            out["score_rps"] = round(entry.get("rps", float("nan")), 4)
            out["score_acc"] = round(entry.get("accuracy", float("nan")), 4)
        else:
            out[f"{kind}_acc"] = round(entry.get("accuracy", float("nan")), 4)
        out[f"{kind}_ece"] = round(entry.get("ece", float("nan")), 4)
    return out
