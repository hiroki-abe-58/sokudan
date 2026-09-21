"""`sokudan` as a `bench_ja` baseline (SOKUDAN_SPEC.md §9).

This deliberately implements the same `Baseline` shape as `LayaBaseline` and returns
the same `BaselineOutput`, so `scripts/run_baseline_ja.py` scores it with the same
metrics module, the same probability floor and the same reliability-diagram code.

That is not tidiness. §9 lists the baselines that have to be present for a number to
mean anything, and a comparison is only fair if both sides go through one scoring
path. Writing a separate evaluation for our own model is how a project ends up
reporting a number nobody else can reproduce.

`bench_ja` is unseen at training time in two senses that both matter: the documents
were never trained on, and none of its three schemas -- nor their option strings --
appear anywhere in the training catalogue (`scripts/build_train_data.py` checks).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sokudan.calibration.temperature import apply_temperature
from sokudan.eval.baselines import BaselineOutput
from sokudan.eval.bench_ja import DEPARTMENTS, URGENCY_LEVELS, BenchItem, bench_questions
from sokudan.schema.question import parse_question
from sokudan.train.dataset import Example
from sokudan.train.loop import TrainConfig, build_collator, run_model


class SokudanBaseline:
    """A trained checkpoint answering the bench_ja questions.

    Args:
        checkpoint: a `model.pt` written by `scripts/train.py`.
        label: the row name in the report.
        temperatures: optional `{(kind, n_options): T}` from Stage 2. Applied exactly
            as §8 Stage 2 specifies -- one temperature per (question type, option
            count) bucket, fitted on validation data, never on `bench_ja`.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        label: str = "sokudan-ja-310m",
        *,
        device: str = "cuda",
        batch_size: int = 16,
        temperatures: dict[tuple[str, int], float] | None = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.name = label
        self.device = device
        self.batch_size = batch_size
        self.temperatures = temperatures or {}
        self._model: Any | None = None
        self._encoding = "separate"
        self._tokenizer: Any | None = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is None:
            from transformers import AutoTokenizer

            from sokudan.config import BACKBONE_MODEL_ID

            blob = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            config = blob.get("config", {})
            # The checkpoint records which arm trained it (`scripts/train.py`), so a
            # joint checkpoint cannot be scored through the separate path by
            # forgetting a flag -- that would report a number for an architecture
            # that was never trained.
            self._encoding = config.get("encoding", "separate")
            if self._encoding == "joint":
                from sokudan.model.joint import SokudanJointModel

                model = SokudanJointModel.from_pretrained_backbone(
                    config.get("backbone", BACKBONE_MODEL_ID)
                )
            else:
                from sokudan.model.sokudan import SokudanModel

                model = SokudanModel.from_pretrained_backbone(
                    config.get("backbone", BACKBONE_MODEL_ID),
                    n_head_layers=config.get("n_head_layers", 2),
                )
            model.load_state_dict(blob["state_dict"])
            model.to(self.device).eval()
            self._model = model
            self._tokenizer = AutoTokenizer.from_pretrained(
                config.get("backbone", BACKBONE_MODEL_ID)
            )
        return self._model, self._tokenizer

    def _temperature_for(self, kind: str, n_options: int) -> float:
        return self.temperatures.get((kind, n_options), 1.0)

    @torch.no_grad()
    def run(self, items: list[BenchItem]) -> BaselineOutput:
        model, tokenizer = self._load()
        collator = build_collator(
            tokenizer,
            TrainConfig(device=self.device, encoding=self._encoding,
                        max_state_tokens=1024),
        )
        questions = bench_questions()

        # Three questions per item, each a separate row. The state is encoded once
        # per row here rather than once per item; §10's caching work is what removes
        # that, and it is a latency question, not an accuracy one.
        specs = [
            ("department", "choice", parse_question(questions["department"])),
            ("urgency", "score", parse_question(questions["urgency"])),
            ("churn", "bool", parse_question(questions["churn"])),
        ]

        results: dict[str, list[np.ndarray]] = {name: [] for name, _, _ in specs}
        latencies: list[float] = []

        for start in range(0, len(items), self.batch_size):
            chunk = items[start:start + self.batch_size]
            started = time.perf_counter()
            for name, kind, question in specs:
                examples = [
                    Example(state=item.state, question=question, label=0,
                            ordered=(kind == "score"), doc_id=item.item_id,
                            domain="bench_ja", attribute=name, kind=kind)
                    for item in chunk
                ]
                batch = collator(examples).to(self.device)
                out = run_model(model, batch)
                probs = out.probs.float().cpu().numpy()
                n_options = int(batch.marker_mask[0].sum())
                temperature = self._temperature_for(kind, n_options)
                if temperature != 1.0:
                    probs = apply_temperature(probs[:, :n_options], temperature)
                else:
                    probs = probs[:, :n_options]
                results[name].append(probs)
            elapsed = time.perf_counter() - started
            latencies.extend([elapsed / len(chunk)] * len(chunk))

        choice_probs = np.vstack(results["department"])
        score_probs = np.vstack(results["urgency"])
        bool_probs = np.vstack(results["churn"])

        assert choice_probs.shape[1] == len(DEPARTMENTS)
        assert score_probs.shape[1] == len(URGENCY_LEVELS)

        return BaselineOutput(
            name=self.name,
            choice_probs=choice_probs,
            score_probs=score_probs,
            bool_p_true=bool_probs[:, 1],
            per_item_latency_s=latencies,
            parse_failures=0,
            parse_attempts=len(items) * 3,
            notes={
                "checkpoint": str(self.checkpoint),
                "description": "head logits read directly; no text generation, nothing to parse",
                "temperatures": {f"{k[0]}/{k[1]}": v for k, v in self.temperatures.items()},
                "unseen_schema": (
                    "bench_ja's three schemas and their option strings do not appear in "
                    "the training catalogue; verified by scripts/build_train_data.py"
                ),
                "latency_caveat": (
                    "measured with the state re-encoded per question; the per-request "
                    "single-encode path of §6.2 is not what this number reflects"
                ),
            },
        )
