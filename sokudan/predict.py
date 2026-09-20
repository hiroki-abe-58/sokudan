"""The three-line entry point (SOKUDAN_SPEC.md §10, §11).

    import sokudan
    agent = sokudan.load("runs/s0/model.pt")
    result = agent.predict({"body": "先月の請求が二重になっています"}, questions)

The response shape follows `laya`'s, which was read off an actual call rather than
guessed -- `{"answers": {qid: {...}}, "usage": {...}}`, with `choice`/`score`/`noul`
naming each primitive's answer. Matching a shape we verified by running the library
beats inventing one from a specification we cannot check.

What this does **not** share with that API is where the probabilities come from. They
are the head's logits read directly, not a model's own report of its confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from sokudan.encoding.question import QuestionEncoderCache
from sokudan.schema.question import Question, is_ordered, parse_questions


@dataclass
class _Prepared:
    question: Question
    kind: str
    labels: list[str]


class Agent:
    """A loaded checkpoint that answers typed questions about a state."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: str = "cuda",
        temperatures: dict[tuple[str, int], float] | None = None,
        max_state_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperatures = temperatures or {}
        self.max_state_tokens = max_state_tokens
        self.cache = QuestionEncoderCache()

    @staticmethod
    def _state_text(state: str | dict[str, Any] | list[Any]) -> str:
        """Accept a string, a JSON-ish dict, or a conversation, like `laya` does."""
        if isinstance(state, str):
            return state
        if isinstance(state, dict):
            return "\n".join(f"{key}: {value}" for key, value in state.items())
        if isinstance(state, list):
            parts = []
            for turn in state:
                if isinstance(turn, dict):
                    parts.append(
                        f"{turn.get('role', '')}: {turn.get('content', turn)}".strip(": ")
                    )
                else:
                    parts.append(str(turn))
            return "\n".join(parts)
        raise TypeError(f"unsupported state type {type(state).__name__}")

    @torch.no_grad()
    def predict(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, dict[str, Any] | Question],
    ) -> dict[str, Any]:
        """Answer every question about one state in a single pass per question batch."""
        from sokudan.calibration.temperature import apply_temperature
        from sokudan.encoding.state import encode_state

        if not questions:
            raise ValueError("no questions given")

        parsed = parse_questions(questions)
        prepared = {
            qid: _Prepared(question, question.type, list(question.labels))
            for qid, question in parsed.items()
        }

        text = self._state_text(state)
        encoded_state = encode_state(text, self.tokenizer, max_tokens=self.max_state_tokens)
        encoded_questions = {
            qid: self.cache.get(item.question, self.tokenizer)
            for qid, item in prepared.items()
        }

        order = list(prepared)
        n = len(order)
        question_len = max(len(encoded_questions[q].input_ids) for q in order)
        n_markers = max(encoded_questions[q].n_markers for q in order)
        pad_id = self.tokenizer.pad_token_id

        question_ids = torch.full((n, question_len), pad_id, dtype=torch.long)
        question_mask = torch.zeros((n, question_len), dtype=torch.long)
        marker_positions = torch.zeros((n, n_markers), dtype=torch.long)
        marker_mask = torch.zeros((n, n_markers), dtype=torch.long)
        ordered = torch.zeros(n, dtype=torch.bool)

        for row, qid in enumerate(order):
            encoded = encoded_questions[qid]
            question_ids[row, : len(encoded.input_ids)] = torch.tensor(encoded.input_ids)
            question_mask[row, : len(encoded.input_ids)] = 1
            positions = encoded.marker_positions
            marker_positions[row, : len(positions)] = torch.tensor(positions)
            marker_positions[row, len(positions):] = positions[0]
            marker_mask[row, : len(positions)] = 1
            ordered[row] = is_ordered(prepared[qid].question)

        device = torch.device(self.device)
        # The state is encoded once and broadcast across every question -- the
        # structural point of §6.2. Not a speed claim until §9 measures both sides.
        state_ids = torch.tensor([encoded_state.input_ids], dtype=torch.long).to(device)
        state_mask = torch.tensor([encoded_state.attention_mask], dtype=torch.long).to(device)

        out = self.model(
            state_ids, state_mask,
            question_ids.to(device), question_mask.to(device),
            marker_positions.to(device), marker_mask.to(device), ordered.to(device),
        )
        probs = out.probs.float().cpu().numpy()

        answers: dict[str, Any] = {}
        for row, qid in enumerate(order):
            item = prepared[qid]
            k = len(item.labels)
            row_probs = probs[row, :k]
            temperature = self.temperatures.get((item.kind, k), 1.0)
            if temperature != 1.0:
                row_probs = apply_temperature(row_probs.reshape(1, -1), temperature)[0]

            entry: dict[str, Any] = {"type": item.kind}
            if item.kind == "score":
                entry["score"] = float((row_probs * range(k)).sum())
                entry["probabilities"] = {str(i): round(float(p), 4)
                                          for i, p in enumerate(row_probs)}
                entry["legend"] = {str(i): label for i, label in enumerate(item.labels)}
            elif item.kind == "bool":
                entry["noul"] = round(float(row_probs[1]), 4)
            else:
                best = int(row_probs.argmax())
                entry["choice"] = item.labels[best]
                entry["probabilities"] = {label: round(float(p), 4)
                                          for label, p in zip(item.labels, row_probs,
                                                              strict=True)}
            if item.kind != "bool":
                entry["confidence"] = round(float(row_probs.max()), 4)
            answers[qid] = entry

        return {
            "model": "sokudan-ja-310m",
            "answers": answers,
            "usage": {
                "state_tokens": encoded_state.n_tokens,
                "state_truncated": encoded_state.truncated,
                "question_tokens": int(question_mask.sum()),
                "output_tokens": 0,
            },
        }


def load(
    checkpoint: str | Path,
    *,
    device: str | None = None,
    temperatures: str | Path | dict[tuple[str, int], float] | None = None,
) -> Agent:
    """Load a checkpoint written by `scripts/train.py`.

    Args:
        checkpoint: path to `model.pt`.
        device: defaults to cuda when available.
        temperatures: a `temperatures.json` from `scripts/calibrate.py`, or a dict.
            Without it the model reports its raw head probabilities, which are
            **not calibrated** -- see the README's Limits section.
    """
    import json

    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID
    from sokudan.model.sokudan import SokudanModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    blob = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    config = blob.get("config", {})
    backbone_id = config.get("backbone", BACKBONE_MODEL_ID)

    model = SokudanModel.from_pretrained_backbone(
        backbone_id, n_head_layers=config.get("n_head_layers", 2)
    )
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()

    parsed: dict[tuple[str, int], float] = {}
    if isinstance(temperatures, (str, Path)):
        data = json.loads(Path(temperatures).read_text(encoding="utf-8"))
        for key, value in data["temperatures"].items():
            kind, _, count = key.partition("/")
            parsed[(kind, int(count))] = float(value)
    elif isinstance(temperatures, dict):
        parsed = temperatures

    return Agent(model, AutoTokenizer.from_pretrained(backbone_id),
                 device=device, temperatures=parsed)
