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

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from sokudan.encoding.question import QuestionEncoderCache, encode_joint
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
        encoding: str = "separate",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperatures = temperatures or {}
        self.max_state_tokens = max_state_tokens
        self.cache = QuestionEncoderCache()
        self.encoding = encoding
        if encoding not in ("separate", "joint"):
            raise ValueError(f"unknown encoding {encoding!r}")

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
        *,
        order_marginalize: bool | Mapping[str, bool] = False,
    ) -> dict[str, Any]:
        """Answer every question about one state in a single pass per question batch.

        `order_marginalize=True` (joint encoding only; off by default, which keeps the
        single pass) asks each question again with its options reordered -- the K cyclic
        shifts of a `choice`, the original and reversed levels of a `score`, the two slot
        orders of a `bool` -- maps every answer back to the original order and averages
        the probabilities before any temperature (`sokudan.order_marginalize`,
        docs/order_marginalization.md). All the reorderings go through the backbone in one
        batch, one row each. A mapping picks the types, e.g.
        `{"choice": False, "score": True, "bool": False}` (types left out are off); the
        other types are asked once, in the same batch. All off is the single pass.
        """
        from sokudan.calibration.temperature import apply_temperature

        if not questions:
            raise ValueError("no questions given")

        parsed = parse_questions(questions)
        prepared = {
            qid: _Prepared(question, question.type, list(question.labels))
            for qid, question in parsed.items()
        }

        from sokudan.order_marginalize import marginalized_types

        text = self._state_text(state)
        order = list(prepared)
        types = marginalized_types(order_marginalize)
        if types:
            if self.encoding != "joint":
                raise NotImplementedError("order_marginalize needs the joint encoding")
            from sokudan.order_marginalize import combine, variants

            expanded: dict[str, _Prepared] = {}
            perms: dict[str, list[tuple[str, list[int]]]] = {}
            for qid in order:
                item = prepared[qid]
                if item.kind not in types:
                    expanded[qid] = item
                    continue
                for i, (question, perm) in enumerate(variants(item.question)):
                    key = f"{qid}\x1f{i}"
                    expanded[key] = _Prepared(question, item.kind, list(item.labels))
                    perms.setdefault(qid, []).append((key, perm))
            rows, n, state_tokens, truncated, question_tokens = self._forward(text, expanded)
            raw = {qid: (combine([rows[key] for key, _ in perms[qid]],
                                 [perm for _, perm in perms[qid]])
                         if qid in perms else rows[qid])
                   for qid in order}
        else:
            raw, n, state_tokens, truncated, question_tokens = self._forward(text, prepared)

        answers: dict[str, Any] = {}
        calibrated: list[str] = []
        for qid in order:
            item = prepared[qid]
            k = len(item.labels)
            row_probs = raw[qid]
            temperature = self.temperatures.get((item.kind, k), 1.0)
            if temperature != 1.0:
                row_probs = apply_temperature(row_probs.reshape(1, -1), temperature)[0]
                calibrated.append(qid)

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

        usage = {
            "state_tokens": state_tokens,
            "state_truncated": truncated,
            "question_tokens": max(question_tokens, 0),
            "backbone_passes": n if self.encoding == "joint" else 2,
            "output_tokens": 0,
        }
        if types:
            usage["order_marginalized"] = (True if types == {"choice", "score", "bool"}
                                           else sorted(types))
        return {
            "model": "sokudan-ja-310m",
            "answers": answers,
            "encoding": self.encoding,
            # v0.2.1 (docs/calibration.md §10): whether a temperature was applied to any
            # answer, and to which (by default only bool answers are calibrated).
            "calibrated": bool(calibrated),
            "calibrated_answers": calibrated,
            "usage": usage,
        }



    def _forward(self, text: str, prepared: dict[str, _Prepared]):
        """One batch through the model: `({qid: probabilities over its K options}, rows,
        state tokens, truncated, question tokens)`."""
        from sokudan.encoding.state import encode_state

        order = list(prepared)
        n = len(order)
        pad_id = self.tokenizer.pad_token_id
        device = torch.device(self.device)

        if self.encoding == "joint":
            # v0.1. The state is re-encoded per question, so there is nothing to
            # cache and latency grows with the number of questions
            # (`docs/architecture.md` §1.2).
            encoded = [
                encode_joint(prepared[qid].question, text, self.tokenizer,
                             max_tokens=self.max_state_tokens,
                             input_order=getattr(self.model, "input_order",
                                                 "question_first"))
                for qid in order
            ]
            state_tokens = max((e.n_state_tokens for e in encoded), default=0)
            truncated = any(e.truncated for e in encoded)
        else:
            encoded_state = encode_state(
                text, self.tokenizer, max_tokens=self.max_state_tokens
            )
            encoded = [self.cache.get(prepared[qid].question, self.tokenizer)
                       for qid in order]
            state_tokens = encoded_state.n_tokens
            truncated = encoded_state.truncated

        sequence_len = max(len(e.input_ids) for e in encoded)
        n_markers = max(e.n_markers for e in encoded)

        input_ids = torch.full((n, sequence_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((n, sequence_len), dtype=torch.long)
        marker_positions = torch.zeros((n, n_markers), dtype=torch.long)
        marker_mask = torch.zeros((n, n_markers), dtype=torch.long)
        ordered = torch.zeros(n, dtype=torch.bool)
        sandwich = getattr(self.model, "input_order", "question_first") == "sandwich"
        back = torch.zeros((n, n_markers), dtype=torch.long) if sandwich else None

        for row, qid in enumerate(order):
            item = encoded[row]
            input_ids[row, : len(item.input_ids)] = torch.tensor(item.input_ids)
            attention_mask[row, : len(item.input_ids)] = 1
            positions = item.marker_positions
            marker_positions[row, : len(positions)] = torch.tensor(positions)
            marker_positions[row, len(positions):] = positions[0]
            if back is not None:
                back[row, : len(positions)] = torch.tensor(item.marker_positions_back)
                back[row, len(positions):] = item.marker_positions_back[0]
            marker_mask[row, : len(positions)] = 1
            ordered[row] = is_ordered(prepared[qid].question)

        if self.encoding == "joint":
            extra = {} if back is None else {"marker_positions_back": back.to(device)}
            out = self.model(
                input_ids.to(device), attention_mask.to(device),
                marker_positions.to(device), marker_mask.to(device), ordered.to(device),
                **extra,
            )
            question_tokens = int(attention_mask.sum()) - state_tokens * n
        else:
            state_ids = torch.tensor(
                [encoded_state.input_ids], dtype=torch.long).to(device)
            state_mask = torch.tensor(
                [encoded_state.attention_mask], dtype=torch.long).to(device)
            out = self.model(
                state_ids, state_mask,
                input_ids.to(device), attention_mask.to(device),
                marker_positions.to(device), marker_mask.to(device), ordered.to(device),
            )
            question_tokens = int(attention_mask.sum())
        probs = out.probs.float().cpu().numpy()
        rows = {qid: probs[row, :len(prepared[qid].labels)]
                for row, qid in enumerate(order)}
        return rows, n, state_tokens, truncated, question_tokens



def _resolve_checkpoint(checkpoint: str | Path) -> tuple[dict, Path | None]:
    """Accept a `.pt` file, a directory of safetensors, or a Hub repo id.

    The published weights are safetensors in a repository, and the training script
    writes a `.pt`. Both have to load through one entry point, or the Quickstart in
    the README describes something the package cannot do.

    Returns the state blob and, when the checkpoint came from a directory or the Hub,
    the directory it came from -- so a sibling `temperatures.json` can be found.
    """
    from pathlib import Path as _Path

    path = _Path(str(checkpoint))

    if path.is_file():
        return torch.load(str(path), map_location="cpu", weights_only=False), path.parent

    if path.is_dir():
        return _load_directory(path), path

    # Not on disk: treat it as `repo_id` or `repo_id@revision`.
    repo_id, _, revision = str(checkpoint).partition("@")
    if "/" not in repo_id:
        raise FileNotFoundError(
            f"{checkpoint!r} is neither a file, a directory, nor a Hub repo id "
            f"(expected something like 'GeneLab/sokudan-ja-310m')"
        )
    from huggingface_hub import snapshot_download

    local = _Path(snapshot_download(
        repo_id, revision=revision or None,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*"],
    ))
    return _load_directory(local), local


def _load_directory(directory: Path) -> dict:
    """Read `model.safetensors` + `config.json` into the shape `torch.load` returns."""
    from safetensors.torch import load_file

    weights = directory / "model.safetensors"
    config_path = directory / "config.json"
    if not weights.exists():
        raise FileNotFoundError(f"{directory} has no model.safetensors")
    config = (
        json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    )
    return {"state_dict": load_file(str(weights)), "config": config}


def read_temperatures(path: str | Path) -> dict[tuple[str, int], float]:
    """`{(type, option count): T}` from a temperatures file: `temperatures.json` from
    `scripts/calibrate.py` or `calibration.json` from `scripts/calibration_heldout.py`
    (docs/calibration.md) -- both hold `{"temperatures": {"bool/2": T, ...}}`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    parsed: dict[tuple[str, int], float] = {}
    for key, value in data["temperatures"].items():
        kind, _, count = key.partition("/")
        parsed[(kind, int(count))] = float(value)
    return parsed


DEFAULT_CALIBRATION = "calibration.json"
"""v0.2.1: the calibration file `load` looks for beside the checkpoint by default. Only its
bool temperatures are applied by default (docs/calibration.md §8-§10)."""


def resolve_temperatures(
    temperatures: str | Path | dict[tuple[str, int], float] | None,
    resolved_dir: Path | None,
) -> dict[tuple[str, int], float]:
    """What `load` applies.

    - `DEFAULT_CALIBRATION` (the default): the `calibration.json` beside the checkpoint
      (the `.pt` file's directory, the safetensors directory, or the Hub download), bool
      temperatures only; no file there means no calibration.
    - `None`: no calibration (the raw head probabilities).
    - a bare file name: that file beside the checkpoint, else the path as given; any other
      path: that file (it must exist); a dict: as given. All of these are applied whole.
    """
    if temperatures is None:
        return {}
    if isinstance(temperatures, dict):
        return dict(temperatures)
    default = str(temperatures) == DEFAULT_CALIBRATION
    path = Path(temperatures)
    if (resolved_dir is not None and not path.exists()
            and path.name == str(temperatures)):
        # A bare name beside a Hub / directory / .pt checkpoint: the file that came with
        # the weights rather than the working directory's.
        candidate = resolved_dir / str(temperatures)
        if candidate.exists():
            path = candidate
    if default:
        if not path.exists():
            return {}
        return {k: t for k, t in read_temperatures(path).items() if k[0] == "bool"}
    return read_temperatures(path)


def load(
    checkpoint: str | Path,
    *,
    device: str | None = None,
    temperatures: str | Path | dict[tuple[str, int], float] | None = DEFAULT_CALIBRATION,
) -> Agent:
    """Load a checkpoint, from disk or from the Hub.

    Args:
        checkpoint: a local `model.pt`, a local directory holding
            `model.safetensors` + `config.json`, or a Hub repo id such as
            `GeneLab/sokudan-ja-310m`. A repo id may carry a revision after `@`
            (`GeneLab/sokudan-ja-310m@seed1`).
        device: defaults to cuda when available.
        temperatures: by default (v0.2.1) the `calibration.json` shipped beside the
            checkpoint, **bool temperatures only** -- score and choice stay raw
            (docs/calibration.md §10). `None` turns calibration off (the raw head
            probabilities, as in v0.2). A path to a `temperatures.json` /
            `calibration.json`, or a dict, applies that file or dict whole. Each
            response says whether a temperature was applied (`calibrated`).
    """
    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID
    from sokudan.model.sokudan import SokudanModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    blob, resolved_dir = _resolve_checkpoint(checkpoint)
    parsed = resolve_temperatures(temperatures, resolved_dir)
    config = blob.get("config", {})
    backbone_id = config.get("backbone", BACKBONE_MODEL_ID)
    encoding = config.get("encoding", "separate")

    # Absent in checkpoints that predate `--local-attention`; None keeps the
    # pretrained window, which is what those were trained at.
    local_attention = config.get("local_attention")

    if encoding == "joint":
        from sokudan.model.joint import SokudanJointModel

        model = SokudanJointModel.from_pretrained_backbone(
            backbone_id, local_attention=local_attention,
            input_order=config.get("input_order", "question_first"),
        )
    else:
        model = SokudanModel.from_pretrained_backbone(
            backbone_id, n_head_layers=config.get("n_head_layers", 2),
            local_attention=local_attention,
        )
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()

    return Agent(model, AutoTokenizer.from_pretrained(backbone_id),
                 device=device, temperatures=parsed, encoding=encoding)
