"""Where the forward pass runs.

`Agent` does everything that does not depend on the array library -- parsing the schema,
encoding the questions and the state, padding them into one batch, temperatures, order
marginalization, building the answers -- and hands a backend the padded batch as numpy
arrays. The backend returns the head's probabilities as numpy. Nothing here imports
torch or MLX; each backend imports its own library when it is loaded.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Batch:
    """One request's questions, padded to a common length. All int64 except `ordered`.

    Padded marker slots repeat the row's first marker position and are 0 in
    `marker_mask`; padded tokens are `pad_id` and 0 in `attention_mask`.
    """

    input_ids: np.ndarray
    """`(N, L)` one sequence per question (joint), or the question sides (separate)."""
    attention_mask: np.ndarray
    marker_positions: np.ndarray
    """`(N, M)`."""
    marker_mask: np.ndarray
    """`(N, M)`, 1 for a real option."""
    ordered: np.ndarray
    """`(N,)` bool, True for `score` rows."""
    marker_positions_back: np.ndarray | None = None
    """`(N, M)`, `sandwich` only: the same slots in the second question block."""
    state_ids: np.ndarray | None = None
    """`(1, L_s)`, separate encoding only: the state, encoded once."""
    state_mask: np.ndarray | None = None

    @classmethod
    def build(cls, encoded: Sequence[Any], ordered: Sequence[bool], pad_id: int, *,
              sandwich: bool = False, state: Any = None) -> Batch:
        """Pad `EncodedJoint` / `EncodedQuestion`s (and an `EncodedState` for the
        separate encoding) into one batch."""
        n = len(encoded)
        sequence_len = max(len(e.input_ids) for e in encoded)
        n_markers = max(e.n_markers for e in encoded)

        input_ids = np.full((n, sequence_len), pad_id, dtype=np.int64)
        attention_mask = np.zeros((n, sequence_len), dtype=np.int64)
        marker_positions = np.zeros((n, n_markers), dtype=np.int64)
        marker_mask = np.zeros((n, n_markers), dtype=np.int64)
        back = np.zeros((n, n_markers), dtype=np.int64) if sandwich else None

        for row, item in enumerate(encoded):
            input_ids[row, : len(item.input_ids)] = item.input_ids
            attention_mask[row, : len(item.input_ids)] = 1
            positions = item.marker_positions
            marker_positions[row, : len(positions)] = positions
            marker_positions[row, len(positions):] = positions[0]
            if back is not None:
                back[row, : len(positions)] = item.marker_positions_back
                back[row, len(positions):] = item.marker_positions_back[0]
            marker_mask[row, : len(positions)] = 1

        state_ids = state_mask = None
        if state is not None:
            state_ids = np.asarray([state.input_ids], dtype=np.int64)
            state_mask = np.asarray([state.attention_mask], dtype=np.int64)
        return cls(input_ids, attention_mask, marker_positions, marker_mask,
                   np.asarray(ordered, dtype=bool), back, state_ids, state_mask)


class Backend(ABC):
    """A loaded model that turns a `Batch` into `(N, M)` probabilities."""

    name: str
    """`"torch"` or `"mlx"`."""
    device: str
    dtype: str
    encoding: str
    """`"joint"` or `"separate"`: how the checkpoint was trained to read its input."""
    input_order: str
    """The joint block order the checkpoint was trained with (`question_first` for
    checkpoints that predate the option, and for the separate encoding)."""
    backbone_id: str
    """The backbone the checkpoint was trained on; its tokenizer encodes the input."""

    @abstractmethod
    def probs(self, batch: Batch) -> np.ndarray:
        """`(N, M)` float32; each row sums to 1 over its real options, 0 on padding."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name} {self.device} {self.dtype}>"
