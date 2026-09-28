"""The backend boundary: the padded numpy batch, and `import sokudan` without torch."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

import numpy as np

from sokudan.backends import Batch


@dataclass
class _Encoded:
    input_ids: list[int]
    marker_positions: list[int]
    marker_positions_back: list[int] | None = None

    @property
    def n_markers(self) -> int:
        return len(self.marker_positions)


def test_rows_are_padded_and_padded_markers_repeat_the_first_position():
    batch = Batch.build([_Encoded([1, 7, 5, 5, 2], [2, 3]),
                         _Encoded([1, 5, 5, 5], [1, 2, 3])], [False, True], pad_id=3)
    assert batch.input_ids.tolist() == [[1, 7, 5, 5, 2], [1, 5, 5, 5, 3]]
    assert batch.attention_mask.tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]
    assert batch.marker_positions.tolist() == [[2, 3, 2], [1, 2, 3]]
    assert batch.marker_mask.tolist() == [[1, 1, 0], [1, 1, 1]]
    assert batch.ordered.tolist() == [False, True]
    assert batch.input_ids.dtype == np.int64 and batch.ordered.dtype == bool
    assert batch.marker_positions_back is None and batch.state_ids is None


def test_sandwich_back_positions_pad_the_same_way():
    batch = Batch.build([_Encoded([1, 5, 2, 5, 2], [1], [3]),
                         _Encoded([1, 5, 5, 2, 5, 5, 2], [1, 2], [4, 5])],
                        [False, False], pad_id=3, sandwich=True)
    assert batch.marker_positions_back.tolist() == [[3, 3], [4, 5]]


def test_import_sokudan_does_not_import_torch():
    code = "import sys, sokudan; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == "False"
