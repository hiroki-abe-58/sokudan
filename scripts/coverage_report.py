"""Line coverage for the modules that carry the arithmetic.

    uv run python scripts/coverage_report.py
    uv run python scripts/coverage_report.py --fail-under 80

`coverage run -m pytest` does not work in this environment: coverage's import hook
makes `torch` execute twice and the second pass dies on
`RuntimeError: function '_has_torch_function' already has a docstring`. Importing
torch *before* coverage starts avoids it, which needs coverage driven in-process
rather than from the command line. That is all this script is.

Only three packages are measured, and on purpose. `sokudan.encoding` decides what the
model is shown, `sokudan.calibration` decides what the probabilities mean, and
`sokudan.model` is the head that produces them; a silent break in any of them changes
published numbers without failing anything. The training loop and the eval harnesses
are excluded because their failures are loud.
"""

from __future__ import annotations

import argparse
import sys

TARGETS = ["sokudan.encoding", "sokudan.calibration", "sokudan.model"]
TESTS = [
    "tests/test_encoding.py", "tests/test_encoding_joint.py",
    "tests/test_temperature.py", "tests/test_metrics.py",
    "tests/test_model.py", "tests/test_model_joint.py",
    "tests/test_schema.py", "tests/test_schema_aug.py",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-under", type=float, default=0.0)
    parser.add_argument("--tests", nargs="*", default=TESTS)
    args = parser.parse_args()

    # Import order is the fix, so it is pinned against the formatter.
    import torch  # noqa: F401, I001  -- before coverage, see the module docstring
    import coverage  # noqa: I001

    cov = coverage.Coverage(source=TARGETS, branch=False)
    cov.start()
    import pytest

    status = pytest.main(["-q", *args.tests])
    cov.stop()

    print()
    total = cov.report(show_missing=True, skip_empty=True)
    if status != 0:
        print(f"\ntests failed (pytest exit {status})")
        return int(status)
    if args.fail_under and total < args.fail_under:
        print(f"\ncoverage {total:.1f}% is below --fail-under {args.fail_under}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
