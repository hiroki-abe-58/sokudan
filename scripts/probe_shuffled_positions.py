"""Thin shim. The probe moved into the package so the wheel carries it.

    uv run python scripts/probe_shuffled_positions.py --out runs/position_bias_ja_shuffled.json

Equivalent to `sokudan probe-position --shuffle`. Kept because `docs/baseline_ja.md`
§6.2b cites this path.
"""

from sokudan.eval.position_bias_shuffled import main

if __name__ == "__main__":
    raise SystemExit(main())
