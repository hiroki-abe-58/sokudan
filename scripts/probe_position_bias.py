"""Thin shim. The probe moved into the package so the wheel carries it.

    uv run python scripts/probe_position_bias.py
        --model laya:convaiinnovations/laya-multilingual --out runs/position_bias.json

Equivalent to `sokudan probe-position`. Kept because `docs/baseline_ja.md` §6.2 and
`docs/baseline_en.md` cite this path.
"""

from sokudan.eval.position_bias import main

if __name__ == "__main__":
    raise SystemExit(main())
