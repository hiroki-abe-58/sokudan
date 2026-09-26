"""The `sokudan` command line.

    sokudan probe-position --model sokudan:runs/v01_seed0/model.pt
        --bench data/bench_en.jsonl --lang en --out runs/position_bias_en.json
    sokudan probe-position --shuffle
        --model laya:convaiinnovations/laya-multilingual --out runs/condition_f.json

Only the position-bias probe is exposed so far, because it is the one piece of this
repository that is useful against a model that is not `sokudan`. `docs/baseline_ja.md`
§6.2 reports `laya-multilingual` putting its argmax on the first presented option in
0, 0, 1, 1 and 0 of 300 items across five schema variants, and §6.2b repeats it with
the options shuffled per item. Anyone checking a third model for the same failure
should not have to clone the repository and read `scripts/`.

The subcommand delegates to `sokudan.eval.position_bias` rather than reimplementing it;
`scripts/probe_position_bias.py` is a shim onto the same function.
Two copies of a measurement procedure is how the copies drift, and a probe that
disagrees with the numbers in the docs is worse than no probe.
"""

from __future__ import annotations

import argparse
import sys

USAGE = """sokudan <command> [options]

commands:
  probe-position    measure position bias on an ordinal question
                    (--shuffle runs condition F: options shuffled per item)
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="sokudan", usage=USAGE, add_help=False,
        description="Japanese System One decision model.",
    )
    parser.add_argument("command", nargs="?", choices=["probe-position"])
    parser.add_argument("-h", "--help", action="store_true", dest="help")
    known, rest = parser.parse_known_args(argv)

    # `--help` with a command reaches that command's own parser; without one it is
    # the top-level listing. Swallowing it here would hide every probe option.
    if known.command is None:
        print(USAGE)
        return 0 if known.help else 2

    if known.command == "probe-position":
        # `--shuffle` selects condition F, which is a different script rather than a
        # flag on the first one: it re-orders the options per item and so cannot
        # report the fixed-schema table the other conditions share.
        if "--shuffle" in rest:
            from sokudan.eval.position_bias_shuffled import main as run
            rest = [a for a in rest if a != "--shuffle"]
        else:
            from sokudan.eval.position_bias import main as run
        return run(rest + (["--help"] if known.help else []))

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
