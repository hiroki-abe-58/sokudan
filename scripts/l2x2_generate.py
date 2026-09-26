"""Main generation for docs/length_2x2.md: batches until a stop rule fires.

    uv run python scripts/l2x2_generate.py --dir <snapshot> --start-epoch <unix time>

Each batch is one run of the v0.1 pipeline (`scripts/l2x2_pipeline.py generate`,
`BATCH` documents requested, seed `SEED_BASE + k`) written to
`data/l2x2/batch_<k>.jsonl`, so a stop never loses finished work. After each batch the
log gets one line, with the wall-clock time read at that moment: batches, documents,
on-target documents (state 450-800 tokens), hit rate, verification pass rate.

Stop rules, checked after every batch (docs/length_2x2.md §3):

- on-target documents, trial included, reach `L_TARGET`;
- the next batch would end past the 3-hour cap (judged by the longest batch so far);
- a batch's verification pass rate falls below `PASS_FLOOR` (half the trial's).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import sokudan.config  # noqa: F401
from scripts.l2x2_docstats import stats

BATCH = 200
SEED_BASE = 20260926
L_TARGET = 824
CAP_S = 3 * 3600
PASS_FLOOR = 0.4516  # 0.9032 / 2, the trial's pass rate halved
TRIAL = Path("data/l2x2/trial_docs.jsonl")
LOG = Path("runs/l2x2/generation.log")


def log(line: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{stamp}  {line}\n")
    print(f"{stamp}  {line}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--start-epoch", type=float, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    trial = stats(TRIAL, tokenizer)
    on_target, kept, requested = trial["hit"], trial["docs_kept"], trial["requested"]
    log(f"start: trial counted, on-target {on_target}, kept {kept}, requested {requested}")
    longest = 0.0
    k = 0
    while True:
        elapsed = time.time() - args.start_epoch
        if on_target >= L_TARGET:
            log(f"STOP target reached: on-target {on_target} >= {L_TARGET}")
            break
        if k and elapsed + longest > CAP_S:
            log(f"STOP cap: elapsed {elapsed:.0f}s + longest batch {longest:.0f}s > {CAP_S}s")
            break
        k += 1
        out = Path(f"data/l2x2/batch_{k:02d}.jsonl")
        began = time.time()
        code = subprocess.run(
            [sys.executable, "scripts/l2x2_pipeline.py", "generate", "--dir", args.dir,
             "--docs", str(BATCH), "--seed", str(SEED_BASE + k), "--out", str(out)],
        ).returncode
        took = time.time() - began
        longest = max(longest, took)
        if code != 0 or not out.exists():
            log(f"STOP batch {k} failed with exit code {code}")
            break
        s = stats(out, tokenizer)
        on_target += s["hit"]
        kept += s["docs_kept"]
        requested += s["requested"]
        log(f"batch {k}: {took:.0f}s, requested {s['requested']}, kept {s['docs_kept']}, "
            f"on-target {s['hit']} ({s['hit_rate']:.3f}), pass {s['pass']['rate']:.4f}, "
            f"endings {s['sentence_final_endings']}/{s['docs_kept']} | cumulative on-target "
            f"{on_target}, kept {kept}, requested {requested}, "
            f"elapsed {time.time() - args.start_epoch:.0f}s")
        if s["pass"]["rate"] < PASS_FLOOR:
            log(f"STOP pass rate {s['pass']['rate']:.4f} < {PASS_FLOOR}")
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
