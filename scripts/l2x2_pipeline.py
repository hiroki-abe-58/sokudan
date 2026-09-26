"""v0.1's data pipeline, re-run with longer states (docs/length_2x2.md).

    uv run python scripts/l2x2_pipeline.py export  --dir <snapshot>
    uv run python scripts/l2x2_pipeline.py generate --dir <snapshot> --docs 30 --out <docs.jsonl>
    uv run python scripts/l2x2_pipeline.py expand  --dir <snapshot> --docs-file <docs.jsonl> \
        --out-dir <dir>

**Why a snapshot.** `data/train_v2b.jsonl` (v0.1's training set) is reproduced byte for
byte only by the code of commit 68d217b: its `scripts/build_intent_train.py
--bool-share 0.63` on `data/docs_v2.jsonl`. Today's scripts produce a different set
(the catalogue, the defaults and the filters have all moved since). And the 68d217b
generator's plans reproduce every one of the 4,833 `docs_v2` documents' gold labels.
So the long data is made by that code, not by today's -- `git archive 68d217b` into a
scratch directory, never a checkout of this repository.

**What is changed** (`PATCHES`, each asserted to apply exactly once):

1. The length instruction in the prompt: v0.1's three buckets (100-550 characters) are
   replaced by one, "1600〜2200文字" -- the longest bucket of the later long-state
   corpus, which put 58-67% of its documents in the 450-800 token target. Nothing is
   said about where in the document anything goes.
2. The validator's character ceiling, 1600 -> 3000, the value the later long-state
   generator uses; at 1600 the prompt would ask for documents the validator rejects.
3. The generation `max_tokens`, 1000 -> 2400. At 1000 the long-state corpora were cut
   off mid-sentence in 8-20% of their longest bucket (measured on
   `data/docs_long*.jsonl`); a cut-off ending would remove exactly the tail this
   experiment is about. It is an output cap, not an instruction.
4. The document id prefix `i2-` -> `l2x-`, so the new documents cannot collide with
   v0.1's ids.

Everything else -- generator and verifier model (the same qwen3 model in both roles,
as in v0.1), temperature 0.9, planning, the training-split attribute pool (v0.1's 19
trainable attributes, no held-out attribute), banned words, the blind verification and
the expansion rules -- is v0.1's code, unchanged. `--val-fraction 0` puts every
document in the training split.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

V01_COMMIT = "68d217b"
ROOT = Path(__file__).resolve().parent.parent

PATCHES = [
    ("sokudan/data/builders/synthetic.py",
     'LENGTHS = [("短", "100〜180文字"), ("中", "200〜320文字"), ("長", "350〜550文字")]',
     'LENGTHS = [("極めて長い", "1600〜2200文字")]  # l2x2: length only'),
    ("scripts/build_intent_corpus.py", "MAX_CHARS = 1600", "MAX_CHARS = 3000  # l2x2"),
    ("scripts/build_intent_corpus.py", "temperature=temperature, max_tokens=1000,",
     "temperature=temperature, max_tokens=2400,  # l2x2: room to finish"),
    ("scripts/build_intent_corpus.py", '"doc_id": f"i2-{index:06d}",',
     '"doc_id": f"l2x-{index:06d}",  # l2x2: own namespace'),
]


def export(directory: Path) -> None:
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    archive = subprocess.run(["git", "archive", V01_COMMIT, "sokudan", "scripts"],
                             cwd=ROOT, check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(directory)], input=archive, check=True)
    for relative, old, new in PATCHES:
        path = directory / relative
        text = path.read_text(encoding="utf-8")
        if text.count(old) != 1:
            raise SystemExit(f"patch does not apply exactly once: {relative}: {old!r}")
        path.write_text(text.replace(old, new), encoding="utf-8")
    shutil.copy(ROOT / ".env", directory / ".env")
    print(f"exported {V01_COMMIT} to {directory}, {len(PATCHES)} patches applied")


def run_in(directory: Path, module: str, *args: str) -> int:
    env = {**os.environ, "PYTHONPATH": str(directory)}
    command = ["uv", "run", "--project", str(ROOT), "python", "-m", module, *args]
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=directory, env=env).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("export")
    p.add_argument("--dir", required=True)
    p = sub.add_parser("generate")
    p.add_argument("--dir", required=True)
    p.add_argument("--docs", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--concurrency", type=int, default=16)
    p = sub.add_parser("expand")
    p.add_argument("--dir", required=True)
    p.add_argument("--docs-file", required=True)
    p.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    directory = Path(args.dir).resolve()
    if args.command == "export":
        export(directory)
        return 0
    if args.command == "generate":
        return run_in(directory, "scripts.build_intent_corpus",
                      "--docs", str(args.docs), "--out", str(Path(args.out).resolve()),
                      "--seed", str(args.seed), "--val-fraction", "0",
                      "--paraphrase-sample", "0", "--concurrency", str(args.concurrency))
    # v0.1's exact expansion flags (reproduces data/train_v2b.jsonl from docs_v2).
    return run_in(directory, "scripts.build_intent_train",
                  "--docs", str(Path(args.docs_file).resolve()),
                  "--out-dir", str(Path(args.out_dir).resolve()), "--bool-share", "0.63")


if __name__ == "__main__":
    sys.exit(main())
