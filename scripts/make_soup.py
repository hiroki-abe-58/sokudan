"""Rebuild the v0.2 soup from its members and check it against the recorded hash.

    uv run python scripts/make_soup.py --out runs/release_candidate/rebuilt/model.pt

v0.2 (`S8_old`, docs/release_candidate.md §3) is the plain average of the v0.1 seed 0-7
checkpoints `runs/v01_seed{0..7}/model.pt`: every floating tensor averaged in float32
(`torch.stack(...).mean(0)`, cast back to the tensor's dtype), integer tensors and the
config taken from the first member -- the rule of `scripts/soup_eval.make_soup`, which
built the file that was selected and benchmarked. This script repeats it and compares:

- each member's SHA-256 with `runs/release_candidate/selection.json`;
- the rebuilt file's SHA-256 with the recorded soup's;
- if the file hashes differ, every tensor (serialisation can differ while the weights
  do not), and says which.

Exit status 0 only when the member hashes match and the weights are identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

MEMBERS = [f"runs/v01_seed{s}/model.pt" for s in range(8)]
SELECTION = Path("runs/release_candidate/selection.json")


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def soup(members: list[str]) -> dict:
    blobs = [torch.load(m, map_location="cpu", weights_only=False) for m in members]
    averaged = {}
    for key, first in blobs[0]["state_dict"].items():
        if torch.is_floating_point(first):
            averaged[key] = torch.stack([b["state_dict"][key].float() for b in blobs]).mean(
                0).to(first.dtype)
        else:
            averaged[key] = first.clone()
    return {"state_dict": averaged, "config": dict(blobs[0]["config"])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/release_candidate/rebuilt/model.pt")
    parser.add_argument("--reference", default="runs/release_candidate/model.pt")
    args = parser.parse_args()
    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    report: dict = {"members": {}}
    for m in MEMBERS:
        name = Path(m).parent.name
        got = sha256(m)
        report["members"][name] = {"sha256": got,
                                   "matches_record": got == selection["member_sha256"][name]}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(soup(MEMBERS), out)
    report["rebuilt_sha256"] = sha256(out)
    report["recorded_sha256"] = selection["sha256"]
    report["file_identical"] = report["rebuilt_sha256"] == report["recorded_sha256"]
    ref = torch.load(args.reference, map_location="cpu", weights_only=False)
    new = torch.load(out, map_location="cpu", weights_only=False)
    differing = [k for k in ref["state_dict"]
                 if not torch.equal(ref["state_dict"][k], new["state_dict"][k])]
    report["tensors_compared"] = len(ref["state_dict"])
    report["tensors_differing"] = differing
    report["config_identical"] = ref["config"] == new["config"]
    ok = (all(v["matches_record"] for v in report["members"].values())
          and not differing and report["config_identical"]
          and set(ref["state_dict"]) == set(new["state_dict"]))
    report["ok"] = ok
    (out.parent / "make_soup_report.json").write_text(json.dumps(report, indent=2),
                                                      encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "members"}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
