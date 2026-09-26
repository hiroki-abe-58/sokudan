"""docs/release_candidate.md Phase 3: build the candidate soups, score them, select by rule.

    uv run python scripts/release_eval.py

Candidates (a soup is the plain average of every floating tensor of its members; integer
tensors and config from the first member, as in `scripts/soup_eval.make_soup`):

- S8_old: old-code v0.1 seeds 0-7 -- the existing `runs/soup/soups/k8_0_7/model.pt`;
- S8_new: new-code seeds 0-7 (`runs/v01new_seed<s>`);
- S16: old 0-7 + new-code 8-15 -- the existing `runs/soup/soups/k16_0_15/model.pt`;
- S24: old 0-7 + new-code 8-15 + new-code 0-7.

If `runs/regression/results.json` does not say the regression check found nothing (a
regression, a break, or too few runs), only S8_old is a candidate. The reference is the
mean of the old-code single seeds 0-7 (`runs/soup/results.json` `scores`). Cutoffs
(non-inferiority, docs/research_protocol.md margins): choice acc >= ref - 0.005, bool acc
>= ref - 0.005, score acc >= ref - 0.01, score RPS <= ref + 0.005, bool ECE (val) <= ref +
0.02. Among those that pass, the highest M1m; within 0.002 of it, the fewest members (then
the higher M1m). The chosen soup is copied to `runs/release_candidate/model.pt` with its
members and SHA-256 hashes. Writes `runs/release_candidate/selection.json`.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

import sokudan.config  # noqa: F401
from scripts.soup_eval import METRICS, PROBS, Scorer, collect, load_data

OLD = [f"v01_seed{s}" for s in range(8)]
LATER = [f"v01_seed{s}" for s in range(8, 16)]
NEW = [f"v01new_seed{s}" for s in range(8)]
CANDIDATES = {"S8_old": OLD, "S8_new": NEW, "S16": OLD + LATER, "S24": OLD + LATER + NEW}
EXISTING = {"S8_old": ("k8_0_7", "runs/soup/soups/k8_0_7/model.pt"),
            "S16": ("k16_0_15", "runs/soup/soups/k16_0_15/model.pt")}
OUT = Path("runs/release_candidate")
# (metric, margin, direction): +1 higher is better, -1 lower is better
CUTOFFS = (("choice_accuracy", 0.005, 1), ("bool_accuracy", 0.005, 1),
           ("score_accuracy", 0.01, 1), ("score_rps", 0.005, -1), ("ece_val", 0.02, -1))
TIE = 0.002


def make_soup(members: list[str], out: Path) -> Path:
    """Average by running sum, one member in memory at a time."""
    if out.exists():
        return out
    first = torch.load(f"runs/{members[0]}/model.pt", map_location="cpu", weights_only=False)
    sums = {k: v.double() for k, v in first["state_dict"].items() if torch.is_floating_point(v)}
    for name in members[1:]:
        blob = torch.load(f"runs/{name}/model.pt", map_location="cpu", weights_only=False)
        for k in sums:
            sums[k] += blob["state_dict"][k].double()
        del blob
    state = {k: ((sums[k] / len(members)).to(v.dtype) if k in sums else v.clone())
             for k, v in first["state_dict"].items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state, "config": dict(first["config"])}, out)
    return out


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def passes(entry: dict, ref: dict) -> dict[str, bool]:
    return {m: (entry[m] >= ref[m] - margin) if sign > 0 else (entry[m] <= ref[m] + margin)
            for m, margin, sign in CUTOFFS}


def select(scores: dict[str, dict], candidates: list[str], ref: dict) -> str | None:
    ok = [c for c in candidates if all(passes(scores[c], ref).values())]
    if not ok:
        return None
    best = max(scores[c]["M1m"] for c in ok)
    # "within 0.002" includes 0.002 itself; the tolerance keeps float rounding out of it.
    close = [c for c in ok if best - scores[c]["M1m"] <= TIE + 1e-12]
    return min(close, key=lambda c: (len(CANDIDATES[c]), -scores[c]["M1m"]))


def regression_found() -> tuple[bool, str]:
    path = Path("runs/regression/results.json")
    if not path.exists():
        return True, "runs/regression/results.json missing (判定不能)"
    test = json.loads(path.read_text(encoding="utf-8")).get("main_test")
    if not isinstance(test, dict):
        return True, f"main test not run: {test}"
    return test["verdict"].startswith("退行あり"), test["verdict"]


def main() -> int:
    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID

    found, verdict = regression_found()
    candidates = ["S8_old"] if found else list(CANDIDATES)
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    data = load_data()
    paths, probs_names = {}, {}
    for c in candidates:
        if c in EXISTING:
            probs_names[c], paths[c] = EXISTING[c]
        else:
            paths[c] = str(make_soup(CANDIDATES[c], OUT / "soups" / c / "model.pt"))
            probs_names[c] = f"release_{c}"
        collect(probs_names[c], paths[c], tokenizer, data)
    score = Scorer(tokenizer, data)
    scores = {}
    for c in candidates:
        d = np.load(PROBS / f"probs_{probs_names[c]}.npz")
        scores[c] = score(d["frozen"], d["long"], d["val"], d["val_n"])
    old = json.loads(Path("runs/soup/results.json").read_text(encoding="utf-8"))["scores"]
    ref = {m: float(np.mean([old[n][m] for n in OLD])) for m in METRICS}
    chosen = select(scores, candidates, ref)
    res = {"regression_verdict": verdict, "candidates": candidates, "reference": ref,
           "scores": scores, "cutoffs": {c: passes(scores[c], ref) for c in candidates},
           "chosen": chosen}
    OUT.mkdir(parents=True, exist_ok=True)
    if chosen is None:
        res["decision"] = "出荷候補なし"
    else:
        shutil.copyfile(paths[chosen], OUT / "model.pt")
        res["decision"] = f"{chosen} を出荷候補とする"
        res["members"] = CANDIDATES[chosen]
        res["source"] = paths[chosen]
        res["sha256"] = sha256(OUT / "model.pt")
        res["sha256_source"] = sha256(paths[chosen])
        res["member_sha256"] = {n: sha256(f"runs/{n}/model.pt") for n in CANDIDATES[chosen]}
    (OUT / "selection.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "member_sha256"},
                     ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
