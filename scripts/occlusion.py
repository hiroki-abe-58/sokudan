"""docs/occlusion.md: which sentence of the state decides the answer, and where is it?

    uv run python scripts/occlusion.py

Exploratory; nothing here is a verdict. For each positive case, every sentence is removed
in turn and P(true) is recomputed: Delta_i = P(true | full) - P(true | without sentence i),
averaged over the 8 seeds of each condition (v0.1 = question_first, sf = state_first).
The deciding sentence is the one with the largest seed-mean Delta for v0.1 (Q1, Q2) or
for each condition separately (Q3). Its position is the centre token of the sentence
divided by the state's token count.

Sentence splitting (`split_sentences`): a sentence ends right after "。", "！", "？" or a
newline (the delimiter stays with the sentence it ends); the text after the last
delimiter is a sentence too. Pieces that are only whitespace are never candidates, but
they stay in the text when another sentence is removed.

Cases:
- M5: frozen held-out `implies_declining` rows asked in the canonical (non-negated)
  direction with label true.
- M2: the 210 M2 rows (400-799 state tokens, frozen + `docs_long_val` held-out), same
  filter (non-negated, label true).

Writes `runs/occlusion/results.json`. Stops after `CAP_S` seconds and writes what it has.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import sokudan.config  # noqa: F401
from scripts.eval_heldout import load_checkpoint, load_rows
from scripts.eval_local_attention import p_true
from scripts.eval_long_states import rows_from_documents
from sokudan.train.loop import TrainConfig, build_collator

DELIMITERS = set("。！？\n")
CAP_S = 60 * 60
CONDITIONS = {
    "v01": [f"runs/v01_seed{s}/model.pt" for s in range(8)],
    "sf": [f"runs/io_sf_seed{s}/model.pt" for s in range(8)],
}


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Character spans of the candidate sentences (see module docstring)."""
    spans, start = [], 0
    for i, ch in enumerate(text):
        if ch in DELIMITERS:
            spans.append((start, i + 1))
            start = i + 1
    if start < len(text):
        spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def relative_centre(offsets: list[tuple[int, int]], span: tuple[int, int]) -> float:
    s, e = span
    hits = [i for i, (a, b) in enumerate(offsets) if a < e and b > s]
    centre = (hits[0] + hits[-1]) / 2 if hits else 0.0
    return centre / max(len(offsets), 1)


def cases(tokenizer) -> dict[str, list[dict]]:
    def positive(r: dict) -> bool:
        return not r.get("negation") and int(r["label"]) == 1

    frozen = load_rows(Path("data/v2/heldout_val_v2.jsonl"))
    m5 = [r for r in frozen if r["attribute"] == "implies_declining" and positive(r)]
    long_rows = [r for r in rows_from_documents(Path("data/docs_long_val.jsonl")) if r["held_out"]]

    def tokens(state: str) -> int:
        return len(tokenizer(state, add_special_tokens=True)["input_ids"])

    m2 = [r for r in frozen + long_rows if 400 <= tokens(r["state"]) < 800 and positive(r)]
    return {"M5": m5, "M2": m2}


def main() -> int:
    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    groups = cases(tokenizer)
    variants: list[dict] = []   # one row per (case, removed sentence or None)
    index: dict[str, list[dict]] = {}
    for group, rows in groups.items():
        index[group] = []
        for case_id, row in enumerate(rows):
            state = row["state"]
            spans = split_sentences(state)
            offsets = tokenizer(state, add_special_tokens=False,
                                return_offsets_mapping=True)["offset_mapping"]
            entry = {"case": case_id, "doc_id": row.get("doc_id"), "attribute": row["attribute"],
                     "n_sentences": len(spans),
                     "positions": [relative_centre(offsets, sp) for sp in spans],
                     "full": len(variants), "without": []}
            variants.append(row)
            for s, e in spans:
                entry["without"].append(len(variants))
                variants.append({**row, "state": state[:s] + state[e:]})
            index[group].append(entry)
    print("cases: " + ", ".join(f"{g} {len(v)}" for g, v in groups.items())
          + f"; variant rows {len(variants)}", flush=True)

    probs: dict[str, list[np.ndarray]] = {c: [] for c in CONDITIONS}
    order = [(c, p) for pair in zip(*CONDITIONS.values(), strict=True)
             for c, p in zip(CONDITIONS, pair, strict=True)]
    stopped_early = False
    for cond, path in order:
        if time.time() - started > CAP_S:
            stopped_early = True
            print(f"cap reached before {path}", flush=True)
            break
        model, encoding, device = load_checkpoint(path)
        config = TrainConfig(device=device, batch_size=32, encoding=encoding,
                             input_order=model.input_order)
        probs[cond].append(p_true(model, variants, build_collator(tokenizer, config), config))
        del model
        torch.cuda.empty_cache()
        print(f"{path} done at {time.time() - started:.0f}s", flush=True)

    mean = {c: np.mean(v, axis=0) for c, v in probs.items() if v}
    results: dict = {"seeds_used": {c: len(v) for c, v in probs.items()},
                     "stopped_early": stopped_early, "seconds": round(time.time() - started),
                     "cases": {}}

    def third(x: float) -> str:
        return "前" if x < 1 / 3 else ("中" if x < 2 / 3 else "後")

    def decide(entry: dict, cond: str) -> dict | None:
        if not entry["without"] or cond not in mean:
            return None
        full = float(mean[cond][entry["full"]])
        deltas = [full - float(mean[cond][j]) for j in entry["without"]]
        k = int(np.argmax(deltas))
        return {"p_full": full, "delta": deltas[k], "sentence": k,
                "position": entry["positions"][k], "third": third(entry["positions"][k])}

    for group, entries in index.items():
        out = []
        for e in entries:
            out.append({**{k: e[k] for k in ("case", "doc_id", "attribute", "n_sentences")},
                        "v01": decide(e, "v01"), "sf": decide(e, "sf")})
        results["cases"][group] = out

    def dist(items: list[dict | None]) -> dict:
        items = [i for i in items if i]
        counts = {t: sum(1 for i in items if i["third"] == t) for t in ("前", "中", "後")}
        n = len(items)
        return {"n": n, **{t: {"n": c, "share": c / n if n else None} for t, c in counts.items()},
                "mean_position": float(np.mean([i["position"] for i in items])) if n else None}

    m5 = results["cases"]["M5"]
    q2 = [c for c in m5 if c["v01"] and c["sf"] and c["v01"]["p_full"] >= 0.5
          and c["sf"]["p_full"] < 0.5]
    results["Q1"] = dist([c["v01"] for c in m5])
    results["Q2"] = dist([c["v01"] for c in q2])
    results["Q3"] = {"v01": dist([c["v01"] for c in results["cases"]["M2"]]),
                     "sf": dist([c["sf"] for c in results["cases"]["M2"]])}
    out = Path("runs/occlusion/results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: results[k] for k in ("seeds_used", "stopped_early", "seconds", "Q1",
                                              "Q2", "Q3")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
