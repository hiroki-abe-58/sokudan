"""Phase 0 of docs/length_2x2.md: measurements that need no training.

    uv run python scripts/l2x2_phase0.py

1. State-length distribution of bench_ja / bench_en, counted exactly as M2 counts
   (modernbert-ja tokenizer, special tokens included). No model is run on them.
2. M5 = held-out AUROC on `implies_declining`, v0.1 and io_sf, 3 seeds each, read from
   the evaluation outputs already on disk (`runs/io/eval_*.json`).
3. Provenance of M2's 210 rows: source file, generator, generation date, document ids.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import sokudan.config  # noqa: F401
from scripts.eval_heldout import load_rows
from scripts.eval_long_states import rows_from_documents

OUT = Path("runs/l2x2/phase0.json")


def lengths(tokenizer, states: list[str]) -> dict:
    a = np.array([len(tokenizer(s, add_special_tokens=True)["input_ids"]) for s in states])
    return {"n": int(len(a)), "mean": round(float(a.mean()), 1), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": int(a.max()),
            "n_ge_400": int((a >= 400).sum())}


def main() -> int:
    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    out: dict = {}

    out["bench_lengths"] = {}
    for name in ("bench_ja", "bench_en"):
        rows = load_rows(Path(f"data/{name}.jsonl"))
        states = [r["state"] for r in rows]
        out["bench_lengths"][name] = {"rows": lengths(tokenizer, states),
                                      "unique_states": lengths(tokenizer, sorted(set(states)))}

    def m5(paths: list[str]) -> list[dict]:
        entries = [e for p in paths for e in json.loads(Path(p).read_text(encoding="utf-8"))]
        return [{"checkpoint": e["checkpoint"], **e["M1_by_attribute"]["implies_declining"]}
                for e in entries]

    out["M5"] = {"qf_short (v0.1)": m5(["runs/io/eval_v01.json"]),
                 "sf_short (io_sf)": m5(["runs/io/eval_seed0.json", "runs/io/eval_seed12.json"])}
    for name, rows in out["M5"].items():
        v = [r["auroc"] for r in rows]
        out["M5"][name] = {"per_seed": rows, "mean": sum(v) / 3, "min": min(v), "max": max(v)}

    frozen = load_rows(Path("data/v2/heldout_val_v2.jsonl"))
    long_rows = [r for r in rows_from_documents(Path("data/docs_long_val.jsonl")) if r["held_out"]]

    def tokens(state: str) -> int:
        return len(tokenizer(state, add_special_tokens=True)["input_ids"])

    band = [("frozen", r) for r in frozen if 400 <= tokens(r["state"]) < 800]
    band += [("docs_long_val", r) for r in long_rows if 400 <= tokens(r["state"]) < 800]
    manifest = json.loads(Path("data/docs_long_val.manifest.json").read_text(encoding="utf-8"))
    ids = sorted({r["doc_id"] for _, r in band})
    out["M2_provenance"] = {
        "n_rows": len(band),
        "by_source": {s: sum(1 for x, _ in band if x == s) for s in ("frozen", "docs_long_val")},
        "docs_long_val": {"file": "data/docs_long_val.jsonl",
                          "generator_model": manifest["generator_model"],
                          "generated_at": manifest["generated_at"], "seed": manifest["seed"]},
        "frozen": {"file": "data/v2/heldout_val_v2.jsonl",
                   "sources": sorted({r.get("source", "") for x, r in band if x == "frozen"})},
        "doc_ids": ids,
        "doc_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "states_sha256": hashlib.sha256("\n".join(sorted({r["state"] for _, r in band}))
                                        .encode()).hexdigest(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    brief = {k: v for k, v in out.items() if k != "M2_provenance"}
    print(json.dumps(brief, ensure_ascii=False, indent=1)[:3000])
    p = out["M2_provenance"]
    print({k: p[k] for k in p if k != "doc_ids"}, "n_docs", len(p["doc_ids"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
