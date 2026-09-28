"""Agreement between two `dump_probs.py` outputs, by question type.

    python scripts/mlx/compare_probs.py --items parity_set.json --ref a.npz --test b.npz

Per type (choice / score / bool) and overall: the share of questions whose argmax agrees,
and the mean and max absolute difference over every probability entry, with the question
where the max occurs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sokudan.schema.question import parse_questions


def compare(items: list[dict], ref: dict[str, np.ndarray],
            test: dict[str, np.ndarray]) -> dict[str, dict]:
    diffs: dict[str, list[np.ndarray]] = {}
    agree: dict[str, list[bool]] = {}
    worst: dict[str, tuple[float, str]] = {}
    for item in items:
        for qid, question in parse_questions(item["questions"]).items():
            key = f"{item['id']}/{qid}"
            a = np.asarray(ref[key], dtype=np.float64)
            b = np.asarray(test[key], dtype=np.float64)
            d = np.abs(a - b)
            for kind in (question.type, "all"):
                diffs.setdefault(kind, []).append(d)
                agree.setdefault(kind, []).append(int(a.argmax()) == int(b.argmax()))
                if d.max() > worst.get(kind, (-1.0, ""))[0]:
                    worst[kind] = (float(d.max()), key)
    out = {}
    for kind in ("choice", "score", "bool", "all"):
        if kind not in diffs:
            continue
        flat = np.concatenate(diffs[kind])
        out[kind] = {
            "questions": len(agree[kind]),
            "argmax_agree": int(sum(agree[kind])),
            "argmax_rate": sum(agree[kind]) / len(agree[kind]),
            "mean_abs": float(flat.mean()),
            "max_abs": float(flat.max()),
            "max_at": worst[kind][1],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    items = json.loads(args.items.read_text(encoding="utf-8"))
    result = compare(items, dict(np.load(args.ref)), dict(np.load(args.test)))
    for kind, r in result.items():
        print(f"{kind:6s} n={r['questions']:4d} argmax {r['argmax_agree']}/{r['questions']} "
              f"({r['argmax_rate']:.4f}) mean_abs={r['mean_abs']:.3e} "
              f"max_abs={r['max_abs']:.3e} at {r['max_at']}")
    if args.json:
        args.json.write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
