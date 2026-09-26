"""Tier-I AUROC on held-out, split by whether a second verifier agrees with the label.

    uv run python scripts/label_ceiling.py

Every held-out pair passed qwen3's verification, so qwen3's verdict is the gold label.
`gemma3:27b` re-judged the same documents with the same prompt
(`data/reverify/heldout/*_gemma.jsonl`). Each held-out row is then

* **agree** -- gemma gives the gold answer too,
* **disagree** -- gemma gives the opposite answer,
* **unsure** -- gemma answered 判断できない or not at all (in (a) only),

and every checkpoint's AUROC is reported on (a) all rows, (b) agree, (c) disagree, for
the frozen yardstick's tier I and the Day-3 tier-I attributes of the twelve-attribute
set. The reading of (b)-(a) was registered before any of this ran
(docs/label_ceiling.md §1).

Gemma's verdicts only partition an evaluation set here; they select no training data.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sokudan.config  # noqa: F401
from scripts.eval_heldout import load_checkpoint, load_rows
from sokudan.calibration.metrics import auroc
from sokudan.data import intent_attributes as ia
from sokudan.train.dataset import Example
from sokudan.train.loop import TrainConfig, build_collator, length_bucketed_batches, run_model

YES, NO = "はい", "いいえ"
DAY3_I = {"implies_decision_already_made", "implies_seeking_exception", "implies_relief"}
GROUPS = {
    "凍結 / I 段": ("data/v2/heldout_val_v2.jsonl", None),
    "Day 3 追加 / I 段": ("data/v4/heldout_val_v4.jsonl", DAY3_I),
    "12 属性 / I 段（参考）": ("data/v4/heldout_val_v4.jsonl", None),
}
MODELS = {"v0.1": "v01", "v0.3": "v03", "v0.4": "v04"}


def gemma_verdicts() -> dict[tuple[str, str, str], dict[str, Any]]:
    """(corpus prefix, doc_id, attribute) -> gold, qwen3 and gemma answers."""
    out = {}
    for prefix in ("v2", "v3", "v4"):
        path = Path(f"data/reverify/heldout/{prefix}_gemma.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for attribute, gold in row["intent_labels"].items():
                out[(prefix, row["doc_id"], attribute)] = {
                    "gold": int(gold),
                    "qwen3": (row.get("verdicts_ref") or {}).get(attribute),
                    "gemma": (row.get("verdicts") or {}).get(attribute),
                }
    return out


def status(entry: dict[str, Any] | None) -> str:
    if entry is None or entry["gemma"] not in (YES, NO):
        return "unsure"
    return "agree" if (entry["gemma"] == YES) == bool(entry["gold"]) else "disagree"


def kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    po = sum(a == b for a, b in pairs) / n
    pa, pb = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe >= 1 else (po - pe) / (1 - pe)


@torch.no_grad()
def score(model: Any, rows: list[dict], collator: Any, config: TrainConfig) -> np.ndarray:
    """P(true) for every row, in the order given."""
    examples = [Example.from_row(r) for r in rows]
    position = {id(e): i for i, e in enumerate(examples)}
    probs = np.zeros(len(rows))
    device = torch.device(config.device)
    model.eval()
    for group in length_bucketed_batches(examples, config.batch_size, collator.tokenizer,
                                         rng=random.Random(0), shuffle=False):
        batch = collator(group).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = run_model(model, batch)
        p = out.probs.float().cpu().numpy()[:, 1]
        for index, example in enumerate(group):
            probs[position[id(example)]] = p[index]
    return probs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default="runs/diagnostics/label_ceiling.json")
    args = parser.parse_args()

    verdicts = gemma_verdicts()
    result: dict[str, Any] = {"agreement": {}, "groups": {}}

    # Which rows, and their agreement status.
    sets: dict[str, list[dict]] = {}
    for name, (path, only) in GROUPS.items():
        rows = [r for r in load_rows(Path(path))
                if (r.get("tier") or ia.BY_ID[r["attribute"]].tier) == "I"
                and (only is None or r["attribute"] in only)]
        for r in rows:
            prefix, _, doc = r["doc_id"].partition("-")
            r["_entry"] = verdicts.get((prefix, doc, r["attribute"]))
            r["_status"] = status(r["_entry"])
        sets[name] = rows

    # Agreement per attribute, on pairs (not views), over both held-out sets.
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for rows in sets.values():
        for r in rows:
            pairs[(r["doc_id"], r["attribute"])] = r["_entry"] or {}
    qwen3_is_gold = sum(1 for e in pairs.values()
                        if e and e["qwen3"] in (YES, NO) and (e["qwen3"] == YES) != bool(e["gold"]))
    print(f"held-out I-tier pairs: {len(pairs)}; pairs where qwen3's original verdict "
          f"differs from gold: {qwen3_is_gold} (expected 0)\n")
    by_attr: dict[str, list] = defaultdict(list)
    unsure: Counter = Counter()
    for (_doc, attribute), e in pairs.items():
        if not e or e["gemma"] not in (YES, NO):
            unsure[attribute] += 1
            continue
        by_attr[attribute].append((bool(e["gold"]), e["gemma"] == YES))
    print(f"{'attribute':34s}{'pairs':>7s}{'agree':>8s}{'kappa':>8s}{'unsure':>8s}")
    all_pairs = []
    for attribute in sorted(by_attr, key=lambda a: (a not in DAY3_I, a)):
        p = by_attr[attribute]
        all_pairs += p
        agree = sum(a == b for a, b in p) / len(p)
        k = kappa(p)
        result["agreement"][attribute] = {"n": len(p), "agree": agree, "kappa": k,
                                          "unsure": unsure[attribute],
                                          "day3": attribute in DAY3_I}
        print(f"{attribute:34s}{len(p):7d}{agree:8.3f}{k:8.3f}{unsure[attribute]:8d}"
              f"{'  (Day 3)' if attribute in DAY3_I else ''}")
    k = kappa(all_pairs)
    agree = sum(a == b for a, b in all_pairs) / len(all_pairs)
    result["agreement"]["_all"] = {"n": len(all_pairs), "agree": agree, "kappa": k}
    print(f"{'all':34s}{len(all_pairs):7d}{agree:8.3f}{k:8.3f}{sum(unsure.values()):8d}\n")

    for name, rows in sets.items():
        counts = Counter(r["_status"] for r in rows)
        pos = {s: np.mean([r["label"] for r in rows if r["_status"] == s]) for s in counts}
        print(f"{name}: {len(rows)} rows  agree {counts['agree']}  disagree {counts['disagree']}"
              f"  unsure {counts['unsure']}   positive rate "
              + "  ".join(f"{s} {pos[s]:.2f}" for s in ("agree", "disagree") if s in pos))
        result["groups"][name] = {"rows": len(rows), "counts": dict(counts),
                                  "positive_rate": pos, "auroc": {}}

    # Score every checkpoint once per held-out file.
    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    files = {path for path, _ in GROUPS.values()}
    for model_name, tag in MODELS.items():
        for seed in range(3):
            checkpoint = f"runs/{tag}_seed{seed}/model.pt"
            model, encoding, device = load_checkpoint(checkpoint)
            config = TrainConfig(device=device, batch_size=args.batch_size, encoding=encoding,
                                 input_order=getattr(model, "input_order", "question_first"))
            collator = build_collator(tokenizer, config)
            for path in files:
                names = [n for n, (p, _) in GROUPS.items() if p == path]
                union = {id(r): r for n in names for r in sets[n]}
                rows = list(union.values())
                probs = score(model, rows, collator, config)
                p_of = {id(r): probs[i] for i, r in enumerate(rows)}
                for n in names:
                    entry = {}
                    for subset, keep in (("a", lambda r: True),
                                         ("b", lambda r: r["_status"] == "agree"),
                                         ("c", lambda r: r["_status"] == "disagree")):
                        chosen = [r for r in sets[n] if keep(r)]
                        g = np.array([r["label"] for r in chosen])
                        p = np.array([p_of[id(r)] for r in chosen])
                        entry[subset] = (float(auroc(p, g)) if len(g) and 0 < g.sum() < len(g)
                                         else None)
                    result["groups"][n]["auroc"][f"{model_name}/s{seed}"] = entry
            del model
            torch.cuda.empty_cache()
            print(f"scored {checkpoint}", flush=True)

    # Summary per group: mean over seeds per model, and the registered (b)-(a).
    print()
    for name, group in result["groups"].items():
        print(f"== {name} ==")
        print(f"  {'model':6s}{'(a) 全事例':>18s}{'(b) 一致のみ':>18s}{'(c) 不一致のみ':>18s}"
              f"{'(b)-(a)':>10s}")
        diffs = []
        for model_name in MODELS:
            vals = {s: [group["auroc"][f"{model_name}/s{i}"][s] for i in range(3)]
                    for s in "abc"}
            cells = []
            for s in "abc":
                v = [x for x in vals[s] if x is not None]
                cells.append(f"{st.mean(v):.4f} ± {st.pstdev(v):.4f}" if v else "—")
            d = [b - a for a, b in zip(vals["a"], vals["b"], strict=True)
                 if a is not None and b is not None]
            diffs += d
            print(f"  {model_name:6s}{cells[0]:>18s}{cells[1]:>18s}{cells[2]:>18s}"
                  f"{st.mean(d):>+10.4f}")
        mean_diff = st.mean(diffs)
        verdict = ("天井はラベル" if mean_diff >= 0.05 else
                   "モデル側の限界" if mean_diff < 0.02 else "どちらとも言えない")
        group["mean_b_minus_a"] = mean_diff
        group["verdict"] = verdict
        print(f"  9 チェックポイント平均の (b)-(a): {mean_diff:+.4f}  -> {verdict}\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for group in result["groups"].values():
        group["positive_rate"] = {k: float(v) for k, v in group["positive_rate"].items()}
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
