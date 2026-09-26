"""Every number the local_attention_1024 gate reads, from one script.

    uv run python scripts/eval_local_attention.py \
        --checkpoints runs/v01_seed0/model.pt runs/v01_seed1/model.pt ... \
        --out runs/diagnostics/la_v01.json

Definitions are in docs/local_attention_1024.md and were fixed before any number
from this script was seen. In short:

- **M1** held-out bool AUROC, pooled over the frozen yardstick
  (`data/v2/heldout_val_v2.jsonl`, 5 held-out attributes).
- **M2** held-out bool AUROC on rows whose *state* is 400-799 tokens (counted with
  special tokens, as `scripts/eval_long_states.py` bins them), pooled over the frozen
  yardstick and the held-out rows of `data/docs_long_val.jsonl`. The frozen yardstick
  alone has 6 such rows; the long-document val set is what makes the band measurable.
- **M3** accuracy on `ends_with_question`, the only positional S attribute in the
  held-out sets (day3 classification), on the frozen yardstick. `uses_bullet_points`
  (structural, same shape per day3) is reported beside it as reference.
- **Guardrails** choice accuracy, score RPS, score accuracy and bool accuracy on
  `data/val_v2.jsonl` through `sokudan.train.loop.evaluate` -- the function that wrote
  each run's report -- so they are the same metric the runs were trained against.

Each checkpoint loads with the window it was trained at (`load_checkpoint` restores
it). `--local-attention` overrides that for an inference-only wiring check and is
written into the output so the result cannot be mistaken for a trained model's.

Added for docs/input_order.md (M1-M3 above are unchanged):

- **M4** accuracy on `includes_greeting` -- the one attribute in the day3 catalogue
  decided by the *opening* of the state (`POSITIONAL_ATTRIBUTES`: "the opening line")
  -- over the val-split documents of `data/docs_all_v5.jsonl`, one row per document
  with the canonical question and the same discard rule as training. Neither v0.1 nor
  the input-order runs train on it (`data/train_v2b.jsonl` has no such row).
- The block order is restored from the checkpoint like the window; `--input-order`
  overrides it for an inference-only wiring check, flagged the same way.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sokudan.config  # noqa: F401
from scripts.eval_heldout import load_checkpoint, load_rows
from scripts.eval_long_states import rows_from_documents
from sokudan.calibration.metrics import auroc
from sokudan.train.dataset import Example, length_bucketed_batches, load_examples
from sokudan.train.loop import TrainConfig, build_collator, evaluate, run_model

FROZEN = Path("data/v2/heldout_val_v2.jsonl")
V4 = Path("data/v4/heldout_val_v4.jsonl")
LONG = Path("data/docs_long_val.jsonl")
VAL = Path("data/val_v2.jsonl")

M2_BAND = (400, 800)
REFERENCE_BINS = [(0, 200), (200, 400), (400, 600), (600, 800)]
POSITIONAL_HELD_OUT = "ends_with_question"
STRUCTURAL_HELD_OUT = "uses_bullet_points"
OPENING_ATTRIBUTE = "includes_greeting"
OPENING_DOCS = Path("data/docs_all_v5.jsonl")


def opening_rows() -> list[dict]:
    """M4's rows: val-split documents only, one per document.

    Same row as `scripts/eval_long_states.rows_from_documents` builds -- canonical
    question form, verdict must agree with the label -- restricted to one attribute,
    because the file also carries attributes the catalogue has since retired.
    """
    from sokudan.data import intent_attributes as ia
    from sokudan.data.schema_aug import LabelledQuestion

    attribute = ia.BY_ID[OPENING_ATTRIBUTE]
    out = []
    for line in OPENING_DOCS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        if doc.get("split") != "val" or OPENING_ATTRIBUTE not in doc["intent_labels"]:
            continue
        gold = doc["intent_labels"][OPENING_ATTRIBUTE]
        answer = (doc.get("verdicts") or {}).get(OPENING_ATTRIBUTE)
        if answer is None or answer == "判断できない" or (answer == "はい") != bool(gold):
            continue  # same discard rule as training
        item = LabelledQuestion(attribute.question(0), gold)
        out.append({
            "doc_id": doc["doc_id"], "domain": doc["domain"],
            "attribute": OPENING_ATTRIBUTE, "tier": attribute.tier, "held_out": False,
            "kind": "bool", "state": doc["state"],
            "question": item.question.model_dump(mode="json"),
            "label": item.label, "n_options": 2,
        })
    return out


@torch.no_grad()
def p_true(model: Any, rows: list[dict], collator: Any, config: TrainConfig) -> np.ndarray:
    """P(true) per row, in row order."""
    examples = [Example.from_row(r) for r in rows]
    index_of = {id(e): i for i, e in enumerate(examples)}
    out = np.full(len(rows), np.nan)
    device = torch.device(config.device)
    model.eval()
    for group in length_bucketed_batches(examples, config.batch_size, collator.tokenizer,
                                         rng=random.Random(0), shuffle=False):
        batch = collator(group).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            probs = run_model(model, batch).probs.float().cpu().numpy()
        for row, example in enumerate(group):
            out[index_of[id(example)]] = probs[row, 1]
    assert not np.isnan(out).any()
    return out


def summary(p: np.ndarray, g: np.ndarray) -> dict[str, Any]:
    entry: dict[str, Any] = {"n": int(len(g)), "n_true": int(g.sum())}
    if len(g):
        entry["accuracy"] = float(((p > 0.5).astype(int) == g).mean())
    if 0 < g.sum() < len(g):
        entry["auroc"] = float(auroc(p, g))
    return entry


VAL_WIDTH = 16  # the per-row option width of `scripts/ens_collect.val_probs`


def val_rows_to_arrays(examples: list[Example], rows: list) -> tuple[np.ndarray, np.ndarray]:
    """`evaluate(rows_out=...)`'s rows, laid out as `ens_collect.val_probs` returns them."""
    index_of = {id(e): i for i, e in enumerate(examples)}
    probs = np.full((len(examples), VAL_WIDTH), np.nan)
    n_opts = np.zeros(len(examples), dtype=int)
    for example, row in rows:
        i = index_of[id(example)]
        probs[i, :len(row)] = row
        n_opts[i] = len(row)
    return probs, n_opts


def evaluate_checkpoint(path: str, local_attention: int | None, data: dict,
                        tokenizer: Any, batch_size: int,
                        input_order: str | None = None,
                        save_probs: Path | None = None) -> dict[str, Any]:
    started = time.time()
    model, encoding, device = load_checkpoint(path, local_attention=local_attention,
                                              input_order=input_order)
    config = TrainConfig(device=device, batch_size=batch_size, encoding=encoding,
                         input_order=model.input_order)
    collator = build_collator(tokenizer, config)

    frozen, long_rows, bullets = data["frozen"], data["long"], data["bullets"]
    p_frozen = p_true(model, frozen, collator, config)
    p_long = p_true(model, long_rows, collator, config)
    p_bullets = p_true(model, bullets, collator, config)
    g_frozen = np.array([int(r["label"]) for r in frozen])
    g_long = np.array([int(r["label"]) for r in long_rows])
    g_bullets = np.array([int(r["label"]) for r in bullets])

    result: dict[str, Any] = {
        "checkpoint": path, "encoding": encoding,
        "local_attention": model.backbone.spec.local_attention,
        "input_order": model.input_order,
        "inference_only_override": local_attention is not None or input_order is not None,
    }
    result["M1"] = summary(p_frozen, g_frozen)
    result["M1_by_attribute"] = {
        attr: summary(p_frozen[idx], g_frozen[idx])
        for attr, idx in data["frozen_by_attr"].items()
    }

    pool_p = np.concatenate([p_frozen, p_long])
    pool_g = np.concatenate([g_frozen, g_long])
    pool_tokens = np.concatenate([data["frozen_tokens"], data["long_tokens"]])
    pool_attr = data["frozen_attrs"] + data["long_attrs"]
    lo, hi = M2_BAND
    band = (pool_tokens >= lo) & (pool_tokens < hi)
    result["M2"] = summary(pool_p[band], pool_g[band])
    result["M2_by_source"] = {
        "frozen": summary(p_frozen[band[:len(frozen)]], g_frozen[band[:len(frozen)]]),
        "docs_long_val": summary(p_long[band[len(frozen):]], g_long[band[len(frozen):]]),
    }
    result["bins_pooled"] = {
        f"{a}-{b}": summary(pool_p[(pool_tokens >= a) & (pool_tokens < b)],
                            pool_g[(pool_tokens >= a) & (pool_tokens < b)])
        for a, b in REFERENCE_BINS
    }
    result["M2_band_by_attribute"] = {}
    for attr in sorted(set(pool_attr)):
        mask = band & np.array([a == attr for a in pool_attr])
        result["M2_band_by_attribute"][attr] = summary(pool_p[mask], pool_g[mask])

    idx = data["frozen_by_attr"][POSITIONAL_HELD_OUT]
    result["M3"] = summary(p_frozen[idx], g_frozen[idx])
    result["M3_reference"] = {
        f"{STRUCTURAL_HELD_OUT} (構造依存, v4 held-out)": summary(p_bullets, g_bullets),
        f"{POSITIONAL_HELD_OUT} in the M2 band (pooled)": result["M2_band_by_attribute"].get(
            POSITIONAL_HELD_OUT, {}),
    }
    result["heldout_bool_accuracy_reference"] = result["M1"]["accuracy"]

    opening = data["opening"]
    p_opening = p_true(model, opening, collator, config)
    result["M4"] = summary(p_opening, np.array([int(r["label"]) for r in opening]))

    # One pass over val serves both the guardrails and, with `save_probs`, the per-row
    # predictions `scripts/soup_eval.collect` would otherwise compute again (A5).
    val_rows: list | None = [] if save_probs is not None else None
    val = evaluate(model, data["val"], collator, config, rows_out=val_rows)
    if save_probs is not None:
        v, n = val_rows_to_arrays(data["val"], val_rows)
        save_probs.parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_probs, frozen=p_frozen, long=p_long, val=v, val_n=n,
                 input_order=np.array(model.input_order))
        result["probs"] = str(save_probs)
    result["guardrails"] = {
        "choice_accuracy": val["choice"]["accuracy"], "choice_n": val["choice"]["n"],
        "score_rps": val["score"]["rps"], "score_accuracy": val["score"]["accuracy"],
        "score_n": val["score"]["n"],
        "bool_accuracy": val["bool"]["accuracy"], "bool_n": val["bool"]["n"],
        "bool_auroc_reference": val["bool"]["auroc"],
    }
    result["seconds"] = round(time.time() - started, 1)
    del model
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--local-attention", type=int, default=None,
                        help="inference-only override; never for a reported result")
    parser.add_argument("--input-order", choices=("question_first", "state_first", "sandwich"),
                        default=None,
                        help="inference-only override; never for a reported result")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", required=True)
    parser.add_argument("--save-probs", action="store_true",
                        help="also write each checkpoint's per-row predictions next to --out "
                             "(<out>.probs<i>.npz), in scripts/soup_eval.collect's format")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)

    def state_tokens(rows: list[dict]) -> np.ndarray:
        cache: dict[str, int] = {}
        for r in rows:
            if r["state"] not in cache:
                cache[r["state"]] = len(tokenizer(r["state"], add_special_tokens=True)
                                        ["input_ids"])
        return np.array([cache[r["state"]] for r in rows])

    frozen = load_rows(FROZEN)
    long_rows = [r for r in rows_from_documents(LONG) if r["held_out"]]
    bullets = [r for r in load_rows(V4) if r["attribute"] == STRUCTURAL_HELD_OUT]
    by_attr: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(frozen):
        by_attr[r["attribute"]].append(i)
    data = {
        "frozen": frozen, "long": long_rows, "bullets": bullets,
        "frozen_tokens": state_tokens(frozen), "long_tokens": state_tokens(long_rows),
        "frozen_attrs": [r["attribute"] for r in frozen],
        "long_attrs": [r["attribute"] for r in long_rows],
        "frozen_by_attr": {k: np.array(v) for k, v in sorted(by_attr.items())},
        "val": load_examples(VAL),
        "opening": opening_rows(),
    }
    print(f"frozen {len(frozen)}, docs_long_val held-out {len(long_rows)}, "
          f"{STRUCTURAL_HELD_OUT} {len(bullets)}, val {len(data['val'])}, "
          f"{OPENING_ATTRIBUTE} {len(data['opening'])}", flush=True)

    results = []
    for i, path in enumerate(args.checkpoints):
        target = Path(f"{args.out}.probs{i}.npz") if args.save_probs else None
        entry = evaluate_checkpoint(path, args.local_attention, data, tokenizer,
                                    args.batch_size, input_order=args.input_order,
                                    save_probs=target)
        results.append(entry)
        brief = {"M1": entry["M1"], "M2": entry["M2"], "M3": entry["M3"],
                 "M4": entry["M4"], "guardrails": entry["guardrails"],
                 "local_attention": entry["local_attention"],
                 "input_order": entry["input_order"], "seconds": entry["seconds"]}
        print(path, json.dumps(brief, ensure_ascii=False), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
