"""docs/ensemble.md: score single models, ensembles and the soup from per-row predictions.

    uv run python scripts/ens_eval.py

Reads `runs/ens/probs_<name>.npz` (scripts/ens_collect.py). An ensemble's prediction is
the mean of its members' probabilities (bool P(true), choice option probabilities, score
level probabilities). Metrics use the same functions as `scripts/eval_local_attention.py`
(`auroc`, `loop._metrics_for_group`); single checkpoints are first checked against the
stored evaluation outputs. z = (value - mu) / sigma against v0.1's 8 seeds, and the
distillation rule of §2 is applied mechanically. Writes `runs/ens/results.json`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import sokudan.config  # noqa: F401
from scripts.eval_heldout import load_rows
from scripts.eval_long_states import rows_from_documents
from sokudan.calibration.metrics import auroc
from sokudan.train.dataset import load_examples
from sokudan.train.loop import _metrics_for_group

REF = {"M1m": (0.831028, 0.012591), "M2": (0.837224, 0.035855), "M5": (0.781811, 0.036470),
       "M1": (0.882193, 0.009724)}
CORE = ("M1m", "M2", "M5")
STORED = {**{f"v01_seed{s}": ("runs/io/eval_v01.json", s) for s in range(3)},
          **{f"v01_seed{s}": (f"runs/nf/eval_v01_seed{s}.json", 0) for s in range(3, 8)},
          "io_sf_seed0": ("runs/io/eval_seed0.json", 0),
          "io_sf_seed1": ("runs/io/eval_seed12.json", 0),
          "io_sf_seed2": ("runs/io/eval_seed12.json", 1),
          **{f"io_sf_seed{s}": (f"runs/nf/eval_io_sf_seed{s}.json", 0) for s in range(3, 8)}}


def main() -> int:
    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    frozen = load_rows(Path("data/v2/heldout_val_v2.jsonl"))
    long_rows = [r for r in rows_from_documents(Path("data/docs_long_val.jsonl")) if r["held_out"]]
    val = load_examples("data/val_v2.jsonl")
    g_frozen = np.array([int(r["label"]) for r in frozen])
    g_long = np.array([int(r["label"]) for r in long_rows])

    def tokens(state: str) -> int:
        return len(tokenizer(state, add_special_tokens=True)["input_ids"])

    cache: dict[str, int] = {}
    for r in frozen + long_rows:
        if r["state"] not in cache:
            cache[r["state"]] = tokens(r["state"])
    band = np.array([400 <= cache[r["state"]] < 800 for r in frozen + long_rows])
    by_attr: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(frozen):
        by_attr[r["attribute"]].append(i)
    val_kind = [e.kind for e in val]
    val_label = [e.label for e in val]

    def score(f: np.ndarray, g: np.ndarray, v: np.ndarray, n: np.ndarray) -> dict:
        attrs = {a: float(auroc(f[idx], g_frozen[idx])) for a, idx in sorted(by_attr.items())
                 if len(idx) >= 30}
        pooled_p = np.concatenate([f, g])[band]
        pooled_g = np.concatenate([g_frozen, g_long])[band]
        out = {"M1": float(auroc(f, g_frozen)), "M1m": float(np.mean(list(attrs.values()))),
               "M2": float(auroc(pooled_p, pooled_g)), "M5": attrs["implies_declining"],
               "ends_with_question": attrs["ends_with_question"], "by_attribute": attrs}
        groups: dict[str, dict[str, list]] = {k: {"p": [], "y": []}
                                              for k in ("choice", "score", "bool")}
        for i, kind in enumerate(val_kind):
            k = kind if kind in groups else "choice"
            groups[k]["p"].append(v[i, :n[i]])
            groups[k]["y"].append(val_label[i])
        m = {k: _metrics_for_group(k, d["p"], d["y"]) for k, d in groups.items()}
        out.update({"choice_accuracy": m["choice"]["accuracy"], "score_rps": m["score"]["rps"],
                    "score_accuracy": m["score"]["accuracy"],
                    "bool_accuracy": m["bool"]["accuracy"]})
        return out

    probs = {}
    for path in sorted(Path("runs/ens").glob("probs_*.npz")):
        d = np.load(path)
        probs[path.stem[len("probs_"):]] = {k: d[k] for k in ("frozen", "long", "val", "val_n")}

    results: dict = {"single_check": {}, "ensembles": {}}
    worst = 0.0
    for name, (file, idx) in STORED.items():
        if name not in probs:
            continue
        mine = score(probs[name]["frozen"], probs[name]["long"], probs[name]["val"],
                     probs[name]["val_n"])
        e = json.loads(Path(file).read_text(encoding="utf-8"))[idx]
        stored = {"M1": e["M1"]["auroc"], "M2": e["M2"]["auroc"],
                  "M5": e["M1_by_attribute"]["implies_declining"]["auroc"],
                  "choice_accuracy": e["guardrails"]["choice_accuracy"],
                  "score_rps": e["guardrails"]["score_rps"],
                  "score_accuracy": e["guardrails"]["score_accuracy"],
                  "bool_accuracy": e["guardrails"]["bool_accuracy"]}
        diff = max(abs(mine[k] - stored[k]) for k in stored)
        worst = max(worst, diff)
        results["single_check"][name] = diff
    results["single_check_max_abs_diff"] = worst
    print(f"single-model check vs stored outputs: max |diff| = {worst:.3e}", flush=True)

    members = {
        "E_qf8": [f"v01_seed{s}" for s in range(8)],
        "E_mix16": [f"v01_seed{s}" for s in range(8)] + [f"io_sf_seed{s}" for s in range(8)],
        "E_sf8": [f"io_sf_seed{s}" for s in range(8)],
        "Soup": ["soup_v01"],
    }
    for name, names in members.items():
        if not all(n in probs for n in names):
            continue
        n_opts = probs[names[0]]["val_n"]
        assert all((probs[m]["val_n"] == n_opts).all() for m in names)
        f = np.mean([probs[m]["frozen"] for m in names], axis=0)
        g = np.mean([probs[m]["long"] for m in names], axis=0)
        v = np.mean([probs[m]["val"] for m in names], axis=0)
        s = score(f, g, v, n_opts)
        s["z"] = {k: (s[k] - REF[k][0]) / REF[k][1] for k in REF}
        z = s["z"]
        meets = sum(z[k] >= 1.0 for k in CORE) >= 2 and all(z[k] >= 0 for k in CORE)
        s["meets_distillation_rule"] = bool(meets)
        s["z_sum_core"] = float(sum(z[k] for k in CORE))
        results["ensembles"][name] = s
        print(f"{name}: " + ", ".join(f"{k} {s[k]:.4f} (z {z[k]:+.2f})" for k in REF)
              + f" | choice {s['choice_accuracy']:.4f} rps {s['score_rps']:.4f} "
              f"score {s['score_accuracy']:.4f} bool {s['bool_accuracy']:.4f} "
              f"| meets {meets}", flush=True)

    e = results["ensembles"]
    candidates = [n for n in ("E_qf8", "E_mix16") if n in e and e[n]["meets_distillation_rule"]]
    if not candidates:
        teacher = None
    elif len(candidates) == 1:
        teacher = candidates[0]
    else:
        gap = e["E_mix16"]["z_sum_core"] - e["E_qf8"]["z_sum_core"]
        teacher = "E_mix16" if gap > 0.1 else "E_qf8"
    results["teacher"] = teacher
    results["soup_candidate"] = bool(e.get("Soup", {}).get("meets_distillation_rule", False))
    print(f"teacher: {teacher}; soup candidate: {results['soup_candidate']}")
    Path("runs/ens/results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
