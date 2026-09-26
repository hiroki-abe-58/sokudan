"""docs/soup.md Phase 4: build the pre-registered soups, score everything, apply the rules.

    uv run python scripts/soup_eval.py build      # soups + per-row predictions
    uv run python scripts/soup_eval.py report     # metrics, tests, curve, rules

`build` averages each fixed subset of `runs/v01_seed<s>/model.pt` (every floating
tensor, heads included; integer tensors and config from the first seed), saves it as
`runs/soup/soups/<name>/model.pt`, and collects per-row predictions for it and for any
single seed not collected yet (`runs/ens/probs_<name>.npz`, the scoring of
`scripts/ens_collect.py`). `report` scores every single and soup with the evaluation's
own metric functions, adds bool calibration (10-bin top-label ECE and mean P(true) -
positive rate on `val_v2` and on the frozen held-out set), runs the main test (k = 4
soups vs 16 singles, one-sided exact permutation, Holm over M1m / M2 / M5), the
independent comparison (soup of seeds 8-15 vs singles 0-7), the k-curve, the final-
candidate and calibration rules, and the head-similarity description. Writes
`runs/soup/results.json`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import sokudan.config  # noqa: F401
from scripts.eval_heldout import load_checkpoint, load_rows
from scripts.eval_local_attention import p_true
from scripts.eval_long_states import rows_from_documents
from scripts.nf_test import holm
from scripts.seed8_stats import exact_permutation_p
from sokudan.calibration.metrics import auroc, ece
from sokudan.train.dataset import load_examples
from sokudan.train.loop import TrainConfig, _metrics_for_group, build_collator

SUBSETS = {
    **{f"k2_{a}_{a + 1}": [a, a + 1] for a in range(0, 16, 2)},
    **{f"k4_{a}_{a + 3}": list(range(a, a + 4)) for a in range(0, 16, 4)},
    "k8_0_7": list(range(8)), "k8_8_15": list(range(8, 16)),
    "k16_0_15": list(range(16)),
}
SINGLES = [f"v01_seed{s}" for s in range(16)]
PROBS = Path("runs/ens")
SOUPS = Path("runs/soup/soups")
CORE = ("M1m", "M2", "M5")
GUARDS = (("choice_accuracy", 1), ("score_rps", -1), ("score_accuracy", 1), ("bool_accuracy", 1))
METRICS = ["M1m", "M2", "M5", "M1", "choice_accuracy", "score_rps", "score_accuracy",
           "bool_accuracy", "ece_val", "gap_val", "ece_heldout", "gap_heldout"]


def make_soup(name: str, seeds: list[int]) -> Path:
    out = SOUPS / name / "model.pt"
    if out.exists():
        return out
    blobs = [torch.load(f"runs/v01_seed{s}/model.pt", map_location="cpu", weights_only=False)
             for s in seeds]
    averaged = {}
    for key, first in blobs[0]["state_dict"].items():
        if torch.is_floating_point(first):
            averaged[key] = torch.stack([b["state_dict"][key].float() for b in blobs]).mean(
                0).to(first.dtype)
        else:
            averaged[key] = first.clone()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": averaged, "config": dict(blobs[0]["config"])}, out)
    return out


def collect(name: str, path: Path | str, tokenizer, data,
            precomputed: Path | None = None) -> None:
    """Per-row predictions of one checkpoint into `runs/ens/probs_<name>.npz`.

    `precomputed` is the `<out>.probs<i>.npz` that `scripts/eval_local_attention.py
    --save-probs` wrote for the same checkpoint in the same pass that produced its
    evaluation output (same functions, batch size and bucketing); when it exists it is
    copied instead of running the model over the three sets again (A5).
    """
    from scripts.ens_collect import val_probs

    target = PROBS / f"probs_{name}.npz"
    if target.exists():
        return
    if precomputed is not None and Path(precomputed).exists():
        d = np.load(precomputed)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez(target, frozen=d["frozen"], long=d["long"], val=d["val"], val_n=d["val_n"],
                 input_order=d["input_order"])
        print(f"collected {name} (from {precomputed})", flush=True)
        return
    model, encoding, device = load_checkpoint(str(path))
    config = TrainConfig(device=device, batch_size=32, encoding=encoding,
                         input_order=model.input_order)
    collator = build_collator(tokenizer, config)
    f = p_true(model, data["frozen"], collator, config)
    g = p_true(model, data["long"], collator, config)
    v, n = val_probs(model, data["val"], collator, config)
    np.savez(target, frozen=f, long=g, val=v, val_n=n, input_order=np.array(model.input_order))
    print(f"collected {name}", flush=True)
    del model
    torch.cuda.empty_cache()


def load_data() -> dict:
    return {"frozen": load_rows(Path("data/v2/heldout_val_v2.jsonl")),
            "long": [r for r in rows_from_documents(Path("data/docs_long_val.jsonl"))
                     if r["held_out"]],
            "val": load_examples("data/val_v2.jsonl")}


class Scorer:
    def __init__(self, tokenizer, data) -> None:
        frozen, long_rows, val = data["frozen"], data["long"], data["val"]
        self.g_frozen = np.array([int(r["label"]) for r in frozen])
        self.g_long = np.array([int(r["label"]) for r in long_rows])
        cache: dict[str, int] = {}
        for r in frozen + long_rows:
            if r["state"] not in cache:
                cache[r["state"]] = len(tokenizer(r["state"], add_special_tokens=True)
                                        ["input_ids"])
        self.band = np.array([400 <= cache[r["state"]] < 800 for r in frozen + long_rows])
        by_attr: dict[str, list[int]] = defaultdict(list)
        for i, r in enumerate(frozen):
            by_attr[r["attribute"]].append(i)
        self.by_attr = {a: np.array(v) for a, v in sorted(by_attr.items()) if len(v) >= 30}
        self.kind = [e.kind for e in val]
        self.label = np.array([e.label for e in val])

    def __call__(self, f: np.ndarray, g: np.ndarray, v: np.ndarray, n: np.ndarray) -> dict:
        attrs = {a: float(auroc(f[idx], self.g_frozen[idx])) for a, idx in self.by_attr.items()}
        pooled_p = np.concatenate([f, g])[self.band]
        pooled_g = np.concatenate([self.g_frozen, self.g_long])[self.band]
        out = {"M1": float(auroc(f, self.g_frozen)), "M1m": float(np.mean(list(attrs.values()))),
               "M2": float(auroc(pooled_p, pooled_g)), "M5": attrs["implies_declining"],
               "by_attribute": attrs}
        groups: dict[str, dict[str, list]] = {k: {"p": [], "y": []}
                                              for k in ("choice", "score", "bool")}
        for i, kind in enumerate(self.kind):
            k = kind if kind in groups else "choice"
            groups[k]["p"].append(v[i, :n[i]])
            groups[k]["y"].append(int(self.label[i]))
        m = {k: _metrics_for_group(k, d["p"], d["y"]) for k, d in groups.items()}
        out.update({"choice_accuracy": m["choice"]["accuracy"], "score_rps": m["score"]["rps"],
                    "score_accuracy": m["score"]["accuracy"],
                    "bool_accuracy": m["bool"]["accuracy"]})
        bool_rows = np.array([k == "bool" for k in self.kind])
        pv = v[bool_rows][:, :2]
        yv = self.label[bool_rows]
        out["ece_val"] = ece(pv, yv, n_bins=10)
        out["gap_val"] = float(pv[:, 1].mean() - yv.mean())
        ph = np.stack([1 - f, f], axis=1)
        out["ece_heldout"] = ece(ph, self.g_frozen, n_bins=10)
        out["gap_heldout"] = float(f.mean() - self.g_frozen.mean())
        return out


def head_similarity() -> dict:
    names = ("scorer.weight", "ordinal.cut.weight", "ordinal.location.weight")
    vecs: dict[str, list[np.ndarray]] = {n: [] for n in names}
    for s in range(16):
        sd = torch.load(f"runs/v01_seed{s}/model.pt", map_location="cpu",
                        weights_only=False)["state_dict"]
        for n in names:
            vecs[n].append(sd[n].float().flatten().numpy())
    out = {}
    for n, vs in vecs.items():
        m = np.stack(vs)
        m = m / np.linalg.norm(m, axis=1, keepdims=True)
        sim = m @ m.T
        off = sim[~np.eye(16, dtype=bool)]
        block = {"within_0_7": sim[:8, :8][~np.eye(8, dtype=bool)].mean(),
                 "within_8_15": sim[8:, 8:][~np.eye(8, dtype=bool)].mean(),
                 "across": sim[:8, 8:].mean()}
        out[n] = {"matrix": sim.round(4).tolist(), "offdiag_mean": float(off.mean()),
                  "offdiag_min": float(off.min()), "offdiag_max": float(off.max()),
                  **{k: float(v) for k, v in block.items()}}
    return out


def report(tokenizer, data) -> dict:
    score = Scorer(tokenizer, data)
    probs = {}
    for name in SINGLES + list(SUBSETS):
        path = PROBS / f"probs_{name}.npz"
        if path.exists():
            d = np.load(path)
            probs[name] = score(d["frozen"], d["long"], d["val"], d["val_n"])
    res: dict = {"scores": probs}
    # single-model check against the stored evaluation outputs
    worst = 0.0
    for s in range(16):
        name = f"v01_seed{s}"
        if name not in probs:
            continue
        if s < 3:
            e = json.loads(Path("runs/io/eval_v01.json").read_text(encoding="utf-8"))[s]
        elif s < 8:
            e = json.loads(Path(f"runs/nf/eval_v01_seed{s}.json").read_text(encoding="utf-8"))[0]
        else:
            e = json.loads(Path(f"runs/soup/eval_v01_seed{s}.json").read_text(encoding="utf-8"))[0]
        stored = {"M1": e["M1"]["auroc"], "M2": e["M2"]["auroc"],
                  "M5": e["M1_by_attribute"]["implies_declining"]["auroc"],
                  **{k: e["guardrails"][k] for k, _ in GUARDS}}
        worst = max(worst, max(abs(probs[name][k] - stored[k]) for k in stored))
    res["single_check_max_abs_diff"] = worst
    singles = [probs[n] for n in SINGLES if n in probs]
    res["n_singles"] = len(singles)
    single_mean = {m: float(np.mean([s[m] for s in singles])) for m in METRICS}
    res["single_mean"] = single_mean
    res["single_sd"] = {m: float(np.std([s[m] for s in singles], ddof=1)) for m in METRICS}
    # curve
    curve = {"1": single_mean}
    for k in (2, 4, 8, 16):
        members = [probs[n] for n in SUBSETS if n.startswith(f"k{k}_") and n in probs]
        if members:
            curve[str(k)] = {m: float(np.mean([x[m] for x in members])) for m in METRICS}
            curve[str(k)]["n_soups"] = len(members)
    res["curve"] = curve
    # main test
    k4 = [probs[n] for n in SUBSETS if n.startswith("k4_") and n in probs]
    if len(k4) == 4 and len(singles) == 16:
        pvals, diffs = {}, {}
        for m in CORE:
            d, p, splits = exact_permutation_p([x[m] for x in k4], [x[m] for x in singles])
            pvals[f"{m} soup(k=4) > single"], diffs[f"{m} soup(k=4) > single"] = p, d
        tests = holm(pvals)
        for name in tests:
            tests[name]["diff"] = diffs[name]
            tests[name]["splits"] = splits
        res["main_test"] = tests
        m1m_rejected = tests["M1m soup(k=4) > single"]["reject"]
        top = probs.get("k16_0_15")
        if top is not None:
            checks = {"M2": top["M2"] >= single_mean["M2"], "M5": top["M5"] >= single_mean["M5"]}
            for g, sign in GUARDS:
                checks[g] = (top[g] >= single_mean[g]) if sign > 0 else (top[g] <= single_mean[g])
            res["k16_checks"] = checks
            res["final_candidate"] = bool(m1m_rejected and all(checks.values()))
            res["ece_gap_k16_minus_single"] = top["ece_val"] - single_mean["ece_val"]
            res["temperature_refit_needed"] = bool(res["ece_gap_k16_minus_single"] >= 0.02)
    else:
        res["main_test"] = "not run: needs 4 k=4 soups and 16 singles"
    # independent comparison
    if "k8_8_15" in probs and all(f"v01_seed{s}" in probs for s in range(8)):
        ref = [probs[f"v01_seed{s}"] for s in range(8)]
        soup = probs["k8_8_15"]
        res["independent"] = {m: {"soup_8_15": soup[m],
                                  "singles_0_7_mean": float(np.mean([r[m] for r in ref])),
                                  "singles_0_7_sd": float(np.std([r[m] for r in ref], ddof=1)),
                                  "z": (soup[m] - np.mean([r[m] for r in ref]))
                                  / np.std([r[m] for r in ref], ddof=1),
                                  "singles_below": int(sum(r[m] < soup[m] for r in ref))}
                              for m in METRICS}
    return res


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "report", "heads"))
    args = parser.parse_args()
    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    data = load_data()
    if args.command == "build":
        available = {s for s in range(16) if Path(f"runs/v01_seed{s}/model.pt").exists()}
        for s in sorted(available):
            collect(f"v01_seed{s}", f"runs/v01_seed{s}/model.pt", tokenizer, data)
        for name, seeds in SUBSETS.items():
            if set(seeds) <= available:
                collect(name, make_soup(name, seeds), tokenizer, data)
        return 0
    out = Path("runs/soup/results.json")
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    if args.command == "heads":
        res["head_similarity"] = head_similarity()
    else:
        heads = res.get("head_similarity")
        res = report(tokenizer, data)
        if heads:
            res["head_similarity"] = heads
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float),
                   encoding="utf-8")
    brief = {k: v for k, v in res.items() if k not in ("scores", "head_similarity")}
    print(json.dumps(brief, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
