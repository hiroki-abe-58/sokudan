"""Does adding a catch-all option ("その他") make sokudan fall apart? (docs/probe_catch_all.md)

    python scripts/probe_catch_all.py run --model sokudan:GeneLab/sokudan-ja-310m \
        --name sokudan_v021 --device cpu
    python scripts/probe_catch_all.py run --model sokudan:GeneLab/sokudan-ja-310m \
        --name sokudan_v021_otherlast --device cpu --other-last
    python scripts/probe_catch_all.py report --names sokudan_v021 sokudan_v021_otherlast \
        --a-from sokudan_v021_otherlast=sokudan_v021

    # the laya-multilingual reference needs `pip install laya` (skipped when absent)
    python scripts/probe_catch_all.py run \
        --model laya:convaiinnovations/laya-multilingual --name laya_multilingual

    # a short smoke run: one condition, two orders
    python scripts/probe_catch_all.py run --model sokudan:GeneLab/sokudan-ja-310m \
        --name smoke --conditions B --n-orders 2 --out probe_smoke

Both commands take `--out DIR` (default `./probe_catch_all_out/`). Dependencies: sokudan's
public API (`sokudan.load`, `Agent.predict`), numpy and the standard library.

The set (`docs/probe_catch_all/set.json`) holds 80 expense states over ten accounts plus
10 states that fit none of them.
Four conditions, each asked in the same 20 random option orders (order j uses
`random.Random(SEED + j)` for every condition, so B and D share their permutations):

- A: the ten accounts with descriptions, no catch-all; the 10 unmatched states are left out.
- B: the ten accounts + "その他: 上記以外".
- C: five accounts (消耗品費, 事務用品費, 会議費, 交際費, 通信費) + その他; states of the other
  five accounts and the unmatched states have gold その他.
- D: B with every description removed (labels only).

`run` writes the raw probabilities (`<out>/<name>.json`); `report` computes the tables
(`<out>/summary.json`). sokudan gets the state as a string (its training input); laya gets
`{"body": state}` (the form its baseline used, docs/baseline_ja.md). sokudan's default
calibration touches bool only, so choice probabilities are raw.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

SET = Path(__file__).resolve().parents[1] / "docs" / "probe_catch_all" / "set.json"
DEFAULT_OUT = Path("probe_catch_all_out")
SEED = 20260928
N_ORDERS = 20
C_ACCOUNTS = ("消耗品費", "事務用品費", "会議費", "交際費", "通信費")


def load_set() -> dict:
    return json.loads(SET.read_text(encoding="utf-8"))


def conditions(s: dict) -> dict[str, dict]:
    accounts = s["accounts"]
    other, other_desc = s["other_label"], s["other_description"]
    items = s["items"]
    matched = [it for it in items if it["gold"] != other]
    with_other = dict(accounts) | {other: other_desc}
    c_labels = {a: accounts[a] for a in C_ACCOUNTS} | {other: other_desc}
    return {
        "A": {"criteria": dict(accounts),
              "items": [(it["id"], it["gold"]) for it in matched]},
        "B": {"criteria": with_other,
              "items": [(it["id"], it["gold"]) for it in items]},
        "C": {"criteria": c_labels,
              "items": [(it["id"], it["gold"] if it["gold"] in C_ACCOUNTS else other)
                        for it in items]},
        "D": {"criteria": {k: "" for k in with_other},
              "items": [(it["id"], it["gold"]) for it in items]},
    }


def orders(k: int) -> list[list[int]]:
    return [random.Random(SEED + j).sample(range(k), k) for j in range(N_ORDERS)]


def make_scorer(spec: str, device: str):
    kind, _, target = spec.partition(":")
    if kind == "sokudan":
        import sokudan

        agent = sokudan.load(target, device=device)
        return lambda state, questions: agent.predict(state, questions)["answers"]
    if kind == "laya":
        try:
            import laya
        except ImportError:
            return None  # the reference run is optional; `pip install laya` to include it

        agent = laya.load(target, device=device)
        return lambda state, questions: agent.predict({"body": state}, questions)["answers"]
    raise SystemExit(f"unknown model spec {spec!r}")


def run(args) -> int:
    s = load_set()
    states = {it["id"]: it["state"] for it in s["items"]}
    score = make_scorer(args.model, args.device)
    if score is None:
        print(f"skipped {args.model}: laya is not installed", flush=True)
        return 0
    n_orders = args.n_orders
    out = {"model": args.model, "seed": SEED, "n_orders": n_orders,
           "other_last": bool(args.other_last), "conditions": {}}
    for name, cond in conditions(s).items():
        if args.conditions and name not in args.conditions:
            continue
        labels = list(cond["criteria"])
        if args.other_last:
            # supplementary (after the pre-registered run): その他 fixed in the last slot,
            # the other options in the same 20 random orders
            if s["other_label"] not in labels:
                continue
            assert labels[-1] == s["other_label"]
            perms = [perm + [len(labels) - 1]
                     for perm in orders(len(labels) - 1)[:n_orders]]
        else:
            perms = orders(len(labels))[:n_orders]
        started = time.perf_counter()
        rows = []
        for item_id, gold in cond["items"]:
            questions = {
                f"o{j:02d}": {"type": "choice", "instructions": s["instructions"],
                              "criteria": {labels[i]: cond["criteria"][labels[i]] for i in perm}}
                for j, perm in enumerate(perms)
            }
            answers = score(states[item_id], questions)
            probs = []
            for j in range(len(perms)):
                p = answers[f"o{j:02d}"].get("probabilities", {})
                probs.append([float(p.get(label, 0.0)) for label in labels])
            rows.append({"id": item_id, "gold": gold, "probs": probs})
        seconds = time.perf_counter() - started
        out["conditions"][name] = {"labels": labels, "orders": perms, "rows": rows,
                                   "seconds": round(seconds, 1)}
        print(f"{name}: {len(rows)} states x {len(perms)} orders, K={len(labels)}, "
              f"{seconds:.1f} s", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.name}.json"
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")
    return 0


# ---------------------------------------------------------------- report

def analyse(cond: dict, s: dict) -> dict:
    other = s["other_label"]
    uncertain = {it["id"] for it in s["items"] if it.get("uncertain")}
    unmatched = {it["id"] for it in s["items"] if it["gold"] == other}
    labels = cond["labels"]
    k = len(labels)
    li = {lab: i for i, lab in enumerate(labels)}
    perms = cond["orders"]
    rows = cond["rows"]
    P = np.array([r["probs"] for r in rows])                  # [n, orders, K], canonical order
    P = P / np.clip(P.sum(-1, keepdims=True), 1e-12, None)
    gold = np.array([li[r["gold"]] for r in rows])
    ids = [r["id"] for r in rows]
    pred = P.argmax(-1)                                       # [n, orders]
    ties = int(((P == P.max(-1, keepdims=True)).sum(-1) > 1).sum())
    correct = pred == gold[:, None]
    per_order = correct.mean(0)
    sure = np.array([i not in uncertain for i in ids])

    # slot of the chosen label, per order
    slot_of = np.array([[perm.index(c) for c in range(k)] for perm in perms])  # [orders, K]
    chosen_slot = np.take_along_axis(np.broadcast_to(slot_of, P.shape[:1] + slot_of.shape),
                                     pred[..., None], -1)[..., 0]
    flips = np.array([len(set(p)) > 1 for p in pred])
    modal_share = np.array([np.bincount(p, minlength=k).max() / len(p) for p in pred])
    entropy = -(P * np.log(np.clip(P, 1e-12, None))).sum(-1) / np.log(k)

    res = {
        "K": k, "n_states": len(rows), "ties": ties,
        "acc_mean": float(per_order.mean()), "acc_min": float(per_order.min()),
        "acc_max": float(per_order.max()),
        "acc_mean_without_uncertain": float(correct[sure].mean()),
        "acc_order_averaged_probs": float((P.mean(1).argmax(-1) == gold).mean()),
        "first_slot_rate": float((chosen_slot == 0).mean()),
        "last_slot_rate": float((chosen_slot == k - 1).mean()),
        "flip_rate": float(flips.mean()),
        "modal_share_mean": float(modal_share.mean()),
        "max_prob_mean": float(P.max(-1).mean()),
        "entropy_norm_mean": float(entropy.mean()),
        "per_state": {i: {"acc": float(c.mean()), "flip": bool(f),
                          "labels": [labels[x] for x in p]}
                      for i, c, f, p in zip(ids, correct, flips, pred, strict=True)},
    }
    gold_is_other = np.array([r["gold"] == other for r in rows])
    groups = {"gold_not_other": ~gold_is_other}
    if other in li:
        o = li[other]
        groups["gold_other"] = gold_is_other
        groups["gold_other_unmatched"] = np.array([i in unmatched for i in ids])
        groups["gold_other_from_accounts"] = gold_is_other & ~groups["gold_other_unmatched"]
        other_slot = slot_of[:, o]                              # [orders]
        res["other"] = {}
        for g, m in groups.items():
            if not m.any():
                continue
            po = P[m][..., o]
            res["other"][g] = {
                "n_states": int(m.sum()),
                "other_rate": float((pred[m] == o).mean()),
                "acc": float(correct[m].mean()),
                "p_other_mean": float(po.mean()),
                "p_other_median": float(np.median(po)),
                "p_other_p90": float(np.quantile(po, 0.9)),
                "p_other_ge_0.5": float((po >= 0.5).mean()),
            }
        for g, m in groups.items():
            if m.any() and g in res["other"]:
                res["other"][g]["flip_rate"] = float(flips[m].mean())
                counts = np.bincount(pred[m].ravel(), minlength=k)
                res["other"][g]["predicted"] = {labels[i]: int(c)
                                                for i, c in enumerate(counts) if c}
        last = other_slot == k - 1
        res["other_by_slot"] = {
            "orders_with_other_last": int(last.sum()),
            "acc_other_last": float(correct[:, last].mean()) if last.any() else None,
            "acc_other_not_last": float(correct[:, ~last].mean()) if (~last).any() else None,
            "other_rate_gold_not_other_other_last":
                float((pred[~gold_is_other][:, last] == o).mean()) if last.any() else None,
            "other_rate_gold_not_other_other_not_last":
                float((pred[~gold_is_other][:, ~last] == o).mean()) if (~last).any() else None,
            "p_other_mean_by_slot_gold_not_other": {
                int(sl): float(P[~gold_is_other][:, other_slot == sl, o].mean())
                for sl in sorted(set(other_slot.tolist()))},
        }
    else:
        res["other"] = {"gold_not_other": {"n_states": int((~gold_is_other).sum()),
                                           "acc": float(correct[~gold_is_other].mean())}}
    # near-miss confusion: decisions of the pair's states, over the labels offered
    conf = {}
    for a, b in s["near_miss_pairs"]:
        if a not in li or b not in li:
            continue
        for g in (a, b):
            m = np.array([r["gold"] == g and r["id"].split("_")[0] != "none" for r in rows])
            # in C, only states whose own account is offered count as that account's
            m &= np.array([_account_of(r["id"], s) == g for r in rows])
            counts = np.bincount(pred[m].ravel(), minlength=k)
            conf[g] = {labels[i]: int(c) for i, c in enumerate(counts) if c}
    res["near_miss"] = conf
    return res


def _account_of(item_id: str, s: dict) -> str:
    for it in s["items"]:
        if it["id"] == item_id:
            return it["gold"]
    raise KeyError(item_id)


def report(args) -> int:
    s = load_set()
    summary = {}
    for name in args.names:
        raw = json.loads((args.out / f"{name}.json").read_text(encoding="utf-8"))
        summary[name] = {c: analyse(v, s) for c, v in raw["conditions"].items()}
        if "A" not in raw["conditions"] and name in args.a_from:
            base = args.out / f"{args.a_from[name]}.json"
            raw_a = json.loads(base.read_text(encoding="utf-8"))
            summary[name]["A"] = analyse(raw_a["conditions"]["A"], s)
        summary[name]["_seconds"] = {c: v["seconds"] for c, v in raw["conditions"].items()}
        # A restricted to the states of C's five accounts (the comparison for C, rule 5)
        if "A" in summary[name]:
            per_a = summary[name]["A"]["per_state"]
            in_c = [i for i in per_a if _account_of(i, s) in C_ACCOUNTS]
            summary[name]["_A_on_C_accounts"] = {
                "n_states": len(in_c), "acc": float(np.mean([per_a[i]["acc"] for i in in_c])),
                "flip_rate": float(np.mean([per_a[i]["flip"] for i in in_c]))}
    path = args.out / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, conds in summary.items():
        print(f"\n## {name}")
        for c in [c for c in "ABCD" if c in conds]:
            r = conds[c]
            o = r["other"]
            line = (f"{c} K={r['K']:>2} n={r['n_states']:>2} acc {r['acc_mean']:.3f} "
                    f"[{r['acc_min']:.3f}, {r['acc_max']:.3f}] "
                    f"sure {r['acc_mean_without_uncertain']:.3f} "
                    f"flip {r['flip_rate']:.3f} first {r['first_slot_rate']:.3f} "
                    f"maxp {r['max_prob_mean']:.3f} H {r['entropy_norm_mean']:.3f} "
                    f"ties {r['ties']}")
            print(line)
            for g, v in o.items():
                print(f"    {g}: {v}")
            if "other_by_slot" in r:
                print(f"    by slot: {r['other_by_slot']}")
            print(f"    near-miss: {r['near_miss']}")
        if "_A_on_C_accounts" in conds:
            print(f"A on C's accounts: {conds['_A_on_C_accounts']}")
    print(f"\nwrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--device", default="cpu")
    r.add_argument("--other-last", action="store_true",
                   help="supplementary: keep その他 in the last slot (B, C, D only)")
    r.add_argument("--conditions", nargs="*", choices=list("ABCD"), default=None,
                   help="run only these conditions (default: all four)")
    r.add_argument("--n-orders", type=int, default=N_ORDERS,
                   help=f"the first N of the {N_ORDERS} seeded orders (default {N_ORDERS})")
    r.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p = sub.add_parser("report")
    p.add_argument("--names", nargs="+", required=True)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--a-from", nargs="*", default=[],
                   help="name=base pairs: take condition A from the base run")
    args = parser.parse_args(argv)
    if args.cmd == "report":
        args.a_from = dict(x.split("=", 1) for x in args.a_from)
    return run(args) if args.cmd == "run" else report(args)


if __name__ == "__main__":
    sys.exit(main())
