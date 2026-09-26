"""Stage the v0.2 upload directory locally and check it loads. Uploads nothing.

    uv run python scripts/export_v02.py      # -> runs/release_candidate/hf/

v0.2 is `S8_old` (docs/release_candidate.md §3): the plain average of v0.1 seeds 0-7.
Written in the v0.1 layout (`scripts/push_to_hub.stage`): `model.safetensors`,
`config.json`, the backbone tokenizer, `temperatures.json` (fitted on `val_v2` by
`scripts/calibrate.py` during the bench run; v0.2 is used uncalibrated by default, as
v0.1 is) and `README.md` (a copy of `docs/model_card_v0.2.md`, when it exists).
`config.json` also records the soup: its members, each member's SHA-256 and the soup
checkpoint's SHA-256.

Checks, written to `runs/release_candidate/hf_check.json`:

1. every tensor read back from `model.safetensors` equals `runs/release_candidate/model.pt`;
2. `sokudan.load(<hf dir>)` and `sokudan.load(model.pt)` give identical `predict` output on
   the first five rows of the frozen held-out set (`data/v2/heldout_val_v2.jsonl`; not
   `bench_ja` / `bench_en`), the state passed as the raw string the evaluation uses (a
   dict would be rendered as `key: value` lines, a different input);
3. for information, the largest difference between those P(true) values and the
   batched bf16 predictions saved for the same soup (`runs/ens/probs_k8_0_7.npz`).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

SOURCE = Path("runs/release_candidate/model.pt")
SELECTION = Path("runs/release_candidate/selection.json")
OUT = Path("runs/release_candidate/hf")
CARD = Path("docs/model_card_v0.2.md")
TEMPERATURES = Path("runs/release_candidate/temperatures.json")
N_ROWS = 5


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def stage() -> dict:
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    selection = json.loads(SELECTION.read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    blob = torch.load(str(SOURCE), map_location="cpu", weights_only=False)
    stored = blob.get("config", {})
    backbone = stored.get("backbone", "sbintuitions/modernbert-ja-310m")
    state = {k: v.contiguous() for k, v in blob["state_dict"].items()}
    save_file(state, str(OUT / "model.safetensors"),
              metadata={"format": "pt", "source": "soup of v0.1 seeds 0-7"})
    config = {
        "model_type": "sokudan",
        "architectures": ["SokudanJointModel"],
        "backbone": backbone,
        "encoding": stored.get("encoding", "joint"),
        "input_order": stored.get("input_order", "question_first"),
        "local_attention": stored.get("local_attention"),
        "n_parameters": sum(v.numel() for v in state.values()),
        "question_types": ["choice", "score", "bool"],
        "bool_aliases": ["noul"],
        "license": "apache-2.0",
        "version": "0.2",
        "soup": {
            "rule": ("plain average of every floating tensor of the members (float32 mean, "
                     "cast back); integer tensors and config from the first member"),
            "members": [f"v0.1 seed {s}" for s in range(8)],
            "member_sha256": {f"v01_seed{s}": selection["member_sha256"][f"v01_seed{s}"]
                              for s in range(8)},
            "soup_checkpoint_sha256": selection["sha256"],
            "rebuild": "scripts/make_soup.py",
        },
        "note": (
            "Custom architecture, the v0.1 joint encoding: [CLS] instructions [SEP] "
            "options+markers [SEP] state [SEP] through the backbone, marker hidden states "
            "through a scorer (choice / bool) or the ordinal head (score). Not an "
            "AutoModel -- load with `sokudan.load`."
        ),
    }
    (OUT / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    AutoTokenizer.from_pretrained(backbone).save_pretrained(str(OUT))
    if TEMPERATURES.exists():
        shutil.copy(TEMPERATURES, OUT / "temperatures.json")
    if CARD.exists():
        shutil.copy(CARD, OUT / "README.md")
    return {"files": sorted(p.name for p in OUT.iterdir()),
            "model_safetensors_sha256": sha256(OUT / "model.safetensors")}


def check() -> dict:
    from safetensors.torch import load_file

    import sokudan
    from scripts.eval_heldout import load_rows

    ref = torch.load(str(SOURCE), map_location="cpu", weights_only=False)["state_dict"]
    loaded = load_file(str(OUT / "model.safetensors"))
    tensors_equal = (set(ref) == set(loaded)
                     and all(torch.equal(ref[k], loaded[k]) for k in ref))
    rows = load_rows(Path("data/v2/heldout_val_v2.jsonl"))[:N_ROWS]
    outputs = {}
    for label, target in (("pt", str(SOURCE)), ("hf", str(OUT))):
        agent = sokudan.load(target)
        outputs[label] = [agent.predict(r["state"], {"q": r["question"]})["answers"]
                          for r in rows]
        del agent
    saved = np.load("runs/ens/probs_k8_0_7.npz")["frozen"][:N_ROWS]
    p_hf = np.array([o["q"]["noul"] for o in outputs["hf"]])
    return {
        "tensors_equal": tensors_equal,
        "n_tensors": len(ref),
        "predict_identical_pt_vs_hf": outputs["pt"] == outputs["hf"],
        "rows": [{"doc_id": r.get("doc_id"), "attribute": r["attribute"],
                  "label": r["label"], "noul_hf": o["q"]["noul"]}
                 for r, o in zip(rows, outputs["hf"], strict=True)],
        "max_abs_diff_vs_saved_batched_bf16": float(np.abs(p_hf - saved).max()),
    }


def main() -> int:
    staged = stage()
    result = {**staged, **check()}
    Path("runs/release_candidate/hf_check.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0 if result["tensors_equal"] and result["predict_identical_pt_vs_hf"] else 1


if __name__ == "__main__":
    sys.exit(main())
