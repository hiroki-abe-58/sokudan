"""`scripts/merge_corpora.py`: renaming, and restricting a label to one source."""

from __future__ import annotations

import json
import sys

import pytest

from scripts import merge_corpora


def _write(path, docs):
    path.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in docs) + "\n",
                    encoding="utf-8")


def _doc(doc_id, labels, split="train"):
    return {"doc_id": doc_id, "split": split, "state": "本文", "domain": "x",
            "intent_labels": dict(labels),
            "verdicts": {k: ("はい" if v else "いいえ") for k, v in labels.items()}}


def _run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["merge_corpora.py", *args])
    return merge_corpora.main()


def _read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_ids_are_namespaced_so_colliding_corpora_merge(tmp_path, monkeypatch):
    a, b, out = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "out.jsonl"
    _write(a, [_doc("i2-000000", {"x": 1})])
    _write(b, [_doc("i2-000000", {"x": 0})])
    assert _run(monkeypatch, "v2=" + str(a), "v3=" + str(b), "--out", str(out)) == 0
    assert [d["doc_id"] for d in _read(out)] == ["v2-i2-000000", "v3-i2-000000"]


def test_labels_only_from_strips_the_attribute_everywhere_else(tmp_path, monkeypatch):
    old, new, out = tmp_path / "old.jsonl", tmp_path / "new.jsonl", tmp_path / "out.jsonl"
    _write(old, [_doc("i2-000000", {"readmitted": 1, "kept": 0}),
                 _doc("i2-000001", {"kept": 1})])
    _write(new, [_doc("i5-000000", {"readmitted": 0, "kept": 1})])
    _run(monkeypatch, "v2=" + str(old), "v5=" + str(new), "--out", str(out),
         "--labels-only-from", "v5=readmitted")
    docs = {d["doc_id"]: d for d in _read(out)}
    # Stripped from the old corpus, label and verdict both...
    assert "readmitted" not in docs["v2-i2-000000"]["intent_labels"]
    assert "readmitted" not in docs["v2-i2-000000"]["verdicts"]
    # ...without touching anything else on the document...
    assert docs["v2-i2-000000"]["intent_labels"] == {"kept": 0}
    # ...and kept in the source it is restricted to.
    assert docs["v5-i5-000000"]["intent_labels"] == {"readmitted": 0, "kept": 1}


def test_labels_only_from_rejects_a_malformed_argument(tmp_path, monkeypatch):
    src, out = tmp_path / "a.jsonl", tmp_path / "out.jsonl"
    _write(src, [_doc("i2-000000", {"x": 1})])
    with pytest.raises(SystemExit):
        _run(monkeypatch, "v2=" + str(src), "--out", str(out), "--labels-only-from", "v5")
