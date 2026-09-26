"""scripts/run_bench_en_sokudan.py asks bench_en's questions with bench_ja's shapes."""

from __future__ import annotations

from sokudan.eval import bench_en, bench_ja
from sokudan.schema.question import parse_question


def test_bench_en_questions_parse_with_the_bench_ja_option_counts():
    en, ja = bench_en.bench_questions(), bench_ja.bench_questions()
    assert en.keys() == ja.keys()
    for key in en:
        q_en, q_ja = parse_question(en[key]), parse_question(ja[key])
        assert type(q_en) is type(q_ja)
        assert len(getattr(q_en, "options", [])) == len(getattr(q_ja, "options", []))


def test_the_runner_swaps_only_the_questions():
    import scripts.run_bench_en_sokudan  # noqa: F401  (import must not run anything)
    import sokudan.eval.sokudan_baseline as sb

    assert sb.bench_questions is bench_ja.bench_questions
