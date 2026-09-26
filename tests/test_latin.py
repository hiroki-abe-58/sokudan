"""`sokudan.data.latin`: English words leaking into Japanese documents."""

from __future__ import annotations

import pytest

from sokudan.data.latin import latin_intrusion_rate, latin_intrusions

BODY = "いつもお世話になっております。先日の件についてご連絡いたします。" * 4


@pytest.mark.parametrize("text", [
    "添付の PDF と Excel をご確認ください。API の仕様は SNS で告知します。",
    "詳細は https://example.com/report.pdf をご覧ください。",
    "報告書は Final_Report_v2.xlsx に保存しました。",
    "圧力は 0.5 MPa、回転数は 1200 rpm、重さは 3 kg です。",
    "Zoom か Teams で打ち合わせ、議事録は Google Drive に置きます。",
    "@GameWorldOfficial の告知と GitHub の InspectionReport を参照。",
    "伝票番号 XXXX-XXXX、担当 xxx 様。",
    "A 社と B 社の見積もりを比較し、ID と URL を控えました。",
])
def test_ordinary_latin_in_japanese_business_text_is_not_an_intrusion(text):
    assert latin_intrusions(text) == []
    assert latin_intrusion_rate(text) == 0.0


@pytest.mark.parametrize("text,words", [
    ("来週の schedule を確認してください。", ["schedule"]),
    ("宿泊日program式017に不具合がございました。", ["program"]),
    ("Cannot 対応できません。", ["Cannot"]),
    ("週末は weekends なので the office is closed.", ["weekends", "the", "closed"]),
    ("窓（Fenster）の修理をお願いします。", ["Fenster"]),
    ("ご対応は directement お願いします。", ["directement"]),
])
def test_prose_words_are_intrusions(text, words):
    assert latin_intrusions(text) == words
    assert latin_intrusion_rate(text) > 0


def test_rate_is_intrusion_characters_over_document_characters():
    text = BODY + "schedule"
    assert latin_intrusion_rate(text) == pytest.approx(len("schedule") / len(text))


def test_rate_of_an_empty_document_is_zero():
    assert latin_intrusion_rate("   ") == 0.0


def test_validate_regenerates_documents_over_the_threshold():
    from scripts.build_intent_corpus import MAX_LATIN_INTRUSION_RATE, validate

    assert validate(BODY + "確認します。", []) is None
    assert validate(BODY + "schedule を確認します。", []) == "latin_intrusion"
    # Acronyms and links alone never trigger it.
    assert validate(BODY + "PDF は https://example.com に置きました。", []) is None
    # 1.0 turns the check off.
    assert validate(BODY + "schedule を確認します。", [], max_latin_rate=1.0) is None
    assert MAX_LATIN_INTRUSION_RATE == 0.005
