"""Detect English words that leaked into a Japanese document.

Latin script is normal in Japanese business text: acronyms (PDF, API, ID), URLs and file
names (https://example.com/report.pdf), units (mm, kg, MPa), product names (Excel, Zoom).
What is not normal is an ordinary English word standing in for a Japanese one --
「Cannot」「weekends」「myself」「program式017」 -- which is how a generator that is less
fluent in Japanese shows it. Measured on 2026-09-24: 6.9% of mistral-small3.2's
documents (docs_v5) against 2.7% of qwen3's (docs_v4), by a cruder count.

So this counts *intrusions*, not Latin characters:

* all-caps tokens are acronyms and never count;
* URLs and file names are removed before tokenising;
* units, version markers and a short list of product names are allowed;
* anything else of three or more letters counts.

The rate is intrusion characters over document characters, so one stray word in a long
document weighs less than the same word in a two-line note.
"""

from __future__ import annotations

import re

_URL = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\w+")
_FILE = re.compile(r"[\w\-]+\.(?:pdf|xlsx?|docx?|csv|zip|pptx?|txt|png|jpe?g|com|jp|net|org)\b",
                   re.IGNORECASE)
_TOKEN = re.compile(r"[A-Za-z][A-Za-z']*")

ALLOWED = frozenset(w.lower() for w in """
    ver rev min max mm cm km kg kgf mg ml ms kb mb gb tb mbps rpm hz khz mhz ghz mpa hpa kpa
    kwh ph db egfr iot wi fi web wifi app apps email mail tel fax api office
    google excel word outlook teams zoom slack line gmail youtube instagram twitter facebook
    amazon apple iphone ipad android windows mac macos chrome safari edge firefox
    github git gitlab jira notion dropbox onedrive sharepoint drive salesforce kintone
    zip pdf csv xlsx docx pptx https http www com jp net org example
""".split())


def latin_intrusions(text: str) -> list[str]:
    """English words in `text` that are not acronyms, links, file names, units or products."""
    stripped = _FILE.sub(" ", _URL.sub(" ", text))
    out = []
    for token in _TOKEN.findall(stripped):
        word = token.strip("'")
        if len(word) < 3 or word.isupper() or word.lower() in ALLOWED:
            continue
        # xxx / XXXX are placeholders; two or more capitals mark an identifier, a
        # handle or a product name (GitHub, iPhone, InspectionReport), not prose.
        if set(word.lower()) == {"x"} or sum(c.isupper() for c in word) >= 2:
            continue
        out.append(word)
    return out


def latin_intrusion_rate(text: str) -> float:
    """Intrusion characters as a share of the document's characters."""
    body = text.strip()
    if not body:
        return 0.0
    return sum(len(w) for w in latin_intrusions(body)) / len(body)
