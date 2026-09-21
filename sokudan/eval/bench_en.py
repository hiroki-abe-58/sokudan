"""`bench_en` — the English twin of `bench_ja`, built to isolate one question.

`docs/baseline_ja.md` §6.2 measured a position bias in `laya-multilingual` on
Japanese: across five conditions that changed only the schema, the first presented
option was chosen 0 or 1 times out of 300 in every one of them, and 「急がない」 drew
0 selections in first position against 250 in last. That is a strong enough finding
to send to the model's authors, and the first question they will ask is whether it is
a Japanese problem or a model problem.

So this is `bench_ja` with one variable changed: the language. Same three schemas,
same label weights (`DEPARTMENT_WEIGHTS`, `URGENCY_WEIGHTS`, `CHURN_TRUE_RATE`
imported from `bench_ja` rather than restated, so they cannot drift apart), same
randomisation axes, same generator, same blind three-valued verification. The options
are the natural English equivalents.

`bench_en` is evaluation-only, exactly like `bench_ja`. It is never trained on.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from sokudan.eval.bench_ja import (
    CHURN_TRUE_RATE,
    DEPARTMENT_WEIGHTS,
    URGENCY_WEIGHTS,
)

# Keys are the English option strings; the *order* matches bench_ja's so that
# "first presented option" means the same slot in both benchmarks.
DEPARTMENTS: dict[str, str] = {
    "Billing": "payments, refunds, invoices, double charges -- anything about money",
    "Technical": "bugs, outages, errors, being unable to log in -- the product not working",
    "Sales": "pricing plans, new contracts, quotes, adding seats -- a purchase decision",
    "Other": "general correspondence that fits none of the above",
}

URGENCY_LEVELS: list[str] = ["Not urgent", "Soon", "Work is blocked"]

# bench_ja weights are keyed by the Japanese strings; remap onto the English ones by
# position so the two benchmarks have the same label distribution by construction.
_DEPT_WEIGHTS = dict(zip(DEPARTMENTS, DEPARTMENT_WEIGHTS.values(), strict=True))
_URG_WEIGHTS = dict(zip(URGENCY_LEVELS, URGENCY_WEIGHTS.values(), strict=True))

INDUSTRIES = [
    "a regional logistics firm", "a dental clinic", "a SaaS startup",
    "a construction contractor", "a university department", "a food wholesaler",
    "a municipal office", "an online retailer", "a staffing agency",
    "a car dealership", "a boutique hotel", "an accounting practice",
]
ROLES = [
    "an office administrator", "a shift supervisor", "the owner",
    "a part-time assistant", "an IT coordinator", "a procurement officer",
    "a finance manager", "a front-desk receptionist",
]
STYLES = [
    "polite and formal", "brisk and businesslike", "faintly irritated but civil",
    "mostly bullet points", "chatty and informal", "terse, almost curt",
    "apologetic and hedging",
]
CHANNELS = [
    "a support email", "a web contact form", "a reply in an existing ticket",
    "a message to a shared team inbox",
]
LENGTHS = [
    ("short", "60-110 words"),
    ("medium", "120-200 words"),
    ("long", "220-340 words"),
]

# A leaked label word turns the benchmark into keyword matching. Case-insensitive,
# and "urgent" is banned because it is the giveaway for the top urgency level.
FORBIDDEN = [
    *DEPARTMENTS.keys(), *URGENCY_LEVELS,
    "cancel", "cancellation", "terminate", "termination", "urgent", "urgency",
]
META_MARKERS = ["here is", "here's", "sure,", "certainly", "```", "subject:", "###"]

MIN_CHARS = 200
MAX_CHARS = 2600

SYSTEM_PROMPT = (
    "You are a data generator that writes English business messages. "
    "You output only the message text that satisfies the given conditions. "
    "You never output commentary, preamble, sign-off notes, or markdown decoration."
)


@dataclass
class BenchItem:
    """One evaluation item: the generated text plus the labels it was written from."""

    item_id: str
    state: str
    department: str
    urgency: int
    churn: bool
    industry: str
    role: str
    style: str
    length: str
    channel: str
    generator_model: str
    generator_temperature: float
    verified: bool | None = None
    verdicts: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id, "state": self.state,
            "department": self.department, "urgency": self.urgency,
            "churn": self.churn, "industry": self.industry, "role": self.role,
            "style": self.style, "length": self.length, "channel": self.channel,
            "generator_model": self.generator_model,
            "generator_temperature": self.generator_temperature,
            "verified": self.verified, "verdicts": self.verdicts,
        }


@dataclass
class LabelDraw:
    department: str
    urgency: int
    churn: bool
    industry: str
    role: str
    style: str
    length_name: str
    length_hint: str
    channel: str


def draw_labels(rng: random.Random) -> LabelDraw:
    """Sample gold labels and surface axes independently, as `bench_ja` does."""
    dept = rng.choices(list(_DEPT_WEIGHTS), weights=list(_DEPT_WEIGHTS.values()))[0]
    urgency = rng.choices(list(_URG_WEIGHTS), weights=list(_URG_WEIGHTS.values()))[0]
    length_name, length_hint = rng.choice(LENGTHS)
    return LabelDraw(
        department=dept,
        urgency=URGENCY_LEVELS.index(urgency),
        churn=rng.random() < CHURN_TRUE_RATE,
        industry=rng.choice(INDUSTRIES),
        role=rng.choice(ROLES),
        style=rng.choice(STYLES),
        length_name=length_name,
        length_hint=length_hint,
        channel=rng.choice(CHANNELS),
    )


_DEPARTMENT_BEHAVIOUR = {
    "Billing": "the message is about money -- an invoice, a refund, a charge that looks wrong",
    "Technical": "the message is about the product not working -- an error, an "
                 "outage, a failed login",
    "Sales": "the message is about a purchase decision -- pricing, a quote, more "
             "seats, a new contract",
    "Other": "the message is ordinary correspondence and fits none of the above",
}
_URGENCY_BEHAVIOUR = [
    "the writer is in no hurry and says so implicitly; nothing is blocked",
    "the writer would like this handled before long, but work is continuing",
    "the writer's work has stopped until this is resolved",
]
_CHURN_BEHAVIOUR = {
    True: (
        "the writer hints, without stating it outright, that they may stop using the "
        "service -- doubt about renewing, a mention of looking at what else is out there"
    ),
    False: "the writer gives no sign of leaving; they expect to keep using the service",
}


def build_generation_prompt(draw: LabelDraw) -> str:
    """One prompt, conditioned on all three gold labels at once."""
    banned = ", ".join(f'"{w}"' for w in sorted(set(FORBIDDEN)))
    return f"""Write one English business message.

## Setting
- Sender: {draw.role} at {draw.industry}
- Channel: {draw.channel}
- Tone: {draw.style}
- Length: {draw.length_name} ({draw.length_hint})

## Conditions (all must hold)
1. {_DEPARTMENT_BEHAVIOUR[draw.department]}
2. {_URGENCY_BEHAVIOUR[draw.urgency]}
3. {_CHURN_BEHAVIOUR[draw.churn]}

## Prohibited (important)
- Do not use any of these words: {banned}. Describe the situation instead of naming
  the category.
- No subject line, no heading, no preamble such as "Here is".
- All company names and personal names must be fictional.

Output only the message body."""


def bench_questions() -> dict[str, dict[str, Any]]:
    """The three `bench_en` questions, in the shape Laya and sokudan both accept."""
    return {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this message?",
            "criteria": dict(DEPARTMENTS),
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": list(URGENCY_LEVELS),
        },
        "churn": {
            "type": "noul",
            "instructions": "Is the sender hinting that they may stop using the service?",
        },
    }


VERIFY_SYSTEM = (
    "You read an English business message and answer questions about it. "
    "You output only JSON and never any commentary."
)


def build_verify_prompt(state: str) -> str:
    """Blind verification: the gold is never shown, only the criteria.

    Same shape as the corpus verifier -- three-valued, criteria supplied so the
    question means the same thing to the checker as it did to the writer.
    """
    options = "\n".join(f"   - {k}: {v}" for k, v in DEPARTMENTS.items())
    levels = "\n".join(f"   - {i}: {name}" for i, name in enumerate(URGENCY_LEVELS))
    return f"""Read the message and answer the three questions.

## Message
{state}

## Questions
1. Which team should handle this? Answer with exactly one of the option names.
{options}
2. How urgent is it? Answer with the index 0, 1 or 2.
{levels}
3. Is the sender hinting they may stop using the service? Answer "yes", "no" or
   "unclear". Answer "yes" only if the message implies it without saying it outright.

## How to answer
- Judge only from the message. If a question cannot be decided, answer "unclear".
- Output only JSON: {{"department": "...", "urgency": 0, "churn": "no"}}"""
