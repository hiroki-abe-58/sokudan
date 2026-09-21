"""Cross-domain boolean attributes that require reading the document (Day 2).

`docs/benchmarks.md` §8 measured the problem this module exists to fix: the
`bool` head scored *higher* with an empty state (0.703) than with the real one
(0.623), because 72 of its 92 boolean schemas were mechanical rewrites of a `choice`
or `score` label. Asking "does this document belong to 請求?" teaches category
membership. It does not teach reading what a writer meant.

So these attributes are generation *conditions*, not derivations. The document is
written to carry the attribute, and the condition is the gold label -- the same trick
`bench_ja` uses, applied to intent rather than to category.

Three tiers, and the tiers are the point (`docs/benchmarks.md` §6):

    S  surface    -- decidable from the words and the shape of the text
    E  explicit   -- an attitude or request the writer put into words
    I  implicit   -- what the writer meant without saying it

`bench_ja`'s boolean ("does the sender *suggest* cancellation?") is tier I. Reporting
one pooled AUROC over all three would let tier S carry a number that says nothing
about tier I, which is exactly the failure Day 1 could not see until it ran the
benchmark. Held-out validation therefore reports per tier, and the Day 2 gate is three
conditions rather than one.

Nothing here may name cancellation, contract termination, switching providers or
competitors -- not in a question, a paraphrase, a negation or a behaviour instruction.
`scripts/check_catalog_leak.py` enforces that statically, over this file's own text,
before any generation runs. The ban stops at the *question*: generated bodies may use
that vocabulary, because a model that has never seen those tokens in context is not
protected from `bench_ja`, it is handicapped on it (§3.2).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from sokudan.schema.question import BoolQuestion

TIERS = ("S", "E", "I")

TIER_NAMES = {
    "S": "表層",
    "E": "明示的態度",
    "I": "非明示の意図",
}


@dataclass(frozen=True)
class IntentAttribute:
    """One yes/no property a document is written to carry, plus how to ask about it."""

    id: str
    tier: str
    forms: list[str]
    """Question texts. `forms[0]` is canonical; the rest are paraphrases. >= 4."""
    negation_form: str
    """Asks about the *absence*, so the gold label inverts.

    Deliberately not a 「〜していないか」 negative interrogative: in Japanese the
    answer 「はい」 to one of those is ambiguous about what is being affirmed, which
    would inject label noise into the very head this module is trying to fix. Every
    negation here keeps a positive predicate over an absence
    (「〜に触れずに書かれているか」), so 「はい」 always means the absence holds.
    """
    true_behaviour: str
    false_behaviour: str
    banned_phrases: list[str] = field(default_factory=list)
    """Phrases the body may not contain *when this attribute is selected*, either way.

    For tier I these carry the load: the attribute is defined as something the writer
    did not say, so the generator saying it outright is a label error, not a stylistic
    one. Banned symmetrically rather than only on the true side -- a phrase banned only
    when true is itself a shortcut, since its absence would then carry information.
    """

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"{self.id}: unknown tier {self.tier!r}")
        if len(self.forms) < 4:
            raise ValueError(f"{self.id}: needs >= 4 question forms, has {len(self.forms)}")
        if len(set(self.forms)) != len(self.forms):
            raise ValueError(f"{self.id}: duplicate question forms")

    def question(self, form_index: int = 0) -> BoolQuestion:
        return BoolQuestion(
            instructions=self.forms[form_index], false_label="いいえ", true_label="はい"
        )

    def negated_question(self) -> BoolQuestion:
        return BoolQuestion(
            instructions=self.negation_form, false_label="いいえ", true_label="はい"
        )

    def behaviour(self, label: int) -> str:
        return self.true_behaviour if label else self.false_behaviour


# ---------------------------------------------------------------------------
# Tier S -- surface. The floor. If these do not transfer, the bug is in training.
# ---------------------------------------------------------------------------

_SURFACE: list[IntentAttribute] = [
    IntentAttribute(
        id="mentions_deadline", tier="S",
        forms=[
            "期限や日付に言及しているか",
            "いつまでに、という時期の指定が書かれているか",
            "この文面には具体的な日付や締切が含まれるか",
            "対応の期限が明示されているか",
        ],
        negation_form="この文面は時期や期限に触れずに書かれているか",
        true_behaviour="具体的な日付・曜日・締切のいずれかを1つ以上、本文に書く",
        false_behaviour="時期にはまったく触れず、タイミングの判断を相手に委ねる書き方にする",
    ),
    IntentAttribute(
        id="mentions_attachment", tier="S",
        forms=[
            "添付や別送の資料に言及しているか",
            "何らかの資料の受け渡しに触れているか",
            "この文面は添付ファイルや共有リンクの存在に言及しているか",
            "書類やデータを送った、または送ってほしい旨が書かれているか",
        ],
        negation_form="この文面は資料の受け渡しに触れずに書かれているか",
        true_behaviour="添付・別送・共有リンクのいずれかに具体的に触れる",
        false_behaviour="資料やファイルの受け渡しには一切触れない",
    ),
    IntentAttribute(
        id="ends_with_question", tier="S",
        forms=[
            "文末が相手への問いかけで終わっているか",
            "最後の一文は質問か",
            "この文面は疑問の形で締めくくられているか",
            "締めくくりが相手への問いになっているか",
        ],
        negation_form="この文面は問いかけ以外の形で締めくくられているか",
        true_behaviour="最後の一文を相手への質問にし、疑問符または疑問の語尾で終える",
        false_behaviour="最後の一文を依頼・報告・挨拶のいずれかで締め、問いかけでは終えない",
    ),
    IntentAttribute(
        id="lists_multiple_issues", tier="S",
        forms=[
            "別種の問題を複数挙げているか",
            "扱っている論点は2つ以上か",
            "この文面は性質の異なる複数の事柄に触れているか",
            "話題が1つに絞られていないか",
        ],
        negation_form="この文面は単一の論点だけを扱っているか",
        true_behaviour="性質の異なる問題を2つ以上、区別がつく形で並べる",
        false_behaviour="話題を1つに絞りきり、他の論点には触れない",
    ),
    IntentAttribute(
        id="cites_numbers", tier="S",
        forms=[
            "金額・件数・番号などの具体的な数値を挙げているか",
            "本文に具体的な数量が書かれているか",
            "この文面は数字を伴う事実を示しているか",
            "件数や金額が特定できる形で書かれているか",
        ],
        negation_form="この文面は具体的な数量を挙げずに書かれているか",
        true_behaviour="金額・件数・回数・管理番号のいずれかを1つ以上、文脈付きで具体的に書く",
        false_behaviour="数量は「いくつか」「何度か」「たびたび」のような語で濁し、数字を書かない",
    ),
    IntentAttribute(
        id="names_third_party", tier="S",
        forms=[
            "自分と相手以外の第三者に言及しているか",
            "書き手と読み手以外の人物や組織が登場するか",
            "この文面には当事者以外の関係者が出てくるか",
            "やり取りの外にいる誰かに触れているか",
        ],
        negation_form="この文面は書き手と読み手の二者だけで完結しているか",
        true_behaviour="別部署・別の担当者・家族・取引先など、当事者以外の第三者を具体的に登場させる",
        false_behaviour="登場人物を書き手と読み手の二者だけに限り、他の関係者は出さない",
    ),
]


# ---------------------------------------------------------------------------
# Tier E -- explicit. The writer put it into words.
# ---------------------------------------------------------------------------

_EXPLICIT: list[IntentAttribute] = [
    IntentAttribute(
        id="states_dissatisfaction", tier="E",
        forms=[
            "不満をはっきり言葉にしているか",
            "困っている、または納得がいかない旨が明言されているか",
            "この文面は書き手の不満を明示的に述べているか",
            "現状への否定的な評価が言葉で示されているか",
        ],
        negation_form="この文面は不満を言葉にせずに書かれているか",
        true_behaviour="不満・困窮・納得のいかなさを、評価の言葉としてはっきり書く",
        false_behaviour="起きた事実の報告に徹し、良し悪しの評価は一切書かない",
    ),
    IntentAttribute(
        id="states_gratitude", tier="E",
        forms=[
            "感謝を述べているか",
            "相手への礼が言葉で示されているか",
            "この文面には謝意の表明が含まれるか",
            "書き手は相手の対応に感謝しているか",
        ],
        negation_form="この文面は謝意を述べずに書かれているか",
        true_behaviour="相手のこれまでの対応や配慮に対する感謝を、明確な言葉で書く",
        false_behaviour="時候や定型の挨拶だけにとどめ、感謝の言葉は書かない",
    ),
    IntentAttribute(
        id="requests_compensation", tier="E",
        forms=[
            "返金・交換・補償を求めているか",
            "金銭または物品による埋め合わせを要求しているか",
            "この文面は損失の穴埋めを求めているか",
            "書き手は具体的な補償を要求しているか",
        ],
        negation_form="この文面は金銭的な埋め合わせに触れずに書かれているか",
        true_behaviour="返金・交換・値引き・補償のいずれかを、明確な要求として書く",
        false_behaviour="対応は求めるが、金銭や物品による埋め合わせには一切触れない",
    ),
    IntentAttribute(
        id="requests_owner_change", tier="E",
        forms=[
            "担当者の変更を求めているか",
            "別の人に引き継いでほしいと書いているか",
            "この文面は窓口の交代を要求しているか",
            "書き手は今の担当から別の担当への変更を望んでいるか",
        ],
        negation_form="この文面は担当の交代を求めずに書かれているか",
        true_behaviour="今の担当ではなく別の担当者への引き継ぎを、明確な要求として書く",
        false_behaviour="今の担当者にそのまま依頼を続ける前提で書き、交代には触れない",
    ),
    IntentAttribute(
        id="requests_phone_callback", tier="E",
        forms=[
            "電話での折り返しを求めているか",
            "口頭での連絡を希望しているか",
            "この文面は電話による回答を要求しているか",
            "書き手は通話での対応を求めているか",
        ],
        negation_form="この文面は通話での連絡を求めずに書かれているか",
        true_behaviour="電話での連絡を明確に求め、都合のつく時間帯も添える",
        false_behaviour="書面やメールでの回答を求めるか、連絡手段をまったく指定しない",
    ),
    IntentAttribute(
        id="demands_urgent_action", tier="E",
        forms=[
            "至急の対応を明確に要求しているか",
            "早急な処置を言葉で求めているか",
            "この文面は急いでほしい旨を明示しているか",
            "書き手は対応を急ぐよう明言しているか",
        ],
        negation_form="この文面は対応を急がせる言葉を使わずに書かれているか",
        true_behaviour="「至急」に相当する要求を、はっきりした言葉で明言する",
        false_behaviour="急がない旨を書くか、対応の時期の判断を相手に委ねる書き方にする",
    ),
    IntentAttribute(
        id="admits_own_fault", tier="E",
        forms=[
            "自分の側の非や手違いを認めているか",
            "書き手は自分の落ち度に言及しているか",
            "この文面には自分側の不備を認める記述があるか",
            "書き手は自らの確認漏れや誤りを認めているか",
        ],
        negation_form="この文面は自分側の落ち度に触れずに書かれているか",
        true_behaviour="自分の確認漏れ・手違い・認識違いを認める一文を具体的に入れる",
        false_behaviour="自分の側の不備にはまったく触れない",
    ),
    IntentAttribute(
        id="blames_recipient", tier="E",
        forms=[
            "相手の非をはっきり指摘しているか",
            "読み手側の落ち度を名指ししているか",
            "この文面は相手側の不備を明示的に責めているか",
            "原因が相手にあると言葉で述べているか",
        ],
        negation_form="この文面は相手側の落ち度を指摘せずに書かれているか",
        true_behaviour=(
            "相手側のどの対応に不備があったのかを具体的に挙げ、"
            "そちらに原因があるとはっきり書く"
        ),
        false_behaviour="原因の所在を特定せず、起きた事象だけを書く",
    ),
    IntentAttribute(
        id="refers_to_past_exchange", tier="E",
        forms=[
            "過去のやり取りや前回の連絡に触れているか",
            "以前の経緯が言及されているか",
            "この文面は今回が初めてではないことを示しているか",
            "前回の対応や連絡に具体的な言及があるか",
        ],
        negation_form="この文面は過去の経緯に触れずに書かれているか",
        true_behaviour="前回の連絡・過去の対応・経緯に、いつ何があったかが分かる形で触れる",
        false_behaviour=(
            "今回が最初の連絡であることが本文から分かるようにし、"
            "前回・先日・いつもといった過去を指す語を一切使わない"
        ),
    ),
    IntentAttribute(
        id="offers_alternative", tier="E",
        forms=[
            "自分から代替案や妥協案を提示しているか",
            "書き手は解決の案を出しているか",
            "この文面には書き手側からの提案が含まれるか",
            "要求だけでなく、落としどころの案が示されているか",
        ],
        negation_form="この文面は書き手からの代案を出さずに書かれているか",
        true_behaviour="書き手の側から具体的な代案・妥協案を1つ提示する",
        false_behaviour="判断や対応の中身は相手に委ね、自分からは案を出さない",
    ),
]


# ---------------------------------------------------------------------------
# Tier I -- implicit. What the writer meant without saying it. The real target:
# `bench_ja`'s boolean is this shape, and every one of these bans the words that
# would turn it into tier E.
# ---------------------------------------------------------------------------

_IMPLICIT: list[IntentAttribute] = [
    IntentAttribute(
        id="implies_running_out_of_patience", tier="I",
        forms=[
            "これ以上は待てないという含みがあるか",
            "書き手の我慢が限界に近いことが読み取れるか",
            "この文面からは、待ち続ける気がもうないことがうかがえるか",
            "明言はされていないが、書き手はもう待つつもりがないと読めるか",
        ],
        negation_form="この文面からは、書き手がまだ待つ姿勢でいると読めるか",
        true_behaviour=(
            "何度も待たされてきた経過や、日程がこれ以上動かせない事情を書き、"
            "我慢が限界に近いことが行間から伝わるようにする。"
            "ただし限界である旨を直接の言葉にはしない"
        ),
        false_behaviour="相手の都合を待つ姿勢が自然に読み取れる、落ち着いた書き方にする",
        banned_phrases=["もう待てません", "もう待て", "我慢の限界", "限界です", "堪忍袋"],
    ),
    IntentAttribute(
        id="implies_disappointment", tier="I",
        forms=[
            "期待外れだったと感じていることが読み取れるか",
            "書き手の落胆が行間からうかがえるか",
            "この文面からは、想定していた水準に届かなかったという感覚が読めるか",
            "明言はされていないが、書き手は期待を裏切られたと感じていると読めるか",
        ],
        negation_form="この文面は期待と結果の差に触れずに書かれているか",
        true_behaviour=(
            "事前に想定していたことと実際の結果の落差が分かる書き方にし、落胆が伝わるようにする。"
            "ただし不満や失望そのものは言葉にしない"
        ),
        false_behaviour=(
            "結果がおおむね想定どおりだったことが読み取れるようにし、"
            "不足や物足りなさを感じさせる記述を入れない"
        ),
        banned_phrases=["失望", "がっかり", "期待外れ", "残念です", "落胆"],
    ),
    IntentAttribute(
        id="implies_declining", tier="I",
        forms=[
            "依頼や提案を断るつもりであることをにおわせているか",
            "書き手は受けない方向に傾いていると読めるか",
            "この文面からは、話を進める気がないことがうかがえるか",
            "明言はされていないが、書き手は辞退する見込みだと読めるか",
        ],
        negation_form="この文面からは、書き手が話を受ける方向でいると読めるか",
        true_behaviour=(
            "条件が合わない事情や手が空かない状況を並べ、受けない方向であることが"
            "行間から伝わるようにする。ただし断りの言葉そのものは使わない"
        ),
        false_behaviour="前向きに引き受ける姿勢が読み取れるよう、進め方の話を中心に書く",
        banned_phrases=["お断り", "辞退", "お受けできません", "見送らせて", "遠慮させて"],
    ),
    IntentAttribute(
        id="implies_escalation", tier="I",
        forms=[
            "上長や別の窓口に相談するつもりをほのめかしているか",
            "書き手はこの相手以外にも話を持っていく気配があるか",
            "この文面からは、話が当事者の外へ広がりうることが読めるか",
            "明言はされていないが、書き手は別のところに掛け合うつもりだと読めるか",
        ],
        negation_form="この文面は当事者の二者間だけで収める前提で書かれているか",
        true_behaviour=(
            "社内で報告を求められている状況や、他に相談できる先がある事実に触れ、"
            "話が外に出うることが行間から伝わるようにする。ただし誰に相談するとは書かない"
        ),
        false_behaviour="この相手とのやり取りだけで解決する前提で、他の相談先には一切触れない",
        banned_phrases=["上長に相談", "上司に相談", "しかるべき機関", "消費生活センター", "弁護士"],
    ),
    IntentAttribute(
        id="implies_reproach_for_delay", tier="I",
        forms=[
            "対応の遅さを、遅いとは書かずに責めているか",
            "返事が来ないことへの咎めが行間から読めるか",
            "この文面からは、待たされたことへの不服がうかがえるか",
            "明言はされていないが、書き手は対応の遅さを問題にしていると読めるか",
        ],
        negation_form="この文面は対応の速さ遅さを問題にせずに書かれているか",
        true_behaviour=(
            "前回いつ連絡したか、何度目の連絡かといった事実を淡々と並べ、"
            "遅さへの咎めが行間から伝わるようにする。ただし遅いとは書かない"
        ),
        false_behaviour=(
            "これまでの対応の速さに不足はなかったことが読み取れる書き方にし、"
            "経過した時間そのものには触れない"
        ),
        banned_phrases=["遅い", "遅すぎ", "いつまで待", "対応が遅れて", "放置"],
    ),
    IntentAttribute(
        id="implies_out_of_depth", tier="I",
        forms=[
            "自分では手に負えないことを、認めたくない書き方で示しているか",
            "書き手が実は対処しきれていないことが行間から読めるか",
            "この文面からは、書き手の手に余る状況だとうかがえるか",
            "明言はされていないが、書き手は自力では収拾がつかないと読めるか",
        ],
        negation_form="この文面からは、書き手が状況を掌握していると読めるか",
        true_behaviour=(
            "自分では原因が特定できていないこと、試した対処がどれも外れていることを"
            "具体的に並べて書き、手に余っている状況が事実の積み重ねから伝わるようにする。"
            "ただし直接的な弱音は書かない"
        ),
        false_behaviour=(
            "原因の見当がついていることと、次に何をするかを具体的に書き、"
            "状況を掌握していることがはっきり分かるようにする"
        ),
        banned_phrases=["お手上げ", "手に負えません", "どうしていいか分かりません", "無理です"],
    ),
    IntentAttribute(
        id="implies_pressure_from_others", tier="I",
        forms=[
            "自分以外の誰かに急かされていることを、直接は書かずに伝えているか",
            "書き手の背後に外からの圧力があることが読み取れるか",
            "この文面からは、書き手が誰かに追い立てられていることがうかがえるか",
            "明言はされていないが、書き手は自分の都合だけで動いていないと読めるか",
        ],
        negation_form="この文面からは、書き手が自分の判断だけで動いていると読めるか",
        true_behaviour=(
            "報告の期日や他の部署の予定といった外側の事情に触れ、"
            "書き手が急かされていることが行間から伝わるようにする。ただし誰に言われたとは書かない"
        ),
        false_behaviour=(
            "日程も進め方も書き手自身が決めたことだと分かる書き方にし、"
            "外からの事情や他者の都合には一切触れない"
        ),
        banned_phrases=["上から言われ", "急かされて", "板挟み", "せっつかれ"],
    ),
    IntentAttribute(
        id="implies_reluctant_compliance", tier="I",
        forms=[
            "納得していないが従う姿勢が読み取れるか",
            "書き手は不承不承で受け入れていると読めるか",
            "この文面からは、割り切れないまま応じている様子がうかがえるか",
            "明言はされていないが、書き手は腑に落ちないまま進めると読めるか",
        ],
        negation_form="この文面からは、書き手が納得したうえで進めると読めるか",
        true_behaviour=(
            "提示された条件に従うとはっきり書いたうえで、決まった経緯や適用範囲を"
            "くり返し確認し、自分の当初の考えにも触れることで、"
            "割り切れていないことが行間から伝わるようにする。ただし不服は言葉にしない"
        ),
        false_behaviour=(
            "提示された方針に納得していることをはっきり示したうえで、"
            "次の段取りを前向きに書く"
        ),
        banned_phrases=["納得できません", "釈然としません", "不本意", "腑に落ちません"],
    ),
]


ATTRIBUTES: list[IntentAttribute] = [*_SURFACE, *_EXPLICIT, *_IMPLICIT]
BY_ID: dict[str, IntentAttribute] = {a.id: a for a in ATTRIBUTES}


# ---------------------------------------------------------------------------
# Held out from training entirely (`docs/benchmarks.md` §6)
# ---------------------------------------------------------------------------

HELD_OUT: frozenset[str] = frozenset({
    "implies_running_out_of_patience",   # I -- closest analogue to the bench boolean
    "implies_declining",                 # I -- an unstated refusal
    "implies_escalation",                # I -- an unstated next move
    "requests_owner_change",             # E -- control: does an explicit ask transfer?
    "ends_with_question",                # S -- floor: if this is 0.5 the wiring is broken
})

TRAINABLE: list[IntentAttribute] = [a for a in ATTRIBUTES if a.id not in HELD_OUT]


# ---------------------------------------------------------------------------
# Pairs that must not be sampled onto the same document
# ---------------------------------------------------------------------------

EXCLUSIVE_PAIRS: list[tuple[str, str]] = [
    # Every tier-I attribute is defined as "without saying it". Putting one on the
    # same document as the explicit attribute it negates makes the two conditions
    # contradict each other, and the contradiction would surface as a tier-I
    # verification failure rather than as the design error it is.
    ("states_dissatisfaction", "implies_disappointment"),
    ("demands_urgent_action", "implies_running_out_of_patience"),
    ("blames_recipient", "implies_reproach_for_delay"),
    ("implies_declining", "implies_reluctant_compliance"),
    # Added after the smoke run. A document that names the other side's fault also
    # reads as let down, and one that lists several failed attempts also reads as
    # out of its depth -- so a false label on the tier-I side was contradicted by
    # the text in both cases.
    ("blames_recipient", "implies_disappointment"),
    ("lists_multiple_issues", "implies_out_of_depth"),
]

MAX_IMPLICIT_PER_DOC = 1
"""At most one tier-I attribute per document.

The pairwise exclusions above were not enough, and the smoke run says why. Tier-I
attributes do not just conflict with their explicit counterparts -- **they entail each
other**. One smoke document carried five of them; it was written to imply reproach
for a slow reply, and in doing so it also, truthfully, implied disappointment,
impatience and outside pressure, all three of which had been drawn false.

Reading the disagreements settles who was wrong: the verifier was right and the gold
was wrong. Independent 50/50 draws over a mutually entailing set produce negative
labels the document contradicts, and discarding those pairs afterwards would have
deleted most of the negatives and taught the head to answer yes -- the §7.3 trap,
reached by a route §3.1 did not anticipate.

Capping at one is the fix that keeps the marginal rate at exactly 50/50, because it
resolves at selection time rather than by flipping a label. The cost is tier-I
training volume: roughly 4,250 pairs instead of 15,000. That is the honest ceiling on
how much intent supervision this corpus can carry without lying in the labels.
"""

_CONFLICTS: dict[str, set[str]] = {}
for _a, _b in EXCLUSIVE_PAIRS:
    _CONFLICTS.setdefault(_a, set()).add(_b)
    _CONFLICTS.setdefault(_b, set()).add(_a)


def conflicts_with(attribute_id: str) -> set[str]:
    return _CONFLICTS.get(attribute_id, set())


def select_attributes(
    rng: random.Random, *, pool: list[IntentAttribute], count: int,
    forced: list[IntentAttribute] | None = None,
) -> list[IntentAttribute]:
    """Draw `count` attributes, never two that conflict.

    Resolved at *selection* time rather than by flipping a label afterwards: a flip
    would move the positive rate off 50/50 for whichever attribute lost, and the
    whole point of the 50/50 draw is that a model cannot profit from a prior.
    """
    chosen: list[IntentAttribute] = list(forced or [])
    blocked: set[str] = set()
    for attribute in chosen:
        blocked |= conflicts_with(attribute.id)
        blocked.add(attribute.id)

    implicit_used = sum(1 for a in chosen if a.tier == "I")
    candidates = [a for a in pool if a.id not in blocked]
    rng.shuffle(candidates)
    # Tier-I first, so the cap is spent on an intent attribute rather than left
    # unused because the draw happened to fill up on surface ones.
    candidates.sort(key=lambda a: a.tier != "I")
    for candidate in candidates:
        if len(chosen) >= count:
            break
        if candidate.id in blocked:
            continue
        if candidate.tier == "I" and implicit_used >= MAX_IMPLICIT_PER_DOC:
            continue
        if candidate.tier == "I":
            implicit_used += 1
        chosen.append(candidate)
        blocked |= conflicts_with(candidate.id)
        blocked.add(candidate.id)

    rng.shuffle(chosen)
    return chosen


def catalog_summary() -> dict[str, object]:
    """Counted facts for the manifest. Nothing here is asserted by hand."""
    by_tier: dict[str, int] = {t: 0 for t in TIERS}
    for attribute in ATTRIBUTES:
        by_tier[attribute.tier] += 1
    held_by_tier: dict[str, int] = {t: 0 for t in TIERS}
    for attribute_id in HELD_OUT:
        held_by_tier[BY_ID[attribute_id].tier] += 1
    return {
        "n_attributes": len(ATTRIBUTES),
        "by_tier": by_tier,
        "held_out": sorted(HELD_OUT),
        "held_out_by_tier": held_by_tier,
        "n_trainable": len(TRAINABLE),
        "trainable_by_tier": {
            t: sum(1 for a in TRAINABLE if a.tier == t) for t in TIERS
        },
        "exclusive_pairs": [list(p) for p in EXCLUSIVE_PAIRS],
        "question_forms_per_attribute": sorted({len(a.forms) for a in ATTRIBUTES}),
    }
