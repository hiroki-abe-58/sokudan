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




# ---------------------------------------------------------------------------
# Day 3 expansion. Tier I, 20 more (8 -> 28).
#
# `docs/v02_attempt.md` records why: tripling the corpus over the same 24
# attributes did not move held-out AUROC. The corpus got bigger, not wider. These
# 20 widen the implicit tier along axes the existing 8 do not touch -- the existing
# set is almost entirely negative affect, so a model can do well on it by learning
# "is this person annoyed, and how".
#
# Every one is a thing the writer means without saying, and every one bans the
# phrases that would turn it into tier E.
# ---------------------------------------------------------------------------

_IMPLICIT_V2: list[IntentAttribute] = [
    IntentAttribute(
        id="implies_decision_already_made", tier="I",
        forms=[
            "もう結論は出ているのに相談の体裁をとっているか",
            "書き手はすでに決めていると読めるか",
            "この文面は意見を求める形をとりながら、答えが決まっていると読めるか",
            "明言はされていないが、相談ではなく事後の通知に近いと読めるか",
        ],
        negation_form="この文面からは、書き手がまだ決めかねていると読めるか",
        true_behaviour=(
            "意見を求める形で書きながら、日程や手配がすでに進んでいることを具体的に書き、"
            "結論が出ていることが行間から伝わるようにする。ただし決定済みとは書かない"
        ),
        false_behaviour="本当に判断がついておらず、相手の意見で結論が変わることが読み取れるようにする",
        banned_phrases=["決定済み", "もう決まって", "決まりました", "決定事項"],
    ),
    IntentAttribute(
        id="implies_seeking_exception", tier="I",
        forms=[
            "規定どおりでない扱いを、そうと言わずに期待しているか",
            "書き手は例外的な対応を望んでいると読めるか",
            "この文面からは、通常の手順の外側を期待していることがうかがえるか",
            "明言はされていないが、特別扱いを求めていると読めるか",
        ],
        negation_form="この文面は通常の手順どおりの対応を前提に書かれているか",
        true_behaviour=(
            "自分の事情の特殊さを丁寧に並べ、通常とは違う扱いを期待していることが"
            "行間から伝わるようにする。ただし例外や特別扱いという語は使わない"
        ),
        false_behaviour="決められた手順のとおりに進めるつもりであることがはっきり読み取れるようにする",
        banned_phrases=["例外", "特別扱い", "特例", "融通"],
    ),
    IntentAttribute(
        id="implies_relief", tier="I",
        forms=[
            "書き手が安心したことが読み取れるか",
            "心配していたことが解消したと読めるか",
            "この文面からは、肩の荷が下りた様子がうかがえるか",
            "明言はされていないが、書き手はほっとしていると読めるか",
        ],
        negation_form="この文面は気がかりが残ったまま書かれているか",
        true_behaviour=(
            "懸念していた事柄が解消したことが分かる書き方にし、安堵が行間から伝わるようにする。"
            "ただし安心したとは書かない"
        ),
        false_behaviour="懸念がまだ残っていることが読み取れるようにする",
        banned_phrases=["安心しました", "ほっと", "一安心", "胸をなでおろ"],
    ),
    IntentAttribute(
        id="implies_first_time", tier="I",
        forms=[
            "この種の手続きが初めてであることを、認めずに示しているか",
            "書き手は不慣れだと読めるか",
            "この文面からは、勝手が分かっていない様子がうかがえるか",
            "明言はされていないが、書き手はこの手順に慣れていないと読めるか",
        ],
        negation_form="この文面からは、書き手がこの手順に慣れていると読めるか",
        # "用語の使い方が微妙にずれる" was too subtle to survive alongside six other
        # conditions -- all three gold=true documents in the smoke test read as familiar.
        # Replaced with something the generator can actually execute and the verifier can
        # actually see. The co-occurrence with `refers_to_past_exchange` was the other half
        # of the failure and is handled by an exclusion pair.
        true_behaviour=(
            "手順・必要書類・提出先のうち二つ以上を、一から教えてもらう前提で尋ねる文を入れ、"
            "相手が当然と考えていそうな前提も念のため確認する。ただし初めて・不慣れとは書かない"
        ),
        false_behaviour="手順を把握している書き方にし、確認は挟まず必要なことだけを簡潔に書く",
        banned_phrases=["初めて", "不慣れ", "慣れていません", "よく分からず"],
    ),
    IntentAttribute(
        id="implies_favour_expected", tier="I",
        forms=[
            "見返りを期待していることを、そうと書かずに示しているか",
            "書き手は貸し借りを意識していると読めるか",
            "この文面からは、次は自分の番だという含みがうかがえるか",
            "明言はされていないが、書き手は何かを期待していると読めるか",
        ],
        negation_form="この文面は見返りを意識せずに書かれているか",
        true_behaviour=(
            "過去に自分が便宜を図った経緯をさりげなく挟み、"
            "見返りの期待が行間から伝わるようにする。ただし見返りや貸しとは書かない"
        ),
        false_behaviour="過去の貸し借りには触れず、今回の用件だけを書く",
        banned_phrases=["見返り", "貸し", "borrow", "恩"],
    ),
    IntentAttribute(
        id="implies_personal_urgency", tier="I",
        forms=[
            "個人的な事情で急いでいることを、そうと書かずに示しているか",
            "書き手の急ぎが仕事以外の理由だと読めるか",
            "この文面からは、私的な都合が背景にあることがうかがえるか",
            "明言はされていないが、急ぐ理由は書き手個人の事情だと読めるか",
        ],
        negation_form="この文面からは、急ぐ理由が業務上のものだと読めるか",
        true_behaviour=(
            "自分の不在期間や予定の変更にさりげなく触れ、私的な事情が背景にあることが"
            "行間から伝わるようにする。ただし個人的な理由とは書かない"
        ),
        false_behaviour="急ぐ理由が業務の都合であることがはっきり読み取れるようにする",
        banned_phrases=["個人的な", "私用", "プライベート", "家庭の事情"],
    ),
    IntentAttribute(
        id="implies_wants_to_be_included", tier="I",
        forms=[
            "次回も声をかけてほしい気持ちが読み取れるか",
            "書き手は関わり続けたいと読めるか",
            "この文面からは、輪に入っていたい様子がうかがえるか",
            "明言はされていないが、書き手は今後も参加したいと読めるか",
        ],
        negation_form="この文面は今回限りの関わりとして書かれているか",
        true_behaviour=(
            "自分の関心や貢献できる点を具体的に添え、今後も関わりたい気持ちが"
            "行間から伝わるようにする。ただし参加したいとは書かない"
        ),
        false_behaviour="今回の用件で完結する関わりとして書く",
        banned_phrases=["参加したい", "呼んでください", "involve", "加わりたい"],
    ),
    IntentAttribute(
        id="implies_pride_in_work", tier="I",
        forms=[
            "自分の仕事に誇りを持っていることがにじんでいるか",
            "書き手は成果に自信があると読めるか",
            "この文面からは、手をかけたことへの自負がうかがえるか",
            "明言はされていないが、書き手は自分の仕事を誇っていると読めるか",
        ],
        negation_form="この文面は自分の仕事を淡々と報告しているか",
        true_behaviour=(
            "工夫した点や手間をかけた箇所を具体的に書き込み、自負が行間から伝わるようにする。"
            "ただし自慢めいた言葉は使わない"
        ),
        false_behaviour="やったことを淡々と事実として並べ、評価を差し挟まない",
        banned_phrases=["自信があ", "誇り", "自負", "よくできた"],
    ),
    IntentAttribute(
        id="implies_peer_comparison", tier="I",
        forms=[
            "他の人や別の部署の扱いと暗に比べているか",
            "書き手は自分だけ違うと感じていると読めるか",
            "この文面からは、周囲との差を意識している様子がうかがえるか",
            "明言はされていないが、書き手は他との比較で不満を持っていると読めるか",
        ],
        negation_form="この文面は他との比較に触れずに書かれているか",
        true_behaviour=(
            "同じ立場の人や別の部署で聞いた話をさりげなく挟み、比較していることが"
            "行間から伝わるようにする。ただし不公平や差別とは書かない"
        ),
        false_behaviour="自分の件だけを扱い、他の人や部署の話は一切出さない",
        banned_phrases=["不公平", "えこひいき", "差別", "同じ扱い"],
    ),
    IntentAttribute(
        id="implies_deadline_is_soft", tier="I",
        forms=[
            "示した期限が実は動かせることを、そうと書かずに示しているか",
            "書き手自身は期限を絶対視していないと読めるか",
            "この文面からは、日付に余裕があることがうかがえるか",
            "明言はされていないが、期限は目安にすぎないと読めるか",
        ],
        negation_form="この文面からは、示された期限が動かせないものだと読めるか",
        true_behaviour=(
            "期限を示しつつ、その後の工程に余白があることや調整の余地を匂わせる書き方にする。"
            "ただし融通がきくとは書かない"
        ),
        false_behaviour="期限が動かせないことの理由まで書き、確定した日付として示す",
        banned_phrases=["融通", "多少なら", "目安です", "厳密ではありません"],
    ),
    IntentAttribute(
        id="implies_prior_commitment", tier="I",
        forms=[
            "以前に約束があったことを、そうと書かずに持ち出しているか",
            "書き手は過去の取り決めを根拠にしていると読めるか",
            "この文面からは、前に交わした話が前提にあることがうかがえるか",
            "明言はされていないが、書き手は約束を盾にしていると読めるか",
        ],
        negation_form="この文面は過去の取り決めに依らずに書かれているか",
        true_behaviour=(
            "以前のやり取りの場面や時期を具体的に描き、約束があったことが"
            "行間から伝わるようにする。ただし約束やお約束とは書かない"
        ),
        false_behaviour="今回の事情だけを根拠にし、過去の取り決めには触れない",
        banned_phrases=["お約束", "約束し", "取り決め", "合意して"],
    ),
]


# ---------------------------------------------------------------------------
# Day 3 expansion. Tier E, 10 more (10 -> 20). Each is a stance put into words,
# and several are the explicit twin of a new tier-I attribute -- which is what
# makes the exclusion pairs below necessary rather than decorative.
# ---------------------------------------------------------------------------

_EXPLICIT_V2: list[IntentAttribute] = [
    IntentAttribute(
        id="states_agreement", tier="E",
        forms=[
            "提案や方針への同意をはっきり述べているか",
            "賛成であることが言葉で示されているか",
            "この文面は相手の案を受け入れると明言しているか",
            "書き手は合意の意思を明確に書いているか",
        ],
        negation_form="この文面は賛否を明言せずに書かれているか",
        true_behaviour="提示された案に同意する旨を、はっきりした言葉で書く",
        false_behaviour="賛否には触れず、確認や事実の共有だけにとどめる",
    ),
    IntentAttribute(
        id="states_disagreement", tier="E",
        forms=[
            "提案や方針への反対をはっきり述べているか",
            "同意できない旨が言葉で示されているか",
            "この文面は相手の案を受け入れないと明言しているか",
            "書き手は異議を明確に書いているか",
        ],
        negation_form="この文面は異議を唱えずに書かれているか",
        true_behaviour="提示された案に同意できない旨を、理由とともにはっきり書く",
        false_behaviour="異議は述べず、提示された案を前提に話を進める",
    ),
    IntentAttribute(
        id="requests_meeting", tier="E",
        forms=[
            "打ち合わせの場を設けることを明確に求めているか",
            "直接話す機会を要求しているか",
            "この文面は会議や面談の設定を求めているか",
            "書き手は対面や通話での場を明示的に求めているか",
        ],
        negation_form="この文面は話す場を設けずに文面だけで済ませる前提で書かれているか",
        true_behaviour="打ち合わせの場を設けるよう明確に求め、候補の日程にも触れる",
        false_behaviour="文面のやり取りだけで済ませる前提で書き、場の設定には触れない",
    ),
    IntentAttribute(
        id="requests_written_record", tier="E",
        forms=[
            "書面や記録に残すことを求めているか",
            "証跡を残すよう要求しているか",
            "この文面は文書化を明示的に求めているか",
            "書き手は記録として残すことを求めているか",
        ],
        negation_form="この文面は記録に残すことを求めずに書かれているか",
        true_behaviour="やり取りや決定を書面・議事録として残すよう、明確に求める",
        false_behaviour="記録の形式には触れず、内容のやり取りだけを書く",
    ),
    IntentAttribute(
        id="offers_help", tier="E",
        forms=[
            "自分から手伝いを申し出ているか",
            "協力を買って出ているか",
            "この文面には書き手からの手助けの申し出が含まれるか",
            "書き手は自分が引き受ける意思を明示しているか",
        ],
        negation_form="この文面は手伝いを申し出ずに書かれているか",
        true_behaviour="自分が手を動かす、または引き受ける旨を具体的に申し出る",
        false_behaviour="依頼や報告にとどめ、自分から手伝う申し出はしない",
    ),
    IntentAttribute(
        id="declines_explicitly", tier="E",
        forms=[
            "依頼や提案をはっきり断っているか",
            "受けられない旨が明言されているか",
            "この文面は明確な辞退を含むか",
            "書き手は断ることを言葉にしているか",
        ],
        negation_form="この文面は断りの言葉を使わずに書かれているか",
        # The first version said only "受けられない旨をはっきり書く" and failed on 13 of 20
        # documents. Reading them showed why: a decline presupposes that somebody asked the
        # writer for something, and most generated documents have the writer doing the
        # asking. With no inbound request in the frame, a refusal is incoherent, so the
        # generator quietly dropped the condition. The frame is now part of the instruction.
        true_behaviour=(
            "相手から依頼または提案を受けた場面として書き、その依頼・提案を指して、"
            "受けられない旨を理由とともにはっきりした言葉で書く"
        ),
        false_behaviour="相手からの依頼・提案を受ける方向で話を進める書き方にする",
    ),
    IntentAttribute(
        id="states_uncertainty", tier="E",
        forms=[
            "判断がつかないことを明言しているか",
            "分からないと言葉で認めているか",
            "この文面は不確かさを明示的に述べているか",
            "書き手は自分に判断できないと書いているか",
        ],
        negation_form="この文面は不確かさを述べずに書かれているか",
        true_behaviour="自分では判断がつかない、または情報が足りないことをはっきり書く",
        false_behaviour="判断できる範囲のことを断定的に書き、不確かさには触れない",
    ),
    IntentAttribute(
        id="praises_specific_person", tier="E",
        forms=[
            "特定の人を名指しで褒めているか",
            "誰かの働きを具体的に評価しているか",
            "この文面には個人への称賛が含まれるか",
            "書き手は特定の担当者の仕事ぶりを明示的に評価しているか",
        ],
        negation_form="この文面は個人を名指しで評価せずに書かれているか",
        true_behaviour="特定の担当者（架空の名前）の働きを具体的に挙げて褒める",
        false_behaviour="個人の評価には踏み込まず、事柄だけを書く",
    ),
    IntentAttribute(
        id="states_cost_concern", tier="E",
        forms=[
            "費用が問題だと明言しているか",
            "金額への懸念が言葉で示されているか",
            "この文面は費用面の心配を明示的に述べているか",
            "書き手は値段が引っかかっていると書いているか",
        ],
        negation_form="この文面は費用への懸念を述べずに書かれているか",
        true_behaviour="金額が負担である、または見合わないと感じている旨をはっきり書く",
        false_behaviour="費用には触れず、内容や進め方だけを書く",
    ),
    IntentAttribute(
        id="sets_condition", tier="E",
        forms=[
            "条件付きで受ける旨を明示しているか",
            "受諾の前提となる条件が書かれているか",
            "この文面は「〜であれば」という条件を明示的に置いているか",
            "書き手は引き受ける条件を言葉にしているか",
        ],
        negation_form="この文面は条件を付けずに書かれているか",
        true_behaviour="引き受ける前提となる条件を具体的に一つ以上書く",
        false_behaviour="条件を付けず、そのまま受けるか、条件の話をしない",
    ),
]


# ---------------------------------------------------------------------------
# Day 3 expansion. Tier S, 6 more (6 -> 12). Surface and structural properties.
# `docs/benchmarks.md` §6 found the surface tier is the weak one because its
# held-out attribute is *positional* -- these add more of that kind on purpose, so
# the weakness is measured rather than represented by a single attribute.
# ---------------------------------------------------------------------------

_SURFACE_V2: list[IntentAttribute] = [
    IntentAttribute(
        id="uses_bullet_points", tier="S",
        forms=[
            "箇条書きを使っているか",
            "本文に行頭記号の並びがあるか",
            "この文面は項目を列挙する形式を含むか",
            "内容が箇条書きの形で整理されているか",
        ],
        negation_form="この文面は地の文だけで書かれているか",
        true_behaviour="本文の一部を行頭記号付きの箇条書きにして、三項目以上並べる",
        false_behaviour="箇条書きを使わず、すべて地の文の段落で書く",
    ),
    IntentAttribute(
        id="mentions_time_of_day", tier="S",
        forms=[
            "時刻に言及しているか",
            "何時頃かが書かれているか",
            "この文面には具体的な時刻の記述があるか",
            "時間帯を特定できる記述が含まれるか",
        ],
        negation_form="この文面は時刻に触れずに書かれているか",
        true_behaviour="午前・午後や具体的な時刻を、本文に一つ以上書く",
        false_behaviour="時刻や時間帯にはまったく触れない",
    ),
    IntentAttribute(
        id="includes_greeting", tier="S",
        forms=[
            "冒頭に挨拶があるか",
            "本題の前に時候や定型の挨拶が置かれているか",
            "この文面は挨拶から始まっているか",
            "書き出しが儀礼的な挨拶になっているか",
        ],
        negation_form="この文面は挨拶を置かず本題から始まっているか",
        true_behaviour="本題の前に、時候または定型の挨拶を一文置く",
        false_behaviour="挨拶を置かず、最初の一文から本題に入る",
    ),
    IntentAttribute(
        id="mentions_location", tier="S",
        forms=[
            "場所や地名に言及しているか",
            "どこで、が書かれているか",
            "この文面には場所を特定する記述があるか",
            "建物名や地域名が本文に出てくるか",
        ],
        negation_form="この文面は場所に触れずに書かれているか",
        true_behaviour="架空の地名・建物名・部屋名のいずれかを、本文に一つ以上書く",
        false_behaviour="場所にはまったく触れない",
    ),
    # `uses_plain_form` was written for this tier and then removed after the 200-document
    # smoke test: 15 of 16 documents told to use 常体 came back in 敬体 (discard 0.533,
    # and every failure in the true->false direction). The register is not something the
    # generator drops by accident -- all thirty domains are business correspondence, where
    # 敬体 is near-obligatory, and the style condition loses to every content condition on
    # the same document. Forcing it would buy a surface attribute at the cost of documents
    # that no longer look like the ones bench_ja is drawn from, which is the wrong trade
    # under "bench_ja の純度 > カバレッジ". Recorded in docs/day3_attribute_expansion.md §7.
    IntentAttribute(
        id="mentions_document_name", tier="S",
        forms=[
            "特定の書類や帳票の名称を挙げているか",
            "文書名が本文に出てくるか",
            "この文面には具体的な書類の名前が含まれるか",
            "様式名や帳票名が特定できる形で書かれているか",
        ],
        negation_form="この文面は書類の名称を挙げずに書かれているか",
        true_behaviour="架空の書類名・様式名・帳票名のいずれかを、本文に一つ以上書く",
        false_behaviour="書類そのものには触れるとしても、名称は挙げない",
    ),
]


# ---------------------------------------------------------------------------
# Retired -- removed from the catalogue, kept for re-testing
# ---------------------------------------------------------------------------
#
# Ten tier-I attributes left the catalogue after the second Day 3 smoke test
# (docs/day3_attribute_expansion.md §8): even with no co-occurring attribute on the
# document, the verifier read their gold=false documents as true 37-70% of the time.
# The reading of the documents was that the register itself carries them --
# 「もしよろしければ…ご検討いただけますと幸いです」 *is* testing the waters.
#
# That reading was never tested against a second generator, so the definitions are
# kept here verbatim rather than deleted. They are **not** in `ATTRIBUTES` or `BY_ID`:
# nothing trains on them, and `build_intent_train.py` still drops their labels as
# retired. Only a corpus run that asks for them (`build_intent_corpus.py
# --include-retired`) sees them, which is the H2 test in night 3 §2.

_RETIRED_DEFINITIONS: list[IntentAttribute] = [
    IntentAttribute(
        id="implies_testing_the_waters", tier="I",
        forms=[
            "本決まりではなく様子を見ていることを、そうと書かずに示しているか",
            "書き手はまだ打診の段階だと読めるか",
            "この文面からは、反応を見ようとしている様子がうかがえるか",
            "明言はされていないが、これは下見のような連絡だと読めるか",
        ],
        negation_form="この文面は本決まりの用件として書かれているか",
        true_behaviour=(
            "条件を幅のある形で示し、相手の反応次第で変えられる余地を残す書き方にすることで、"
            "打診段階であることが行間から伝わるようにする。ただし仮やお試しとは書かない"
        ),
        false_behaviour="条件が確定した用件として、具体的に詰めた書き方にする",
        banned_phrases=["仮に", "お試し", "打診", "検討段階"],
    ),
    IntentAttribute(
        id="implies_seeking_validation", tier="I",
        forms=[
            "助言ではなく同意がほしいことが読み取れるか",
            "書き手は背中を押してほしいだけだと読めるか",
            "この文面からは、意見より承認を求めている様子がうかがえるか",
            "明言はされていないが、書き手は賛同を期待していると読めるか",
        ],
        negation_form="この文面は率直な意見を求めて書かれているか",
        true_behaviour=(
            "自分の考えを根拠まで丁寧に固めて示し、問いの形だけを残す書き方にすることで、"
            "同意がほしいことが行間から伝わるようにする。ただし賛成してほしいとは書かない"
        ),
        false_behaviour="判断材料を並べたうえで、反対意見も歓迎する姿勢が読み取れるようにする",
        banned_phrases=["賛成してほしい", "後押し", "背中を押"],
    ),
    IntentAttribute(
        id="implies_closing_the_topic", tier="I",
        forms=[
            "この話を終わらせたい意図が読み取れるか",
            "書き手はこれ以上やり取りを続けたくないと読めるか",
            "この文面からは、議論を打ち切りたい様子がうかがえるか",
            "明言はされていないが、書き手はこの件を閉じたいと読めるか",
        ],
        negation_form="この文面はやり取りを続ける前提で書かれているか",
        true_behaviour=(
            "論点をまとめ直して結論めいた形に整え、次の問いを残さない書き方にすることで、"
            "打ち切りたい意図が行間から伝わるようにする。ただし終わりにしたいとは書かない"
        ),
        false_behaviour="論点を開いたままにし、やり取りが続く前提で書く",
        banned_phrases=["以上で", "これで終わり", "打ち切り", "議論は不要"],
    ),
    IntentAttribute(
        id="implies_doubts_feasibility", tier="I",
        forms=[
            "計画の実現性を疑っていることを、そうと書かずに示しているか",
            "書き手はうまくいかないと思っていると読めるか",
            "この文面からは、提示された計画への疑いがうかがえるか",
            "明言はされていないが、書き手は無理だと考えていると読めるか",
        ],
        negation_form="この文面からは、書き手が計画を実現できると見ていると読めるか",
        true_behaviour=(
            "必要な条件や前提を細かく列挙し、それが揃う見込みの薄さが"
            "行間から伝わるようにする。ただし無理や難しいとは書かない"
        ),
        false_behaviour="計画が実行できる見通しであることが読み取れるようにする",
        banned_phrases=["無理です", "難しいと思", "できないと思", "非現実的"],
    ),
    IntentAttribute(
        id="implies_partial_disclosure", tier="I",
        forms=[
            "事情の一部を伏せていることが読み取れるか",
            "書き手は何かを言わずにいると読めるか",
            "この文面からは、説明されていない事情があることがうかがえるか",
            "明言はされていないが、話していないことがあると読めるか",
        ],
        negation_form="この文面は事情を一通り説明しきっているか",
        true_behaviour=(
            "経緯の説明に不自然な飛びを残し、触れられていない部分があることが"
            "行間から伝わるようにする。ただし言えない事情があるとは書かない"
        ),
        false_behaviour="経緯を順を追って説明しきり、抜けがないようにする",
        banned_phrases=["詳細は控え", "事情があり", "言えない", "伏せ"],
    ),
    IntentAttribute(
        id="implies_someone_else_decides", tier="I",
        forms=[
            "自分に決定権がないことを、そうと書かずに示しているか",
            "書き手は決められる立場にないと読めるか",
            "この文面からは、別の誰かの承認が要ることがうかがえるか",
            "明言はされていないが、書き手だけでは決まらないと読めるか",
        ],
        negation_form="この文面からは、書き手自身が決められる立場だと読めるか",
        true_behaviour=(
            "持ち帰って確認する必要や、返答に時間がかかる事情を書き、"
            "決定権が別にあることが行間から伝わるようにする。ただし権限がないとは書かない"
        ),
        false_behaviour="書き手自身の判断でその場で決められることがはっきり読み取れるようにする",
        banned_phrases=["権限がありません", "決裁", "承認をとって", "上に確認"],
    ),
    IntentAttribute(
        id="implies_budget_constraint", tier="I",
        forms=[
            "予算が厳しいことを、そうと書かずに示しているか",
            "書き手は費用を強く気にしていると読めるか",
            "この文面からは、使える金額に限りがあることがうかがえるか",
            "明言はされていないが、金額が制約になっていると読めるか",
        ],
        negation_form="この文面は費用の制約に触れずに書かれているか",
        true_behaviour=(
            "必要最小限の範囲や段階的な進め方を繰り返し確認する書き方にし、"
            "金額の制約が行間から伝わるようにする。ただし予算や高いとは書かない"
        ),
        false_behaviour="費用にはこだわらず、内容そのものを中心に書く",
        banned_phrases=["予算", "高い", "値引き", "安く", "コストが"],
    ),
    IntentAttribute(
        id="implies_protecting_colleague", tier="I",
        forms=[
            "同僚をかばっていることが読み取れるか",
            "書き手は誰かの落ち度を隠していると読めるか",
            "この文面からは、特定の人を守ろうとする様子がうかがえるか",
            "明言はされていないが、書き手は誰かの責任を引き受けていると読めるか",
        ],
        negation_form="この文面は誰かをかばう様子なく書かれているか",
        true_behaviour=(
            "原因を組織や仕組みの側に寄せて書き、特定の人物への言及を避けることで、"
            "かばっていることが行間から伝わるようにする。ただしかばうとは書かない"
        ),
        false_behaviour="原因の所在を事実どおりに書き、誰かを守ろうとする様子を出さない",
        banned_phrases=["かばう", "庇", "悪くありません", "責任は私に"],
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
        id="implies_will_follow_up", tier="I",
        forms=[
            "返事がなければまた連絡するつもりを、そうと書かずに示しているか",
            "書き手は放置させない構えだと読めるか",
            "この文面からは、これで終わりにする気がないことがうかがえるか",
            "明言はされていないが、書き手は再度連絡すると読めるか",
        ],
        negation_form="この文面は一度きりの連絡として書かれているか",
        true_behaviour=(
            "次の確認の時期や自分の予定を書き添え、また連絡が来ることが"
            "行間から伝わるようにする。ただし再度連絡しますとは書かない"
        ),
        false_behaviour="今回で完結する連絡であることが読み取れるようにする",
        banned_phrases=["また連絡", "再度ご連絡", "追って連絡", "改めてご連絡"],
    ),
]


# The rule for putting a retired attribute back, from 2026-09-24 on: its false->true
# rate -- how often a document written *without* it is read as having it -- must be
# below this, measured on a re-generation of at least ~200 documents per attribute.
#
# This replaced a 30%-discard rule. Discard could not have caught what retired these
# attributes: five of the ten were already under 30% discard in the qwen3 smoke test
# that retired them, and they were retired for false->true >= 0.35. A re-test has to
# clear the bar the attribute originally failed. (docs/day3_attribute_expansion.md §9)
READMIT_MAX_FALSE_TO_TRUE = 0.35

# Retired attributes put back after a re-test showed a different generator can write
# them (night 3 §3). Changing this set is the whole of re-admitting an attribute, so
# the decision is reviewable in a diff.
READMITTED: frozenset[str] = frozenset({
    # Night 3 §3, under the rule then in force: discard < 30% when mistral-small3.2
    # wrote them and qwen3 verified (docs_v5, ~233 documents each). NOT re-judged under
    # READMIT_MAX_FALSE_TO_TRUE -- the new rule is not applied retroactively, and v0.4
    # was trained with all five. Under it only three would qualify (marked *).
    "implies_doubts_feasibility",     # discard 0.235, false->true 0.286 *
    "implies_someone_else_decides",   # discard 0.249, false->true 0.374
    "implies_budget_constraint",      # discard 0.204, false->true 0.321 *
    "implies_pressure_from_others",   # discard 0.228, false->true 0.272 *
    "implies_will_follow_up",         # discard 0.225, false->true 0.375
})

ATTRIBUTES: list[IntentAttribute] = [
    *_SURFACE, *_EXPLICIT, *_IMPLICIT,
    *_SURFACE_V2, *_EXPLICIT_V2, *_IMPLICIT_V2,
    *[a for a in _RETIRED_DEFINITIONS if a.id in READMITTED],
]
BY_ID: dict[str, IntentAttribute] = {a.id: a for a in ATTRIBUTES}

RETIRED: list[IntentAttribute] = [
    a for a in _RETIRED_DEFINITIONS if a.id not in READMITTED
]
# Every attribute that was ever retired, re-admitted or not -- for reports that
# compare the original ten across re-tests.
RETIRED_DEFINITIONS: list[IntentAttribute] = list(_RETIRED_DEFINITIONS)
ALL_ATTRIBUTES: list[IntentAttribute] = [*ATTRIBUTES, *RETIRED]
ALL_BY_ID: dict[str, IntentAttribute] = {a.id: a for a in ALL_ATTRIBUTES}


# Why each one left, from the second smoke test (qwen3 wrote and qwen3 verified): how
# often a document written *without* the attribute was read as having it, over all of
# the attribute's gold=false documents. This is the attribute's overall rate -- not the
# per-partner "absent" rate `scripts/diagnose_implications.py` reports, which is what
# showed that no single exclusion pair explained it.
RETIRED_SOLO_FALSE_TO_TRUE: dict[str, float] = {
    "implies_testing_the_waters": 0.696,
    "implies_seeking_validation": 0.667,
    "implies_closing_the_topic": 0.667,
    "implies_doubts_feasibility": 0.500,
    "implies_partial_disclosure": 0.444,
    "implies_someone_else_decides": 0.440,
    "implies_budget_constraint": 0.421,
    "implies_protecting_colleague": 0.409,
    "implies_pressure_from_others": 0.391,
    "implies_will_follow_up": 0.370,
}



# ---------------------------------------------------------------------------
# Held out from training entirely (`docs/benchmarks.md` §6)
# ---------------------------------------------------------------------------

HELD_OUT: frozenset[str] = frozenset({
    # The original five stay held out, unchanged. Rotating them would make every
    # number measured against v0.1 incomparable, and the whole point of the Day 3
    # expansion is to find out whether a wider corpus beats v0.1 on the same ruler.
    "implies_running_out_of_patience",   # I -- closest analogue to the bench boolean
    "implies_declining",                 # I -- an unstated refusal
    "implies_escalation",                # I -- an unstated next move
    "requests_owner_change",             # E -- control: does an explicit ask transfer?
    "ends_with_question",                # S -- floor: if this is 0.5 the wiring is broken
    # Seven more, so each tier pool is wide enough to read on its own.
    "implies_decision_already_made",     # I -- an unstated decision behind a question
    "implies_seeking_exception",         # I -- an unstated ask for special treatment
    "implies_relief",                    # I -- the only *positive* implicit held out
    "states_disagreement",               # E
    "requests_written_record",           # E
    "offers_help",                       # E
    "uses_bullet_points",                # S -- structural, like ends_with_question
})

TRAINABLE: list[IntentAttribute] = [a for a in ATTRIBUTES if a.id not in HELD_OUT]

# Surface attributes whose answer sits at a fixed place in the text -- the last sentence,
# the opening line -- rather than anywhere in it. They are exempt from the S gate, which
# is judged per attribute (docs/benchmarks.md §9): with `local_attention: 128`, a fixed
# location can fall outside what the option markers attend to, so their AUROC measures
# where the model looks rather than whether it reads. An attribute detectable wherever
# it occurs (bullet points, a date, a number) is not positional, whatever its shape.
POSITIONAL_ATTRIBUTES: frozenset[str] = frozenset({
    "ends_with_question",   # the final sentence
    "includes_greeting",    # the opening line
})


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
    # Day 3. Each of these is an explicit attribute paired with the implicit one it
    # would contradict: a document that says the thing outright cannot also be a
    # document that only implies it. Derived from the implication graph in
    # docs/day3_attribute_expansion.md rather than collected ad hoc.
    ("declines_explicitly", "implies_declining"),
    ("states_uncertainty", "implies_out_of_depth"),
    ("states_disagreement", "implies_reluctant_compliance"),
    ("states_agreement", "implies_declining"),
    ("offers_help", "implies_out_of_depth"),
    ("refers_to_past_exchange", "implies_prior_commitment"),
    ("mentions_deadline", "implies_deadline_is_soft"),
    ("sets_condition", "implies_seeking_exception"),
    ("states_gratitude", "implies_favour_expected"),
    ("demands_urgent_action", "implies_personal_urgency"),
    ("admits_own_fault", "implies_first_time"),
    # Found by the 200-document smoke test rather than derived: a document that refers to
    # an earlier exchange states outright that this is not the first time, so the two
    # conditions contradict. Both smoke-test failures of `implies_first_time` carried it.
    ("refers_to_past_exchange", "implies_first_time"),
    # Measured, not derived. `scripts/diagnose_implications.py` compares each tier-I
    # attribute's violation rate when a given attribute is present against when it is
    # absent, over the 1,163 tier-I observations in the second smoke corpus. Each pair
    # below violates 25-43% of the time together and **0-25% apart**, which is what
    # separates an entailment from two attributes that merely share a domain.
    ("admits_own_fault", "implies_disappointment"),            # 0.429 vs 0.000
    ("requests_owner_change", "implies_running_out_of_patience"),  # 0.333 vs 0.000
    ("states_disagreement", "implies_running_out_of_patience"),    # 0.300 vs 0.000
    ("offers_help", "implies_relief"),                         # 0.333 vs 0.000
    ("offers_help", "implies_decision_already_made"),          # 0.250 vs 0.000
    ("requests_phone_callback", "implies_deadline_is_soft"),   # 0.600 vs 0.250
    ("states_dissatisfaction", "implies_peer_comparison"),
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

# The pairs that involved a retired attribute, exactly as they stood when the attributes
# were dropped. The high-lift partners `scripts/diagnose_implications.py` found for them
# afterwards are deliberately *not* added: re-testing with a new generator and new pairs
# at once would leave no way to say which of the two made the difference.
RETIRED_EXCLUSIVE_PAIRS: list[tuple[str, str]] = [
    ("declines_explicitly", "implies_closing_the_topic"),
    ("states_uncertainty", "implies_doubts_feasibility"),
    ("states_cost_concern", "implies_budget_constraint"),
    ("states_disagreement", "implies_doubts_feasibility"),
    ("states_agreement", "implies_doubts_feasibility"),
    ("requests_meeting", "implies_closing_the_topic"),
    ("praises_specific_person", "implies_protecting_colleague"),
]

_CONFLICTS: dict[str, set[str]] = {}
for _a, _b in [*EXCLUSIVE_PAIRS, *RETIRED_EXCLUSIVE_PAIRS]:
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
