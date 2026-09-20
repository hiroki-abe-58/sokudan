"""Label-conditioned synthetic training data (SOKUDAN_SPEC.md §7.4b).

Same trick as `bench_ja`: choose the gold labels first, then have a local LLM write a
Japanese document that matches them. The generation condition *is* the ground truth,
so there is no annotation step and no teacher API.

Three constraints shape this catalogue, and all three came from measurements or
decisions made earlier today rather than from taste:

**Nothing here may overlap `bench_ja`.** `bench_ja` is the held-out test set and its
whole purpose is to measure generalisation to schemas never seen in training. So no
department routing over 請求/技術/営業/その他, no three-level urgency over
急がない/早めに/業務が止まっている, no churn-suggestion boolean -- not the schemas, not
their option strings, not close paraphrases. Where a domain is conceptually adjacent
(an ordinal severity, say) the level count and the wording are both different.

**`score` levels span K = 2 to 7.** The dynamic-K cumulative link (§6.3) is the piece
`docs/baseline_ja.md` identified as the real gap, and a head that only ever saw K=3
would not demonstrate anything about K.

**Label distributions are deliberately skewed, not uniform.** The measured failure in
`docs/baseline_ja.md` §6.3 was a model whose mean P(true) sat at 0.434 against a gold
rate of 0.297 -- it leaned to one side regardless of the text. Training only on
balanced labels is how a model acquires that habit without anyone noticing.

Generation cost is why documents carry several attributes: the local endpoint
saturates near 3 documents/second, so 24k separately-generated examples would take
over two hours. One document labelled on three attributes, each expanded into a few
schema variants (§7.2), turns a few thousand generations into tens of thousands of
training examples -- which is what §7.2 recommends doing anyway.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from sokudan.schema.question import BoolQuestion, ChoiceQuestion, Question, ScoreQuestion


@dataclass(frozen=True)
class Attribute:
    """One labelled question attached to a generated document."""

    name: str
    kind: str                      # "choice" | "score" | "bool"
    instructions: str
    labels: list[str]              # for bool: [false_label, true_label]
    behaviour: list[str]           # how the writer should express each label
    weights: list[float]           # sampling weights -- deliberately uneven
    descriptions: dict[str, str] = field(default_factory=dict)   # choice only
    surface_forms: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.behaviour) or len(self.labels) != len(self.weights):
            raise ValueError(f"{self.name}: labels, behaviour and weights must align")
        if self.kind == "bool" and len(self.labels) != 2:
            raise ValueError(f"{self.name}: a bool attribute needs exactly 2 labels")
        if self.kind == "score" and len(self.labels) < 2:
            raise ValueError(f"{self.name}: a score attribute needs at least 2 levels")

    def sample(self, rng: random.Random) -> int:
        return rng.choices(range(len(self.labels)), weights=self.weights)[0]

    def to_question(self) -> Question:
        if self.kind == "choice":
            return ChoiceQuestion(
                instructions=self.instructions,
                criteria={label: self.descriptions.get(label, "") for label in self.labels},
            )
        if self.kind == "score":
            return ScoreQuestion(instructions=self.instructions, criteria=list(self.labels))
        if self.kind == "bool":
            return BoolQuestion(
                instructions=self.instructions,
                false_label=self.labels[0],
                true_label=self.labels[1],
            )
        raise ValueError(f"{self.name}: unknown attribute kind {self.kind!r}")


@dataclass(frozen=True)
class Domain:
    """A document type plus the attributes a generated instance is labelled on."""

    name: str
    document: str                  # what the writer is asked to produce
    contexts: list[str]            # randomisation axis: who is writing, about what
    attributes: list[Attribute]
    banned_words: list[str] = field(default_factory=list)


# Shared across domains so a document does not read like the last one.
STYLES = [
    "丁寧な敬体", "事務的で短い常体", "やや感情的な敬体", "箇条書き中心",
    "口語的で崩れた文体", "非常に формально… ではなく、硬い書き言葉", "端的で無駄のない文体",
]
LENGTHS = [("短", "100〜180文字"), ("中", "200〜320文字"), ("長", "350〜550文字")]


CATALOG: list[Domain] = [
    Domain(
        name="review_request",
        document="社内レビュー依頼のメッセージ",
        contexts=["ソフトウェア開発", "出版", "製薬", "広告制作", "建築設計"],
        attributes=[
            Attribute(
                name="review_kind", kind="choice",
                instructions="このレビュー依頼が求めているものは",
                labels=["仕様", "実装", "文言", "法務"],
                descriptions={
                    "仕様": "要件や設計の妥当性の確認",
                    "実装": "成果物の中身そのものの確認",
                    "文言": "表記や言い回しの確認",
                    "法務": "契約・権利・規制の観点の確認",
                },
                behaviour=[
                    "要件や設計方針が妥当か見てほしい、と読める内容にする",
                    "できあがった中身そのものを見てほしい、と読める内容にする",
                    "表記ゆれや言い回しを整えてほしい、と読める内容にする",
                    "権利関係や規制に触れないか見てほしい、と読める内容にする",
                ],
                weights=[0.3, 0.35, 0.2, 0.15],
                surface_forms={
                    "仕様": ["要件", "設計", "仕様確認"],
                    "実装": ["成果物", "中身", "実装内容"],
                    "文言": ["表記", "コピー", "文面"],
                    "法務": ["リーガル", "法務確認", "権利関係"],
                },
            ),
            Attribute(
                name="blocks_others", kind="bool",
                instructions="この依頼が滞ると他の人の作業が止まるか",
                labels=["止まらない", "止まる"],
                behaviour=[
                    "自分の作業だけの話で、他の人は待っていないことが読み取れるようにする",
                    "後工程の人が待っている状況が具体的に読み取れるようにする",
                ],
                weights=[0.62, 0.38],
            ),
        ],
    ),
    Domain(
        name="job_application",
        document="求人への応募メール",
        contexts=["製造業", "IT", "小売", "物流", "介護", "飲食"],
        attributes=[
            Attribute(
                name="role_family", kind="choice",
                instructions="応募者が希望している職種の系統は",
                labels=["技術職", "デザイン職", "事務職", "現場職"],
                descriptions={
                    "技術職": "設計・開発・保守などの専門技術",
                    "デザイン職": "意匠・表現・UIなどの設計",
                    "事務職": "書類・調整・経理などの内勤",
                    "現場職": "製造・配送・接客などの実作業",
                },
                behaviour=[
                    "設計や開発の経験を中心に書く", "意匠や表現の経験を中心に書く",
                    "書類作成や調整の経験を中心に書く", "現場での実作業の経験を中心に書く",
                ],
                weights=[0.3, 0.15, 0.3, 0.25],
                surface_forms={
                    "技術職": ["エンジニア", "技術系", "開発職"],
                    "デザイン職": ["デザイナー", "意匠職"],
                    "事務職": ["内勤", "バックオフィス", "管理部門"],
                    "現場職": ["フィールド職", "実務職"],
                },
            ),
            Attribute(
                name="experience_fit", kind="score",
                instructions="募集職種に対する経験の充足度は",
                labels=["未経験", "基礎のみ", "実務経験あり", "即戦力"],
                behaviour=[
                    "その分野の経験がまったくないことが読み取れるようにする",
                    "学習や研修どまりで実務はないことが読み取れるようにする",
                    "数年の実務経験があることが読み取れるようにする",
                    "同職種で長く主導的に働いてきたことが読み取れるようにする",
                ],
                weights=[0.3, 0.3, 0.25, 0.15],
                surface_forms={
                    "未経験": ["経験なし", "これから"],
                    "即戦力": ["すぐ活躍できる", "十分な経験"],
                },
            ),
        ],
    ),
    Domain(
        name="product_review",
        document="通販サイトの商品レビュー",
        contexts=["家電", "食品", "衣類", "書籍", "工具", "化粧品"],
        attributes=[
            Attribute(
                name="star", kind="score",
                instructions="このレビューの星の数は",
                labels=["星1", "星2", "星3", "星4", "星5"],
                behaviour=[
                    "強い不満を具体的に書く", "不満が上回るが一部は認める書き方にする",
                    "良い点と悪い点を同程度に書く", "おおむね満足だが小さな不満も書く",
                    "はっきり満足していることを書く",
                ],
                # J-shaped, as real review distributions are.
                weights=[0.18, 0.09, 0.13, 0.22, 0.38],
                surface_forms={"星1": ["1", "★1"], "星5": ["5", "★5"]},
            ),
            Attribute(
                name="mentions_shipping", kind="bool",
                instructions="配送や梱包について触れているか",
                labels=["触れていない", "触れている"],
                behaviour=[
                    "商品自体の話だけを書き、配送や梱包には一切触れない",
                    "配送の速さか梱包の状態に必ず触れる",
                ],
                weights=[0.55, 0.45],
            ),
        ],
    ),
    Domain(
        name="moderation",
        document="コミュニティサイトへの投稿",
        contexts=["ゲーム掲示板", "料理レシピ", "地域情報", "趣味サークル", "求職掲示板"],
        attributes=[
            Attribute(
                name="violation", kind="choice",
                instructions="この投稿に含まれる問題は",
                labels=["問題なし", "誹謗中傷", "宣伝", "個人情報"],
                descriptions={
                    "問題なし": "ガイドライン上の問題は見当たらない",
                    "誹謗中傷": "特定の相手を貶める表現",
                    "宣伝": "商品やサービスへの誘導",
                    "個人情報": "第三者の個人情報の掲載",
                },
                behaviour=[
                    "ごく普通の話題の投稿にする",
                    "特定の人物を強く貶す表現を含める（過度に攻撃的にはしない）",
                    "商品やサイトへの露骨な誘導を含める",
                    "第三者の連絡先や住所らしき記述を含める（架空のもの）",
                ],
                weights=[0.55, 0.16, 0.19, 0.10],
                surface_forms={
                    "問題なし": ["違反なし", "正常"],
                    "誹謗中傷": ["中傷", "攻撃的表現"],
                    "宣伝": ["スパム", "商業的誘導"],
                },
            ),
            Attribute(
                name="severity", kind="score",
                instructions="対応の必要度は",
                labels=["様子見", "注意喚起", "即時対応"],
                behaviour=[
                    "放置しても実害がなさそうな書き方にする",
                    "一声かけるべき程度の書き方にする",
                    "すぐ消すべき深刻さが読み取れる書き方にする",
                ],
                weights=[0.5, 0.32, 0.18],
            ),
        ],
    ),
    Domain(
        name="meeting_minutes",
        document="社内打ち合わせの議事メモ",
        contexts=["営業定例", "開発進捗", "採用会議", "予算会議", "品質会議"],
        attributes=[
            Attribute(
                name="has_decision", kind="bool",
                instructions="決定事項が含まれているか",
                labels=["含まれていない", "含まれている"],
                behaviour=[
                    "議論や共有だけで、何も決まらなかったことが読み取れるようにする",
                    "はっきり決まったことが読み取れるようにする",
                ],
                weights=[0.42, 0.58],
            ),
            Attribute(
                name="owner_named", kind="bool",
                instructions="次の作業の担当者が明示されているか",
                labels=["明示されていない", "明示されている"],
                behaviour=[
                    "誰がやるかを書かずに終える", "担当者名を必ず書く（架空の氏名）",
                ],
                weights=[0.48, 0.52],
            ),
        ],
    ),
    Domain(
        name="phishing",
        document="受信した不審かもしれないメール",
        contexts=["銀行を名乗る", "宅配業者を名乗る", "社内情シスを名乗る",
                  "取引先を名乗る", "サブスク事業者を名乗る"],
        attributes=[
            Attribute(
                name="is_phishing", kind="bool",
                instructions="このメールは詐欺の疑いがあるか",
                labels=["疑いなし", "疑いあり"],
                behaviour=[
                    "正規の事務連絡として自然な内容にする。不自然な誘導は入れない",
                    "偽サイトらしき誘導や不自然な急かしを含める（実在URLは書かない）",
                ],
                weights=[0.6, 0.4],
            ),
            Attribute(
                name="risk", kind="score",
                instructions="利用者にとっての危険度を7段階で表すと",
                labels=["1", "2", "3", "4", "5", "6", "7"],
                behaviour=[
                    "まったく害のない連絡にする", "ごくわずかに不自然な点だけ残す",
                    "軽い違和感がある程度にする", "判断に迷う程度の不審さにする",
                    "はっきり不審な点を複数入れる", "明らかな詐欺の特徴を多く入れる",
                    "資格情報を直接要求する露骨な内容にする",
                ],
                weights=[0.2, 0.14, 0.14, 0.14, 0.14, 0.12, 0.12],
            ),
        ],
    ),
    Domain(
        name="internal_request",
        document="社内申請の申請理由欄",
        contexts=["備品購入", "出張", "研修受講", "残業", "在宅勤務"],
        attributes=[
            Attribute(
                name="approval_route", kind="choice",
                instructions="この申請の承認先は",
                labels=["直属上長", "部門長", "経理", "人事", "情報システム", "役員"],
                descriptions={
                    "直属上長": "日常的な範囲の申請",
                    "部門長": "部門全体に関わる申請",
                    "経理": "金銭の処理が伴う申請",
                    "人事": "勤務条件や在籍に関わる申請",
                    "情報システム": "端末や権限に関わる申請",
                    "役員": "全社的な判断を要する申請",
                },
                behaviour=[
                    "日常業務の範囲に収まる内容にする", "部門全体に影響する内容にする",
                    "金銭処理が中心の内容にする", "勤務条件に関わる内容にする",
                    "端末や権限に関わる内容にする", "全社判断が要る規模の内容にする",
                ],
                weights=[0.3, 0.18, 0.2, 0.14, 0.12, 0.06],
            ),
            Attribute(
                name="complete", kind="score",
                instructions="申請内容の記載は十分か",
                labels=["不備あり", "不備なし"],
                behaviour=[
                    "金額や日付など必要な情報が抜けている状態にする",
                    "必要な情報が過不足なく書かれている状態にする",
                ],
                weights=[0.45, 0.55],
            ),
        ],
    ),
    Domain(
        name="survey_freetext",
        document="顧客満足度アンケートの自由記述",
        contexts=["宿泊施設", "通信回線", "保険", "フィットネス", "学習サービス"],
        attributes=[
            Attribute(
                name="recommendation", kind="score",
                instructions="この回答者の推奨度合いは",
                labels=["批判的", "中立", "推奨的"],
                behaviour=[
                    "他人には勧めないという姿勢が読み取れるようにする",
                    "可もなく不可もない姿勢が読み取れるようにする",
                    "他人に勧めたい姿勢が読み取れるようにする",
                ],
                weights=[0.3, 0.38, 0.32],
                surface_forms={"批判的": ["否定的"], "推奨的": ["肯定的", "好意的"]},
            ),
            Attribute(
                name="topic", kind="choice",
                instructions="主に言及している側面は",
                labels=["価格", "品質", "対応", "手続き", "設備"],
                descriptions={
                    "価格": "料金や費用対効果", "品質": "提供物そのものの良し悪し",
                    "対応": "人の応対", "手続き": "申込や解約などの手順",
                    "設備": "施設や機器の状態",
                },
                behaviour=[
                    "料金の話を中心にする", "提供物の質の話を中心にする",
                    "スタッフの応対の話を中心にする", "手続きの煩雑さの話を中心にする",
                    "設備や機器の話を中心にする",
                ],
                weights=[0.26, 0.24, 0.22, 0.16, 0.12],
            ),
        ],
    ),
    Domain(
        name="incident_report",
        document="システム障害の一次報告",
        contexts=["社内基幹システム", "ECサイト", "社内ネットワーク", "決済基盤", "社内メール"],
        attributes=[
            Attribute(
                name="impact_scope", kind="score",
                instructions="利用者への影響の広さは",
                labels=["影響なし", "一部の利用者", "広範囲", "全面停止"],
                behaviour=[
                    "内部で検知しただけで利用者への影響がないことを書く",
                    "特定条件の利用者だけが影響を受けたことを書く",
                    "多くの利用者が影響を受けたことを書く",
                    "サービス全体が止まったことを書く",
                ],
                weights=[0.22, 0.4, 0.26, 0.12],
            ),
            Attribute(
                name="cause_identified", kind="bool",
                instructions="原因が特定できているか",
                labels=["未特定", "特定済み"],
                behaviour=[
                    "調査中で原因がまだ分からないことを書く",
                    "原因がはっきり分かったことを書く",
                ],
                weights=[0.55, 0.45],
            ),
        ],
    ),
    Domain(
        name="contract_clause",
        document="契約書の一条項とそれに対する社内コメント",
        contexts=["業務委託", "賃貸借", "秘密保持", "販売代理", "共同研究"],
        attributes=[
            Attribute(
                name="clause_type", kind="choice",
                instructions="この条項が定めている事柄は",
                labels=["報酬", "期間", "秘密保持", "解除", "責任範囲", "知的財産"],
                descriptions={
                    "報酬": "対価や支払条件", "期間": "契約の始期・終期・更新",
                    "秘密保持": "情報の取り扱い", "解除": "契約を終わらせる条件",
                    "責任範囲": "損害の負担", "知的財産": "成果物の権利帰属",
                },
                behaviour=[
                    "対価や支払条件を定める条項にする", "契約期間や更新を定める条項にする",
                    "情報の取り扱いを定める条項にする", "契約終了の条件を定める条項にする",
                    "損害の負担を定める条項にする", "成果物の権利帰属を定める条項にする",
                ],
                weights=[0.22, 0.16, 0.16, 0.16, 0.16, 0.14],
            ),
            Attribute(
                name="risk_level", kind="score",
                instructions="自社にとっての不利さの程度は",
                labels=["軽微", "注意", "重大"],
                behaviour=[
                    "一般的でほぼ問題のない書き方にする",
                    "交渉の余地を残したいと感じる書き方にする",
                    "明らかに自社に不利な書き方にする",
                ],
                weights=[0.45, 0.35, 0.2],
            ),
        ],
    ),
    Domain(
        name="social_post",
        document="企業アカウント宛てのSNS投稿",
        contexts=["飲食チェーン", "鉄道", "ゲーム会社", "化粧品ブランド", "家電メーカー"],
        attributes=[
            Attribute(
                name="sentiment", kind="score",
                instructions="この投稿の感情の向きは",
                labels=["とても否定的", "やや否定的", "中立", "やや肯定的", "とても肯定的"],
                behaviour=[
                    "強い怒りや失望を書く", "軽い不満を書く", "事実の報告だけを書く",
                    "軽い好意を書く", "強い賞賛を書く",
                ],
                weights=[0.16, 0.22, 0.24, 0.22, 0.16],
            ),
            Attribute(
                name="is_ironic", kind="bool",
                instructions="皮肉や反語が使われているか",
                labels=["使われていない", "使われている"],
                behaviour=[
                    "額面どおりの素直な表現だけを使う",
                    "言葉と本心が逆になる皮肉を含める",
                ],
                weights=[0.7, 0.3],
            ),
        ],
    ),
    Domain(
        name="expense_claim",
        document="経費精算の申請メモ",
        contexts=["営業活動", "研修", "出張", "備品", "接待"],
        attributes=[
            Attribute(
                name="expense_category", kind="choice",
                instructions="この支出の費目は",
                labels=["交通費", "宿泊費", "会議費", "消耗品費", "通信費", "研修費"],
                descriptions={
                    "交通費": "移動にかかった費用", "宿泊費": "宿泊にかかった費用",
                    "会議費": "打ち合わせや会食の費用", "消耗品費": "文具や備品の購入",
                    "通信費": "回線や通話の費用", "研修費": "受講や教材の費用",
                },
                behaviour=[
                    "移動の話を中心にする", "宿泊の話を中心にする",
                    "打ち合わせの飲食の話を中心にする", "文具や備品購入の話を中心にする",
                    "回線や通話の話を中心にする", "受講や教材の話を中心にする",
                ],
                weights=[0.3, 0.14, 0.18, 0.18, 0.1, 0.1],
            ),
            Attribute(
                name="policy_violation", kind="bool",
                instructions="社内規程に反する可能性があるか",
                labels=["問題なし", "疑いあり"],
                behaviour=[
                    "金額も用途も常識的な範囲に収める",
                    "上限超過や私的利用を疑わせる記述を含める",
                ],
                weights=[0.72, 0.28],
            ),
        ],
    ),
    Domain(
        name="training_feedback",
        document="社内研修の受講後アンケート",
        contexts=["新人研修", "安全講習", "語学研修", "管理職研修", "システム操作研修"],
        attributes=[
            Attribute(
                name="difficulty", kind="score",
                instructions="受講者が感じた難易度を6段階で表すと",
                labels=["1", "2", "3", "4", "5", "6"],
                behaviour=[
                    "簡単すぎて退屈だったことを書く", "やや物足りなかったことを書く",
                    "ちょうどよかったことを書く", "やや難しかったことを書く",
                    "かなり難しかったことを書く", "まったくついていけなかったことを書く",
                ],
                weights=[0.12, 0.16, 0.26, 0.22, 0.14, 0.10],
            ),
            Attribute(
                name="would_attend_again", kind="bool",
                instructions="また受講したいと考えているか",
                labels=["考えていない", "考えている"],
                behaviour=[
                    "もう受けたくない姿勢を書く", "また受けたい姿勢を書く",
                ],
                weights=[0.38, 0.62],
            ),
        ],
    ),
    Domain(
        name="supplier_message",
        document="取引先からの連絡",
        contexts=["部品供給", "印刷", "清掃", "警備", "システム保守"],
        attributes=[
            Attribute(
                name="message_intent", kind="choice",
                instructions="この連絡の主な用件は",
                labels=["納期連絡", "仕様確認", "値上げ通知", "担当者変更", "謝罪"],
                descriptions={
                    "納期連絡": "いつ届くかの連絡", "仕様確認": "内容の確認",
                    "値上げ通知": "価格改定の知らせ", "担当者変更": "窓口が変わる知らせ",
                    "謝罪": "不手際のお詫び",
                },
                behaviour=[
                    "いつ納品できるかを中心に書く", "内容の確認を中心に書く",
                    "価格改定の知らせを中心に書く", "窓口変更の知らせを中心に書く",
                    "不手際のお詫びを中心に書く",
                ],
                weights=[0.3, 0.22, 0.16, 0.16, 0.16],
            ),
            Attribute(
                name="needs_reply", kind="bool",
                instructions="こちらからの返信が必要か",
                labels=["不要", "必要"],
                behaviour=[
                    "報告だけで返事を求めない書き方にする",
                    "確認や返答を明確に求める書き方にする",
                ],
                weights=[0.4, 0.6],
            ),
        ],
    ),
]


def catalog_summary() -> dict[str, object]:
    """Facts about the catalogue, for the dataset manifest. Counted, never asserted."""
    score_ks = sorted({
        len(a.labels) for d in CATALOG for a in d.attributes if a.kind == "score"
    })
    kinds: dict[str, int] = {}
    for domain in CATALOG:
        for attribute in domain.attributes:
            kinds[attribute.kind] = kinds.get(attribute.kind, 0) + 1
    return {
        "n_domains": len(CATALOG),
        "n_attributes": sum(len(d.attributes) for d in CATALOG),
        "attributes_by_kind": kinds,
        "score_level_counts": score_ks,
        "choice_option_counts": sorted({
            len(a.labels) for d in CATALOG for a in d.attributes if a.kind == "choice"
        }),
    }


SYSTEM_PROMPT = (
    "あなたは日本語の業務文面を書くデータ生成器です。"
    "指示された条件を正確に満たす文面だけを出力します。"
    "解説・前置き・後書き・マークダウンの装飾は一切出力しません。"
)


def build_prompt(
    domain: Domain, chosen: dict[str, int], rng: random.Random
) -> tuple[str, dict[str, str]]:
    """Prompt for one document, conditioned on every attribute's gold label."""
    style = rng.choice(STYLES)
    length_name, length_hint = rng.choice(LENGTHS)
    context = rng.choice(domain.contexts)

    conditions = []
    for index, attribute in enumerate(domain.attributes, start=1):
        conditions.append(f"{index}. {attribute.behaviour[chosen[attribute.name]]}")

    banned = set(domain.banned_words)
    for attribute in domain.attributes:
        banned.update(attribute.labels)
        for forms in attribute.surface_forms.values():
            banned.update(forms)
    # Single characters and digits cannot be banned without making the task absurd.
    banned = {w for w in banned if len(w) >= 2 and not w.isdigit()}
    banned_text = "、".join(f"「{w}」" for w in sorted(banned))

    prompt = f"""日本語で「{domain.document}」を1件だけ書いてください。

## 設定
- 分野・状況: {context}
- 文体: {style}
- 分量: {length_name}（{length_hint}）

## 内容の条件（すべて満たすこと）
{chr(10).join(conditions)}

## 禁止事項（重要）
- 次の語を本文中で使わないでください: {banned_text}
  分類ラベルそのものを書かず、**状況の描写だけ**で内容が分かるようにしてください。
- 見出しや「本文:」のような前置きを付けないでください。
- 会社名・人名はすべて架空のものにしてください。

本文だけを出力してください。"""

    meta = {"context": context, "style": style, "length": length_name}
    return prompt, meta


def banned_words_for(domain: Domain) -> list[str]:
    """The words a generated document must not contain, for validation."""
    banned = set(domain.banned_words)
    for attribute in domain.attributes:
        banned.update(attribute.labels)
        for forms in attribute.surface_forms.values():
            banned.update(forms)
    return sorted(w for w in banned if len(w) >= 2 and not w.isdigit())
