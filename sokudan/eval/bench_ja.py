"""`bench_ja` — the Japanese evaluation set (SOKUDAN_SPEC.md §4.2).

Built by **label-conditioned generation**: the gold labels are chosen first, then a
local LLM is asked to write a Japanese business message that matches them. The
generation condition *is* the ground truth, so no annotation and no teacher API.

Two things keep this honest:

1. The generator is told not to use the option labels verbatim, so the benchmark
   does not degenerate into keyword matching.
2. `bench_ja` is a **held-out test set**. It must never enter training (§4.2).

The schema defined here is deliberately worded differently from the synthetic
*training* schemas built in §7.4(b), so that zero-shot generalisation to unseen
schemas can be measured honestly.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any

# --------------------------------------------------------------------------
# Label spaces. These three questions are attached to every item (§4.2).
# --------------------------------------------------------------------------

DEPARTMENTS: dict[str, str] = {
    "請求": "支払い・返金・請求書・料金の二重引き落としなど金銭処理に関するもの",
    "技術": "不具合・障害・エラー・ログイン不能など製品が動かないことに関するもの",
    "営業": "料金プラン・新規契約・見積もり・増席など購入判断に関するもの",
    "その他": "上のどれにも当てはまらない一般的な連絡",
}

URGENCY_LEVELS: list[str] = ["急がない", "早めに", "業務が止まっている"]

# Non-uniform on purpose: a real support inbox is skewed, and a uniform set would
# make the majority-class baseline meaningless. Exact realised counts are recorded
# in the manifest, not assumed from these weights.
DEPARTMENT_WEIGHTS = {"請求": 0.35, "技術": 0.30, "営業": 0.20, "その他": 0.15}
URGENCY_WEIGHTS = {"急がない": 0.30, "早めに": 0.45, "業務が止まっている": 0.25}
CHURN_TRUE_RATE = 0.30

# --------------------------------------------------------------------------
# Randomisation axes, so 300 items do not read like 300 copies of one email.
# --------------------------------------------------------------------------

INDUSTRIES = [
    "食品卸", "建設", "学習塾", "人材派遣", "医療機器商社", "地方銀行", "アパレルEC",
    "物流倉庫", "自治体", "ソフトウェア受託", "旅行代理店", "印刷", "不動産管理",
    "農協", "歯科医院", "リサイクル", "介護施設", "水産加工", "広告代理店", "税理士法人",
]

ROLES = [
    "経理担当", "情報システム部の担当者", "店長", "代表取締役", "総務の派遣社員",
    "現場のアルバイト", "部長", "個人事業主", "購買担当", "事務員", "支店長", "新入社員",
]

STYLES = [
    "丁寧だが事務的な敬体",
    "やや苛立ちのにじむ敬体",
    "非常に丁寧で回りくどい敬体",
    "短く素っ気ない常体",
    "口語的で崩れた文体（誤字や半角カナが混じる）",
    "箇条書き中心の簡潔な文体",
    "感情的で句読点が多い文体",
]

LENGTHS = [
    ("短文", "80〜150文字程度、1段落"),
    ("中", "200〜350文字程度、2段落"),
    ("長文", "400〜600文字程度、3段落以上"),
]

CHANNELS = ["メール", "問い合わせフォームの自由記述", "サポートチャットの書き込み"]


# --------------------------------------------------------------------------
# The question schema attached to every bench_ja item.
# --------------------------------------------------------------------------

def bench_questions() -> dict[str, dict[str, Any]]:
    """The three bench_ja questions, in the shape both Laya and sokudan accept."""
    return {
        "department": {
            "type": "choice",
            "instructions": "この問い合わせはどの部署が担当すべきか",
            "criteria": dict(DEPARTMENTS),
        },
        "urgency": {
            "type": "score",
            "instructions": "この依頼の緊急度は",
            "criteria": list(URGENCY_LEVELS),
        },
        "churn": {
            "type": "noul",
            "instructions": "送信者は解約・契約終了を示唆しているか",
        },
    }


@dataclass
class BenchItem:
    """One evaluation item: the generated text plus the labels it was generated from."""

    item_id: str
    state: str
    department: str          # gold choice label (key of DEPARTMENTS)
    urgency: int             # gold ordinal index into URGENCY_LEVELS
    churn: bool              # gold boolean
    # Provenance: what the generator was conditioned on, for auditing the set.
    industry: str
    role: str
    style: str
    length: str
    channel: str
    generator_model: str
    generator_temperature: float

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


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
    """Sample gold labels and the surface-variation axes independently."""
    dept = rng.choices(list(DEPARTMENT_WEIGHTS), weights=list(DEPARTMENT_WEIGHTS.values()))[0]
    urg_name = rng.choices(list(URGENCY_WEIGHTS), weights=list(URGENCY_WEIGHTS.values()))[0]
    length_name, length_hint = rng.choice(LENGTHS)
    return LabelDraw(
        department=dept,
        urgency=URGENCY_LEVELS.index(urg_name),
        churn=rng.random() < CHURN_TRUE_RATE,
        industry=rng.choice(INDUSTRIES),
        role=rng.choice(ROLES),
        style=rng.choice(STYLES),
        length_name=length_name,
        length_hint=length_hint,
        channel=rng.choice(CHANNELS),
    )


# --------------------------------------------------------------------------
# Generation prompt.
# --------------------------------------------------------------------------

_URGENCY_BEHAVIOUR = {
    0: "締め切りに触れず、「お手すきの際に」「急ぎではありませんが」といった含みで、"
       "業務は問題なく回っていることが読み取れるように書く",
    1: "今週中・数日中といった緩い期限があり、放置すると困るが今はまだ回っている様子を書く",
    2: "既に業務が止まっている・顧客に影響が出ていることが具体的な状況描写から読み取れるように書く",
}

_CHURN_BEHAVIOUR = {
    True: "他社への乗り換えや契約の見直し・打ち切りを検討していることが、"
          "文面から読み取れるように匂わせる（ただし「解約」という単語は使わない）",
    False: "今後も使い続ける前提で書く。乗り換えや契約終了の話は一切出さない",
}

SYSTEM_PROMPT = (
    "あなたは日本語の業務文面を書くデータ生成器です。"
    "指示された条件を正確に満たす文面だけを出力します。"
    "解説・前置き・後書き・マークダウンの装飾は一切出力しません。"
)


def build_generation_prompt(draw: LabelDraw) -> str:
    """The user-side prompt. Labels go in as *behaviour*, never as vocabulary."""
    forbidden = "、".join(f"「{w}」" for w in [*DEPARTMENTS, *URGENCY_LEVELS, "解約", "緊急"])
    return f"""日本語の「{draw.channel}」の本文を1件だけ書いてください。

## 書き手の設定
- 業種: {draw.industry}
- 立場: {draw.role}
- 文体: {draw.style}
- 分量: {draw.length_name}（{draw.length_hint}）

## 内容の条件（必ず満たすこと）
1. 用件の内容は次のものにしてください: {DEPARTMENTS[draw.department]}
2. 緊急度の表現: {_URGENCY_BEHAVIOUR[draw.urgency]}
3. 契約継続の姿勢: {_CHURN_BEHAVIOUR[draw.churn]}

## 禁止事項（重要）
- 次の語を本文中で使わないでください: {forbidden}
  分類ラベルそのものを書かず、**状況の描写だけ**で用件が分かるようにしてください。
- 「件名:」「本文:」のような見出しを付けないでください。
- 宛名と署名は入れても入れなくても構いませんが、会社名・氏名は架空のものにしてください。

本文だけを出力してください。"""
