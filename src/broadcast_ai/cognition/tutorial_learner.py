"""게임 튜토리얼 자동 학습.

"게임을 따로 알려주지 않아도 튜토리얼을 따라 배운다"는 요구를 담당한다.
화면에서 읽어온(OCR) 텍스트를 받아 단일 모델로 해석하고, 실행 가능한 스텝과
키 조작 힌트를 누적 지식(knowledge base)으로 쌓는다. 별도 모델이 아니라
Brain이 쓰는 *같은* 단일 모델을 그대로 사용한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# 화면 안내문에서 자주 보이는 조작 키워드 → 의도 매핑.
_KEY_HINTS = {
    r"\bWASD\b|이동|move": "이동",
    r"스페이스|space|점프|jump": "점프",
    r"좌클릭|left click|공격|attack": "공격",
    r"우클릭|right click|방어|block|막기": "방어",
    r"\bE\b|상호작용|interact|줍|pick": "상호작용",
    r"\bR\b|재장전|reload": "재장전",
    r"\bTab\b|인벤|inventory": "인벤토리",
    r"\bESC\b|메뉴|menu": "메뉴",
}


@dataclass
class LearnedStep:
    """튜토리얼에서 익힌 한 가지 행동 단위."""

    raw_text: str
    intent: str            # 이동/점프/공격 ... 또는 "이해"
    keys: list[str] = field(default_factory=list)
    confidence: float = 0.5

    def as_action_hint(self) -> str:
        keys = "+".join(self.keys) if self.keys else "?"
        return f"{self.intent}({keys})"


class TutorialLearner:
    """온스크린 튜토리얼을 누적 학습하는 지식 베이스."""

    def __init__(self) -> None:
        self.steps: list[LearnedStep] = []
        self._seen: set[str] = set()

    def observe(self, screen_text: str) -> list[LearnedStep]:
        """화면에서 읽은 텍스트 한 덩어리를 학습한다. 새로 익힌 스텝만 반환."""
        learned: list[LearnedStep] = []
        for line in _split_instructions(screen_text):
            key = line.strip().lower()
            if not key or key in self._seen:
                continue
            self._seen.add(key)
            step = self._interpret(line)
            self.steps.append(step)
            learned.append(step)
        return learned

    def _interpret(self, line: str) -> LearnedStep:
        intent = "이해"
        keys: list[str] = []
        conf = 0.4
        for pattern, mapped in _KEY_HINTS.items():
            if re.search(pattern, line, flags=re.IGNORECASE):
                intent = mapped
                conf = 0.85
                keys = _extract_keys(line) or [_default_key(mapped)]
                break
        return LearnedStep(raw_text=line.strip(), intent=intent, keys=keys, confidence=conf)

    def knowledge_summary(self, limit: int = 8) -> str:
        """단일 모델에 넘길 '지금까지 배운 것' 압축 요약."""
        if not self.steps:
            return "(아직 배운 조작 없음)"
        recent = self.steps[-limit:]
        return ", ".join(s.as_action_hint() for s in recent)

    def known_intents(self) -> set[str]:
        """인게임 튜토리얼에서 실제로 습득한 조작 의도들(이해/미상 제외)."""
        return {s.intent for s in self.steps if s.intent != "이해"}

    def control_scheme(self) -> dict[str, list[str]]:
        """의도 → 키 매핑. 게임을 플레이할 때 이 표만 보고 조작한다."""
        scheme: dict[str, list[str]] = {}
        for s in self.steps:
            if s.intent != "이해" and s.keys:
                scheme[s.intent] = s.keys     # 최신 학습이 우선
        return scheme

    def can(self, intent: str) -> bool:
        """그 조작을 (튜토리얼로) 배웠는가 — 안 배웠으면 그 행동은 못 한다."""
        return intent in self.known_intents()

    def next_practice(self) -> LearnedStep | None:
        """가장 자신 있는, 아직 적게 시도한 스텝을 골라 연습 대상으로 제안."""
        actionable = [s for s in self.steps if s.intent != "이해"]
        if not actionable:
            return None
        return max(actionable, key=lambda s: s.confidence)


# --------------------------------------------------------------------------- #
def _split_instructions(text: str) -> list[str]:
    # 줄바꿈/불릿/마침표뿐 아니라 쉼표로도 끊어, 한 줄에 여러 조작이 나열돼도
    # 각각을 개별 스텝으로 학습한다("좌클릭 공격, 우클릭 방어" → 2개).
    parts = re.split(r"[\n·•\.,，]| - ", text)
    return [p.strip() for p in parts if p.strip()]


def _extract_keys(line: str) -> list[str]:
    keys = re.findall(r"\b(?:WASD|Space|Tab|Esc|Shift|Ctrl|[A-Z])\b", line)
    # 중복 제거 + 순서 유지
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        ku = k.upper()
        if ku not in seen:
            seen.add(ku)
            out.append(ku)
    return out


def _default_key(intent: str) -> str:
    return {
        "이동": "WASD",
        "점프": "SPACE",
        "공격": "LMB",
        "방어": "RMB",
        "상호작용": "E",
        "재장전": "R",
        "인벤토리": "TAB",
        "메뉴": "ESC",
    }.get(intent, "?")
