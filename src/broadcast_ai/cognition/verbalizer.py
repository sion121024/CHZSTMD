"""발화 렌더러.

단일 모델이 *결정한* 발화 의도(감정·에너지·말할지 여부)를 받아 한국어 구어체
문장으로 표면화한다. '무엇을/어떤 톤으로 말할지'의 판단은 신경망(Intent 헤드)이
하고, 이 모듈은 그 의도를 자연스러운 문장으로 옮기는 렌더러일 뿐이다
(음성 합성에서 보코더가 하는 역할과 같다).
"""

from __future__ import annotations

import random

from ..model.unified import Intent

# 감정별 말머리/어미 색깔. 에너지에 따라 강도가 바뀐다.
_OPENERS = {
    "excited": ["오 미쳤다", "와 이거 봐", "가자가자", "개쩐다"],
    "amused": ["ㅋㅋㅋ", "아 진짜", "오~"],
    "focused": ["자, 집중", "좋아", "침착하게"],
    "surprised": ["헉", "어어?", "잠깐"],
    "frustrated": ["아 씨", "음...", "에이"],
    "neutral": ["음~", "그래", "오케이"],
}
_TAILS = {
    "excited": ["가보자고!", "이거지!", "텐션 올라간다!"],
    "amused": ["웃기다 진짜 ㅋㅋ", "귀엽네", "그치 그치"],
    "focused": ["가본다", "이대로 간다", "각 나왔어"],
    "surprised": ["방금 뭐였어?", "깜짝이야", "봤어 방금?"],
    "frustrated": ["다시 가자", "괜찮아 만회한다", "한 번 더"],
    "neutral": ["계속 가보자", "오케이", "좋네"],
}


class Verbalizer:
    def __init__(self, name: str = "하루", seed: int = 99) -> None:
        self.name = name
        self._rng = random.Random(seed)

    def _pick(self, table: dict, emotion: str) -> str:
        return self._rng.choice(table.get(emotion, table["neutral"]))

    def say_chat(self, intent: Intent, viewer_text: str) -> str:
        opener = self._pick(_OPENERS, intent.emotion)
        tail = self._pick(_TAILS, intent.emotion)
        snippet = viewer_text.strip()[:24]
        if snippet.endswith("?") or "어때" in snippet or "뭐" in snippet:
            return f"{opener}, {snippet} 라… 내 생각엔 충분히 해볼 만해. {tail}"
        return f"{opener}, {snippet} {tail}"

    def say_game_event(self, intent: Intent, event: str) -> str:
        opener = self._pick(_OPENERS, intent.emotion)
        tail = self._pick(_TAILS, intent.emotion)
        return f"{opener}! {event.strip()[:28]} {tail}"

    def say_tutorial(self, intent: Intent, learned_hint: str, raw: str) -> str:
        opener = self._pick(_OPENERS, intent.emotion)
        if learned_hint:
            return f"{opener}, 방금 화면 보고 {learned_hint} 익혔어. 바로 따라 해볼게!"
        return f"{opener}, '{raw.strip()[:24]}' 이거구나. 기억했어!"
