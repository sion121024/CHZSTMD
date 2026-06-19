"""Brain — 단일 통합 모델을 쓰는 인지 허브.

대화/게임/튜토리얼 무엇이든 *같은* UnifiedAgent를 틱(tick)해 발화 의도를 얻고,
Verbalizer로 한국어로 표면화한다. 모델은 하나뿐이며, 이 모듈은 입력을 관측
(Observation)으로 만들고 결과를 말로 옮긴다.
"""

from __future__ import annotations

from typing import Iterator

from ..model.unified import Observation, UnifiedAgent
from .persona import Persona
from .verbalizer import Verbalizer
from .tutorial_learner import TutorialLearner


def _stream_text(text: str) -> Iterator[str]:
    """문자열을 작은 조각으로 흘려 저지연 TTS 경로를 그대로 쓰게 한다."""
    buf = ""
    for ch in text:
        buf += ch
        if ch in " ,.!?…~\n" or len(buf) >= 4:
            yield buf
            buf = ""
    if buf:
        yield buf


def _sentiment(text: str) -> float:
    pos = sum(text.count(w) for w in ("ㅋ", "좋", "굿", "개잘", "사랑", "최고", "ㅎㅎ"))
    neg = sum(text.count(w) for w in ("싫", "노잼", "별로", "망", "ㅠ", "안돼"))
    return max(-1.0, min(1.0, 0.3 * (pos - neg)))


class Brain:
    def __init__(self, agent: UnifiedAgent, persona: Persona | None = None) -> None:
        self.agent = agent
        self.persona = persona or Persona()
        self.verbalizer = Verbalizer(name=self.persona.name)
        self.tutorial = TutorialLearner()
        self.last_intent = None
        # 학습된 대화 모델(선택). 없으면 Verbalizer 템플릿으로 폴백.
        self.dialogue = None
        self.history: list[tuple[str, str]] = []

    # ---- 대화 ----------------------------------------------------------------
    def respond_chat(self, viewer_text: str) -> Iterator[str]:
        obs = Observation(
            chat_activity=1.0,
            chat_sentiment=_sentiment(viewer_text),
            audience=0.5,
            excite_drive=0.5,
            speaking=1.0,
        )
        out = self.agent.tick(obs)   # 감정·에너지·모션은 항상 단일 에이전트가 구동
        self.last_intent = out.intent

        if self.dialogue is not None and self.dialogue.available():
            # 학습된 from-scratch 대화 모델로 유창하게 응답.
            text = self.dialogue.reply(self.history, viewer_text)
            self.history.append(("user", viewer_text))
            self.history.append(("bot", text))
            self.history = self.history[-12:]
        else:
            text = self.verbalizer.say_chat(out.intent, viewer_text)
        yield from _stream_text(text)

    # ---- 게임 이벤트 코멘터리 ------------------------------------------------
    def comment_game(self, event: str, excited: float = 0.7) -> Iterator[str]:
        obs = Observation(excite_drive=excited, speaking=1.0, target_visible=1.0)
        out = self.agent.tick(obs)
        self.last_intent = out.intent
        text = self.verbalizer.say_game_event(out.intent, event)
        yield from _stream_text(text)

    # ---- 튜토리얼 자동 학습 + 반응 -------------------------------------------
    def learn_from_screen(self, screen_text: str) -> Iterator[str]:
        learned = self.tutorial.observe(screen_text)
        hint = ", ".join(s.as_action_hint() for s in learned) if learned else ""
        obs = Observation(focus_drive=0.8, speaking=1.0)
        out = self.agent.tick(obs)
        self.last_intent = out.intent
        text = self.verbalizer.say_tutorial(out.intent, hint, screen_text)
        yield from _stream_text(text)
