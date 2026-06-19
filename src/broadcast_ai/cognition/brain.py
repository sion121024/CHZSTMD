"""Brain — 단일 모델을 쓰는 인지 허브.

대화 / 게임 행동 / 튜토리얼 해석을 *하나의* LocalModel로 처리한다. 작업이 무엇이든
같은 모델 인스턴스에 task 태그만 바꿔 흘려보낸다. 출력은 스트리밍이라 호출 측이
첫 토큰부터 바로 말하기/움직이기를 시작할 수 있다.
"""

from __future__ import annotations

from typing import Iterator

from ..model.local_model import LocalModel, Turn
from .persona import Persona
from .tutorial_learner import TutorialLearner


class Brain:
    def __init__(self, model: LocalModel, persona: Persona | None = None) -> None:
        self.model = model
        self.persona = persona or Persona()
        self.tutorial = TutorialLearner()
        self.history: list[Turn] = []
        self._system = self.persona.system_prompt()

    # ---- 대화 (시청자 채팅 등) -----------------------------------------------
    def respond_chat(self, user_text: str) -> Iterator[str]:
        """시청자 입력에 스트리밍으로 답한다."""
        yield from self._stream_and_record(user_text, task="chat")

    # ---- 게임 행동 결정 -------------------------------------------------------
    def decide_game_action(self, situation: str) -> Iterator[str]:
        """현재 게임 상황 설명을 받아 다음 행동을 말로 결정한다.

        지금까지 튜토리얼로 배운 조작을 컨텍스트로 함께 넣는다(같은 모델).
        """
        learned = self.tutorial.knowledge_summary()
        prompt = f"[배운 조작: {learned}] 상황: {situation}"
        yield from self._stream_and_record(prompt, task="game")

    # ---- 튜토리얼 학습 + 반응 -------------------------------------------------
    def learn_from_screen(self, screen_text: str) -> Iterator[str]:
        """화면에서 읽은 튜토리얼 텍스트를 학습하고, 배운 소감을 말한다."""
        learned = self.tutorial.observe(screen_text)
        if learned:
            hint = ", ".join(s.as_action_hint() for s in learned)
            prompt = f"방금 익힌 조작: {hint}. 원문: {screen_text[:80]}"
        else:
            prompt = screen_text[:80]
        yield from self._stream_and_record(prompt, task="tutorial")

    # ---- 공통 스트리밍 + 히스토리 기록 ---------------------------------------
    def _stream_and_record(self, user_text: str, task) -> Iterator[str]:
        self.history.append(Turn("user", user_text))
        collected: list[str] = []
        for piece in self.model.stream(self._system, self.history, user_text, task=task):
            collected.append(piece)
            yield piece
        self.history.append(Turn("assistant", "".join(collected)))
        # 컨텍스트 폭주 방지(저지연 유지): 최근 turn만 남긴다.
        if len(self.history) > 16:
            self.history = self.history[-16:]
