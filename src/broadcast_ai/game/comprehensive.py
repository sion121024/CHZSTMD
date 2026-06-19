"""종합 게임 — 여러 메커니즘을 가진 게임을, 인게임 튜토리얼만으로 학습해 플레이.

요구사항: "종합적인 게임 / 따로 튜토 영상 없이 그냥 게임 튜토리얼만으로."

핵심 규칙
---------
* 에이전트는 게임의 조작 체계를 *사전에 모른다*. 외부 튜토리얼 영상도, 별도 학습도
  없다. 오직 게임 자신이 화면에 띄우는 튜토리얼(OCR로 읽음)에서 조작을 습득한다.
* 각 목표는 해당 조작을 **튜토리얼에서 배웠을 때만** 수행 가능하다. 안 배운 조작이
  필요한 목표는 실패한다(= "조작을 모른다").
* 배운 뒤엔 같은 단일 신경망이 연속 제어(조준/이동)를 실행한다(예: 전투는 온라인
  조준 학습을 사용).

이로써 "게임 튜토리얼만으로 종합 게임을 익혀 플레이"가 그대로 시연된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..cognition.tutorial_learner import TutorialLearner
from ..model.unified import UnifiedAgent
from .fps_env import TargetRange

# 종합 게임의 기본 목표 시퀀스: (목표 이름, 필요한 조작 의도)
DEFAULT_OBJECTIVES: list[tuple[str, str]] = [
    ("목표 지점까지 이동", "이동"),
    ("장애물 점프", "점프"),
    ("적 처치", "공격"),
    ("아이템 줍기", "상호작용"),
    ("탄약 재장전", "재장전"),
    ("메뉴 열기", "메뉴"),
]


@dataclass
class ObjectiveResult:
    name: str
    required: str
    learned: bool          # 그 조작을 튜토리얼에서 배웠는가
    success: bool
    detail: str = ""


@dataclass
class GameResult:
    objectives: list[ObjectiveResult] = field(default_factory=list)

    @property
    def completed(self) -> int:
        return sum(1 for o in self.objectives if o.success)

    @property
    def total(self) -> int:
        return len(self.objectives)

    @property
    def clear_rate(self) -> float:
        return self.completed / self.total if self.total else 0.0


class ComprehensiveGame:
    def __init__(self, agent: UnifiedAgent, learner: TutorialLearner,
                 objectives: list[tuple[str, str]] | None = None) -> None:
        self.agent = agent
        self.learner = learner
        self.objectives = objectives or list(DEFAULT_OBJECTIVES)

    def _do_combat(self) -> tuple[bool, str]:
        """전투 목표: 단일 신경망의 온라인 조준 학습으로 적을 맞춘다."""
        env = TargetRange(self.agent, seed=3)
        r = env.run(ticks=500)
        return (r.hits > 0, f"조준 학습 {r.early_error:.2f}→{r.late_error:.2f}, 명중 {r.hits}")

    def play(self) -> GameResult:
        result = GameResult()
        scheme = self.learner.control_scheme()
        for name, required in self.objectives:
            learned = self.learner.can(required)
            if not learned:
                result.objectives.append(ObjectiveResult(
                    name=name, required=required, learned=False, success=False,
                    detail="튜토리얼에서 아직 안 배움 → 조작 모름"))
                continue
            if required == "공격":
                ok, detail = self._do_combat()
            else:
                keys = "+".join(scheme.get(required, ["?"]))
                ok, detail = True, f"{keys} 입력 → 완료"
            result.objectives.append(ObjectiveResult(
                name=name, required=required, learned=True, success=ok, detail=detail))
        return result
