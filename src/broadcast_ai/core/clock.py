"""지연(latency) 측정 도구.

방송 AI의 체감 품질은 "반응속도"가 좌우한다. 각 파이프라인 단계의 시간을 재서
예산을 넘으면 경고하고, 다음 단계가 출력을 축약하도록 신호를 줄 수 있게 한다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


def now_ms() -> float:
    return time.perf_counter() * 1000.0


@dataclass
class Stage:
    name: str
    start_ms: float
    end_ms: float | None = None

    @property
    def elapsed_ms(self) -> float:
        end = self.end_ms if self.end_ms is not None else now_ms()
        return end - self.start_ms


@dataclass
class LatencyBudget:
    """한 번의 사용자 반응(턴)에 대한 단계별 지연 추적기."""

    target_first_audio_ms: int = 250
    target_think_ms: int = 120
    stages: list[Stage] = field(default_factory=list)
    _origin: float = field(default_factory=now_ms)

    def start(self, name: str) -> Stage:
        s = Stage(name=name, start_ms=now_ms())
        self.stages.append(s)
        return s

    def end(self, stage: Stage) -> None:
        stage.end_ms = now_ms()

    def mark_first_audio(self) -> float:
        """첫 음성이 나간 시점까지의 전체 지연(ms)."""
        return now_ms() - self._origin

    def over_budget(self) -> list[str]:
        warnings: list[str] = []
        for s in self.stages:
            if s.name == "think" and s.elapsed_ms > self.target_think_ms:
                warnings.append(f"think {s.elapsed_ms:.0f}ms > {self.target_think_ms}ms")
        ttfa = self.mark_first_audio()
        if ttfa > self.target_first_audio_ms:
            warnings.append(f"first-audio {ttfa:.0f}ms > {self.target_first_audio_ms}ms")
        return warnings

    def report(self) -> str:
        parts = [f"{s.name}={s.elapsed_ms:.0f}ms" for s in self.stages]
        return " · ".join(parts) if parts else "(no stages)"
