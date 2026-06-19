"""FPS 사격장 — 실시간 제어 + 온라인 조준 학습 시연 환경.

움직이는 타깃을 향해 에이전트가 매 틱 시점(crosshair)을 옮긴다. 단일 모델의
은닉 특징에서 "어디로 봐야 타깃이 중앙에 오는지"를 NLMS로 실시간 학습한다.
처음엔 못 맞추다가 수십~수백 틱 안에 추적에 성공한다 — 고정 스크립트(에임봇)가
아니라 학습되는 정책이다. 동시에 추론 처리량(Hz)을 재 FPS 실시간 가능성을 보인다.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from ..model.unified import Observation, UnifiedAgent


@dataclass
class FPSResult:
    ticks: int
    early_error: float          # 초반 평균 조준 오차
    late_error: float           # 후반 평균 조준 오차 (학습 후)
    hits: int                   # 타깃을 중앙에 두고 발사 성공 횟수
    infer_hz: float             # 초당 추론 횟수(=실시간 능력)
    error_curve: list[float] = field(default_factory=list)

    @property
    def improved(self) -> bool:
        return self.late_error < self.early_error

    @property
    def fps_capable(self) -> bool:
        # 일반적 FPS 게임 틱레이트(60~240Hz) 이상이면 실시간 가능.
        return self.infer_hz >= 240.0


class TargetRange:
    """1인칭 사격장. crosshair와 움직이는 타깃의 오프셋을 다룬다."""

    def __init__(self, agent: UnifiedAgent, seed: int = 0,
                 sensitivity: float = 1.0, hit_radius: float = 0.06) -> None:
        self.agent = agent
        self.sensitivity = sensitivity
        self.hit_radius = hit_radius
        self.cx = 0.0
        self.cy = 0.0
        self._t = 0.0
        self._seed = seed

    def _target_pos(self, t: float) -> tuple[float, float]:
        # 타깃이 리사주 곡선으로 예측 불가하게 움직인다.
        tx = 0.6 * math.sin(t * 1.7 + self._seed)
        ty = 0.5 * math.sin(t * 2.3 + self._seed * 0.5)
        return tx, ty

    def run(self, ticks: int = 600, dt: float = 1.0 / 120.0,
            learn: bool = True) -> FPSResult:
        errors: list[float] = []
        hits = 0
        t0 = time.perf_counter()

        for k in range(ticks):
            self._t += dt
            tx, ty = self._target_pos(self._t)
            off_x = tx - self.cx
            off_y = ty - self.cy
            err = math.hypot(off_x, off_y)
            errors.append(err)

            obs = Observation(
                target_dx=_clip(off_x), target_dy=_clip(off_y),
                target_visible=1.0,
                target_dist=_clip(err),
                enemy_count=0.2, health=1.0, ammo=1.0,
                last_aim_dx=self.agent.last_control.aim_dx,
                last_aim_dy=self.agent.last_control.aim_dy,
                focus_drive=0.8,
            )
            out = self.agent.tick(obs, dt=dt)

            # 시점 적용: 학습된 조준이 crosshair를 타깃 쪽으로 옮긴다.
            self.cx = _clip(self.cx + out.control.aim_dx * self.sensitivity)
            self.cy = _clip(self.cy + out.control.aim_dy * self.sensitivity)

            # 온라인 학습: 이번 틱 이상적인 시점 이동량 = 직전 오프셋.
            if learn:
                self.agent.learn_aim(_clip(off_x), _clip(off_y))

            # 중앙에 두고 발사하면 명중.
            new_off = math.hypot(tx - self.cx, ty - self.cy)
            if new_off <= self.hit_radius and out.control.fire:
                hits += 1

        elapsed = time.perf_counter() - t0
        infer_hz = ticks / elapsed if elapsed > 0 else float("inf")

        half = max(1, ticks // 4)
        early = sum(errors[:half]) / half
        late = sum(errors[-half:]) / half
        return FPSResult(
            ticks=ticks, early_error=early, late_error=late, hits=hits,
            infer_hz=infer_hz, error_curve=errors,
        )


def _clip(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return lo if v < lo else hi if v > hi else v
