"""사람처럼 움직이는 모션 합성기.

"진짜 사람처럼 버츄얼이 움직인다"는 요구의 핵심. 무의식적 생체 모션을 절차적으로
생성한다 — 전부 순수 파이썬 수학이라 GPU 없이도 CPU에서 가볍게 60fps로 돈다.

레이어
-----
1) 호흡: 느린 사인파로 흉부 스케일/머리 끄덕임에 미세하게 반영.
2) 깜빡임: 평균 발생률을 둔 의사난수 타이밍 + 빠른 감음/뜸 곡선.
3) 미세 흔들림(idle sway): 서로 다른 주기의 사인파를 겹쳐 만든 의사-펄린 노이즈로
   머리/무게중심이 끊임없이 살짝 움직이게 한다(완전 정지 = 로봇 같음 방지).
4) 제스처: 말하거나 감정이 고조될 때 팔/눈썹/미소를 얹는다.
"""

from __future__ import annotations

import math
import random

from .rig import Pose


def _pseudo_perlin(t: float, seed: float) -> float:
    """서로 무관한 주기의 사인파를 합쳐 만든 부드러운 노이즈(-1..1 근처)."""
    return (
        0.55 * math.sin(t * 0.7 + seed)
        + 0.30 * math.sin(t * 1.7 + seed * 2.3)
        + 0.15 * math.sin(t * 3.1 + seed * 0.7)
    )


class MotionSynth:
    """시간에 따라 생동감 있는 idle/말하기 포즈를 만들어낸다."""

    def __init__(self, fps: int = 60, blink_per_min: float = 17.0,
                 seed: int = 7) -> None:
        self.fps = max(1, fps)
        self.blink_per_min = blink_per_min
        self._rng = random.Random(seed)
        self._t = 0.0
        # 다음 깜빡임까지 남은 시간(초). 지수분포로 자연스러운 간격.
        self._next_blink = self._draw_blink_interval()
        self._blink_phase = -1.0  # <0 이면 깜빡임 중 아님

        # 말하기/감정 상태(외부에서 갱신).
        self.speaking = False
        self.energy = 0.0  # 0..1, 감정 고조도 → 제스처 크기

    def _draw_blink_interval(self) -> float:
        rate = max(self.blink_per_min, 1.0) / 60.0  # 초당 발생률
        # 지수분포 표본.
        return self._rng.expovariate(rate)

    def set_speaking(self, speaking: bool, energy: float = 0.4) -> None:
        self.speaking = speaking
        self.energy = max(0.0, min(1.0, energy))

    def tick(self) -> Pose:
        """다음 한 프레임의 idle/말하기 포즈를 반환한다."""
        dt = 1.0 / self.fps
        self._t += dt
        t = self._t

        # --- 호흡 (분당 ~14회) ---
        breath_cycle = math.sin(t * 2.0 * math.pi * (14.0 / 60.0))
        breath = 0.5 + 0.5 * breath_cycle
        breath_pitch = 0.015 * breath_cycle  # 숨쉬며 머리 아주 살짝 끄덕

        # --- 미세 흔들림 ---
        sway_x = _pseudo_perlin(t, seed=1.0)
        sway_y = _pseudo_perlin(t, seed=9.0)
        head_yaw = 0.05 * sway_x
        head_roll = 0.03 * sway_y
        body_sway = 0.04 * _pseudo_perlin(t, seed=4.0)

        # --- 깜빡임 ---
        eye_blink = self._update_blink(dt)

        # --- 말하기/감정 제스처 ---
        brow = 0.0
        smile = 0.05
        arm_l = arm_r = 0.0
        head_pitch = breath_pitch
        if self.speaking:
            # 말할 때 머리/눈썹이 박자에 맞춰 약간 움직이고 미소가 늘어난다.
            beat = math.sin(t * 8.0)
            head_pitch += 0.03 * beat * (0.5 + self.energy)
            head_yaw += 0.04 * math.sin(t * 5.3) * self.energy
            brow = 0.15 * (0.5 + 0.5 * beat) * (0.4 + self.energy)
            smile = 0.12 + 0.2 * self.energy
            # 감정이 셀수록 손제스처가 커진다.
            arm_l = 0.10 * self.energy * math.sin(t * 3.0)
            arm_r = 0.10 * self.energy * math.sin(t * 3.0 + 0.6)

        return Pose(
            head_yaw=head_yaw,
            head_pitch=head_pitch,
            head_roll=head_roll,
            body_sway=body_sway,
            breath=breath,
            eye_blink=eye_blink,
            brow_raise=brow,
            smile=smile,
            arm_l=arm_l,
            arm_r=arm_r,
        )

    def _update_blink(self, dt: float) -> float:
        # 깜빡임 진행 중이면 곡선 진행.
        if self._blink_phase >= 0.0:
            self._blink_phase += dt
            dur = 0.12  # 한 번 깜빡이는 데 걸리는 시간(초)
            if self._blink_phase >= dur:
                self._blink_phase = -1.0
                return 0.0
            # 0→1→0 빠른 감음/뜸 (사인 반주기).
            return math.sin((self._blink_phase / dur) * math.pi)

        # 대기: 카운트다운.
        self._next_blink -= dt
        if self._next_blink <= 0.0:
            self._blink_phase = 0.0
            self._next_blink = self._draw_blink_interval()
            return 0.0
        return 0.0
