"""모션 어댑터 — AI가 생성한 모션을 리그 포즈로 옮긴다.

중요: 더 이상 사인파 같은 *고정 애니메이션*을 만들지 않는다. 움직임은 단일
UnifiedAgent의 모션 헤드(신경망 forward)가 매 프레임 생성한다. 이 어댑터는 그
신경망 출력(MotionPose, 12채널)을 렌더러가 이해하는 Rig.Pose로 변환할 뿐이다.
"""

from __future__ import annotations

import math

from ..model.unified import MotionPose
from .rig import Pose


def _unit(v: float) -> float:
    """-1..1 신호를 0..1로 부드럽게 매핑(호흡/깜빡임처럼 단방향 채널용)."""
    return 0.5 + 0.5 * math.tanh(v)


class MotionAdapter:
    """신경망 모션 채널 → 아바타 Pose 변환기."""

    def to_pose(self, motion: MotionPose, energy: float = 0.0) -> Pose:
        m = motion.get
        return Pose(
            head_yaw=0.18 * m("head_yaw"),
            head_pitch=0.14 * m("head_pitch"),
            head_roll=0.12 * m("head_roll") + 0.05 * m("lean"),
            body_sway=0.12 * m("body_sway"),
            breath=_unit(1.2 * m("breath")),          # 0..1, 네트워크가 만든 호흡
            eye_blink=max(0.0, math.tanh(2.0 * m("eye_blink"))),  # 감을 때만 양수
            brow_raise=max(0.0, m("brow")),
            mouth_open=0.0,    # 립싱크가 발화 중 덮어쓴다
            mouth_wide=0.0,
            smile=max(0.0, 0.1 + 0.4 * energy),
            arm_l=0.25 * m("arm_l"),
            arm_r=0.25 * m("arm_r"),
        )
