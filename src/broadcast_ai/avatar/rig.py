"""아바타 리그(골격/표정 상태).

한 프레임의 포즈를 표현한다. 렌더러(VRM/Live2D/Unity 등)는 이 Pose를 받아
실제 메시를 그린다. 여기서는 렌더러에 독립적인 수치 상태만 다룬다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Pose:
    """한 프레임의 아바타 상태. 값 범위는 대체로 -1..1 또는 0..1."""

    # 머리/상체
    head_yaw: float = 0.0     # 좌우
    head_pitch: float = 0.0   # 끄덕임
    head_roll: float = 0.0    # 갸웃
    body_sway: float = 0.0    # 미세한 무게중심 이동
    breath: float = 0.0       # 호흡에 따른 흉부 스케일(0..1)

    # 얼굴
    eye_blink: float = 0.0    # 0=뜸, 1=감음 (양쪽)
    wink_l: float = 0.0       # 왼눈만 추가로 감음(윙크 등 제스처)
    wink_r: float = 0.0       # 오른눈만 추가로 감음
    brow_raise: float = 0.0
    mouth_open: float = 0.0   # 립싱크에서 갱신
    mouth_wide: float = 0.0   # 입모양(아/이 구분)
    smile: float = 0.0

    # 팔(제스처)
    arm_l: float = 0.0
    arm_r: float = 0.0

    def blended(self, other: "Pose", w: float) -> "Pose":
        """두 포즈를 w(0..1)로 선형 보간. 모션 합성에 사용."""
        def lerp(a: float, b: float) -> float:
            return a * (1.0 - w) + b * w

        return Pose(
            head_yaw=lerp(self.head_yaw, other.head_yaw),
            head_pitch=lerp(self.head_pitch, other.head_pitch),
            head_roll=lerp(self.head_roll, other.head_roll),
            body_sway=lerp(self.body_sway, other.body_sway),
            breath=lerp(self.breath, other.breath),
            eye_blink=lerp(self.eye_blink, other.eye_blink),
            wink_l=lerp(self.wink_l, other.wink_l),
            wink_r=lerp(self.wink_r, other.wink_r),
            brow_raise=lerp(self.brow_raise, other.brow_raise),
            mouth_open=lerp(self.mouth_open, other.mouth_open),
            mouth_wide=lerp(self.mouth_wide, other.mouth_wide),
            smile=lerp(self.smile, other.smile),
            arm_l=lerp(self.arm_l, other.arm_l),
            arm_r=lerp(self.arm_r, other.arm_r),
        )


@dataclass
class Rig:
    """아바타 리그. 현재 포즈를 보관하고 렌더 프레임을 만들어낸다."""

    pose: Pose = field(default_factory=Pose)

    def apply(self, pose: Pose) -> None:
        self.pose = pose

    def to_render_frame(self) -> dict:
        """렌더러로 보낼 직렬화 가능한 프레임."""
        p = self.pose
        base_open = 1.0 - max(0.0, min(1.0, p.eye_blink))
        open_l = base_open * (1.0 - max(0.0, min(1.0, p.wink_l)))
        open_r = base_open * (1.0 - max(0.0, min(1.0, p.wink_r)))
        return {
            "head": [round(p.head_yaw, 4), round(p.head_pitch, 4), round(p.head_roll, 4)],
            "body_sway": round(p.body_sway, 4),
            "breath": round(p.breath, 4),
            "eyes": {"blink": round(p.eye_blink, 4), "brow": round(p.brow_raise, 4),
                     "open_l": round(open_l, 4), "open_r": round(open_r, 4)},
            "mouth": {"open": round(p.mouth_open, 4), "wide": round(p.mouth_wide, 4),
                      "smile": round(p.smile, 4)},
            "arms": [round(p.arm_l, 4), round(p.arm_r, 4)],
        }
