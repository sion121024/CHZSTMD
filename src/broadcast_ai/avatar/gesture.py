"""제스처 — '명령으로 부르는' 짧은 동작(윙크/손인사/끄덕임 등).

설계 의도: 평상시 움직임은 여전히 신경망이 *생성*한다(고정 애니메이션 아님).
제스처는 그 위에 잠깐 *덧씌우는* 의도적 동작이다 — VTuber의 핫키 모션처럼,
"윙크해줘" 같은 요청이 오면 잠깐 재생하고 끝나면 다시 AI 모션으로 돌아간다.

각 제스처는 위상 p(0..1)에 대한 함수로, Pose 필드에 더할 '델타'를 돌려준다.
실시간 시계(perf_counter)로 지속시간을 추적하므로 호출 주기와 무관하게 동작한다.
"""

from __future__ import annotations

import math
import time

from .rig import Pose


def _bump(p: float) -> float:
    """0→1→0 부드러운 종(0..1)."""
    return math.sin(math.pi * max(0.0, min(1.0, p)))


def _osc(p: float, cycles: float) -> float:
    """페이드 인/아웃 봉투가 걸린 진동(-1..1) — 끄덕임/도리도리/손흔들기용."""
    return math.sin(2 * math.pi * cycles * p) * math.sin(math.pi * p)


# 각 제스처: (지속시간 초, p→Pose 델타 dict). 값은 AI 모션 위에 가산된다.
def _wink(p):       # 한쪽 눈만 깜빡 + 살짝 미소
    c = math.sin(math.pi * p) ** 0.55      # 빨리 감았다 뜸
    return {"wink_r": c, "smile": 0.25 * _bump(p), "head_roll": 0.08 * _bump(p)}


def _nod(p):        # 끄덕끄덕(두 번)
    return {"head_pitch": 0.55 * _osc(p, 2)}


def _shake(p):      # 도리도리(두 번)
    return {"head_yaw": 0.6 * _osc(p, 2)}


def _tilt(p):       # 갸웃
    return {"head_roll": 0.5 * _bump(p), "smile": 0.15 * _bump(p)}


def _wave(p):       # 손 흔들어 인사 — 팔 올리고 좌우로
    raise_ = 0.5 * _bump(p)
    return {"arm_r": raise_ + 0.45 * _osc(p, 3), "smile": 0.3 * _bump(p),
            "head_roll": 0.1 * _bump(p)}


def _surprise(p):   # 놀람 — 눈썹 올리고 입 벌리고 살짝 젖힘
    b = _bump(p)
    return {"brow_raise": 1.0 * b, "mouth_open": 0.6 * b, "head_pitch": -0.2 * b}


def _bounce(p):     # 폴짝(신남) — 몸 들썩 + 미소
    return {"body_sway": 0.0, "head_pitch": -0.25 * abs(_osc(p, 2)),
            "smile": 0.4 * _bump(p), "arm_l": 0.2 * _bump(p), "arm_r": 0.2 * _bump(p)}


GESTURES: dict[str, tuple[float, object]] = {
    "wink": (0.55, _wink),
    "nod": (1.1, _nod),
    "shake": (1.1, _shake),
    "tilt": (1.2, _tilt),
    "wave": (1.5, _wave),
    "surprise": (0.9, _surprise),
    "bounce": (1.0, _bounce),
}

# 채팅/음성 명령 키워드 → 제스처 (한국어/영어). 부분일치.
GESTURE_KW: dict[str, list[str]] = {
    "wink": ["윙크", "wink", "윙크해", "눈 찡긋", "찡긋"],
    "wave": ["손 흔들", "손흔들", "인사", "안녕 손", "빠이", "바이", "hi", "hello", "wave", "손 인사"],
    "nod": ["끄덕", "응응", "고개 끄덕", "nod", "그래그래"],
    "shake": ["도리도리", "절레", "고개 저어", "아니아니", "shake"],
    "tilt": ["갸웃", "갸우뚱", "고개 갸", "tilt"],
    "surprise": ["놀라", "놀람", "깜짝", "surprise", "헉 해", "리액션"],
    "bounce": ["폴짝", "방방", "신나", "점프", "bounce", "뛰어"],
}


def detect_gesture(text: str) -> str | None:
    """입력 텍스트에서 요청된 제스처를 찾는다(없으면 None)."""
    t = text.lower()
    for name, kws in GESTURE_KW.items():
        if any(k.lower() in t for k in kws):
            return name
    return None


_SET_FIELDS = {"wink_l", "wink_r"}   # 가산 대신 '최대' 적용(눈은 가산하면 어색)
_CLAMP = {
    "wink_l": (0.0, 1.0), "wink_r": (0.0, 1.0), "smile": (0.0, 1.0),
    "mouth_open": (0.0, 1.0), "brow_raise": (-1.0, 1.0),
}


class GestureController:
    """현재 활성 제스처를 추적하고 Pose에 덧씌운다."""

    def __init__(self) -> None:
        self._name: str | None = None
        self._dur: float = 0.0
        self._start: float = 0.0

    @property
    def active(self) -> str | None:
        return self._name

    def trigger(self, name: str) -> bool:
        """제스처를 시작한다. 알 수 없는 이름이면 False."""
        if name not in GESTURES:
            return False
        self._name = name
        self._dur, _ = GESTURES[name]
        self._start = time.perf_counter()
        return True

    def apply(self, pose: Pose) -> None:
        """활성 제스처의 이번 프레임 델타를 Pose에 얹는다(끝났으면 해제)."""
        if self._name is None:
            return
        p = (time.perf_counter() - self._start) / self._dur
        if p >= 1.0:
            self._name = None
            return
        _, fn = GESTURES[self._name]
        for field, delta in fn(p).items():
            cur = getattr(pose, field, 0.0)
            new = max(cur, delta) if field in _SET_FIELDS else cur + delta
            lo, hi = _CLAMP.get(field, (-1.0, 1.0))
            setattr(pose, field, max(lo, min(hi, new)))
