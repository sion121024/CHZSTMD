"""UnifiedAgent — 처음부터 설계한 단 하나의 신경망.

이 한 개의 네트워크가 매 틱마다 동시에 출력한다:
  1) 게임 제어 (FPS): 시점 이동(aim) · 이동(move) · 버튼(발사/점프/재장전/숙이기)
  2) 아바타 모션 : 12개 포즈 채널의 속도 → 적분해 사람처럼 살아있는 움직임
  3) 발화 의도   : 감정 분포 · 에너지 · 말하기 게이트

핵심 설계
---------
* "기존 모델"을 적재하지 않는다. 트렁크(입력 임베딩 + GRU)와 모든 헤드를 직접 구현한
  가중치로 구성한다. 사전학습 체크포인트 없음.
* "고정 애니메이션"이 아니다. 모션은 매 프레임 네트워크 forward로 생성된다. 내부의
  확률적 드라이브(OU 잡음)와 GRU의 시간 동역학, 그리고 감정/에너지/게임 맥락이
  섞여 비반복적이고 반응적인 움직임이 나온다.
* "모델을 나누지 않는다." 제어·모션·발화의도가 *같은* 은닉 상태 h_t를 공유한다.
* FPS 실시간: 차원이 작아 한 틱 추론이 수백 µs~수 ms. 조준은 온라인(NLMS)으로
  실시간 학습돼 타깃을 빠르게 추적한다.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ..nn.layers import GRUCell, Linear, matvec, sigmoid, softmax, tanh
from ..nn.seed import seeded_rng

# ---- 차원 ------------------------------------------------------------------
OBS_DIM = 24
HIDDEN = 48
N_BUTTONS = 4         # 발사, 점프, 재장전, 숙이기
N_MOTION = 12         # 포즈 채널
EMOTIONS = ["neutral", "excited", "focused", "surprised", "amused", "frustrated"]

MOTION_CHANNELS = [
    "head_yaw", "head_pitch", "head_roll", "body_sway", "breath", "eye_blink",
    "brow", "mouth_open", "mouth_wide", "arm_l", "arm_r", "lean",
]


# ---- 입출력 자료구조 -------------------------------------------------------
@dataclass
class Observation:
    """한 틱의 관측. 게임/자기수용/사회/내부드라이브를 고정 순서 벡터로 직렬화."""

    # 게임 (8)
    target_dx: float = 0.0        # 가장 가까운 적까지 조준 오프셋 x (-1..1)
    target_dy: float = 0.0        # 조준 오프셋 y
    target_visible: float = 0.0   # 적이 보이는가 (0/1)
    target_dist: float = 1.0      # 거리 (0..1, 가까울수록 0)
    enemy_count: float = 0.0      # 적 수 (정규화)
    health: float = 1.0           # 체력 (0..1)
    ammo: float = 1.0             # 탄약 (0..1)
    under_fire: float = 0.0       # 피격 중 (0/1)
    # 자기수용 — 직전 행동 피드백 (5)
    last_aim_dx: float = 0.0
    last_aim_dy: float = 0.0
    last_move_x: float = 0.0
    last_move_y: float = 0.0
    firing: float = 0.0
    # 사회/맥락 (3)
    chat_activity: float = 0.0
    chat_sentiment: float = 0.0
    audience: float = 0.0
    # 내부 드라이브 (4): 감정 자극 + 발화 활성 + (트렁크가 모션으로 변환할) 잡음 위상
    excite_drive: float = 0.0
    focus_drive: float = 0.0
    speaking: float = 0.0
    bias: float = 1.0
    # 모션용 확률 드라이브 (3) — 네트워크가 이를 유기적 움직임으로 변환
    noise_slow: float = 0.0
    noise_fast1: float = 0.0
    noise_fast2: float = 0.0

    def to_vector(self) -> list[float]:
        return [
            self.target_dx, self.target_dy, self.target_visible, self.target_dist,
            self.enemy_count, self.health, self.ammo, self.under_fire,
            self.last_aim_dx, self.last_aim_dy, self.last_move_x, self.last_move_y,
            self.firing,
            self.chat_activity, self.chat_sentiment, self.audience,
            self.excite_drive, self.focus_drive, self.speaking, self.bias,
            self.noise_slow, self.noise_fast1, self.noise_fast2,
        ]
        # 길이 23 → bias 자리에서 24 맞춤은 아래 _pad에서 보정


@dataclass
class Control:
    aim_dx: float = 0.0       # 시점 이동량 x (이번 틱)
    aim_dy: float = 0.0       # 시점 이동량 y
    move_x: float = 0.0       # 좌우 이동 (-1..1)
    move_y: float = 0.0       # 전후 이동 (-1..1)
    fire: bool = False
    jump: bool = False
    reload: bool = False
    crouch: bool = False


@dataclass
class MotionPose:
    values: dict[str, float] = field(default_factory=dict)

    def get(self, name: str) -> float:
        return self.values.get(name, 0.0)


@dataclass
class Intent:
    emotion: str = "neutral"
    emotion_probs: dict[str, float] = field(default_factory=dict)
    energy: float = 0.0
    speak: bool = False


@dataclass
class AgentOutput:
    control: Control
    motion: MotionPose
    intent: Intent
    hidden_norm: float = 0.0


# --------------------------------------------------------------------------- #
class UnifiedAgent:
    """하나의 신경망. 제어·모션·발화의도를 같은 은닉 상태에서 낸다."""

    def __init__(self, seed: int = 1234) -> None:
        rng = seeded_rng(seed)
        self._rng = rng

        # --- 공유 트렁크 (단일 모델) ---
        self.embed = Linear(OBS_DIM, HIDDEN, rng)
        self.gru = GRUCell(HIDDEN, HIDDEN, rng)

        # --- 헤드 (모두 같은 h_t를 읽는다) ---
        self.head_aim = Linear(HIDDEN, 2, rng)
        self.head_move = Linear(HIDDEN, 2, rng)
        self.head_btn = Linear(HIDDEN, N_BUTTONS, rng)
        self.head_motion = Linear(HIDDEN, N_MOTION, rng)
        self.head_emotion = Linear(HIDDEN, len(EMOTIONS), rng)
        self.head_energy = Linear(HIDDEN, 1, rng)
        self.head_speak = Linear(HIDDEN, 1, rng)

        # --- 온라인 조준 학습기 (NLMS): 2 x HIDDEN 선형 리드아웃 ---
        # 처음엔 0으로 시작해 실시간으로 "어디를 봐야 타깃이 중앙에 오는지" 학습.
        self.aim_readout = [[0.0] * HIDDEN for _ in range(2)]
        self.aim_lr = 0.5

        # --- 상태 ---
        self.h = [0.0] * HIDDEN
        self._last_h = self.h
        self.pose = [0.0] * N_MOTION          # 적분되는 지속 포즈
        self._neutral = [0.0] * N_MOTION
        # OU 잡음 상태(모션 드라이브). 네트워크가 이걸 움직임으로 바꾼다.
        self._ou = [0.0, 0.0, 0.0]

        self.last_control = Control()

    # ---- 파라미터 수(직접 만든 모델의 규모) ----------------------------------
    @property
    def num_params(self) -> int:
        def lin(l: Linear) -> int:
            return l.n_in * l.n_out + l.n_out

        gru = sum(len(W) * len(W[0]) for W in (self.gru.Wz, self.gru.Wr, self.gru.Wn))
        gru += 3 * self.gru.h
        heads = sum(lin(l) for l in (
            self.head_aim, self.head_move, self.head_btn, self.head_motion,
            self.head_emotion, self.head_energy, self.head_speak))
        readout = 2 * HIDDEN
        return lin(self.embed) + gru + heads + readout

    def reset(self) -> None:
        self.h = [0.0] * HIDDEN
        self.pose = [0.0] * N_MOTION
        self._ou = [0.0, 0.0, 0.0]

    # ---- 내부 모션 드라이브 갱신 (OU 잡음) ----------------------------------
    def _advance_drives(self) -> tuple[float, float, float]:
        # Ornstein-Uhlenbeck: 평균회귀 잡음. slow=느린 호흡류, fast=미세 떨림.
        # 빠른 잡음(떨림)을 줄여 미세한 흔들림이 덜 어색하게.
        thetas = (0.02, 0.12, 0.2)
        sigmas = (0.16, 0.3, 0.3)
        for i in range(3):
            self._ou[i] += -thetas[i] * self._ou[i] + sigmas[i] * self._rng.gauss(0, 1)
            self._ou[i] = max(-3.0, min(3.0, self._ou[i]))
        return self._ou[0], self._ou[1], self._ou[2]

    # ---- 한 틱 forward (제어 + 모션 + 의도 동시) ----------------------------
    def tick(self, obs: Observation, dt: float = 1.0 / 60.0) -> AgentOutput:
        # 모션용 확률 드라이브 주입(고정 사인 아님 — 네트워크가 변환).
        ns, nf1, nf2 = self._advance_drives()
        obs.noise_slow, obs.noise_fast1, obs.noise_fast2 = ns, nf1, nf2

        x = _pad(obs.to_vector(), OBS_DIM)
        emb = tanh(self.embed(x))
        self.h = self.gru.step(emb, self.h)
        self._last_h = self.h
        h = self.h

        control = self._control(h)
        motion = self._motion(h, dt, energy_hint=obs.excite_drive)
        intent = self._intent(h)

        self.last_control = control
        hn = math.sqrt(sum(v * v for v in h))
        return AgentOutput(control=control, motion=motion, intent=intent, hidden_norm=hn)

    # ---- 제어 헤드 (FPS) -----------------------------------------------------
    def _control(self, h: list[float]) -> Control:
        base = tanh(self.head_aim(h))                  # 기본 정책(작게)
        learned = matvec(self.aim_readout, h)          # 온라인 학습된 조준
        aim_dx = _clamp(0.25 * base[0] + learned[0], -1.0, 1.0)
        aim_dy = _clamp(0.25 * base[1] + learned[1], -1.0, 1.0)

        mv = tanh(self.head_move(h))
        btn = sigmoid(self.head_btn(h))
        return Control(
            aim_dx=aim_dx, aim_dy=aim_dy,
            move_x=mv[0], move_y=mv[1],
            fire=btn[0] > 0.5, jump=btn[1] > 0.5,
            reload=btn[2] > 0.5, crouch=btn[3] > 0.5,
        )

    def learn_aim(self, desired_dx: float, desired_dy: float) -> float:
        """NLMS 온라인 업데이트: 조준을 실시간으로 학습한다.

        desired = 타깃을 중앙에 두려면 이번 틱에 시점을 얼마나 움직여야 하는가.
        은닉 특징 h에서 그 값으로 가는 선형 사상을 학습한다. 반환값은 학습 전 오차.
        """
        h = self._last_h
        pred = matvec(self.aim_readout, h)
        err = (desired_dx - pred[0], desired_dy - pred[1])
        denom = sum(v * v for v in h) + 1e-6
        scale = self.aim_lr / denom
        for o in range(2):
            e = err[o]
            row = self.aim_readout[o]
            for i, hi in enumerate(h):
                row[i] += scale * e * hi
        return math.hypot(err[0], err[1])

    # ---- 모션 헤드 (AI 생성, 고정 애니메이션 아님) --------------------------
    def _motion(self, h: list[float], dt: float, energy_hint: float) -> MotionPose:
        vel = tanh(self.head_motion(h))
        # 에너지가 높을수록 움직임 진폭이 커진다(감정→모션, 같은 모델 안에서).
        gain = 1.0 + 1.2 * max(0.0, min(1.0, energy_hint))
        # 중립 복귀는 약하게(0.045) — 중립에 붙박이지 않고 천천히 더 멀리 배회해서
        # '살아있는' 게 눈에 보이게. 적분(저역통과)이라 진폭만 커지고 떨림은 안 는다.
        damp = 0.045
        for i in range(N_MOTION):
            # 속도 적분(×5.5) — 천천히, 하지만 분명히 움직이게
            self.pose[i] += vel[i] * gain * dt * 5.5
            self.pose[i] += (self._neutral[i] - self.pose[i]) * damp
            self.pose[i] = _clamp(self.pose[i], -1.0, 1.0)
        return MotionPose(values={name: self.pose[i] for i, name in enumerate(MOTION_CHANNELS)})

    # ---- 발화 의도 헤드 ------------------------------------------------------
    def _intent(self, h: list[float]) -> Intent:
        probs = softmax(self.head_emotion(h))
        emo_idx = max(range(len(EMOTIONS)), key=lambda i: probs[i])
        energy = sigmoid(self.head_energy(h))[0]
        speak = sigmoid(self.head_speak(h))[0] > 0.5
        return Intent(
            emotion=EMOTIONS[emo_idx],
            emotion_probs={e: round(p, 4) for e, p in zip(EMOTIONS, probs)},
            energy=round(energy, 4),
            speak=speak,
        )


# --------------------------------------------------------------------------- #
def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _pad(v: list[float], n: int) -> list[float]:
    if len(v) < n:
        return v + [0.0] * (n - len(v))
    return v[:n]
