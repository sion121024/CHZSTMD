"""신경망 기본 연산 — 직접 구현(순수 파이썬).

성능 메모: 차원이 작아(입력~24, 은닉~48) 한 틱당 곱셈-덧셈이 ~2만 회 수준이라
순수 파이썬으로도 수백~수천 Hz가 나온다(FPS 실시간에 충분). zip/sum을 써서
파이썬 루프 오버헤드를 줄인다.
"""

from __future__ import annotations

import math
import random

from .seed import xavier

Vector = list
Matrix = list


def tanh(v: Vector) -> Vector:
    return [math.tanh(x) for x in v]


def sigmoid(v: Vector) -> Vector:
    out = []
    for x in v:
        if x >= 0:
            z = math.exp(-x)
            out.append(1.0 / (1.0 + z))
        else:
            z = math.exp(x)
            out.append(z / (1.0 + z))
    return out


def softmax(v: Vector) -> Vector:
    m = max(v)
    exps = [math.exp(x - m) for x in v]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def matvec(W: Matrix, x: Vector) -> Vector:
    """행렬(out×in) · 벡터(in) → 벡터(out). 내적은 zip/sum으로."""
    return [sum(wi * xi for wi, xi in zip(row, x)) for row in W]


def add(a: Vector, b: Vector) -> Vector:
    return [ai + bi for ai, bi in zip(a, b)]


def hadamard(a: Vector, b: Vector) -> Vector:
    return [ai * bi for ai, bi in zip(a, b)]


class Linear:
    """완전연결층 y = W·x + b."""

    def __init__(self, n_in: int, n_out: int, rng: random.Random,
                 bias: float = 0.0) -> None:
        self.W = xavier(rng, n_out, n_in)
        self.b = [bias] * n_out
        self.n_in = n_in
        self.n_out = n_out

    def __call__(self, x: Vector) -> Vector:
        return add(matvec(self.W, x), self.b)


class GRUCell:
    """Gated Recurrent Unit — 직접 구현.

    시간적 상태 h를 유지하므로 (1) 부드럽고 끊김 없는 모션과 (2) 과거 맥락을 반영한
    제어가 자연스럽게 나온다. 입력 x와 이전 은닉 h를 받아 새 은닉 h'을 낸다.
    """

    def __init__(self, n_in: int, n_hidden: int, rng: random.Random) -> None:
        self.n_in = n_in
        self.h = n_hidden
        nio = n_in + n_hidden
        self.Wz = xavier(rng, n_hidden, nio)
        self.Wr = xavier(rng, n_hidden, nio)
        self.Wn = xavier(rng, n_hidden, nio)
        self.bz = [0.0] * n_hidden
        self.br = [0.0] * n_hidden
        self.bn = [0.0] * n_hidden

    def step(self, x: Vector, h: Vector) -> Vector:
        xh = x + h  # 리스트 연결(concat)
        z = sigmoid(add(matvec(self.Wz, xh), self.bz))
        r = sigmoid(add(matvec(self.Wr, xh), self.br))
        xrh = x + hadamard(r, h)
        n = tanh(add(matvec(self.Wn, xrh), self.bn))
        # h' = (1-z)⊙n + z⊙h
        return [(1.0 - zi) * ni + zi * hi for zi, ni, hi in zip(z, n, h)]
