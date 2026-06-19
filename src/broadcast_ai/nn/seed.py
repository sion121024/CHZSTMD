"""결정론적 가중치 초기화용 난수.

같은 seed면 항상 같은 네트워크가 만들어져 동작이 재현 가능하다.
"""

from __future__ import annotations

import math
import random


def seeded_rng(seed: int) -> random.Random:
    return random.Random(seed)


def xavier(rng: random.Random, n_out: int, n_in: int) -> list[list[float]]:
    """Xavier/Glorot 균등 초기화. 신호가 층을 지나며 폭주/소멸하지 않게 한다."""
    limit = math.sqrt(6.0 / (n_in + n_out))
    return [[rng.uniform(-limit, limit) for _ in range(n_in)] for _ in range(n_out)]
