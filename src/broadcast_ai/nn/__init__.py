"""처음부터(from scratch) 구현한 신경망 — 순수 파이썬.

외부 ML 프레임워크(torch/transformers)도, 사전학습 가중치도 쓰지 않는다.
선형층/GRU 셀/활성함수를 직접 구현한다. 차원을 작게 유지해 CPU 순수 파이썬으로도
FPS 게임에 충분한 고주파(>240Hz) 추론이 가능하다.
"""

from .layers import Linear, GRUCell, tanh, sigmoid, softmax
from .seed import seeded_rng

__all__ = ["Linear", "GRUCell", "tanh", "sigmoid", "softmax", "seeded_rng"]
