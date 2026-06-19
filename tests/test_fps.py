"""FPS 실시간 제어 + 온라인 조준 학습 환경 검증."""

from broadcast_ai.model.unified import UnifiedAgent
from broadcast_ai.game.fps_env import TargetRange


def test_aim_improves_over_session():
    a = UnifiedAgent(seed=11)
    env = TargetRange(a, seed=1)
    r = env.run(ticks=800)
    assert r.improved                      # 후반 조준 오차 < 초반
    assert r.late_error < r.early_error


def test_inference_is_realtime_fast():
    a = UnifiedAgent(seed=12)
    env = TargetRange(a, seed=2)
    r = env.run(ticks=400)
    # 추론 처리량이 양수이고, 일반 게임 틱레이트 이상이어야 한다.
    assert r.infer_hz > 0
    assert r.infer_hz >= 120.0             # 보수적 하한(대개 수백~천 Hz)


def test_learning_off_does_not_improve_like_learning_on():
    a1 = UnifiedAgent(seed=20)
    on = TargetRange(a1, seed=3).run(ticks=600, learn=True)
    a2 = UnifiedAgent(seed=20)
    off = TargetRange(a2, seed=3).run(ticks=600, learn=False)
    # 학습을 켰을 때 후반 오차가 더 낮다(=실제로 학습이 일한다).
    assert on.late_error <= off.late_error
