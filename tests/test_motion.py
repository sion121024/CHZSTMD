"""AI가 생성한 모션 검증 — 고정 애니메이션이 아니라 신경망 출력."""

from broadcast_ai.model.unified import UnifiedAgent, Observation
from broadcast_ai.avatar.motion import MotionAdapter


def _run(energy=0.5, n=180, seed=3):
    a = UnifiedAgent(seed=seed)
    adapter = MotionAdapter()
    poses = []
    for _ in range(n):
        out = a.tick(Observation(excite_drive=energy))
        poses.append(adapter.to_pose(out.motion, energy=out.intent.energy))
    return poses


def test_motion_is_alive_and_non_repeating():
    """정지여도 네트워크가 계속 미세하게 움직임을 만든다(완전 정지=고정클립 아님)."""
    poses = _run()
    yaws = [p.head_yaw for p in poses]
    assert max(yaws) != min(yaws)              # 실제로 움직인다
    # 단순 주기 반복이 아님: 앞부분과 뒷부분이 동일하지 않다.
    assert yaws[:20] != yaws[-20:]


def test_motion_is_bounded():
    poses = _run(n=300)
    for p in poses:
        assert -0.5 < p.head_yaw < 0.5
        assert 0.0 <= p.breath <= 1.0
        assert 0.0 <= p.eye_blink <= 1.0


def test_energy_increases_movement_amplitude():
    calm = _run(energy=0.0, seed=5)
    hyped = _run(energy=1.0, seed=5)
    span_calm = max(p.arm_l for p in calm) - min(p.arm_l for p in calm)
    span_hyped = max(p.arm_l for p in hyped) - min(p.arm_l for p in hyped)
    assert span_hyped >= span_calm     # 에너지↑ → 제스처 진폭↑


def test_blink_channel_active():
    poses = _run(n=400)
    assert max(p.eye_blink for p in poses) > 0.0
