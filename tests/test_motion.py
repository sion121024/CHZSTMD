"""사람 같은 모션 합성 검증."""

from broadcast_ai.avatar.motion import MotionSynth


def test_idle_motion_is_never_perfectly_still():
    """정지 상태여도 머리/몸이 미세하게 계속 움직여야 한다(로봇 방지)."""
    m = MotionSynth(fps=60, seed=1)
    yaws = []
    for _ in range(120):  # 2초
        p = m.tick()
        yaws.append(p.head_yaw)
    # 값이 변하고, 합리적 범위 안.
    assert max(yaws) != min(yaws)
    assert all(abs(y) < 0.3 for y in yaws)


def test_breathing_oscillates_in_range():
    m = MotionSynth(fps=60, seed=2)
    breaths = [m.tick().breath for _ in range(180)]
    assert min(breaths) >= 0.0 and max(breaths) <= 1.0
    assert max(breaths) - min(breaths) > 0.3  # 실제로 숨쉬어야


def test_blinking_happens():
    """충분한 시간 동안 적어도 한 번은 깜빡여야 한다."""
    m = MotionSynth(fps=60, blink_per_min=30.0, seed=3)
    max_blink = 0.0
    for _ in range(60 * 10):  # 10초
        max_blink = max(max_blink, m.tick().eye_blink)
    assert max_blink > 0.5


def test_speaking_adds_gesture_energy():
    m = MotionSynth(fps=60, seed=4)
    m.set_speaking(False)
    idle = [abs(m.tick().arm_l) for _ in range(120)]
    m.set_speaking(True, energy=1.0)
    talk = [abs(m.tick().arm_l) for _ in range(120)]
    assert max(talk) > max(idle)  # 말할 때 손제스처가 커진다
