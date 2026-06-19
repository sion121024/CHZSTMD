"""단일 통합 모델(from scratch) 검증 — 제어/모션/의도 + 온라인 조준 학습."""

from broadcast_ai.model.unified import UnifiedAgent, Observation, EMOTIONS


def test_from_scratch_single_model_has_real_params():
    a = UnifiedAgent(seed=1)
    assert a.num_params > 1000          # 실제 규모의 직접 구현 네트워크


def test_one_tick_produces_control_motion_intent_together():
    a = UnifiedAgent(seed=1)
    out = a.tick(Observation(target_visible=1.0, target_dx=0.3))
    # 같은 한 번의 forward가 셋 다 낸다(모델 1개).
    assert hasattr(out.control, "aim_dx")
    assert out.motion.values
    assert out.intent.emotion in EMOTIONS
    assert 0.0 <= out.intent.energy <= 1.0


def test_online_aim_learning_reduces_error():
    """은닉 특징에서 목표 조준으로 가는 사상을 NLMS로 학습 → 오차 감소."""
    a = UnifiedAgent(seed=2)
    desired = (0.4, -0.3)
    first_err = None
    last_err = None
    for i in range(200):
        a.tick(Observation(target_dx=desired[0], target_dy=desired[1], target_visible=1.0))
        e = a.learn_aim(desired[0], desired[1])
        if i == 0:
            first_err = e
        last_err = e
    assert last_err < first_err          # 실시간으로 조준을 학습


def test_determinism_same_seed():
    a = UnifiedAgent(seed=42)
    b = UnifiedAgent(seed=42)
    oa = a.tick(Observation(target_dx=0.2))
    ob = b.tick(Observation(target_dx=0.2))
    assert abs(oa.control.aim_dx - ob.control.aim_dx) < 1e-9


def test_reset_clears_state():
    a = UnifiedAgent(seed=3)
    for _ in range(10):
        a.tick(Observation(excite_drive=0.9))
    a.reset()
    assert all(v == 0.0 for v in a.h)
    assert all(v == 0.0 for v in a.pose)
