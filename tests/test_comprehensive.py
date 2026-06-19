"""종합 게임을 인게임 튜토리얼만으로 학습해 플레이하는지 검증."""

from broadcast_ai.runtime import BroadcastStreamer


def _learn_full_tutorial(s: BroadcastStreamer) -> None:
    s.screen.script_frames([
        "WASD 키로 이동",
        "스페이스바로 점프",
        "좌클릭 공격, 우클릭 방어",
        "E로 상호작용, R로 재장전",
        "ESC로 메뉴",
    ])
    s.watch_and_learn()


def test_cannot_play_before_learning():
    """튜토리얼을 안 봤으면 조작을 몰라 아무 목표도 못 깬다."""
    s = BroadcastStreamer(seed=1)
    r = s.play_comprehensive()
    assert r.completed == 0
    assert all(o.learned is False for o in r.objectives)


def test_learns_from_ingame_tutorial_only_then_clears():
    """외부 영상 없이 인게임 튜토리얼만 보고 종합 게임을 클리어한다."""
    s = BroadcastStreamer(seed=1)
    _learn_full_tutorial(s)
    r = s.play_comprehensive()
    assert r.completed == r.total          # 전부 클리어
    assert r.clear_rate == 1.0


def test_partial_tutorial_limits_capability():
    """일부만 배우면 그만큼만 할 수 있다(배운 만큼만 플레이)."""
    s = BroadcastStreamer(seed=1)
    s.screen.script_frames(["WASD 키로 이동", "스페이스바로 점프"])
    s.watch_and_learn()
    r = s.play_comprehensive()
    learned_intents = {o.required for o in r.objectives if o.success}
    assert "이동" in learned_intents
    assert "점프" in learned_intents
    assert "메뉴" not in learned_intents   # 안 배운 건 못 함


def test_combat_uses_single_model_aim_learning():
    """전투 목표는 같은 단일 신경망의 온라인 조준 학습으로 수행된다."""
    s = BroadcastStreamer(seed=2)
    _learn_full_tutorial(s)
    r = s.play_comprehensive()
    combat = next(o for o in r.objectives if o.required == "공격")
    assert combat.learned
    assert "조준" in combat.detail
