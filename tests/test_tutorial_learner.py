"""게임 튜토리얼 자동 학습 검증."""

from broadcast_ai.cognition.tutorial_learner import TutorialLearner


def test_learns_movement_from_screen_text():
    tl = TutorialLearner()
    learned = tl.observe("WASD 키로 이동할 수 있습니다")
    assert len(learned) == 1
    assert learned[0].intent == "이동"
    assert "WASD" in learned[0].keys


def test_learns_multiple_actions_and_dedupes():
    tl = TutorialLearner()
    tl.observe("스페이스바를 눌러 점프하세요")
    tl.observe("좌클릭으로 공격, 우클릭으로 방어합니다")
    # 같은 안내문을 다시 봐도 중복 학습하지 않음.
    again = tl.observe("스페이스바를 눌러 점프하세요")
    assert again == []
    intents = {s.intent for s in tl.steps}
    assert {"점프", "공격", "방어"} <= intents


def test_knowledge_summary_and_next_practice():
    tl = TutorialLearner()
    tl.observe("WASD 이동 · 스페이스 점프 · 좌클릭 공격")
    summary = tl.knowledge_summary()
    assert "이동" in summary
    nxt = tl.next_practice()
    assert nxt is not None
    assert nxt.intent != "이해"


def test_unknown_text_is_understood_not_actioned():
    tl = TutorialLearner()
    learned = tl.observe("환영합니다! 모험을 시작하세요")
    assert learned[0].intent == "이해"
