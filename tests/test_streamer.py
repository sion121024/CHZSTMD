"""런타임 통합 검증 — 단일 모델 / 저지연 발화 / AI 모션 / FPS."""

from broadcast_ai import Config
from broadcast_ai.runtime import BroadcastStreamer


def make() -> BroadcastStreamer:
    return BroadcastStreamer(Config())


def test_single_model_drives_everything():
    s = make()
    # 대화도, 게임도, 모션도 전부 같은 하나의 에이전트.
    assert s.brain.agent is s.agent
    assert s.agent.num_params > 1000


def test_chat_produces_streaming_audio_with_latency():
    s = make()
    res = s.on_chat("하루야 안녕!")
    assert res.text.strip()
    assert len(res.audio_chunks) >= 1
    assert res.first_audio_ms > 0.0
    assert abs(res.total_audio_ms - sum(c.duration_ms for c in res.audio_chunks)) < 1e-6


def test_first_audio_before_full_text():
    s = make()
    order = []
    s.on_chat("이 게임 어때?", on_event=lambda k, d: order.append(k) if k in ("token", "first_audio") else None)
    assert "first_audio" in order
    assert order.index("first_audio") >= 1     # 토큰 일부만 보고도 음성 시작(저지연)


def test_render_frame_is_ai_driven_and_well_formed():
    s = make()
    frame = s.render_frame()
    assert set(frame.keys()) == {"head", "body_sway", "breath", "eyes", "mouth", "arms"}
    # 연속 프레임이 서로 다르다(네트워크가 매 프레임 생성).
    f2 = s.render_frame()
    assert frame != f2


def test_fps_play_runs_and_learns():
    s = make()
    r = s.play_fps(ticks=400)
    assert r.infer_hz > 0
    assert r.improved


def test_tutorial_learning_end_to_end():
    s = make()
    s.screen.script_frames(["WASD 키로 이동", "스페이스바로 점프"])
    results = s.watch_and_learn()
    assert len(results) == 2
    assert "이동" in s.brain.tutorial.knowledge_summary()
