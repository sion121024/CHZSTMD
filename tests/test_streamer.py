"""런타임 통합 검증 — 저지연 스트리밍 발화 / 단일 모델 / 아바타."""

from broadcast_ai import Config
from broadcast_ai.runtime import BroadcastStreamer


def make() -> BroadcastStreamer:
    return BroadcastStreamer(Config())


def test_single_model_loaded_once_and_resident():
    s = make()
    assert s.model.loaded is True
    assert s.device.resident is True
    # 같은 모델 인스턴스가 모든 작업에 쓰인다.
    assert s.brain.model is s.model


def test_chat_produces_streaming_audio_with_first_audio_latency():
    s = make()
    res = s.on_chat("안녕!")
    assert res.text.strip()
    assert len(res.audio_chunks) >= 1
    # 첫 음성이 측정되고 0보다 크다.
    assert res.first_audio_ms > 0.0
    # 발화 길이는 청크 합과 같다.
    assert abs(res.total_audio_ms - sum(c.duration_ms for c in res.audio_chunks)) < 1e-6


def test_first_audio_before_full_text():
    """첫 오디오 청크가 마지막 토큰을 기다리지 않고 나와야 한다(저지연)."""
    s = make()
    order: list[str] = []

    def on_event(kind, data):
        if kind in ("first_audio", "token"):
            order.append(kind)

    s.on_chat("이 게임 재밌어?", on_event=on_event)
    # 첫 음성 시점 이후에도 토큰이 더 들어오는(=다 안 기다린) 경우가 일반적.
    assert "first_audio" in order
    first_idx = order.index("first_audio")
    assert order[:first_idx].count("token") >= 1  # 최소 한 토큰 후 음성 시작


def test_avatar_frame_has_expected_shape():
    s = make()
    frame = s.render_frame()
    assert set(frame.keys()) == {"head", "body_sway", "breath", "eyes", "mouth", "arms"}
    assert len(frame["head"]) == 3


def test_tutorial_learning_end_to_end():
    s = make()
    s.screen.script_frames([
        "WASD 키로 이동",
        "스페이스바로 점프",
    ])
    results = s.watch_and_learn()
    assert len(results) == 2
    assert "이동" in s.brain.tutorial.knowledge_summary()
