"""방송용 AI 버추얼 스트리머 — 엔드투엔드 데모.

외부 API/가중치 없이도 전체 파이프라인을 시연한다:
  1) 단일 로컬 모델 + 디바이스 상태
  2) 시청자 채팅에 저지연 스트리밍 반응 (첫 음성 지연 측정)
  3) 게임 튜토리얼을 화면(OCR 시뮬)에서 스스로 학습
  4) 배운 조작으로 게임 상황 판단
  5) 사람처럼 움직이는 아바타 프레임 샘플
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from broadcast_ai import Config  # noqa: E402
from broadcast_ai.runtime import BroadcastStreamer  # noqa: E402


def section(title: str) -> None:
    print("\n" + "=" * 64)
    print(f"▶ {title}")
    print("=" * 64)


def main() -> None:
    streamer = BroadcastStreamer(Config())

    section("1) 시스템 상태 (완전 로컬 / 단일 모델)")
    print(streamer.status())
    print(f"디바이스: {streamer.device.summary()}")
    print(f"단일 모델이 8B라면 예산 적합? {streamer.device.fits_params(8.0)}")

    section("2) 시청자 채팅에 저지연 스트리밍 반응")
    for msg in ["하루야 안녕!", "이 게임 어때?", "방금 그거 개잘했다 ㅋㅋ"]:
        print(f"\n시청자> {msg}")
        tokens: list[str] = []
        res = streamer.on_chat(msg, on_event=lambda k, d: tokens.append(d) if k == "token" else None)
        print(f"하루> {res.text}")
        print(f"   첫 음성 {res.first_audio_ms:.0f}ms · 발화 {res.total_audio_ms:.0f}ms "
              f"· 청크 {len(res.audio_chunks)}개"
              + (f" · ⚠ {res.warnings}" if res.warnings else " · ✅ 지연 예산 내"))

    section("3) 게임 튜토리얼 자동 학습 (누가 안 알려줘도)")
    # 화면에 순차적으로 뜨는 튜토리얼 안내문(실전에선 OCR로 읽어옴).
    streamer.screen.script_frames([
        "WASD 키로 이동할 수 있습니다",
        "스페이스바를 눌러 점프하세요",
        "좌클릭으로 공격, 우클릭으로 방어합니다",
        "E 키로 아이템을 줍습니다 · R 키로 재장전",
    ])
    learn_results = streamer.watch_and_learn()
    for r in learn_results:
        print(f"하루> {r.text}")
    print(f"\n📚 지금까지 배운 조작: {streamer.brain.tutorial.knowledge_summary()}")
    nxt = streamer.brain.tutorial.next_practice()
    if nxt:
        print(f"🎮 가장 자신 있는 연습 대상: {nxt.as_action_hint()} (확신도 {nxt.confidence:.2f})")

    section("4) 배운 조작으로 게임 상황 판단")
    res = streamer.react_to_situation("앞에 적이 달려오고 체력은 충분함")
    print(f"하루> {res.text}")

    section("5) 사람처럼 움직이는 아바타 프레임 (말하는 중 vs 정지)")
    # 말하는 동안 몇 프레임, 그리고 idle 몇 프레임을 샘플링.
    streamer.motion.set_speaking(True, energy=0.7)
    print("[말하는 중]")
    for i in range(3):
        f = streamer.render_frame()
        print(f"  frame {i}: head={f['head']} 입벌림={f['mouth']['open']:.2f} "
              f"눈깜빡={f['eyes']['blink']:.2f} 호흡={f['breath']:.2f}")
    streamer.motion.set_speaking(False)
    print("[정지 idle — 그래도 미세하게 살아있음]")
    for i in range(3):
        f = streamer.render_frame()
        print(f"  frame {i}: head={f['head']} body_sway={f['body_sway']:.3f} "
              f"눈깜빡={f['eyes']['blink']:.2f} 호흡={f['breath']:.2f}")

    print("\n데모 완료 — 전부 로컬, 외부 API 0건. ✅")


if __name__ == "__main__":
    main()
