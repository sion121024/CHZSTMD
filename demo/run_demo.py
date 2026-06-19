"""방송용 AI 버추얼 스트리머 — 엔드투엔드 데모 (재설계판).

처음부터 설계한 *단 하나의* 신경망이:
  1) FPS 게임을 실시간으로 제어하고 조준을 온라인 학습하며
  2) 아바타를 매 프레임 AI로 움직이고 (고정 애니메이션 아님)
  3) 시청자 채팅에 저지연으로 반응하고
  4) 게임 튜토리얼을 화면에서 스스로 배운다.
모두 로컬, 외부 API/사전학습 모델 없음.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from broadcast_ai import Config  # noqa: E402
from broadcast_ai.runtime import BroadcastStreamer  # noqa: E402
from broadcast_ai.model.unified import Observation  # noqa: E402


def section(title: str) -> None:
    print("\n" + "=" * 66)
    print(f"▶ {title}")
    print("=" * 66)


def main() -> None:
    s = BroadcastStreamer(Config(), seed=2025)

    section("1) 시스템 — 처음부터 설계한 단일 모델")
    print(s.status())
    print("→ 제어·모션·발화의도가 모두 같은 신경망 하나에서 나온다 (모델 분리 없음).")

    section("2) FPS 게임 — 실시간 제어 + 조준 온라인 학습")
    r = s.play_fps(ticks=1500)
    print(f"추론 처리량 : {r.infer_hz:,.0f} Hz  "
          f"→ FPS 실시간 가능: {'예 ✅' if r.fps_capable else '아니오'} "
          f"(일반 FPS 60~240Hz)")
    print(f"조준 오차   : 초반 {r.early_error:.3f} → 후반 {r.late_error:.3f}  "
          f"({'학습으로 개선 ✅' if r.improved else '변화 없음'})")
    print(f"명중        : {r.hits}회 (움직이는 타깃 중앙에 두고 발사)")
    # 막 학습한 직후의 한 마디
    reaction = s.comment(f"방금 {r.hits}번 맞췄어")
    print(f"하루> {reaction.text}")

    section("3) 아바타 — AI가 매 프레임 모션을 *생성* (고정 클립 아님)")
    print("[정지 idle — 그래도 네트워크가 끊임없이 미세하게 움직임]")
    prev = None
    for i in range(4):
        f = s.render_frame()
        same = "(직전과 동일)" if f == prev else "(매 프레임 다름)"
        prev = f
        print(f"  f{i}: head={f['head']} body_sway={f['body_sway']:+.3f} "
              f"호흡={f['breath']:.2f} 깜빡={f['eyes']['blink']:.2f} {same}")

    section("4) 시청자 채팅 — 저지연 스트리밍 반응")
    for msg in ["하루야 안녕!", "방금 에임 미쳤다 ㅋㅋ", "이 게임 어려워?"]:
        print(f"\n시청자> {msg}")
        res = s.on_chat(msg)
        print(f"하루> {res.text}")
        print(f"   ⤷ 첫 음성 {res.first_audio_ms:.0f}ms · 발화 {res.total_audio_ms:.0f}ms "
              f"· 청크 {len(res.audio_chunks)}개"
              + (f" · ⚠ {res.warnings}" if res.warnings else " · ✅ 지연 예산 내"))

    section("5) 종합 게임 — 인게임 튜토리얼만으로 학습 (외부 영상 없음)")
    # 튜토리얼 학습 전: 조작을 전혀 모른다.
    before = s.play_comprehensive()
    print(f"튜토리얼 학습 전 클리어: {before.completed}/{before.total} (조작을 모름)")

    # 게임 자신이 띄우는 튜토리얼 화면(OCR)만 보고 스스로 학습.
    print("\n[게임의 인게임 튜토리얼 팝업을 화면에서 읽는 중...]")
    s.screen.script_frames([
        "WASD 키로 이동할 수 있습니다",
        "스페이스바를 눌러 점프하세요",
        "좌클릭으로 공격, 우클릭으로 방어합니다",
        "E 키로 아이템을 줍습니다, R 키로 재장전",
        "ESC로 메뉴를 엽니다",
    ])
    for r2 in s.watch_and_learn():
        print(f"하루> {r2.text}")
    print(f"\n📚 스스로 익힌 조작: {s.brain.tutorial.knowledge_summary()}")

    # 학습 후: 종합 게임을 직접 플레이.
    after = s.play_comprehensive()
    print(f"\n인게임 튜토리얼 학습 후 클리어: {after.completed}/{after.total}")
    for o in after.objectives:
        mark = "✅" if o.success else "❌"
        print(f"  {mark} {o.name} (필요:{o.required}) — {o.detail}")
    react = s.comment(f"튜토리얼만 보고 {after.completed}개 다 깼다")
    print(f"하루> {react.text}")

    section("6) Live2D 아바타 적용 (있을 때) — AI 모션 → 실제 파라미터")
    model_dir = Path(__file__).resolve().parents[1] / "assets/avatar/gothic_lolita"
    # 폴더 안의 어떤 *.model3.json 이든 자동으로 찾는다(파일명 달라도 OK).
    if list(model_dir.glob("*.model3.json")):
        model = s.load_avatar(str(model_dir))
        print(model.summary())
        s._speaking = True; s._viseme_open = 0.7; s._energy = 0.8
        for i in range(3):
            fr = s.render_live2d_frame()
            sel = {k: fr[k] for k in ("ParamAngleX", "ParamEyeLOpen", "ParamMouthOpenY", "ParamBreath") if k in fr}
            print(f"  f{i} → {sel}")
        s._speaking = False
        n = s.export_live2d_motion("/tmp/live2d_motion.jsonl", seconds=2.0)
        print(f"AI 생성 모션 {n}프레임을 Live2D 파라미터 JSONL로 저장 → Cubism 런타임이 재생.")
    else:
        print("모델 폴더에 .model3.json 이 없습니다(유료 에셋은 git 제외). "
              "assets/avatar/gothic_lolita/ 에 모델 파일을 넣으면 자동 인식됩니다.")

    print("\n데모 완료 — 단일 모델 · AI 모션 · FPS 실시간 · 종합게임 자가학습 · Live2D 연동 · 외부 API 0건. ✅")


if __name__ == "__main__":
    main()
