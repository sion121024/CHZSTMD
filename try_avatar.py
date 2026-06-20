"""youling 같은 Live2D 모델을 인식/연결해 보는 가장 간단한 실행기.

코드를 수정할 필요 없습니다. 모델 폴더(또는 .model3.json) 경로만 인자로 주세요.

사용법 (Windows CMD, 이 저장소 폴더에서):
    python try_avatar.py "C:\\경로\\youling폴더"
  또는 파일을 직접:
    python try_avatar.py "C:\\경로\\youling.model3.json"

  (탐색기에서 폴더 주소창을 복사하거나, 파일을 Shift+우클릭 → "경로로 복사")
"""

import sys
from pathlib import Path

# src 경로 자동 추가 — PYTHONPATH 설정조차 필요 없게.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from broadcast_ai.runtime import BroadcastStreamer  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print("사용법: python try_avatar.py \"<youling 폴더 또는 .model3.json 경로>\"")
        print("예시:  python try_avatar.py \"C:\\Users\\drmak\\...\\youling\"")
        raise SystemExit(1)

    path = sys.argv[1]
    s = BroadcastStreamer()
    try:
        model = s.load_avatar(path)
    except Exception as e:
        print("❌ 인식 실패:\n", e)
        raise SystemExit(2)

    print("✅ 인식 성공:", model.summary())
    # AI가 매 프레임 생성하는 실제 Live2D 파라미터 예시
    frame = s.render_live2d_frame()
    sample = {k: frame[k] for k in list(frame)[:8]}
    print("AI가 만든 파라미터(예시):", sample)
    print("\n→ 이 값들을 Cubism 런타임의 setParameterValueById 에 넣으면 youling이 움직입니다.")


if __name__ == "__main__":
    main()
