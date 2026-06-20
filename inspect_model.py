"""Live2D 모델 내부 들여다보기 — 표정/파트/표식 컨트롤을 찾는 도구.

무료버전 표식(워터마크)을 끄려면 그게 어떤 '파트' 또는 '파라미터'인지 알아야 한다.
이 스크립트가 전부 나열하고, 표식으로 의심되는 후보를 표시한다.

사용법 (Windows):
    python inspect_model.py "C:\\경로\\youling폴더"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from broadcast_ai.avatar.live2d import Live2DModel  # noqa: E402


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "assets/avatar/gothic_lolita"
    m = Live2DModel.load(path)
    print("=" * 64)
    print(m.summary())
    print(f"파트 {len(m.part_ids)}개 · 표정 {len(m.expressions)}개")
    print("=" * 64)

    print("\n[표정 (.exp3.json)] — 뷰어에서 숫자키 1~0 으로 전환")
    for i, (name, f) in enumerate(m.expressions, 1):
        print(f"  {i if i < 10 else 0}: {name}   ({Path(f).name})")

    cands = m.watermark_candidates()
    print("\n[무료버전 표식(워터마크) 후보] — 이게 표식이면 자동으로 숨깁니다")
    if cands:
        for kind, pid, name in cands:
            print(f"  ★ {kind}: {pid}  ({name})")
    else:
        print("  (이름에 표식 키워드가 없음 — 아래 전체 목록에서 직접 찾아 알려주세요)")

    print("\n[파트 목록] (표식이 여기 있을 수 있음)")
    for pid in m.part_ids:
        print(f"  - {pid}  ({m.part_names.get(pid, '')})")

    print("\n[파라미터 목록]")
    for pid in m.parameter_ids:
        print(f"  - {pid}  ({m.parameter_names.get(pid, '')})")

    print("\n→ 표식이 위 후보로 안 잡히면, 파트/파라미터 목록에서 의심되는 줄을 "
          "그대로 복사해 알려주세요. 정확히 끄도록 연결해 드립니다.")


if __name__ == "__main__":
    main()
