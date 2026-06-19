"""화면 캡처 소스.

게임 화면을 프레임으로 가져온다. 실제 환경에서는 `mss`로 모니터/창을 캡처하고,
없으면(헤드리스) 스크립트로 주입한 가짜 프레임을 내보내 파이프라인을 검증한다.
어느 쪽이든 외부 네트워크는 쓰지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass
class Frame:
    index: int
    # 실제로는 픽셀 배열. 폴백에서는 화면에 보이는 텍스트(튜토리얼 안내문)를 직접 담아
    # OCR 단계를 시뮬레이션한다.
    text_overlay: str = ""
    width: int = 1920
    height: int = 1080


class ScreenSource:
    def __init__(self) -> None:
        self._sct = None
        self._scripted: list[str] = []
        self._i = 0
        try:
            import mss  # type: ignore

            self._sct = mss.mss()
        except Exception:
            self._sct = None

    @property
    def live(self) -> bool:
        return self._sct is not None

    def script_frames(self, overlays: list[str]) -> None:
        """헤드리스 테스트/시연용: 프레임마다 보일 텍스트를 미리 넣는다."""
        self._scripted = list(overlays)
        self._i = 0

    def grab(self) -> Frame | None:
        """다음 프레임을 가져온다. 더 없으면 None."""
        if self.live:
            shot = self._sct.grab(self._sct.monitors[0])
            frame = Frame(index=self._i, width=shot.width, height=shot.height)
            self._i += 1
            return frame
        if self._i < len(self._scripted):
            frame = Frame(index=self._i, text_overlay=self._scripted[self._i])
            self._i += 1
            return frame
        return None

    def frames(self) -> Iterator[Frame]:
        while True:
            f = self.grab()
            if f is None:
                return
            yield f
