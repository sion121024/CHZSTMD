"""온디바이스 OCR.

화면 프레임에서 텍스트(튜토리얼 안내문, UI 라벨)를 읽는다. 실제로는 `pytesseract`
같은 로컬 OCR을 쓰고, 없으면 폴백 프레임에 심어둔 text_overlay를 그대로 돌려준다.
외부 OCR API는 쓰지 않는다.
"""

from __future__ import annotations

from .screen import Frame


class OCR:
    def __init__(self) -> None:
        self._engine = None
        try:
            import pytesseract  # type: ignore

            self._engine = pytesseract
        except Exception:
            self._engine = None

    @property
    def live(self) -> bool:
        return self._engine is not None

    def read(self, frame: Frame) -> str:
        """프레임에서 텍스트를 추출한다."""
        if self.live and not frame.text_overlay:
            try:
                from PIL import Image  # type: ignore  # noqa: F401

                # 실제 구현에서는 frame의 픽셀을 PIL Image로 변환 후 OCR.
                # 여기서는 인터페이스만 고정한다.
                return self._engine.image_to_string(_frame_to_image(frame), lang="kor+eng")
            except Exception:
                return ""
        # 폴백: 시뮬레이션된 화면 텍스트.
        return frame.text_overlay


def _frame_to_image(frame: Frame):  # pragma: no cover - 실제 픽셀 경로
    raise NotImplementedError("실제 캡처 픽셀 → PIL 변환은 mss/Pillow 설치 시 구현")
