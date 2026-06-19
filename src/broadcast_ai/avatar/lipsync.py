"""립싱크.

발화 텍스트/오디오를 입모양(viseme)으로 변환해 입이 말과 맞아 떨어지게 한다.
오디오 진폭이 있으면 그걸로 입 벌림을 구동하고, 없으면 글자(자모)에서 모음을 뽑아
입모양을 만든다. 둘 다 로컬 계산이다.
"""

from __future__ import annotations

from dataclasses import dataclass


# 한글 모음/영문 모음 → (입 벌림, 입 너비). 아=크게벌림, 이=넓게, 우=오므림.
_VOWEL_SHAPES = {
    "ㅏ": (1.0, 0.3), "ㅑ": (1.0, 0.3), "ㅓ": (0.8, 0.2), "ㅕ": (0.8, 0.2),
    "ㅗ": (0.6, 0.0), "ㅛ": (0.6, 0.0), "ㅜ": (0.5, 0.0), "ㅠ": (0.5, 0.0),
    "ㅡ": (0.3, 0.6), "ㅣ": (0.3, 0.9), "ㅐ": (0.7, 0.7), "ㅔ": (0.6, 0.7),
    "a": (1.0, 0.3), "e": (0.6, 0.7), "i": (0.3, 0.9), "o": (0.6, 0.0), "u": (0.5, 0.0),
}

# 한글 호환 자모 모음 추출용(완성형 음절 → 종성/중성 분해는 단순화).
_HANGUL_BASE = 0xAC00
_MEDIAL = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"


@dataclass
class Viseme:
    mouth_open: float
    mouth_wide: float


def _vowel_of_syllable(ch: str) -> str | None:
    code = ord(ch)
    if _HANGUL_BASE <= code <= 0xD7A3:
        medial_idx = ((code - _HANGUL_BASE) % (21 * 28)) // 28
        v = _MEDIAL[medial_idx]
        # 복모음은 대표 단모음으로 근사.
        return {"ㅘ": "ㅏ", "ㅙ": "ㅐ", "ㅚ": "ㅗ", "ㅝ": "ㅓ",
                "ㅞ": "ㅔ", "ㅟ": "ㅣ", "ㅢ": "ㅡ", "ㅒ": "ㅐ", "ㅖ": "ㅔ"}.get(v, v)
    low = ch.lower()
    return low if low in "aeiou" else None


class LipSync:
    def __init__(self, smoothing: float = 0.45) -> None:
        self.smoothing = smoothing
        self._open = 0.0
        self._wide = 0.0

    def reset(self) -> None:
        self._open = 0.0
        self._wide = 0.0

    def from_text_piece(self, piece: str) -> Viseme:
        """방금 나온 텍스트 조각의 대표 모음으로 입모양을 만든다(스트리밍용)."""
        target_open, target_wide = 0.0, 0.0
        for ch in piece:
            v = _vowel_of_syllable(ch)
            if v and v in _VOWEL_SHAPES:
                target_open, target_wide = _VOWEL_SHAPES[v]
        return self._smooth(target_open, target_wide)

    def from_audio_amplitude(self, amplitude: float) -> Viseme:
        """오디오 청크의 진폭(0..1)으로 입 벌림을 구동한다."""
        target_open = max(0.0, min(1.0, amplitude))
        return self._smooth(target_open, self._wide * 0.6)

    def rest(self) -> Viseme:
        """무발화 상태로 입을 천천히 닫는다."""
        return self._smooth(0.0, 0.0)

    def _smooth(self, target_open: float, target_wide: float) -> Viseme:
        s = self.smoothing
        self._open = self._open * (1 - s) + target_open * s
        self._wide = self._wide * (1 - s) + target_wide * s
        return Viseme(mouth_open=round(self._open, 4), mouth_wide=round(self._wide, 4))
