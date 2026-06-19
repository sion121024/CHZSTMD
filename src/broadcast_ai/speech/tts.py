"""저지연 스트리밍 TTS.

"말이 빠르다"의 핵심. 모델이 토큰을 흘리는 즉시, 문장 끝을 기다리지 않고 짧은
글자 청크가 모이면 바로 음성으로 합성한다. 첫 음성(TTFA)이 빨리 나가야 사람이
"반응이 빠르다"고 느낀다.

실제로는 Piper/XTTS 같은 온디바이스 TTS를 붙이고, 없으면 진폭 엔벌로프만 만들어
립싱크/타이밍을 검증한다. 외부 음성 API는 쓰지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from ..config import SpeechConfig


@dataclass
class AudioChunk:
    """합성된 음성 한 조각."""

    text: str
    duration_ms: float
    # 립싱크 구동용 진폭 엔벌로프(0..1). 실제 PCM 대신/함께 제공.
    amplitude: float
    sample_rate: int
    pcm: bytes = b""  # 실제 백엔드가 있으면 PCM이 채워진다.


class StreamingTTS:
    def __init__(self, cfg: SpeechConfig) -> None:
        self.cfg = cfg
        self._backend = self._load_backend()
        self._buf = ""

    def _load_backend(self):
        try:
            from piper import PiperVoice  # type: ignore  # noqa: F401

            # 실제 적재는 음성 모델 경로가 필요. 여기서는 인터페이스만 둔다.
            return None
        except Exception:
            return None

    @property
    def live(self) -> bool:
        return self._backend is not None

    def feed(self, text_piece: str) -> Iterator[AudioChunk]:
        """모델이 흘린 텍스트 조각을 받아, 청크가 차면 즉시 음성을 내보낸다."""
        self._buf += text_piece
        # 청크 경계: 글자 수가 차거나 문장부호가 나오면 끊어서 바로 말한다.
        while self._should_flush():
            cut = self._cut_point()
            chunk_text, self._buf = self._buf[:cut], self._buf[cut:]
            if chunk_text.strip():
                yield self._synthesize(chunk_text)

    def flush(self) -> Iterator[AudioChunk]:
        """발화 끝: 버퍼에 남은 텍스트를 모두 합성한다."""
        if self._buf.strip():
            yield self._synthesize(self._buf)
        self._buf = ""

    def _should_flush(self) -> bool:
        if len(self._buf) >= self.cfg.chunk_chars:
            return True
        return any(p in self._buf for p in ".!?…\n,~")

    def _cut_point(self) -> int:
        # 문장부호 위치 우선, 없으면 청크 길이.
        for i, ch in enumerate(self._buf):
            if ch in ".!?…\n":
                return i + 1
        for i, ch in enumerate(self._buf):
            if ch in ",~" and i >= 6:
                return i + 1
        return min(len(self._buf), self.cfg.chunk_chars)

    def _synthesize(self, text: str) -> AudioChunk:
        if self.live:
            return self._synthesize_real(text)
        return self._synthesize_fallback(text)

    def _synthesize_real(self, text: str) -> AudioChunk:  # pragma: no cover - 백엔드 필요
        # PiperVoice.synthesize(text) → PCM. 진폭 엔벌로프 계산 후 반환.
        raise NotImplementedError("Piper/XTTS 음성 모델 적재 시 구현")

    def _synthesize_fallback(self, text: str) -> AudioChunk:
        # 글자 수 기반으로 발화 길이를 추정(말속도 반영). 진폭은 모음 비율로 근사.
        n = max(1, len(text.strip()))
        base_ms = n * 75.0 / max(self.cfg.speaking_rate, 0.1)  # 글자당 ~75ms
        vowels = sum(1 for c in text if c in "아야어여오요우유으이aeiouAEIOU")
        amp = 0.35 + 0.5 * min(1.0, vowels / n)
        return AudioChunk(
            text=text,
            duration_ms=round(base_ms, 1),
            amplitude=round(amp, 3),
            sample_rate=self.cfg.sample_rate,
        )
