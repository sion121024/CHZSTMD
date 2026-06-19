"""노래 합성(Singing Voice Synthesis) — 음정·박자·비브라토 직접 제어.

말하기 TTS와 다른 경로다. TTS는 프로소디를 자동 생성해 멜로디를 넣을 수 없지만,
여기서는 **악보**(음정 + 길이 + 가사 모음)를 받아 음성을 합성한다. 음정(f0),
박자(bpm·음표 길이), 비브라토, 모음 음색(포먼트)을 전부 코드로 제어한다.

소스-필터 방식: 성대 진동(하모닉 소스)을 모음별 포먼트로 필터링해 "아/에/이/오/우"
음색을 만든다. 전부 로컬 DSP라 노트북 CPU에서 돈다(외부 API 없음). 신경망 브레인이
"부를지/감정/멜로디"를 정할 수 있고, 실제 오디오 렌더는 이 DSP가 담당한다
(보코더가 TTS의 렌더러인 것과 같은 관계).

numpy가 있으면 벡터화로 빠르게, 없으면 순수 파이썬으로 동작한다.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass, field

try:
    import numpy as _np
except Exception:  # pragma: no cover - numpy 없을 때
    _np = None

# 모음별 포먼트 (F1, F2, F3) Hz — 음색을 결정.
VOWEL_FORMANTS = {
    "a": (800, 1150, 2600), "아": (800, 1150, 2600), "라": (800, 1150, 2600),
    "e": (500, 1800, 2550), "에": (500, 1800, 2550),
    "i": (300, 2300, 3000), "이": (300, 2300, 3000),
    "o": (500, 900, 2500), "오": (500, 900, 2500),
    "u": (350, 800, 2400), "우": (350, 800, 2400),
}


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


@dataclass
class Note:
    """음표 하나. midi=음정(60=가운데 도), beats=길이(4분음표=1), lyric=가사."""
    midi: int          # 0 또는 음수면 쉼표
    beats: float = 1.0
    lyric: str = "아"

    @property
    def is_rest(self) -> bool:
        return self.midi <= 0


@dataclass
class Score:
    bpm: float = 120.0
    notes: list[Note] = field(default_factory=list)

    def duration_sec(self) -> float:
        return sum(n.beats for n in self.notes) * 60.0 / self.bpm


@dataclass
class SingResult:
    samples: list  # float -1..1 (numpy array 또는 list)
    sample_rate: int
    lipsync: list  # [(t_sec, vowel, mouth_open), ...] — 아바타 립싱크용


class SingingSynth:
    def __init__(self, sample_rate: int = 22050, vibrato_rate: float = 5.5,
                 vibrato_depth: float = 0.006, max_harmonics: int = 48) -> None:
        self.sr = sample_rate
        self.vib_rate = vibrato_rate
        self.vib_depth = vibrato_depth      # 음정 흔들림(±0.6%)
        self.maxh = max_harmonics

    def _vowel(self, lyric: str) -> tuple[float, float, float]:
        for ch in lyric:
            if ch in VOWEL_FORMANTS:
                return VOWEL_FORMANTS[ch]
        return VOWEL_FORMANTS["아"]

    def render(self, score: Score) -> SingResult:
        if _np is None:
            return self._render_py(score)
        return self._render_np(score)

    # ---- numpy 경로(빠름) ----------------------------------------------------
    def _render_np(self, score: Score) -> SingResult:
        np = _np
        sr = self.sr
        out = []
        lips = []
        t_cursor = 0.0
        for note in score.notes:
            dur = note.beats * 60.0 / score.bpm
            n = max(1, int(dur * sr))
            t = np.arange(n) / sr
            if note.is_rest:
                out.append(np.zeros(n))
                lips.append((round(t_cursor, 3), "_", 0.0))
                t_cursor += dur
                continue

            f0 = midi_to_hz(note.midi)
            vib = 1.0 + self.vib_depth * np.sin(2 * np.pi * self.vib_rate * t) * np.minimum(1.0, t / 0.25)
            inst_f0 = f0 * vib
            phase = 2 * np.pi * np.cumsum(inst_f0) / sr

            F1, F2, F3 = self._vowel(note.lyric)
            sig = np.zeros(n)
            kmax = min(self.maxh, int((sr / 2) / f0))
            for k in range(1, kmax + 1):
                fk = k * f0
                w = (1.0 * math.exp(-((fk - F1) / 90.0) ** 2)
                     + 0.7 * math.exp(-((fk - F2) / 110.0) ** 2)
                     + 0.4 * math.exp(-((fk - F3) / 170.0) ** 2)
                     + 0.15 / k)            # 저차 하모닉 기본 성분
                if w < 1e-4:
                    continue
                sig += w * np.sin(k * phase)

            env = self._adsr(n, sr)
            peak = np.max(np.abs(sig)) or 1.0
            out.append((sig / peak) * env * 0.9)
            lips.append((round(t_cursor, 3), note.lyric, 0.6 + 0.3 * (note.midi % 12) / 12))
            t_cursor += dur

        samples = np.concatenate(out) if out else np.zeros(1)
        m = np.max(np.abs(samples)) or 1.0
        samples = (samples / m) * 0.95
        return SingResult(samples=samples, sample_rate=sr, lipsync=lips)

    def _adsr(self, n: int, sr: int):
        np = _np
        env = np.ones(n)
        a = min(n, int(0.02 * sr))
        r = min(n, int(0.05 * sr))
        if a > 0:
            env[:a] = np.linspace(0, 1, a)
        if r > 0:
            env[-r:] = np.linspace(1, 0, r)
        return env

    # ---- 순수 파이썬 경로(numpy 없을 때) -------------------------------------
    def _render_py(self, score: Score) -> SingResult:  # pragma: no cover
        sr = self.sr
        out: list[float] = []
        lips = []
        t_cursor = 0.0
        for note in score.notes:
            dur = note.beats * 60.0 / score.bpm
            n = max(1, int(dur * sr))
            if note.is_rest:
                out.extend([0.0] * n)
                t_cursor += dur
                continue
            f0 = midi_to_hz(note.midi)
            F1, F2, F3 = self._vowel(note.lyric)
            kmax = min(self.maxh, int((sr / 2) / f0))
            phase = 0.0
            buf = [0.0] * n
            for i in range(n):
                tt = i / sr
                vib = 1.0 + self.vib_depth * math.sin(2 * math.pi * self.vib_rate * tt)
                phase += 2 * math.pi * f0 * vib / sr
                s = 0.0
                for k in range(1, kmax + 1):
                    fk = k * f0
                    w = (math.exp(-((fk - F1) / 90.0) ** 2)
                         + 0.7 * math.exp(-((fk - F2) / 110.0) ** 2)
                         + 0.4 * math.exp(-((fk - F3) / 170.0) ** 2) + 0.15 / k)
                    if w < 1e-4:
                        continue
                    s += w * math.sin(k * phase)
                buf[i] = s
            peak = max(abs(x) for x in buf) or 1.0
            a = int(0.02 * sr); r = int(0.05 * sr)
            for i in range(n):
                e = 1.0
                if i < a:
                    e = i / a
                elif i > n - r:
                    e = (n - i) / r
                buf[i] = buf[i] / peak * e * 0.9
            out.extend(buf)
            lips.append((round(t_cursor, 3), note.lyric, 0.7))
            t_cursor += dur
        m = max((abs(x) for x in out), default=1.0) or 1.0
        out = [x / m * 0.95 for x in out]
        return SingResult(samples=out, sample_rate=sr, lipsync=lips)

    # ---- WAV 저장 ------------------------------------------------------------
    def to_wav(self, result: SingResult, path: str) -> None:
        with wave.open(path, "w") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(result.sample_rate)
            if _np is not None and not isinstance(result.samples, list):
                data = (result.samples * 32767).astype("<i2").tobytes()
            else:
                import struct
                data = b"".join(struct.pack("<h", int(max(-1, min(1, s)) * 32767))
                                for s in result.samples)
            w.writeframes(data)
