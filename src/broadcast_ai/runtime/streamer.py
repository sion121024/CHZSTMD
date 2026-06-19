"""BroadcastStreamer — 모든 조각을 묶는 방송 AI 런타임.

한 번의 반응 흐름
-----------------
입력(채팅/게임상황/화면텍스트)
  → Brain.스트리밍 생성 (단일 로컬 모델)
  → 토큰이 나오는 즉시 StreamingTTS로 음성 청크 합성  ← 첫 음성까지의 지연(TTFA) 측정
  → 각 청크/조각으로 LipSync viseme 갱신
  → MotionSynth가 idle/말하기 모션을 얹어 매 프레임 Pose 생성
  → Rig → 렌더러로 송출

핵심: 토큰을 전부 기다렸다가 말하지 않는다. 첫 조각이 나오자마자 입을 열어
"말이 빠르고 사람 같은" 반응을 만든다. 전부 로컬, 외부 API 없음.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

from ..config import Config
from ..core.clock import LatencyBudget, now_ms
from ..gpu.device import detect_device
from ..model.local_model import LocalModel
from ..cognition.brain import Brain
from ..cognition.persona import Persona
from ..avatar.rig import Rig, Pose
from ..avatar.motion import MotionSynth
from ..avatar.lipsync import LipSync
from ..speech.tts import StreamingTTS, AudioChunk
from ..perception.screen import ScreenSource
from ..perception.ocr import OCR


@dataclass
class SpeakResult:
    text: str
    audio_chunks: list[AudioChunk] = field(default_factory=list)
    first_audio_ms: float = 0.0       # 입력~첫 음성까지(TTFA)
    total_audio_ms: float = 0.0       # 합성된 음성 총 길이
    warnings: list[str] = field(default_factory=list)


class BroadcastStreamer:
    def __init__(self, config: Config | None = None,
                 persona: Persona | None = None) -> None:
        self.cfg = config or Config()
        self.device = detect_device(self.cfg.gpu, self.cfg.model)

        # 단일 모델: 한 번 적재해 상주.
        self.model = LocalModel(self.cfg.model, self.device)
        self.model.load()

        self.brain = Brain(self.model, persona)
        self.motion = MotionSynth(fps=self.cfg.avatar.fps,
                                  blink_per_min=self.cfg.avatar.blink_per_min)
        self.lipsync = LipSync()
        self.tts = StreamingTTS(self.cfg.speech)
        self.rig = Rig()
        self.screen = ScreenSource()
        self.ocr = OCR()

        # 현재 입모양(립싱크가 갱신, 렌더 프레임에 합성).
        self._cur_viseme_open = 0.0
        self._cur_viseme_wide = 0.0

    # ---- 상태 요약 ------------------------------------------------------------
    def status(self) -> str:
        live = []
        live.append(self.model.info())
        live.append(f"TTS={'실음성' if self.tts.live else '폴백엔벌로프'}")
        live.append(f"OCR={'실OCR' if self.ocr.live else '시뮬'}")
        live.append(f"화면캡처={'라이브' if self.screen.live else '스크립트'}")
        return " | ".join(live)

    # ---- 핵심: 스트리밍 발화 --------------------------------------------------
    def speak(self, token_stream: Iterator[str],
              on_event: Callable[[str, object], None] | None = None) -> SpeakResult:
        """모델 토큰 스트림을 소비하며 즉시 음성/립싱크/모션을 구동한다."""
        budget = LatencyBudget(
            target_first_audio_ms=self.cfg.latency.target_first_audio_ms,
            target_think_ms=self.cfg.latency.target_think_ms,
        )
        self.motion.set_speaking(True, energy=0.5)
        self.lipsync.reset()

        result = SpeakResult(text="")
        first_audio_marked = False
        think = budget.start("think")

        def emit(kind: str, data: object) -> None:
            if on_event:
                on_event(kind, data)

        for piece in token_stream:
            result.text += piece
            emit("token", piece)
            # 텍스트 조각 → 입모양(오디오 청크 전에도 입이 먼저 반응).
            v = self.lipsync.from_text_piece(piece)
            self._cur_viseme_open, self._cur_viseme_wide = v.mouth_open, v.mouth_wide

            for chunk in self.tts.feed(piece):
                if not first_audio_marked:
                    budget.end(think)
                    result.first_audio_ms = budget.mark_first_audio()
                    first_audio_marked = True
                    emit("first_audio", result.first_audio_ms)
                result.audio_chunks.append(chunk)
                result.total_audio_ms += chunk.duration_ms
                # 오디오 진폭으로 입 벌림을 다시 정교화.
                av = self.lipsync.from_audio_amplitude(chunk.amplitude)
                self._cur_viseme_open = av.mouth_open
                emit("audio", chunk)

        # 남은 버퍼 flush.
        for chunk in self.tts.flush():
            if not first_audio_marked:
                budget.end(think)
                result.first_audio_ms = budget.mark_first_audio()
                first_audio_marked = True
                emit("first_audio", result.first_audio_ms)
            result.audio_chunks.append(chunk)
            result.total_audio_ms += chunk.duration_ms
            emit("audio", chunk)

        self.motion.set_speaking(False)
        self.lipsync.reset()
        self._cur_viseme_open = 0.0
        self._cur_viseme_wide = 0.0
        result.warnings = budget.over_budget()
        return result

    # ---- 고수준 행동 ----------------------------------------------------------
    def on_chat(self, viewer_text: str, **kw) -> SpeakResult:
        """시청자 채팅에 반응해 말한다."""
        return self.speak(self.brain.respond_chat(viewer_text), **kw)

    def react_to_situation(self, situation: str, **kw) -> SpeakResult:
        """게임 상황을 보고 다음 행동을 말로 결정한다."""
        return self.speak(self.brain.decide_game_action(situation), **kw)

    def study_screen_text(self, screen_text: str, **kw) -> SpeakResult:
        """화면 튜토리얼 텍스트를 스스로 학습하고 소감을 말한다."""
        return self.speak(self.brain.learn_from_screen(screen_text), **kw)

    def watch_and_learn(self, **kw) -> list[SpeakResult]:
        """화면을 프레임 단위로 보며 튜토리얼을 자동 학습한다(누가 안 알려줘도)."""
        results: list[SpeakResult] = []
        for frame in self.screen.frames():
            text = self.ocr.read(frame)
            if text.strip():
                results.append(self.study_screen_text(text, **kw))
        return results

    # ---- 아바타 렌더 프레임 ---------------------------------------------------
    def render_frame(self) -> dict:
        """현재 시점의 아바타 포즈 한 프레임. fps에 맞춰 호출."""
        pose: Pose = self.motion.tick()
        # 립싱크로 입모양 덮어쓰기(말 중이면 입이 음성과 맞물림).
        pose.mouth_open = max(pose.mouth_open, self._cur_viseme_open)
        pose.mouth_wide = max(pose.mouth_wide, self._cur_viseme_wide)
        self.rig.apply(pose)
        return self.rig.to_render_frame()


# --------------------------------------------------------------------------- #
def main() -> None:  # pragma: no cover - CLI 진입점
    """`broadcast-ai` 콘솔 스크립트. 간단한 상호작용 데모."""
    from .. import __version__

    streamer = BroadcastStreamer()
    print(f"방송용 AI 버추얼 스트리머 v{__version__}")
    print(streamer.status())
    print("-" * 60)
    print("채팅을 입력하면 AI가 반응합니다. (빈 줄/Ctrl-D 종료)")
    try:
        while True:
            try:
                msg = input("시청자> ").strip()
            except EOFError:
                break
            if not msg:
                break
            res = streamer.on_chat(msg)
            print(f"{streamer.brain.persona.name}> {res.text}")
            print(f"   ⤷ 첫 음성 {res.first_audio_ms:.0f}ms · "
                  f"발화 {res.total_audio_ms:.0f}ms · 청크 {len(res.audio_chunks)}개")
    except KeyboardInterrupt:
        pass
    print("\n방송 종료. 수고했어!")


if __name__ == "__main__":  # pragma: no cover
    main()
