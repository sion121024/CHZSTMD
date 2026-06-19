"""BroadcastStreamer — 단일 통합 모델을 중심으로 모든 것을 묶는 런타임.

하나의 UnifiedAgent가
  * 매 프레임 아바타 모션을 *생성*하고 (고정 애니메이션 아님)
  * FPS 게임 제어를 실시간으로 출력/학습하며
  * 발화 의도(감정·에너지·말하기)를 결정한다.

제어 루프(빠름, 매 프레임)와 발화(느림, 문장 단위)는 분리돼 있어, 말하는 중에도
게임 반응이 끊기지 않는다 — FPS 실시간성의 핵심.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

from ..config import Config
from ..core.clock import LatencyBudget
from ..gpu.device import detect_device
from ..model.unified import UnifiedAgent, Observation, AgentOutput, Control
from ..cognition.brain import Brain
from ..cognition.persona import Persona
from ..avatar.rig import Rig, Pose
from ..avatar.motion import MotionAdapter
from ..avatar.lipsync import LipSync
from ..avatar.live2d import Live2DModel, Live2DAvatar
from ..speech.tts import StreamingTTS, AudioChunk
from ..perception.screen import ScreenSource
from ..perception.ocr import OCR
from ..game.fps_env import TargetRange, FPSResult
from ..game.comprehensive import ComprehensiveGame, GameResult


@dataclass
class SpeakResult:
    text: str
    audio_chunks: list[AudioChunk] = field(default_factory=list)
    first_audio_ms: float = 0.0
    total_audio_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)


class BroadcastStreamer:
    def __init__(self, config: Config | None = None,
                 persona: Persona | None = None, seed: int = 1234) -> None:
        self.cfg = config or Config()
        self.device = detect_device(self.cfg.gpu, self.cfg.model)

        # 처음부터 설계한 단일 모델 — 한 번 만들어 상주.
        self.agent = UnifiedAgent(seed=seed)

        self.brain = Brain(self.agent, persona)
        self.adapter = MotionAdapter()
        self.lipsync = LipSync()
        self.tts = StreamingTTS(self.cfg.speech)
        self.rig = Rig()
        self.screen = ScreenSource()
        self.ocr = OCR()

        self._speaking = False
        self._energy = 0.2
        self._viseme_open = 0.0
        self._viseme_wide = 0.0

        # Live2D 아바타(선택): load_avatar()로 실제 모델을 연결한다.
        self.live2d_model: Live2DModel | None = None
        self.live2d: Live2DAvatar | None = None

    # ---- 상태 ----------------------------------------------------------------
    def status(self) -> str:
        av = self.live2d_model.summary() if self.live2d_model else "기본 리그(렌더러 미연결)"
        return (
            f"단일모델: from-scratch 신경망, 파라미터 {self.agent.num_params:,}개 "
            f"(사전학습/외부 모델 없음) | "
            f"{self.device.summary()} | "
            f"TTS={'실음성' if self.tts.live else '폴백'} | "
            f"OCR={'실OCR' if self.ocr.live else '시뮬'} | "
            f"아바타: {av}"
        )

    # ---- Live2D 아바타 연결 --------------------------------------------------
    def load_avatar(self, model3_path: str) -> Live2DModel:
        """사용자 Live2D 모델(.model3.json)을 연결한다. 이후 AI 모션이 이 모델의
        실제 파라미터를 구동한다."""
        self.live2d_model = Live2DModel.load(model3_path)
        self.live2d = Live2DAvatar(self.live2d_model)
        return self.live2d_model

    # ---- 한 프레임의 포즈 계산(공용) ----------------------------------------
    def _tick_pose(self, context: Observation | None = None) -> Pose:
        obs = context or Observation()
        obs.speaking = 1.0 if self._speaking else 0.0
        obs.excite_drive = max(obs.excite_drive, self._energy)
        out = self.agent.tick(obs)
        self._energy = 0.7 * self._energy + 0.3 * out.intent.energy

        pose: Pose = self.adapter.to_pose(out.motion, energy=self._energy)
        if self._speaking:
            pose.mouth_open = max(pose.mouth_open, self._viseme_open)
            pose.mouth_wide = max(pose.mouth_wide, self._viseme_wide)
        self.rig.apply(pose)
        return pose

    # ---- 매 프레임 아바타 (AI가 모션 생성) -----------------------------------
    def render_frame(self, context: Observation | None = None) -> dict:
        """한 프레임: 에이전트를 틱해 모션을 생성하고 리그 포즈로 변환한다."""
        self._tick_pose(context)
        return self.rig.to_render_frame()

    def render_live2d_frame(self, context: Observation | None = None) -> dict[str, float]:
        """한 프레임을 *연결된 Live2D 모델의 실제 파라미터 값*으로 반환한다.

        결과 dict를 Cubism 런타임의 setParameterValueById(id, value)에 그대로 넣으면
        AI가 생성한 모션/립싱크/표정이 아바타에 적용된다.
        """
        if self.live2d is None:
            raise RuntimeError("먼저 load_avatar(model3_path)로 Live2D 모델을 연결하세요.")
        pose = self._tick_pose(context)
        lip = self._viseme_open if self._speaking else None
        return self.live2d.apply(pose, lipsync_open=lip)

    def export_live2d_motion(self, path: str, seconds: float = 5.0,
                             fps: int | None = None) -> int:
        """AI가 생성한 모션을 Live2D 파라미터 프레임(JSONL)으로 저장한다.

        Cubism 런타임(Web/Unity/Native)이 이 프레임들을 그대로 재생하면 아바타가
        움직인다 — 렌더러 없이도 엔드투엔드 적용을 증명/전달할 수 있다.
        """
        import json

        if self.live2d is None:
            raise RuntimeError("먼저 load_avatar(model3_path)로 Live2D 모델을 연결하세요.")
        fps = fps or self.cfg.avatar.fps
        n = int(seconds * fps)
        with open(path, "w", encoding="utf-8") as f:
            for i in range(n):
                frame = self.render_live2d_frame()
                f.write(json.dumps({"t": round(i / fps, 4), "params": frame},
                                   ensure_ascii=False) + "\n")
        return n

    # ---- 단일 제어 틱 (외부 게임 연동용) -------------------------------------
    def control_tick(self, obs: Observation, dt: float = 1.0 / 120.0) -> Control:
        """게임 관측을 받아 이번 프레임의 FPS 제어를 반환한다."""
        return self.agent.tick(obs, dt=dt).control

    # ---- FPS 사격장 (실시간 제어 + 온라인 조준 학습) -------------------------
    def play_fps(self, ticks: int = 600, learn: bool = True,
                 seed: int = 0) -> FPSResult:
        env = TargetRange(self.agent, seed=seed)
        return env.run(ticks=ticks, learn=learn)

    # ---- 종합 게임 (인게임 튜토리얼만으로 학습해 플레이) ---------------------
    def play_comprehensive(self, objectives=None) -> GameResult:
        """지금까지 *인게임 튜토리얼로* 습득한 조작 체계로 종합 게임을 플레이한다.

        외부 튜토 영상/사전지식 없음. 못 배운 조작이 필요한 목표는 실패한다.
        """
        game = ComprehensiveGame(self.agent, self.brain.tutorial, objectives)
        return game.play()

    # ---- 저지연 스트리밍 발화 -------------------------------------------------
    def speak(self, token_stream: Iterator[str],
              on_event: Callable[[str, object], None] | None = None) -> SpeakResult:
        budget = LatencyBudget(
            target_first_audio_ms=self.cfg.latency.target_first_audio_ms,
            target_think_ms=self.cfg.latency.target_think_ms,
        )
        self._speaking = True
        self._energy = max(self._energy, 0.5)
        self.lipsync.reset()
        think = budget.start("think")
        result = SpeakResult(text="")
        first = False

        def emit(kind, data):
            if on_event:
                on_event(kind, data)

        def consume_chunk(chunk: AudioChunk):
            nonlocal first
            if not first:
                budget.end(think)
                result.first_audio_ms = budget.mark_first_audio()
                first = True
                emit("first_audio", result.first_audio_ms)
            result.audio_chunks.append(chunk)
            result.total_audio_ms += chunk.duration_ms
            av = self.lipsync.from_audio_amplitude(chunk.amplitude)
            self._viseme_open = av.mouth_open
            emit("audio", chunk)

        for piece in token_stream:
            result.text += piece
            emit("token", piece)
            v = self.lipsync.from_text_piece(piece)
            self._viseme_open, self._viseme_wide = v.mouth_open, v.mouth_wide
            for chunk in self.tts.feed(piece):
                consume_chunk(chunk)
        for chunk in self.tts.flush():
            consume_chunk(chunk)

        self._speaking = False
        self._viseme_open = self._viseme_wide = 0.0
        result.warnings = budget.over_budget()
        return result

    # ---- 고수준 행동 ----------------------------------------------------------
    def on_chat(self, viewer_text: str, **kw) -> SpeakResult:
        return self.speak(self.brain.respond_chat(viewer_text), **kw)

    def comment(self, event: str, **kw) -> SpeakResult:
        return self.speak(self.brain.comment_game(event), **kw)

    def study_screen_text(self, screen_text: str, **kw) -> SpeakResult:
        return self.speak(self.brain.learn_from_screen(screen_text), **kw)

    def watch_and_learn(self, **kw) -> list[SpeakResult]:
        results: list[SpeakResult] = []
        for frame in self.screen.frames():
            text = self.ocr.read(frame)
            if text.strip():
                results.append(self.study_screen_text(text, **kw))
        return results


# --------------------------------------------------------------------------- #
def main() -> None:  # pragma: no cover
    from .. import __version__

    s = BroadcastStreamer()
    print(f"방송용 AI 버추얼 스트리머 v{__version__}")
    print(s.status())
    print("-" * 60)
    try:
        while True:
            try:
                msg = input("시청자> ").strip()
            except EOFError:
                break
            if not msg:
                break
            res = s.on_chat(msg)
            print(f"{s.brain.persona.name}> {res.text}")
            print(f"   ⤷ 첫 음성 {res.first_audio_ms:.0f}ms · 청크 {len(res.audio_chunks)}개")
    except KeyboardInterrupt:
        pass
    print("\n방송 종료. 수고했어!")


if __name__ == "__main__":  # pragma: no cover
    main()
