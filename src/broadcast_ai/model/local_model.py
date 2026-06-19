"""단일 로컬 모델 (하나의 모델로 모든 인지 작업 처리).

핵심 아이디어
-------------
* 모델은 *한 개*다. 대화든, 게임 행동 결정이든, 튜토리얼 이해든 같은 인스턴스에
  task 태그가 붙은 프롬프트를 흘려보낼 뿐 — 작업별로 모델을 따로 두지 않는다.
* 모델은 생성 시 *한 번* 적재되어 상주한다. 재적재가 없으므로 매 턴 첫 토큰이 빠르다.
* 시스템/페르소나 프리픽스의 KV 캐시를 재사용한다(reuse_kv_cache).
* 출력은 토큰 단위로 *스트리밍*한다. 호출 측은 첫 토큰이 나오는 즉시 TTS를 시작해
  "말이 빠른" 느낌을 만든다.

가중치(weights_path)가 없으면 외부 의존성 0인 결정론적 폴백 생성기를 쓴다.
따라서 이 파일은 어떤 환경에서도 import/실행 가능하다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Literal

from ..config import ModelConfig
from ..gpu.device import Device

Task = Literal["chat", "game", "tutorial"]


@dataclass
class Turn:
    role: str          # "system" | "user" | "assistant"
    content: str


class LocalModel:
    """온디바이스 단일 모델. 외부 API를 호출하지 않는다."""

    def __init__(self, cfg: ModelConfig, device: Device) -> None:
        self.cfg = cfg
        self.device = device
        self._backend = None         # 실제 llama_cpp / transformers 핸들
        self._kv_prefix: str | None = None
        self.loaded = False
        self.using_fallback = True

    # ---- 적재 (한 번만, 상주) -------------------------------------------------
    def load(self) -> None:
        """모델을 한 번 적재한다. 재호출은 무시(상주 유지)."""
        if self.loaded:
            return
        if self.cfg.weights_path:
            self._backend = self._load_real_backend(self.cfg.weights_path)
            self.using_fallback = self._backend is None
        self.loaded = True

    def _load_real_backend(self, path: str):
        """실제 로컬 백엔드 적재. 미설치/실패 시 None → 폴백."""
        try:
            from llama_cpp import Llama  # type: ignore

            n_gpu_layers = -1 if self.device.is_gpu else 0
            return Llama(
                model_path=path,
                n_ctx=self.cfg.context_tokens,
                n_gpu_layers=n_gpu_layers,   # GPU 친화: 가능한 레이어 전부 GPU로
                logits_all=False,
                verbose=False,
            )
        except Exception:
            return None

    # ---- 스트리밍 생성 --------------------------------------------------------
    def stream(self, system: str, history: list[Turn], user: str,
               task: Task = "chat") -> Iterator[str]:
        """토큰(또는 짧은 조각)을 순차적으로 내보낸다.

        호출 측은 이 제너레이터를 소비하며 즉시 음성 합성/자막을 시작한다.
        """
        if not self.loaded:
            self.load()

        if self._backend is not None and not self.using_fallback:
            yield from self._stream_real(system, history, user, task)
        else:
            yield from self._stream_fallback(system, history, user, task)

    def _stream_real(self, system: str, history: list[Turn], user: str, task: Task):
        prompt = self._build_prompt(system, history, user, task)
        # KV 캐시 재사용을 위해 안정적인 프리픽스(system+페르소나)를 앞에 고정한다.
        stream = self._backend.create_completion(
            prompt=prompt,
            max_tokens=self.cfg.max_reply_tokens,
            temperature=self.cfg.temperature,
            stream=True,
            stop=["\n<user>", "<user>"],
        )
        for chunk in stream:
            piece = chunk["choices"][0]["text"]
            if piece:
                yield piece

    # ---- 오프라인 폴백 (의존성 0, 결정론적) ----------------------------------
    def _stream_fallback(self, system: str, history: list[Turn], user: str, task: Task):
        """가중치 없이도 파이프라인 전체를 시연/테스트할 수 있는 생성기.

        실제 LLM의 자리표시자. 작업별로 다른 '성격'의 텍스트를 토큰처럼 흘려보낸다.
        """
        text = _fallback_response(task, user, history)
        # 토큰 스트리밍을 흉내내어 저지연 경로를 실제처럼 검증할 수 있게 한다.
        for token in _tokenize_like_stream(text):
            # 아주 짧은 인공 지연: 실제 디코딩 속도를 모사(반응성 테스트용).
            time.sleep(0.004)
            yield token

    # ---- 프롬프트 조립 --------------------------------------------------------
    def _build_prompt(self, system: str, history: list[Turn], user: str, task: Task) -> str:
        lines = [f"<system task={task}>\n{system}\n</system>"]
        for t in history[-8:]:
            lines.append(f"<{t.role}>{t.content}</{t.role}>")
        lines.append(f"<user>{user}</user>")
        lines.append("<assistant>")
        return "\n".join(lines)

    def info(self) -> str:
        mode = "실모델" if (self.loaded and not self.using_fallback) else "오프라인 폴백"
        return f"LocalModel({mode}) on {self.device.summary()}"


# --------------------------------------------------------------------------- #
# 폴백 텍스트 생성 — 외부 의존성 없이 작업별 성격을 흉내낸다.
# --------------------------------------------------------------------------- #
def _tokenize_like_stream(text: str) -> Iterator[str]:
    """문자열을 실제 토크나이저처럼 작은 조각으로 나눠 스트리밍."""
    buf = ""
    for ch in text:
        buf += ch
        if ch in " ,.!?…~\n" or len(buf) >= 4:
            yield buf
            buf = ""
    if buf:
        yield buf


def _fallback_response(task: Task, user: str, history: list[Turn]) -> str:
    user = user.strip()
    if task == "tutorial":
        return (
            f"오케이, 방금 화면에 뜬 설명 읽었어. '{user[:40]}' 이 부분이 핵심이네. "
            "한 번 그대로 따라 해볼게!"
        )
    if task == "game":
        return (
            f"좋아, 지금 상황 보면 {user[:30]} 이렇게 가는 게 맞는 것 같아. 가보자!"
        )
    # chat
    if not user:
        return "음~ 채팅 천천히 올려줘도 돼, 다 보고 있어!"
    if user.endswith("?") or "뭐" in user or "어때" in user:
        return f"오 좋은 질문! {user[:24]} 라… 내 생각엔 충분히 해볼 만하다고 봐."
    return f"ㅋㅋ {user[:24]} 그치, 나도 완전 공감해. 자 계속 가보자고!"
