"""런타임 설정.

모든 값은 "로컬 우선 / 저지연" 기본값으로 잡혀 있다. 외부 API 키 같은 항목은
존재하지 않는다 — 이 시스템은 네트워크 없이 동작한다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


@dataclass(frozen=True)
class ModelConfig:
    """단일 로컬 모델 설정. 하나의 모델로 모든 인지 작업을 처리한다."""

    # 가중치 경로(.gguf 등). 없으면 결정론적 오프라인 폴백으로 동작한다.
    weights_path: str | None = field(default_factory=lambda: os.environ.get("BAI_MODEL_PATH"))
    # 컨텍스트 길이. 짧게 잡을수록 첫 토큰 지연(TTFT)이 줄어 반응이 빨라진다.
    context_tokens: int = field(default_factory=lambda: _env_int("BAI_CTX", 4096))
    # 한 턴 최대 출력 토큰. 방송 발화는 짧게 끊어 빠르게 — 저지연의 핵심.
    max_reply_tokens: int = field(default_factory=lambda: _env_int("BAI_MAX_REPLY", 96))
    temperature: float = field(default_factory=lambda: _env_float("BAI_TEMP", 0.7))
    # KV 캐시 재사용: 시스템 프롬프트/페르소나 프리픽스를 매 턴 다시 계산하지 않는다.
    reuse_kv_cache: bool = True


@dataclass(frozen=True)
class GPUConfig:
    """GPU 친화 설정. 모델을 한 번 적재해 상주시키고 메모리 예산에 맞춰 조정한다."""

    prefer_gpu: bool = field(default_factory=lambda: os.environ.get("BAI_GPU", "1") != "0")
    # 사용할 VRAM 상한(GB). 0이면 자동 감지값을 사용한다.
    vram_budget_gb: float = field(default_factory=lambda: _env_float("BAI_VRAM_GB", 0.0))
    # 양자화: int4가 가장 빠르고 가볍다 (방송 실시간성에 적합).
    quantization: str = field(default_factory=lambda: os.environ.get("BAI_QUANT", "int4"))


@dataclass(frozen=True)
class SpeechConfig:
    """저지연 스트리밍 TTS 설정."""

    sample_rate: int = 22050
    # 이 글자 수가 쌓이면 즉시 한 청크를 합성한다 — 문장 끝을 안 기다려 빠르게 말한다.
    chunk_chars: int = field(default_factory=lambda: _env_int("BAI_TTS_CHUNK", 18))
    voice: str = field(default_factory=lambda: os.environ.get("BAI_VOICE", "ko-stream-1"))
    speaking_rate: float = field(default_factory=lambda: _env_float("BAI_RATE", 1.08))


@dataclass(frozen=True)
class AvatarConfig:
    """사람처럼 움직이는 아바타 설정."""

    fps: int = field(default_factory=lambda: _env_int("BAI_FPS", 60))
    # 호흡/깜빡임/미세 흔들림 같은 무의식 모션을 켜 생동감을 준다.
    idle_motion: bool = True
    blink_per_min: float = field(default_factory=lambda: _env_float("BAI_BLINK", 17.0))


@dataclass(frozen=True)
class LatencyConfig:
    """반응속도 예산. 각 단계가 이 시간을 넘으면 경고/축약한다."""

    # 첫 음성이 나가기까지 목표(ms). 사람이 "빠르다"고 느끼는 임계.
    target_first_audio_ms: int = field(default_factory=lambda: _env_int("BAI_TTFA", 250))
    # 인지(생각) 단계 목표(ms).
    target_think_ms: int = field(default_factory=lambda: _env_int("BAI_THINK_MS", 120))


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    gpu: GPUConfig = field(default_factory=GPUConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    avatar: AvatarConfig = field(default_factory=AvatarConfig)
    latency: LatencyConfig = field(default_factory=LatencyConfig)

    def with_overrides(self, **kwargs) -> "Config":
        return replace(self, **kwargs)
