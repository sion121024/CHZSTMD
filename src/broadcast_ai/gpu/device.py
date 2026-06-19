"""GPU 감지 및 메모리 예산 계산.

이 모듈은 "GPU 친화" 요구를 담당한다. torch가 없어도(이 환경처럼) 안전하게
CPU로 떨어지며, 단일 모델을 한 번만 적재해 상주시키기 위한 적재 계획을 만든다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..config import GPUConfig, ModelConfig

# 양자화별 가중치당 바이트 수(근사). 적재 가능 여부 추정에 쓴다.
_BYTES_PER_PARAM = {
    "fp16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}


@dataclass(frozen=True)
class Device:
    """선택된 추론 디바이스와 적재 계획."""

    backend: str          # "cuda" | "metal" | "cpu"
    name: str
    total_vram_gb: float
    usable_vram_gb: float
    quantization: str
    # 한 번 적재 후 상주. 재적재(reload)는 지연의 가장 큰 원인이므로 금지.
    resident: bool = True

    @property
    def is_gpu(self) -> bool:
        return self.backend in ("cuda", "metal")

    def fits_params(self, num_params_billion: float) -> bool:
        """주어진 파라미터 규모의 모델이 예산 안에 들어오는지."""
        bytes_per = _BYTES_PER_PARAM.get(self.quantization, 2.0)
        needed_gb = num_params_billion * 1e9 * bytes_per / (1024 ** 3)
        # KV 캐시/활성화 여유 20%.
        return needed_gb * 1.2 <= max(self.usable_vram_gb, _cpu_ram_gb() if not self.is_gpu else 0.0)

    def recommended_batch(self) -> int:
        """방송은 1인 1스트림이므로 배치는 작게 — 지연 최소화가 우선."""
        if not self.is_gpu:
            return 1
        if self.usable_vram_gb >= 24:
            return 4
        if self.usable_vram_gb >= 12:
            return 2
        return 1

    def summary(self) -> str:
        loc = "GPU" if self.is_gpu else "CPU"
        return (
            f"[{loc}] {self.name} · backend={self.backend} · "
            f"quant={self.quantization} · usable={self.usable_vram_gb:.1f}GB · "
            f"batch={self.recommended_batch()} · resident={self.resident}"
        )


def _cpu_ram_gb() -> float:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
    except (ValueError, OSError, AttributeError):
        return 8.0


def _try_torch_cuda():
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            return props.name, props.total_memory / (1024 ** 3)
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "Apple Metal (MPS)", _cpu_ram_gb()  # 통합 메모리
    except Exception:
        pass
    return None


def detect_device(gpu_cfg: GPUConfig, model_cfg: ModelConfig | None = None) -> Device:
    """현재 환경에서 가장 빠른 추론 디바이스를 선택한다."""
    quant = gpu_cfg.quantization

    if gpu_cfg.prefer_gpu:
        found = _try_torch_cuda()
        if found is not None:
            name, total = found
            backend = "metal" if "Metal" in name else "cuda"
            budget = gpu_cfg.vram_budget_gb or (total * 0.85)
            usable = min(budget, total * 0.95)
            return Device(backend=backend, name=name, total_vram_gb=total,
                          usable_vram_gb=usable, quantization=quant)

    # CPU 폴백. 양자화는 그대로 두되(메모리 절감), 작은 모델을 권장.
    return Device(
        backend="cpu",
        name="CPU (torch/CUDA 미탑재 — 오프라인 폴백)",
        total_vram_gb=0.0,
        usable_vram_gb=0.0,
        quantization=quant,
    )
