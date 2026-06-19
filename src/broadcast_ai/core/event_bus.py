"""비동기 이벤트 버스.

지각(perception) → 인지(brain) → 발화(speech) → 아바타(avatar) 단계를 느슨하게
연결해 동시에 굴린다. 한 단계가 다음 단계를 블로킹하지 않으므로 반응이 빨라진다.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class Event:
    topic: str
    payload: Any = None
    ts_ms: float = 0.0


Handler = Callable[[Event], Awaitable[None]]


class EventBus:
    """토픽 기반 발행/구독. 단일 프로세스, 단일 이벤트 루프 전제."""

    def __init__(self) -> None:
        self._subs: dict[str, list[Handler]] = defaultdict(list)
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._running = False

    def subscribe(self, topic: str, handler: Handler) -> None:
        self._subs[topic].append(handler)

    async def publish(self, topic: str, payload: Any = None) -> None:
        from .clock import now_ms

        await self._queue.put(Event(topic=topic, payload=payload, ts_ms=now_ms()))

    async def run(self) -> None:
        """버스를 돌린다. 핸들러는 동시 실행돼 서로를 막지 않는다."""
        self._running = True
        while self._running:
            event = await self._queue.get()
            handlers = self._subs.get(event.topic, ())
            if handlers:
                await asyncio.gather(*(h(event) for h in handlers))
            self._queue.task_done()

    def stop(self) -> None:
        self._running = False
