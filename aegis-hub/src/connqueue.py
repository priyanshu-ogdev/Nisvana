"""src/connqueue.py — Per-Connection Bounded Queue with Drop-Oldest Backpressure.

Design contract:
  - Every connection (node OR dashboard) owns exactly one ConnQueue.
  - enqueue() is synchronous — never blocks, never raises, never awaits.
    Drop-oldest: when full, the stalest frame is evicted and the new one inserted.
  - drain() is a long-lived coroutine — one asyncio task per connection.
    It is the ONLY place that calls ws.send(). No asyncio.gather() over sends
    anywhere else — a slow consumer cannot hostage a fast one.
  - On connection close: cancel the drain task; ConnQueue becomes unreachable.

Backpressure invariant:
  If node B is stalled, only B's dropped_frames counter increments.
  Nodes A and C continue at full rate. This is the core isolation property.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Optional

logger = logging.getLogger("aegis.hub.connqueue")

DEFAULT_MAXSIZE = 256   # ~2.56s at 100fps — enough for transient stalls


class ConnQueue:
    """
    Bounded async queue with drop-oldest eviction and an independent drain task.

    Usage:
        cq = ConnQueue(ws)
        drain_task = asyncio.create_task(cq.drain())
        ...
        cq.enqueue(frame_bytes)     # from router — synchronous, no await
        ...
        drain_task.cancel()         # on disconnect
    """

    def __init__(
        self,
        ws,                          # websockets ServerConnection (or any .send() object)
        maxsize: int = DEFAULT_MAXSIZE,
        label: str = "conn",         # for log messages
    ) -> None:
        self.ws = ws
        self.label = label
        self._q: asyncio.Queue[bytes | str] = asyncio.Queue(maxsize=maxsize)
        self.dropped: int = 0
        self._enqueued: int = 0
        self._started_at: float = time.monotonic()

    # ------------------------------------------------------------------
    # Producer side — called by router (sync, no await)
    # ------------------------------------------------------------------

    def enqueue(self, item: bytes | str) -> None:
        """
        Enqueue a message with drop-oldest eviction.

        Synchronous — safe to call from any coroutine without blocking it.
        Never raises; silently drops the oldest item when the queue is full.
        """
        if self._q.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._q.get_nowait()   # evict oldest (stale) frame
            self.dropped += 1
            if self.dropped % 50 == 1:   # log every 50th drop to avoid spam
                logger.warning(
                    f"ConnQueue[{self.label}]: dropped {self.dropped} frames total "
                    f"(queue full at {self._q.maxsize})"
                )
        with contextlib.suppress(asyncio.QueueFull):
            self._q.put_nowait(item)
        self._enqueued += 1

    # ------------------------------------------------------------------
    # Consumer side — one long-lived task per connection
    # ------------------------------------------------------------------

    async def drain(self) -> None:
        """
        Long-lived drain coroutine. One asyncio.Task per connection.

        Runs until cancelled (on disconnect) or until ws.send() raises.
        This is the ONLY coroutine that calls ws.send() for this connection.
        """
        import websockets.exceptions as _wse
        try:
            while True:
                item = await self._q.get()
                try:
                    await self.ws.send(item)
                except (_wse.ConnectionClosed, OSError):
                    logger.debug(f"ConnQueue[{self.label}]: connection closed during drain")
                    return
                except Exception as e:
                    logger.debug(f"ConnQueue[{self.label}]: send error: {e}")
                    return
        except asyncio.CancelledError:
            pass   # normal shutdown

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def depth(self) -> int:
        return self._q.qsize()

    @property
    def maxsize(self) -> int:
        return self._q.maxsize

    def stats(self) -> dict:
        elapsed = time.monotonic() - self._started_at
        return {
            "label":    self.label,
            "depth":    self._q.qsize(),
            "maxsize":  self._q.maxsize,
            "dropped":  self.dropped,
            "enqueued": self._enqueued,
            "rate_fps": round(self._enqueued / max(elapsed, 1e-6), 1),
        }
