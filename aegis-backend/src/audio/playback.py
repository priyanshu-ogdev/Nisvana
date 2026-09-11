"""audio/playback.py — Jitter buffer for downstream mesh audio playback.

Network packets arrive with variable latency (jitter). Without buffering,
audio pops and clicks. This module absorbs jitter and provides smooth
frame-aligned PCM to the DAC.

Design (per Phase 1 v6.1 spec):
  - asyncio.Queue(maxsize=50) ≈ 500ms at 10ms/frame
  - Drop-oldest policy: when full, the oldest (stalest) frame is evicted
    and the newest is inserted, keeping the buffer real-time aligned.
  - Gate 4: if downstream is paused for N seconds then resumed, all
    buffered frames are dropped instantly (catch-up, zero accumulated delay).

Usage:
    buf = MeshPlaybackBuffer()
    await buf.push_network_audio(pcm_frame)   # from downstream_task
    frame = await buf.pop_for_dac()           # from playback_task → DAC
"""
from __future__ import annotations
import asyncio
import logging
import numpy as np

logger = logging.getLogger(__name__)

FRAME_SIZE = 480
DEFAULT_MAX_QUEUE = 50    # 500ms at 10ms/frame


class MeshPlaybackBuffer:
    """
    Async jitter buffer for mesh network audio.

    Thread safety: designed for use within a single asyncio event loop.
    push_network_audio() can be called from any coroutine.
    pop_for_dac() blocks until a frame is available.
    """

    def __init__(
        self,
        frame_size: int = FRAME_SIZE,
        max_queue_frames: int = DEFAULT_MAX_QUEUE,
    ) -> None:
        self.frame_size = frame_size
        self.max_queue_frames = max_queue_frames
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=max_queue_frames)
        self._pushed: int = 0
        self._dropped: int = 0
        self._silence_frame = np.zeros(frame_size, dtype=np.float32)

    # ------------------------------------------------------------------
    # Producer side (called by downstream_task)
    # ------------------------------------------------------------------

    async def push_network_audio(self, pcm: np.ndarray) -> None:
        """
        Push a decoded PCM frame into the jitter buffer.

        If the queue is full (consumer stalled / network burst), the oldest
        frame is evicted (drop-oldest) and the new frame is inserted so that
        playback stays aligned with real-time. This prevents accumulating
        delay after a pause.
        """
        if self._queue.full():
            try:
                self._queue.get_nowait()  # evict oldest
                self._dropped += 1
                if self._dropped % 10 == 1:  # log every 10th drop to avoid spam
                    logger.warning(
                        f"MeshPlaybackBuffer: queue full — dropped oldest frame "
                        f"(total dropped: {self._dropped})"
                    )
            except asyncio.QueueEmpty:
                pass

        try:
            self._queue.put_nowait(pcm.astype(np.float32))
            self._pushed += 1
        except asyncio.QueueFull:
            # Race: another coroutine filled the slot we just freed. Drop silently.
            self._dropped += 1

    # ------------------------------------------------------------------
    # Consumer side (called by playback_task → DAC)
    # ------------------------------------------------------------------

    async def pop_for_dac(self) -> np.ndarray:
        """
        Block until a PCM frame is ready, then return it.
        Returns silence if the queue is empty and a 10ms timeout elapses
        (avoids DAC underruns during network gaps).
        """
        try:
            frame = await asyncio.wait_for(self._queue.get(), timeout=0.010)
            return frame
        except asyncio.TimeoutError:
            # Network gap: output silence to keep DAC clock alive
            return self._silence_frame.copy()

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def flush(self) -> int:
        """
        Drain all buffered frames immediately (e.g., on reconnect or pause recovery).
        Returns the number of frames flushed.
        """
        flushed = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                flushed += 1
            except asyncio.QueueEmpty:
                break
        if flushed:
            logger.info(f"MeshPlaybackBuffer: flushed {flushed} frames (catch-up)")
        return flushed

    def stats(self) -> dict:
        """Return buffer diagnostics for telemetry."""
        return {
            "queue_depth": self._queue.qsize(),
            "max_queue": self.max_queue_frames,
            "total_pushed": self._pushed,
            "total_dropped": self._dropped,
            "buffer_ms": round(self._queue.qsize() * (self.frame_size / 48000) * 1000, 1),
        }
