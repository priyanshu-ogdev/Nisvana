"""
backend/transport.py — Bidirectional Audio Streaming Transport Layer

Rev 3 P2: High-speed async streaming transport for tactical radio audio.

Provides:
- AudioChunkMessage: wire-efficient packaging of 10ms PCM audio chunks,
  sequence numbering, timestamping, and routing metadata.
- AsyncStreamHandler: asynchronous, backpressure-aware queue pairing
  for bidirectional client-to-engine and engine-to-client streaming.
"""

from dataclasses import dataclass, field
import struct
import time
from typing import Any, Dict, Optional
import queue
import numpy as np


@dataclass
class AudioChunkMessage:
    """
    Standard binary audio packet for network and inter-process transmission.
    """
    session_id: str
    sequence_number: int
    timestamp_ns: int
    audio_data: np.ndarray  # float32 1D array
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_pcm16_bytes(self) -> bytes:
        """Converts float32 audio [-1.0, 1.0] to 16-bit PCM bytes."""
        clipped = np.clip(self.audio_data, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        return pcm16.tobytes()

    @classmethod
    def from_pcm16_bytes(
        cls,
        data: bytes,
        session_id: str,
        sequence_number: int = 0,
        timestamp_ns: Optional[int] = None,
    ) -> "AudioChunkMessage":
        """Reconstructs AudioChunkMessage from raw 16-bit PCM bytes."""
        pcm16 = np.frombuffer(data, dtype=np.int16)
        float32 = pcm16.astype(np.float32) / 32767.0
        ts = timestamp_ns if timestamp_ns is not None else time.time_ns()
        return cls(
            session_id=session_id,
            sequence_number=sequence_number,
            timestamp_ns=ts,
            audio_data=float32,
        )


class AsyncStreamHandler:
    """
    Manages inbound and outbound queues for a single active radio session stream.
    Enforces bounded buffers to prevent memory growth under client backpressure.
    """
    def __init__(self, session_id: str, max_queue_depth: int = 50):
        self.session_id = session_id
        self.max_queue_depth = max_queue_depth
        self.inbound_queue: queue.Queue = queue.Queue(maxsize=max_queue_depth)
        self.outbound_queue: queue.Queue = queue.Queue(maxsize=max_queue_depth)
        self.dropped_inbound_count = 0
        self.dropped_outbound_count = 0

    def push_inbound(self, msg: AudioChunkMessage) -> bool:
        """Pushes incoming raw audio chunk. Drops oldest if full (real-time stream)."""
        try:
            self.inbound_queue.put_nowait(msg)
            return True
        except queue.Full:
            try:
                _ = self.inbound_queue.get_nowait()
                self.inbound_queue.put_nowait(msg)
                self.dropped_inbound_count += 1
                return True
            except Exception:
                return False

    def pop_inbound(self, timeout: float = 0.05) -> Optional[AudioChunkMessage]:
        """Fetches next audio chunk from inbound queue."""
        try:
            return self.inbound_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def push_outbound(self, msg: AudioChunkMessage) -> bool:
        """Pushes enhanced audio chunk to transmit queue."""
        try:
            self.outbound_queue.put_nowait(msg)
            return True
        except queue.Full:
            try:
                _ = self.outbound_queue.get_nowait()
                self.outbound_queue.put_nowait(msg)
                self.dropped_outbound_count += 1
                return True
            except Exception:
                return False

    def pop_outbound(self, timeout: float = 0.05) -> Optional[AudioChunkMessage]:
        """Fetches next enhanced audio chunk to transmit to radio."""
        try:
            return self.outbound_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear(self) -> None:
        """Empties all queues."""
        while not self.inbound_queue.empty():
            try:
                self.inbound_queue.get_nowait()
            except queue.Empty:
                break
        while not self.outbound_queue.empty():
            try:
                self.outbound_queue.get_nowait()
            except queue.Empty:
                break
