"""audio/ringbuffer.py — Lock-free SPSC ring buffer (numpy-backed)."""
from __future__ import annotations
import numpy as np
import threading


class SpscRingBuffer:
    """
    Single-Producer, Single-Consumer ring buffer for real-time audio.
    Producer (audio input thread) and Consumer (DSP thread) never block each other.
    Capacity: ~500ms at 48kHz / 10ms frames.

    NOT thread-safe for multiple producers or consumers.
    Only the producer writes; only the consumer reads.
    """

    def __init__(self, capacity_frames: int, frame_size: int, dtype=np.float32) -> None:
        self._capacity = capacity_frames
        self._frame_size = frame_size
        self._buf = np.zeros((capacity_frames, frame_size), dtype=dtype)
        self._write_idx = 0
        self._read_idx = 0
        self._lock = threading.Lock()  # only for index arithmetic

    @classmethod
    def from_ms(cls, ms: int, sample_rate: int, frame_size: int, dtype=np.float32) -> "SpscRingBuffer":
        """Create a ring buffer sized to hold `ms` milliseconds of audio."""
        frames_per_ms = sample_rate / 1000.0
        capacity = int(frames_per_ms * ms / frame_size) + 1
        return cls(capacity, frame_size, dtype)

    def write(self, frame: np.ndarray) -> bool:
        """Write one frame. Returns False if full (producer overrun)."""
        with self._lock:
            next_idx = (self._write_idx + 1) % self._capacity
            if next_idx == self._read_idx:
                return False  # full
            self._buf[self._write_idx] = frame
            self._write_idx = next_idx
            return True

    def read(self) -> np.ndarray | None:
        """Read one frame. Returns None if empty."""
        with self._lock:
            if self._read_idx == self._write_idx:
                return None
            frame = self._buf[self._read_idx].copy()
            self._read_idx = (self._read_idx + 1) % self._capacity
            return frame

    def available(self) -> int:
        with self._lock:
            return (self._write_idx - self._read_idx) % self._capacity

    def clear(self) -> None:
        with self._lock:
            self._read_idx = self._write_idx
