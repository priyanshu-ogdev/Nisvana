"""
backend/session_manager.py — Multi-User Tactical Audio Session State Manager

Rev 3 P2: High-throughput per-user streaming state tracking.

ARCHITECTURAL PRINCIPLE:
Model weights (DeepFilterNet3, Classifier, etc.) are shared across all users
and kept in GPU VRAM (or RAM). Each concurrent tactical radio user session
maintains only lightweight causal streaming state:
  - Recurrent hidden state (GRU h_n)
  - Causal input context buffers
  - Rolling classifier context window
  - Mode decision & hysteresis counters
  - Output crossfade cache

Per-session footprint is tiny (~10-20 KB), meaning 100 concurrent streams
consume < 2 MB of state memory, fitting effortlessly in the 4-8 GB inference
envelope on Blackwell / edge hardware.
"""

from dataclasses import dataclass, field
import threading
import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch


@dataclass
class UserSession:
    """
    Lightweight per-user streaming state.
    """
    session_id: str
    user_id: str = ""
    sample_rate: int = 48000
    chunk_samples: int = 480
    classifier_window_sec: float = 0.2

    # Persistent streaming recurrent state (torch tensors)
    hidden_state: Optional[torch.Tensor] = None
    input_context: Optional[torch.Tensor] = None
    hidden_context: Optional[torch.Tensor] = None

    # Classifier rolling context buffer
    classifier_buffer: np.ndarray = field(init=False)
    classifier_buffer_filled: int = 0

    # Decision & hysteresis tracking
    current_mode: str = "primary"
    bypass_confirm_counter: int = 0
    last_prediction: Dict[str, Any] = field(default_factory=lambda: {
        "mode": "primary",
        "category": "speech_dominant",
        "estimated_snr_db": 10.0,
    })

    # Crossfade cache (last real output per mode)
    last_output_per_mode: Dict[str, np.ndarray] = field(default_factory=dict)

    # Activity & telemetry
    created_at: float = field(default_factory=time.monotonic)
    last_active_at: float = field(default_factory=time.monotonic)
    total_chunks: int = 0
    total_latency_ms: float = 0.0

    def __post_init__(self):
        window_len = int(self.classifier_window_sec * self.sample_rate)
        self.classifier_buffer = np.zeros(window_len, dtype=np.float32)

    def touch(self) -> None:
        """Updates last active timestamp."""
        self.last_active_at = time.monotonic()
        self.total_chunks += 1

    def reset_state(self) -> None:
        """Clears state buffers between transmissions or on connection resume."""
        self.hidden_state = None
        self.input_context = None
        self.hidden_context = None
        self.classifier_buffer.fill(0.0)
        self.classifier_buffer_filled = 0
        self.bypass_confirm_counter = 0
        self.current_mode = "primary"
        self.last_prediction = {
            "mode": "primary",
            "category": "speech_dominant",
            "estimated_snr_db": 10.0,
        }
        self.last_output_per_mode.clear()

    def get_memory_footprint_bytes(self) -> int:
        """Estimates total in-memory size of this session's state tensors and buffers."""
        size = 0
        for t in (self.hidden_state, self.input_context, self.hidden_context):
            if t is not None and isinstance(t, torch.Tensor):
                size += t.element_size() * t.nelement()
        if self.classifier_buffer is not None:
            size += self.classifier_buffer.nbytes
        for arr in self.last_output_per_mode.values():
            if isinstance(arr, np.ndarray):
                size += arr.nbytes
        return size + 1024  # Base object overhead


class SessionManager:
    """
    Thread-safe registry of concurrent tactical radio sessions.
    """
    def __init__(self, idle_timeout_sec: float = 300.0):
        self._sessions: Dict[str, UserSession] = {}
        self._lock = threading.Lock()
        self.idle_timeout_sec = idle_timeout_sec

    def create_session(
        self,
        session_id: str,
        user_id: str = "",
        sample_rate: int = 48000,
        chunk_samples: int = 480,
    ) -> UserSession:
        """Creates and registers a new tactical audio session."""
        with self._lock:
            session = UserSession(
                session_id=session_id,
                user_id=user_id,
                sample_rate=sample_rate,
                chunk_samples=chunk_samples,
            )
            self._sessions[session_id] = session
            return session

    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Retrieves an active session by ID."""
        with self._lock:
            return self._sessions.get(session_id)

    def get_or_create_session(
        self,
        session_id: str,
        user_id: str = "",
        sample_rate: int = 48000,
        chunk_samples: int = 480,
    ) -> UserSession:
        """Gets existing session or creates a new one atomically."""
        with self._lock:
            if session_id in self._sessions:
                return self._sessions[session_id]
            session = UserSession(
                session_id=session_id,
                user_id=user_id,
                sample_rate=sample_rate,
                chunk_samples=chunk_samples,
            )
            self._sessions[session_id] = session
            return session

    def close_session(self, session_id: str) -> bool:
        """Terminates and removes a session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def list_active_sessions(self) -> List[UserSession]:
        """Returns snapshot of currently registered sessions."""
        with self._lock:
            return list(self._sessions.values())

    def active_session_count(self) -> int:
        """Returns number of active sessions."""
        with self._lock:
            return len(self._sessions)

    def cleanup_idle_sessions(self, timeout_sec: Optional[float] = None) -> int:
        """Removes sessions that have been inactive longer than timeout."""
        timeout = timeout_sec or self.idle_timeout_sec
        now = time.monotonic()
        removed = 0
        with self._lock:
            to_remove = [
                sid for sid, s in self._sessions.items()
                if (now - s.last_active_at) > timeout
            ]
            for sid in to_remove:
                del self._sessions[sid]
                removed += 1
        return removed

    def total_memory_usage_kb(self) -> float:
        """Computes aggregate memory consumption across all active sessions."""
        with self._lock:
            total_bytes = sum(s.get_memory_footprint_bytes() for s in self._sessions.values())
            return total_bytes / 1024.0
