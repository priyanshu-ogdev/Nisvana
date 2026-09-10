"""
Project AEGIS — Multi-User Tactical Audio Backend Subsystem

Rev 3 P2: High-throughput, multi-user backend session management and batched
inference. Serves concurrent tactical radio streams within a 4-8GB VRAM envelope
by sharing model weights across sessions and batching frame inference on GPU.
"""

from .session_manager import UserSession, SessionManager
from .batch_inference import BatchInferenceEngine, BatchInferenceConfig
from .transport import AudioChunkMessage, AsyncStreamHandler

__all__ = [
    "UserSession",
    "SessionManager",
    "BatchInferenceEngine",
    "BatchInferenceConfig",
    "AudioChunkMessage",
    "AsyncStreamHandler",
]
