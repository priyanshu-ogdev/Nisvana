"""
tests/test_backend_subsystem.py — Multi-User Backend Subsystem Verification

Tests:
1. UserSession state isolation and memory footprint (< 50 KB).
2. SessionManager lifecycle: create, get, close, cleanup, thread concurrency.
3. BatchInferenceEngine: shared weights processing multiple concurrent user streams.
4. Intelligibility floor enforcement across sessions.
5. AudioChunkMessage PCM-16 serialization roundtrip.
6. AsyncStreamHandler bounded queuing and backpressure handling.
"""

import time
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from backend.session_manager import UserSession, SessionManager
from backend.batch_inference import BatchInferenceEngine, BatchInferenceConfig
from backend.transport import AudioChunkMessage, AsyncStreamHandler


class TestUserSessionAndManager:
    """Tests lightweight per-user session state tracking and registry."""

    def test_user_session_state_footprint_is_minimal(self):
        """Validates that a single user session consumes < 50 KB of memory."""
        session = UserSession(session_id="tactical-soldier-01", user_id="operator_alpha")
        bytes_used = session.get_memory_footprint_bytes()
        assert bytes_used < 50 * 1024, f"Session state {bytes_used} bytes exceeded 50KB budget"

    def test_session_manager_crud_lifecycle(self):
        """Tests session creation, retrieval, and termination."""
        mgr = SessionManager(idle_timeout_sec=5.0)
        s1 = mgr.create_session("sess-1", "user-1")
        assert mgr.active_session_count() == 1
        assert mgr.get_session("sess-1") is s1

        # Idempotent get_or_create
        s1_again = mgr.get_or_create_session("sess-1")
        assert s1_again is s1
        assert mgr.active_session_count() == 1

        # Close session
        closed = mgr.close_session("sess-1")
        assert closed is True
        assert mgr.get_session("sess-1") is None
        assert mgr.active_session_count() == 0

    def test_session_manager_idle_cleanup(self):
        """Tests that idle sessions are properly cleaned up."""
        mgr = SessionManager(idle_timeout_sec=0.1)
        s1 = mgr.create_session("sess-idle-1")
        s2 = mgr.create_session("sess-active-2")

        time.sleep(0.15)
        # Touch s2 to keep it alive
        s2.touch()

        removed = mgr.cleanup_idle_sessions(timeout_sec=0.1)
        assert removed == 1
        assert mgr.get_session("sess-idle-1") is None
        assert mgr.get_session("sess-active-2") is not None


class TestBatchInferenceExecution:
    """Tests multi-user streaming inference with shared model weights."""

    @pytest.fixture(autouse=True)
    def setup_engine(self):
        self.config = BatchInferenceConfig(chunk_samples=480, sample_rate=48000)
        self.engine = BatchInferenceEngine(self.config)
        self.mgr = SessionManager()

    def test_process_session_frame_single(self):
        """Validates single-stream frame enhancement and metadata generation."""
        session = self.mgr.create_session("sess-audio-1")
        audio = np.random.randn(480).astype(np.float32) * 0.1

        enhanced, meta = self.engine.process_session_frame(session, audio)
        assert len(enhanced) == 480
        assert not np.isnan(enhanced).any()
        assert meta["session_id"] == "sess-audio-1"
        assert "mode" in meta
        assert "category" in meta

    def test_intelligibility_floor_applied_on_speech_frames(self):
        """Validates that dry-mix floor preserves voice during speech dominant frames."""
        session = self.mgr.create_session("sess-speech-floor")
        # Strong speech-like clean signal
        audio = np.sin(np.linspace(0, 10, 480, dtype=np.float32)) * 0.5

        enhanced, meta = self.engine.process_session_frame(session, audio)
        # Verify enhanced output is not zeroed
        energy_in = np.mean(audio ** 2)
        energy_out = np.mean(enhanced ** 2)
        assert energy_out > 0.01 * energy_in, "Intelligibility floor failed to retain signal energy"

    def test_multi_user_batch_execution(self):
        """Processes 8 concurrent user streams in a single batch tick."""
        sessions = [self.mgr.create_session(f"stream-{i}") for i in range(8)]
        batch_reqs = [(s, np.random.randn(480).astype(np.float32) * 0.1) for s in sessions]

        results = self.engine.process_batch(batch_reqs)
        assert len(results) == 8
        for i, (enhanced, meta) in enumerate(results):
            assert len(enhanced) == 480
            assert meta["session_id"] == f"stream-{i}"


class TestAudioTransportLayer:
    """Tests binary audio message serialization and queue handler."""

    def test_pcm16_serialization_roundtrip(self):
        """Verifies float32 to PCM16 and back produces accurate audio."""
        orig_audio = np.sin(np.linspace(0, 20, 480, dtype=np.float32)) * 0.8
        msg = AudioChunkMessage(
            session_id="radio-channel-5",
            sequence_number=42,
            timestamp_ns=1000000,
            audio_data=orig_audio,
        )

        pcm_bytes = msg.to_pcm16_bytes()
        assert len(pcm_bytes) == 480 * 2  # 16-bit = 2 bytes per sample

        recovered = AudioChunkMessage.from_pcm16_bytes(
            pcm_bytes,
            session_id="radio-channel-5",
            sequence_number=42,
            timestamp_ns=1000000,
        )

        np.testing.assert_allclose(recovered.audio_data, orig_audio, atol=1e-4)

    def test_async_stream_handler_queues_and_backpressure(self):
        """Tests queue pushing, popping, and overflow handling."""
        handler = AsyncStreamHandler(session_id="stream-test", max_queue_depth=5)
        audio = np.zeros(480, dtype=np.float32)

        # Push 5 messages (fills queue)
        for i in range(5):
            msg = AudioChunkMessage("stream-test", i, time.time_ns(), audio)
            assert handler.push_inbound(msg) is True

        # Push 6th message — should drop oldest and succeed without blocking
        msg6 = AudioChunkMessage("stream-test", 5, time.time_ns(), audio)
        assert handler.push_inbound(msg6) is True
        assert handler.dropped_inbound_count == 1

        # Pop should return 2nd message (0 was dropped)
        popped = handler.pop_inbound(timeout=0.01)
        assert popped is not None
        assert popped.sequence_number == 1
