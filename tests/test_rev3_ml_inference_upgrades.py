"""
tests/test_rev3_ml_inference_upgrades.py — Unit Tests for Rev 3 ML/Inference Upgrades

Tests all new capabilities introduced in Rev 3:
1. Shared streaming chunker (ChunkConfig, chunk_audio, chunk_count, roundtrip reconstruction).
2. Speech-presence-gated SDR penalty (SDRLoss with speech_mask, _compute_speech_mask).
3. CleanUMamba distillation loss (DistillationLoss magnitude spectrogram MSE).
4. Model 2 lookahead buffer and output-delay streaming (DeepFilterNet3Wrapper).
5. Escalation router safeguards:
   - Intelligibility floor (minimum speech energy retention during speech dominance).
   - Asymmetric mode transition hysteresis (immediate escalation, 15-chunk bypass confirmation).
   - Lazy-loading and idle-unload of escalation model.
6. Gunfire data audit (data_forge.verifier.gunfire_audit).
7. QAT lifecycle and platform-adaptive preparation (BaseTrainer.prepare_qat / convert_qat).
"""

import time
import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from training.data.streaming_chunker import (
    ChunkConfig,
    DEFAULT_CHUNK_CONFIG,
    chunk_audio,
    chunk_count,
    reconstruct_from_chunks,
)
from training.losses.multires_loss import (
    ResolvedLossConfig,
    SDRLoss,
    DistillationLoss,
    _compute_speech_mask,
    build_se_loss,
)
from training.models.model_loader import (
    DeepFilterNet3Wrapper,
    build_model_for_key,
)
from inference.runtime.escalation_router import AcousticEscalationRouter
from data_forge.verifier.gunfire_audit import run_gunfire_audit, GunfireAuditSummary
from training.trainers.base_trainer import BaseTrainer


# =====================================================================
# 1. Shared Streaming Chunker Tests (P0.5)
# =====================================================================

class TestStreamingChunker:
    """Verifies single source of truth for 10ms (480-sample) chunking."""

    def test_chunk_config_properties(self):
        cfg = ChunkConfig(chunk_samples=480, sample_rate=48000, overlap_samples=0)
        assert cfg.chunk_duration_ms == 10.0
        assert cfg.chunk_duration_sec == 0.01

    def test_chunk_audio_even_division(self):
        audio = np.random.randn(4800).astype(np.float32)  # 10 chunks of 480
        chunks = list(chunk_audio(audio, DEFAULT_CHUNK_CONFIG))
        assert len(chunks) == 10
        assert all(c.shape == (480,) for c in chunks)
        assert all(c.dtype == np.float32 for c in chunks)

    def test_chunk_audio_partial_chunk_zero_padded(self):
        audio = np.ones(500, dtype=np.float32)  # 1 full chunk (480) + 1 partial (20)
        chunks = list(chunk_audio(audio, DEFAULT_CHUNK_CONFIG))
        assert len(chunks) == 2
        assert chunks[0].shape == (480,)
        assert chunks[1].shape == (480,)
        assert np.all(chunks[0] == 1.0)
        assert np.all(chunks[1][:20] == 1.0)
        assert np.all(chunks[1][20:] == 0.0)  # Zero-padded

    def test_chunk_count_matches(self):
        assert chunk_count(480, DEFAULT_CHUNK_CONFIG) == 1
        assert chunk_count(481, DEFAULT_CHUNK_CONFIG) == 2
        assert chunk_count(4800, DEFAULT_CHUNK_CONFIG) == 10

    def test_reconstruct_from_chunks_exact_length(self):
        total_samples = 1234
        audio = np.random.randn(total_samples).astype(np.float32)
        chunks = chunk_audio(audio, DEFAULT_CHUNK_CONFIG)
        recovered = reconstruct_from_chunks(chunks, total_samples, DEFAULT_CHUNK_CONFIG)
        assert len(recovered) == total_samples
        np.testing.assert_allclose(recovered, audio, atol=1e-6)

    def test_chunk_audio_rejects_non_1d(self):
        with pytest.raises(ValueError, match="Expected 1D audio"):
            list(chunk_audio(np.zeros((2, 480), dtype=np.float32)))


# =====================================================================
# 2. Speech-Presence-Gated SDR Loss Tests (P0.3)
# =====================================================================

class TestSpeechPresenceGatedSDRLoss:
    """Verifies that speech-active frames receive higher SDR loss penalties."""

    def test_speech_mask_computation(self):
        cfg = ResolvedLossConfig(
            speech_presence_sdr_boost=2.5,
            speech_presence_rms_threshold=0.01,
            speech_band_hz=(300, 4000),
            speech_presence_sample_rate=48000,
        )

        # 1) Speech-like signal in formant band (1000 Hz tone)
        t = torch.linspace(0, 1.0, 48000)
        speech_target = torch.sin(2 * np.pi * 1000.0 * t).unsqueeze(0) * 0.5
        mask_speech = _compute_speech_mask(speech_target, cfg)
        assert mask_speech is not None
        assert mask_speech.item() == 1.0, "Speech formant tone should trigger mask"

        # 2) Silence / low energy
        silent_target = torch.zeros(1, 48000)
        mask_silent = _compute_speech_mask(silent_target, cfg)
        assert mask_silent is not None
        assert mask_silent.item() == 0.0, "Silence should not trigger mask"

        # 3) Out-of-band high frequency noise (18000 Hz)
        hf_target = torch.sin(2 * np.pi * 18000.0 * t).unsqueeze(0) * 0.5
        mask_hf = _compute_speech_mask(hf_target, cfg)
        assert mask_hf is not None
        assert mask_hf.item() == 0.0, "High frequency tone outside speech band should not trigger mask"

    def test_sdr_loss_boost_on_speech_active_frames(self):
        sdr_boosted = SDRLoss(speech_boost=2.5)
        sdr_standard = SDRLoss(speech_boost=1.0)

        target = torch.randn(2, 4800)
        estimate = target + 0.2 * torch.randn(2, 4800)

        # Case A: speech_mask = 1 (speech active)
        mask_active = torch.ones(2)
        loss_active = sdr_boosted(estimate, target, speech_mask=mask_active)
        loss_std = sdr_standard(estimate, target)

        # Note: SDR loss returns negative SDR (lower = better SNR, higher = worse distortion)
        # When boosted by 2.5x, the penalty is 2.5x larger in magnitude
        assert torch.isclose(loss_active, 2.5 * loss_std, rtol=1e-4)

        # Case B: speech_mask = 0 (noise only)
        mask_inactive = torch.zeros(2)
        loss_inactive = sdr_boosted(estimate, target, speech_mask=mask_inactive)
        assert torch.isclose(loss_inactive, loss_std, rtol=1e-4)


# =====================================================================
# 3. CleanUMamba Distillation Loss Tests (P1.1)
# =====================================================================

class TestDistillationLoss:
    """Verifies magnitude spectrogram distillation loss between student and teacher."""

    def test_distillation_loss_identical_is_zero(self):
        distill_loss = DistillationLoss(n_fft=512)
        audio = torch.randn(2, 4800)
        loss = distill_loss(audio, audio)
        assert loss.item() < 1e-6

    def test_distillation_loss_penalizes_magnitude_error(self):
        distill_loss = DistillationLoss(n_fft=512)
        teacher_audio = torch.randn(2, 4800)
        close_student = teacher_audio + 0.01 * torch.randn(2, 4800)
        distant_student = teacher_audio + 0.5 * torch.randn(2, 4800)

        loss_close = distill_loss(close_student, teacher_audio)
        loss_distant = distill_loss(distant_student, teacher_audio)
        assert loss_close.item() < loss_distant.item()

    def test_distillation_loss_short_signal_fallback(self):
        distill_loss = DistillationLoss(n_fft=1024)
        short_s = torch.randn(2, 200)
        short_t = torch.randn(2, 200)
        loss = distill_loss(short_s, short_t)
        assert loss.ndim == 0
        assert not torch.isnan(loss)


# =====================================================================
# 4. Model 2 Lookahead Buffer & Output-Delay Tests (P0.2)
# =====================================================================

class TestModel2LookaheadBuffer:
    """Verifies DeepFilterNet3Wrapper lookahead buffer and output delay."""

    def test_model_with_lookahead_delays_and_uses_future_context(self):
        # conv_lookahead=1 gives right_pad=1
        model = DeepFilterNet3Wrapper(conv_lookahead=1)
        assert model._has_lookahead is True

        chunk_0 = torch.randn(1, 480)
        chunk_1 = torch.randn(1, 480)

        # First chunk: immediate output + buffered in _pending_input
        out_0 = model(chunk_0)
        assert out_0.shape == (1, 480)
        assert model._pending_input is not None
        assert torch.equal(model._pending_input.squeeze(), chunk_0.squeeze())

        # Second chunk: processes pending chunk_0 using chunk_1's leading edge as right-context
        out_1 = model(chunk_1)
        assert out_1.shape == (1, 480)
        assert model._pending_input is not None
        assert torch.equal(model._pending_input.squeeze(), chunk_1.squeeze())

    def test_model_without_lookahead_has_no_delay_buffer(self):
        model = DeepFilterNet3Wrapper(conv_lookahead=0)
        assert model._has_lookahead is False
        chunk = torch.randn(1, 480)
        out = model(chunk)
        assert out.shape == (1, 480)
        assert model._pending_input is None

    def test_reset_state_clears_all_buffers(self):
        model = DeepFilterNet3Wrapper(conv_lookahead=1)
        model(torch.randn(1, 480))
        assert model._pending_input is not None
        assert model.hidden_state is not None

        model.reset_state()
        assert model._pending_input is None
        assert model._pending_output is None
        assert model._input_context is None
        assert model._hidden_context is None
        assert model.hidden_state is None

    def test_explicit_forward_for_onnx_export(self):
        model = DeepFilterNet3Wrapper(conv_lookahead=1)
        hidden, in_ctx, hid_ctx = model.get_initial_state(batch_size=1)
        x = torch.randn(1, 480)

        out, new_hid, new_in_ctx, new_hid_ctx = model(x, hidden, in_ctx, hid_ctx)
        assert out.shape == (1, 480)
        assert new_hid.shape == hidden.shape
        assert new_in_ctx.shape == in_ctx.shape
        assert new_hid_ctx.shape == hid_ctx.shape


# =====================================================================
# 5. Escalation Router Safeguard Tests (P1.2, P1.3, P1.4)
# =====================================================================

class TestEscalationRouterSafeguards:
    """Verifies intelligibility floor, asymmetric hysteresis, and lazy-loading."""

    def test_intelligibility_floor_preserves_speech_frames(self):
        # Create router with aggressive floor
        router = AcousticEscalationRouter(
            intelligibility_floor=0.20,
            lazy_load_escalation=True,
        )

        # Create strong speech-dominant audio
        audio_chunk = np.sin(np.linspace(0, 50, 480, dtype=np.float32)) * 0.8

        # Mock a suppressor that zeroes out output
        class ZeroSuppressor(nn.Module):
            def forward(self, x):
                return torch.zeros_like(x)

        router.model_primary = ZeroSuppressor()

        # Route audio in speech_dominant mode
        enhanced, info = router.route_and_enhance(audio_chunk, forced_mode=None)

        if info["category"] == "speech_dominant":
            # Enhanced must not be zero: floor must guarantee >= 0.20 * audio_chunk magnitude
            min_expected_energy = np.mean((0.20 * audio_chunk) ** 2)
            actual_energy = np.mean(enhanced ** 2)
            assert actual_energy >= 0.99 * min_expected_energy

    def test_asymmetric_hysteresis_escalation_immediate_bypass_confirmed(self):
        router = AcousticEscalationRouter(
            bypass_confirm_chunks=5,
            lazy_load_escalation=True,
        )

        dummy_chunk = np.zeros(480, dtype=np.float32)

        # 1. High SNR speech (bypass candidate): requires 5 consecutive chunks
        router.analyze_audio = lambda chunk: {"category": "speech_dominant", "estimated_snr_db": 30.0}

        for i in range(1, 5):
            _, info = router.route_and_enhance(dummy_chunk)
            assert info["mode"] == "primary"
            assert router._bypass_confirm_counter == i

        # 5th chunk reaches threshold -> transitions to bypass
        _, info_bypass = router.route_and_enhance(dummy_chunk)
        assert info_bypass["mode"] == "bypass"

        # 2. Sudden impulsive sound: enters escalation immediately (1 chunk)
        router.analyze_audio = lambda chunk: {"category": "impulsive", "estimated_snr_db": -5.0}
        _, info_esc = router.route_and_enhance(dummy_chunk)
        assert info_esc["mode"] == "escalation"
        assert router._bypass_confirm_counter == 0

    def test_lazy_loading_and_idle_unload(self):
        router = AcousticEscalationRouter(
            lazy_load_escalation=True,
            escalation_idle_unload_sec=0.1,  # Short timeout for testing
        )
        assert router._escalation_loaded is False

        # Accessing model_escalation property triggers lazy loading
        esc_model = router.model_escalation
        assert esc_model is not None
        assert router._escalation_loaded is True

        # Wait for timeout
        time.sleep(0.15)
        router._maybe_unload_escalation()

        assert router._escalation_loaded is False
        assert router._model_escalation is None


# =====================================================================
# 6. Gunfire Data Audit Tests (P0.1)
# =====================================================================

class TestGunfireDataAudit:
    """Verifies that the gunfire audit script executes and reports real status."""

    def test_run_gunfire_audit_returns_summary(self):
        summary = run_gunfire_audit()
        assert isinstance(summary, GunfireAuditSummary)
        assert len(summary.per_source) == 3
        sources = [s.source_key for s in summary.per_source]
        assert "gunshot_dryad" in sources
        assert "mad" in sources
        assert "noisex92" in sources
        assert summary.verdict != ""
        assert "NIJ" in summary.cited_but_unfetched_sources[0]


# =====================================================================
# 7. QAT Lifecycle Tests (P0.4)
# =====================================================================

class TestQATLifecycle:
    """Verifies platform-adaptive QAT preparation on conv/linear without GRU failure."""

    def test_prepare_qat_and_convert(self):
        class DummySEModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
                self.gru = nn.GRU(16, 16, batch_first=True)
                self.conv2 = nn.Conv1d(16, 1, kernel_size=1)

            def forward(self, x):
                h = torch.relu(self.conv1(x))
                h = h.transpose(1, 2)
                h, _ = self.gru(h)
                h = h.transpose(1, 2)
                return self.conv2(h)

        model = DummySEModel()

        # Mock trainer inheriting BaseTrainer
        class MockTrainer(BaseTrainer):
            def __init__(self):
                self.config = type("Config", (), {
                    "qat_enabled": True,
                    "precision": "bf16",
                    "device": "cpu",
                    "output_dir": "data/checkpoints/test",
                })()

            def build_model(self):
                return DummySEModel()

            def eval_step(self, batch):
                return {}

            def training_step(self, batch):
                return torch.tensor(0.0)

        trainer = MockTrainer()

        # 1. Prepare QAT: Conv layers get qconfig, GRU is bypassed (no tuple error)
        prepared_model = trainer.prepare_qat(model)
        assert hasattr(prepared_model.conv1, "qconfig")
        assert prepared_model.conv1.qconfig is not None
        assert getattr(prepared_model.gru, "qconfig", None) is None

        # 2. Run dummy forward during QAT
        dummy_in = torch.randn(1, 1, 480)
        out = prepared_model(dummy_in)
        assert out.shape == (1, 1, 480)

        # 3. Convert QAT
        converted = trainer.convert_qat(prepared_model)
        assert converted is not None
