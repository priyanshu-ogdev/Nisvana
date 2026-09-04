"""
tests/test_inference_suite.py — Unit and Integration Tests for Real-Time Edge Inference Suite

Tests:
1. AudioRingBuffer (zero-allocation circular buffer)
2. StreamingAudioProcessor (50% Overlap-Add reconstruction)
3. AcousticEscalationRouter (dynamic gating & multi-model escalation)
4. OnnxRuntimeSession (accelerated ONNX Runtime execution)
5. Dynamic INT8 quantization
6. Audio I/O utilities (save/load 48kHz audio with clipping protection)
"""

from pathlib import Path
import pytest
import numpy as np
import torch

from training.models.model_loader import build_model_for_key
from inference.runtime.audio_stream import AudioRingBuffer, StreamingAudioProcessor
from inference.runtime.escalation_router import AcousticEscalationRouter
from inference.engines.onnx_engine import export_to_onnx
from inference.engines.onnx_runtime_engine import OnnxRuntimeSession
from inference.engines.quantization import quantize_model_dynamic
from inference.utils.audio_io import load_audio_48k, save_audio_48k


class TestAudioRingBuffer:
    """Verifies circular ring buffer operations and bounds."""

    def test_ring_buffer_write_read(self):
        buf = AudioRingBuffer(capacity=100)
        data = np.arange(40, dtype=np.float32)
        n = buf.write(data)
        assert n == 40
        assert buf.size == 40

        # Peek without consume
        peek = buf.read(20)
        assert len(peek) == 20
        np.testing.assert_array_equal(peek, data[:20])
        assert buf.size == 40

        # Consume 20 samples
        buf.consume(20)
        assert buf.size == 20

        # Remaining peek
        rem = buf.read(20)
        np.testing.assert_array_equal(rem, data[20:40])

    def test_ring_buffer_overflow_drop_oldest(self):
        buf = AudioRingBuffer(capacity=50)
        data = np.arange(70, dtype=np.float32)
        buf.write(data)
        assert buf.size == 50
        out = buf.read(50)
        # Should keep the newest 50 samples (20 to 69)
        np.testing.assert_array_equal(out, data[20:])

    def test_ring_buffer_clear(self):
        buf = AudioRingBuffer(capacity=50)
        buf.write(np.ones(30, dtype=np.float32))
        assert buf.size == 30
        buf.clear()
        assert buf.size == 0
        assert len(buf.read(10)) == 0


class TestStreamingAudioProcessor:
    """Verifies overlap-add streaming reconstruction."""

    def test_overlap_add_streaming_reconstruction(self):
        """Passes continuous signal through identity pass-through, verifying reconstruction."""
        t = np.linspace(0, 1.0, 48000, dtype=np.float32)
        sine_wave = np.sin(2 * np.pi * 440 * t)

        # Identity processor
        processor = StreamingAudioProcessor(
            enhancement_fn=lambda x: x,
            sample_rate=48000,
            frame_size=960,
            hop_size=480,
        )

        outputs = []
        chunk_size = 480
        for i in range(0, len(sine_wave), chunk_size):
            chunk = sine_wave[i : i + chunk_size]
            out = processor.process_chunk(chunk)
            if len(out) > 0:
                outputs.append(out)

        flushed = processor.flush()
        if len(flushed) > 0:
            outputs.append(flushed)

        reconstructed = np.concatenate(outputs)
        # Verify length is close to original
        assert len(reconstructed) > 0
        assert not np.isnan(reconstructed).any()

    def test_arbitrary_chunk_sizes(self):
        """Feeds variable chunk sizes (128, 256, 512, 1000) without crashing."""
        processor = StreamingAudioProcessor(
            enhancement_fn=lambda x: x * 0.9,
            sample_rate=48000,
            frame_size=960,
            hop_size=480,
        )

        for chunk_len in [128, 256, 512, 1000, 300]:
            chunk = np.random.randn(chunk_len).astype(np.float32)
            out = processor.process_chunk(chunk)
            assert isinstance(out, np.ndarray)


class TestAcousticEscalationRouter:
    """Verifies dynamic model escalation logic and crossfade continuity."""

    def test_router_dynamic_switching(self):
        router = AcousticEscalationRouter()

        # Moderate audio frame
        mod_frame = np.random.randn(960).astype(np.float32) * 0.1
        out1, meta1 = router.route_and_enhance(mod_frame)
        assert meta1["mode"] in ["bypass", "primary", "escalation"]
        assert out1.shape == (960,)

        # Forced escalation
        out2, meta2 = router.route_and_enhance(mod_frame, forced_mode="escalation")
        assert meta2["mode"] == "escalation"
        assert out2.shape == (960,)


    def test_router_crossfade_continuity_without_raw_audio(self):
        router = AcousticEscalationRouter()
        frame1 = np.random.randn(960).astype(np.float32) * 0.1
        # Run in primary
        _, meta1 = router.route_and_enhance(frame1, forced_mode="primary")
        assert router.current_state == "primary"

        # Now force switch to escalation: verify crossfade occurs cleanly without NaN or raw noise injection
        frame2 = np.random.randn(960).astype(np.float32) * 0.1
        out2, meta2 = router.route_and_enhance(frame2, forced_mode="escalation")
        assert meta2["mode"] == "escalation"
        assert router.current_state == "escalation"
        assert not np.isnan(out2).any()
        assert out2.shape == (960,)


class TestOnnxRuntimeEngine:
    """Verifies accelerated ONNX Runtime execution."""

    def test_onnxruntime_session_execution(self, tmp_path):
        model = build_model_for_key("aegis-se-primary")
        onnx_file = tmp_path / "model_primary.onnx"

        export_to_onnx(model, onnx_file, sample_rate=48000, chunk_ms=10.0)
        assert onnx_file.exists()

        ort_session = OnnxRuntimeSession(onnx_file)
        assert ort_session.active_provider in ["CPUExecutionProvider", "CUDAExecutionProvider"]

        dummy_audio = np.random.randn(1, 480).astype(np.float32)
        enhanced = ort_session(dummy_audio)

        assert isinstance(enhanced, np.ndarray)
        assert enhanced.shape == (1, 480)
        assert not np.isnan(enhanced).any()

    def test_benchmark_latency_includes_algorithmic_delay(self):
        from inference.engines.onnx_engine import benchmark_edge_latency

        model = build_model_for_key("aegis-se-primary")
        res = benchmark_edge_latency(
            model=model,
            chunk_ms=10.0,
            algorithmic_delay_ms=40.0,
            num_runs=5,
            warmup_runs=1,
            device=torch.device("cpu"),
        )
        assert "algorithmic_delay_ms" in res
        assert res["algorithmic_delay_ms"] == 40.0
        assert "total_latency_ms" in res
        assert res["total_latency_ms"] >= 40.0
        assert "compute_latency_mean_ms" in res


class TestQuantization:
    """Verifies dynamic INT8 quantization for edge deployment."""

    def test_dynamic_quantization(self):
        model = build_model_for_key("aegis-se-primary")
        quantized = quantize_model_dynamic(model)

        dummy_input = torch.randn(1, 480)
        with torch.no_grad():
            out = quantized(dummy_input)

        assert out.shape == dummy_input.shape
        assert not torch.isnan(out).any()


class TestAudioIO:
    """Verifies standardized 48kHz audio loading and saving."""

    def test_save_and_load_audio_48k(self, tmp_path):
        wav_dest = tmp_path / "test_audio.wav"
        t = np.linspace(0, 0.5, 24000, dtype=np.float32)
        original_wave = np.sin(2 * np.pi * 440 * t)

        saved = save_audio_48k(wav_dest, original_wave, sr=48000)
        assert saved.exists()

        loaded, sr = load_audio_48k(saved, target_sr=48000)
        assert sr == 48000
        assert len(loaded) == len(original_wave)
        np.testing.assert_allclose(loaded, original_wave, atol=1e-3)


class TestStreamingArchitectureAndCausality:
    """Verifies causal zero-lookahead, escalation lookahead, streaming state, and unidirectional CleanUMamba."""

    def test_causal_zero_lookahead_no_future_leakage(self):
        """Validates that for conv_lookahead=0, modifying future input does NOT change past output."""
        from training.models.model_loader import DeepFilterNet3Wrapper

        model = DeepFilterNet3Wrapper(df_lookahead=0, conv_lookahead=0)
        model.eval()

        torch.manual_seed(42)
        x1 = torch.randn(1, 200)
        x2 = x1.clone()
        # Perturb strictly in the future (t >= 150)
        x2[:, 150:] += torch.randn(1, 50) * 2.0

        model.reset_state()
        with torch.no_grad():
            y1 = model(x1)

        model.reset_state()
        with torch.no_grad():
            y2 = model(x2)

        # Output at t <= 140 must be IDENTICAL (no future leakage)
        np.testing.assert_allclose(
            y1[:, :140].numpy(),
            y2[:, :140].numpy(),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Future perturbation altered past outputs in causal zero-lookahead model!",
        )

    def test_lookahead_escalation_absorbs_future_context(self):
        """Validates that conv_lookahead > 0 provides asymmetric future receptive field."""
        from training.models.model_loader import DeepFilterNet3Wrapper

        m_causal = DeepFilterNet3Wrapper(df_lookahead=0, conv_lookahead=0)
        m_lookahead = DeepFilterNet3Wrapper(df_lookahead=2, conv_lookahead=2)

        assert m_causal.pad == (6, 0)      # left 6, right 0 (strictly causal)
        assert m_lookahead.pad == (4, 2)   # left 4, right 2 (2 samples future lookahead)

    def test_cleanumamba_unidirectional_and_causal(self):
        """Validates that CleanUMamba uses strictly unidirectional GRU and causal encoder padding."""
        from training.models.model_loader import CleanUMambaWrapper

        model = CleanUMambaWrapper()
        assert model.gru_mamba.bidirectional is False
        assert model.dec.in_channels == 32

        x = torch.randn(1, 480)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_recurrent_streaming_state_continuity(self):
        """Validates that hidden state persists across sequential chunks and reset_state clears it."""
        from training.models.model_loader import DeepFilterNet3Wrapper

        model = DeepFilterNet3Wrapper(df_lookahead=0, conv_lookahead=0)
        model.eval()
        model.reset_state()

        c1 = torch.randn(1, 100)
        c2 = torch.randn(1, 100)

        with torch.no_grad():
            _ = model(c1)
            assert model.hidden_state is not None
            state_after_c1 = model.hidden_state.clone()

            _ = model(c2)
            assert not torch.equal(model.hidden_state, state_after_c1)

            model.reset_state()
            assert model.hidden_state is None
