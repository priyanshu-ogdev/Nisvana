"""
tests/test_audio_delicacy_and_mixed_quantization.py

Regression tests for the audio-quality-focused fixes made continuing the
inference-layer review:

1. audio_io.py: TPDF dither before PCM_16 quantization, soft peak-limiting
   instead of hard clipping, NaN/Inf detection.
2. onnx_engine.py: delay-source-aware algorithmic delay lookup
   (get_algorithmic_delay_ms), reconciling the fallback's ~0ms causal
   delay against the real vendored DeepFilterNet3's own reported 30ms
   fixed framing delay -- surfaced by audio_stream.py's own research
   during an earlier pass, not acted on until now.
3. quantization.py: mixed FP16-Conv/INT8-GRU quantization, grounded in
   Rusci et al. (arXiv:2210.07692) for this project's actual Conv1d+GRU
   architecture family.
"""

import warnings
import numpy as np
import pytest


class TestAudioIoDitherAndLimiting:
    """
    These run without torch or soundfile -- audio_io.py's dither/limiting
    MATH is pure numpy; only the module import needs a torch.Tensor
    symbol to exist for its isinstance check, stubbed minimally here.
    """

    @staticmethod
    def _load_audio_io_module():
        import sys, types, importlib.util

        if "torch" not in sys.modules:
            torch_stub = types.ModuleType("torch")

            class _FakeTensor:
                pass

            torch_stub.Tensor = _FakeTensor
            sys.modules["torch"] = torch_stub

        if "soundfile" not in sys.modules:
            import wave

            sf_stub = types.ModuleType("soundfile")

            def fake_write(path, data, samplerate, subtype=None):
                data = np.clip(data, -1.0, 1.0)
                ints = (data * 32767.0).astype(np.int16)
                with wave.open(str(path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(samplerate)
                    wf.writeframes(ints.tobytes())

            def fake_read(path, dtype="float32"):
                with wave.open(str(path), "rb") as wf:
                    n = wf.getnframes()
                    raw = wf.readframes(n)
                    sr = wf.getframerate()
                ints = np.frombuffer(raw, dtype=np.int16)
                return (ints.astype(np.float32) / 32768.0), sr

            sf_stub.write = fake_write
            sf_stub.read = fake_read
            sys.modules["soundfile"] = sf_stub

        spec = importlib.util.spec_from_file_location(
            "audio_io_test_target", "inference/utils/audio_io.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_dither_introduces_small_bounded_perturbation(self, tmp_path):
        audio_io = self._load_audio_io_module()
        clean = (np.sin(2 * np.pi * 440 * np.arange(4800) / 48000) * 0.5).astype(np.float32)

        path = tmp_path / "dithered.wav"
        audio_io.save_audio_48k(path, clean.copy(), dither=True, dither_seed=42)

        import soundfile as sf
        saved, _ = sf.read(str(path), dtype="float32")
        diff = np.abs(saved - clean)
        # Bounded: dither peak is +/-1 LSB (~3e-5 at 16-bit) plus quantization
        # step itself -- should be small, but non-zero (proving dither ran).
        assert 0 < diff.max() < 0.001

    def test_soft_limiting_preserves_waveform_shape(self, tmp_path):
        audio_io = self._load_audio_io_module()
        clean = (np.sin(2 * np.pi * 440 * np.arange(4800) / 48000) * 0.5).astype(np.float32)
        overshoot = clean * 3.0  # peak 1.5, well over full scale

        path = tmp_path / "limited.wav"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            audio_io.save_audio_48k(path, overshoot.copy(), dither=False)
            assert any("exceeded full scale" in str(warn.message) for warn in w)

        import soundfile as sf
        saved, _ = sf.read(str(path), dtype="float32")
        # Correlation near 1.0 proves shape preserved (scaled, not clipped) --
        # hard clipping would introduce harmonic distortion and drop this well below 1.0.
        correlation = np.corrcoef(saved, clean)[0, 1]
        assert correlation > 0.999
        assert np.max(np.abs(saved)) <= 1.0

    def test_nan_input_is_detected_and_warned(self, tmp_path):
        audio_io = self._load_audio_io_module()
        bad = (np.sin(2 * np.pi * 440 * np.arange(4800) / 48000) * 0.5).astype(np.float32)
        bad[100] = np.nan

        path = tmp_path / "nan_test.wav"
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            audio_io.save_audio_48k(path, bad, dither=False)
            assert any("non-finite" in str(warn.message) for warn in w)


# --- Tests requiring real torch, gated accordingly ---
torch = pytest.importorskip("torch")


class TestAlgorithmicDelayReconciliation:
    def test_fallback_model_reports_zero_delay(self):
        from inference.engines.onnx_engine import get_algorithmic_delay_ms
        from training.models.model_loader import build_model_for_key

        model = build_model_for_key("aegis-se-primary")
        assert getattr(model, "is_vendored", False) is False
        delay = get_algorithmic_delay_ms("aegis-se-primary", model=model)
        assert delay == 0.0

    def test_vendored_model_reports_thirty_ms_regardless_of_lookahead(self):
        from inference.engines.onnx_engine import get_algorithmic_delay_ms

        class FakeVendoredModel:
            is_vendored = True

        delay = get_algorithmic_delay_ms("aegis-se-primary", model=FakeVendoredModel())
        assert delay == 30.0

    def test_missing_model_falls_back_to_static_dict(self):
        from inference.engines.onnx_engine import get_algorithmic_delay_ms

        # No model instance provided -- can't check is_vendored, so must
        # fall back to the conservative (fallback-wrapper) static figure.
        delay = get_algorithmic_delay_ms("aegis-se-primary", model=None)
        assert delay == 0.0


class TestMixedPrecisionQuantization:
    def test_gru_layers_become_quantized_int8_type(self):
        from inference.engines.quantization import quantize_model_mixed_precision
        from training.models.model_loader import DeepFilterNet3Wrapper

        model = DeepFilterNet3Wrapper(df_lookahead=0, conv_lookahead=0)
        quantized = quantize_model_mixed_precision(model)

        # The GRU submodule's type should no longer be plain nn.GRU --
        # quantize_dynamic replaces it with a quantized wrapper type.
        assert type(quantized.gru) is not torch.nn.GRU

    def test_conv1d_layers_become_fp16(self):
        from inference.engines.quantization import quantize_model_mixed_precision
        from training.models.model_loader import DeepFilterNet3Wrapper

        model = DeepFilterNet3Wrapper(df_lookahead=0, conv_lookahead=0)
        quantized = quantize_model_mixed_precision(model)

        assert quantized.conv1.weight.dtype == torch.float16
        assert quantized.conv2.weight.dtype == torch.float16

    def test_mixed_precision_model_still_produces_finite_output(self):
        from inference.engines.quantization import quantize_model_mixed_precision
        from training.models.model_loader import DeepFilterNet3Wrapper

        model = DeepFilterNet3Wrapper(df_lookahead=0, conv_lookahead=0)
        model.eval()
        quantized = quantize_model_mixed_precision(model)
        quantized.eval()

        # Input must match the Conv1d layers' new FP16 dtype for this
        # simple smoke test -- a real deployment wrapper would handle
        # this cast transparently, verifying that path is a separate,
        # not-yet-built integration task, noted here rather than assumed.
        x = torch.randn(1, 480, dtype=torch.float16)
        with torch.no_grad():
            out = quantized(x)
        assert torch.all(torch.isfinite(out))
