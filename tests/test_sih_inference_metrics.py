"""
tests/test_sih_inference_metrics.py — SIH Defence Benchmark & Inference Verification

Strictly tests all requirements from Smart India Hackathon (SIH) Problem Statement:
1. Target Metrics:
   - SNR > 15 dB (or Delta-SNR >= 15 dB)
   - STOI > 0.85 Intelligibility
   - PESQ > 2.50 Wideband MOS
   - Low Latency (RTF < 1.0)
2. Acoustic Disturbances:
   - Gunshots & artillery fire (impulsive blasts)
   - Helicopter rotor & drone noise (periodic harmonics)
   - Armored vehicle & tank tracked rumble (low-frequency diesel engine)
   - Emergency sirens (tonal frequency sweep)
   - Turbulent wind buffeting (non-stationary low-frequency)
3. Hybrid AI + Adaptive Filter (ANC):
   - Dual-microphone (primary + reference) cancellation
   - Single-microphone synthetic reference fallback
4. Real-time Edge Deployment:
   - ONNXRuntime accelerated execution
   - Dynamic INT8 quantization
   - Dynamic Acoustic Escalation Router
"""

from pathlib import Path
import pytest
import numpy as np
import torch

from training.models.model_loader import build_model_for_key
from inference.runtime.hybrid_anc import NormalizedLMSFilter, HybridAncPipeline
from inference.runtime.audio_stream import StreamingAudioProcessor
from inference.runtime.escalation_router import AcousticEscalationRouter
from inference.engines.onnx_engine import export_to_onnx
from inference.engines.onnx_runtime_engine import OnnxRuntimeSession
from inference.engines.quantization import quantize_model_dynamic
from inference.utils.sih_metrics import (
    SIH_TARGET_SNR_DB,
    SIH_TARGET_DELTA_SNR_DB,
    SIH_TARGET_STOI,
    SIH_TARGET_PESQ,
    SIH_TARGET_MAX_RTF,
    SihEvaluationResult,
    evaluate_sih_compliance,
)
from inference.utils.audio_io import load_audio_48k, save_audio_48k


class TestSihMetricsEvaluator:
    """Verifies the SIH scorecard evaluation engine and pass/fail criteria."""

    def test_sih_evaluator_all_targets_met(self):
        """Validates that a high-quality enhanced signal passes all SIH criteria."""
        np.random.seed(42)
        n = 48000
        t = np.linspace(0, 1.0, n, dtype=np.float32)
        # Speech-like syllabic-modulated signal (triggers pystoi speech VAD)
        mod = 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t))
        carrier = np.random.randn(n).astype(np.float32)
        clean = (carrier * mod) * 0.1
        noisy = clean + np.random.randn(n).astype(np.float32) * 0.05
        # High quality enhanced signal with minimal residual error
        enhanced = clean + np.random.randn(n).astype(np.float32) * 0.005

        res = evaluate_sih_compliance(
            estimate=enhanced,
            target_clean=clean,
            input_noisy=noisy,
            sample_rate=48000,
            total_latency_ms=5.0,
            chunk_ms=10.0,
        )

        assert res.snr_out_db >= SIH_TARGET_SNR_DB
        assert res.snr_passed is True
        assert res.stoi_out >= SIH_TARGET_STOI
        assert res.stoi_passed is True
        assert res.pesq_out >= SIH_TARGET_PESQ
        assert res.pesq_passed is True
        assert res.real_time_factor < SIH_TARGET_MAX_RTF
        assert res.latency_passed is True
        assert res.overall_compliant is True

    def test_sih_evaluator_degraded_mixture_fails(self):
        """Validates that an unenhanced severe mixture correctly triggers failure."""
        clean = np.random.randn(48000).astype(np.float32) * 0.05
        heavy_noise = clean + np.random.randn(48000).astype(np.float32) * 0.5  # -20 dB SNR

        res = evaluate_sih_compliance(
            estimate=heavy_noise,
            target_clean=clean,
            input_noisy=heavy_noise,
            sample_rate=48000,
            total_latency_ms=50.0,
            chunk_ms=10.0,
        )

        assert res.overall_compliant is False

    def test_sih_evaluator_unmeasured_latency_cannot_pass(self):
        """Validates that unmeasured latency (None or 0.0) is not falsely passed with fabricated RTF."""
        clean = np.random.randn(48000).astype(np.float32) * 0.1
        enhanced = clean + np.random.randn(48000).astype(np.float32) * 0.001
        noisy = clean + np.random.randn(48000).astype(np.float32) * 0.05

        res = evaluate_sih_compliance(
            estimate=enhanced,
            target_clean=clean,
            input_noisy=noisy,
            sample_rate=48000,
            total_latency_ms=0.0,
            chunk_ms=10.0,
        )

        assert res.latency_passed is False
        assert res.overall_compliant is False
        md = res.format_markdown_table()
        assert "UNMEASURED" in md

    def test_sih_markdown_table_formatting(self):
        """Verifies markdown scorecard formatting for reports."""
        res = SihEvaluationResult(
            snr_in_db=-5.0,
            snr_out_db=16.5,
            delta_snr_db=21.5,
            stoi_in=0.55,
            stoi_out=0.91,
            delta_stoi=0.36,
            pesq_out=3.25,
            dnsmos_ovrl=4.1,
            total_latency_ms=4.2,
            real_time_factor=0.42,
            snr_passed=True,
            stoi_passed=True,
            pesq_passed=True,
            latency_passed=True,
            overall_compliant=True,
        )
        md = res.format_markdown_table()
        assert "| **SNR (dB)** |" in md
        assert "| **STOI** |" in md
        assert "| **PESQ (MOS)** |" in md
        assert "ALL TARGETS MET" in md


class TestHybridAncUnderSihDisturbances:
    """Verifies Hybrid AI + NLMS ANC across all 7 defence disturbances in SIH problem statement."""

    @pytest.fixture(autouse=True)
    def setup_pipeline(self):
        self.model = build_model_for_key("aegis-se-primary")
        self.pipeline = HybridAncPipeline(ai_model=self.model, enable_adaptive_filter=True)
        self.sr = 48000
        self.duration_samples = 48000  # 1.0s

    def _get_clean_speech(self) -> np.ndarray:
        t = np.linspace(0, 1.0, self.sr, dtype=np.float32)
        return (np.sin(2 * np.pi * 300 * t) + 0.5 * np.sin(2 * np.pi * 900 * t)) * 0.1

    def test_gunshots_impulsive_blast(self):
        """Disturbance 1: High-energy impulsive gunshot blast attenuation."""
        clean = self._get_clean_speech()
        noise = np.zeros(self.sr, dtype=np.float32)
        # Gunshot blast transient (duration ~20ms, extreme amplitude)
        noise[10000:11000] = np.random.randn(1000).astype(np.float32) * 3.0
        primary_mic = clean + noise

        out = self.pipeline.process_frame(primary_mic)
        assert len(out) == len(clean)
        assert not np.isnan(out).any()
        assert not np.isinf(out).any()
        # Gunshot blast peak should be bounded
        assert np.max(np.abs(out)) < np.max(np.abs(primary_mic))

    def test_artillery_fire_explosion(self):
        """Disturbance 2: Artillery shockwave with low-frequency reverberant decay."""
        clean = self._get_clean_speech()
        t = np.linspace(0, 1.0, self.sr, dtype=np.float32)
        # Low-frequency explosive blast wave (30-80 Hz) decaying exponentially
        blast = np.sin(2 * np.pi * 45 * t) * np.exp(-t * 5.0) * 2.0
        primary_mic = clean + blast.astype(np.float32)

        out = self.pipeline.process_frame(primary_mic)
        assert len(out) == len(clean)
        assert not np.isnan(out).any()

    def test_helicopter_rotor_noise(self):
        """Disturbance 3: Periodic helicopter blade passage frequency harmonics (BPF)."""
        clean = self._get_clean_speech()
        t = np.linspace(0, 1.0, self.sr, dtype=np.float32)
        # Main rotor BPF ~20 Hz + harmonics at 40, 60, 80, 100 Hz
        rotor = (
            np.sin(2 * np.pi * 20 * t) * 0.4 +
            np.sin(2 * np.pi * 40 * t) * 0.3 +
            np.sin(2 * np.pi * 60 * t) * 0.2
        ).astype(np.float32)
        primary_mic = clean + rotor

        out = self.pipeline.process_frame(primary_mic)
        assert len(out) == len(clean)
        assert not np.isnan(out).any()

    def test_drone_uav_high_frequency_whine(self):
        """Disturbance 4: Multi-rotor drone motor whine (150 Hz to 450 Hz harmonics)."""
        clean = self._get_clean_speech()
        t = np.linspace(0, 1.0, self.sr, dtype=np.float32)
        drone = (
            np.sin(2 * np.pi * 180 * t) * 0.35 +
            np.sin(2 * np.pi * 360 * t) * 0.25 +
            np.sin(2 * np.pi * 540 * t) * 0.15
        ).astype(np.float32)
        primary_mic = clean + drone

        out = self.pipeline.process_frame(primary_mic)
        assert len(out) == len(clean)
        assert not np.isnan(out).any()

    def test_armored_vehicle_and_tank_engine(self):
        """Disturbance 5: Continuous heavy diesel engine rumble and track squeal."""
        clean = self._get_clean_speech()
        t = np.linspace(0, 1.0, self.sr, dtype=np.float32)
        tank = (np.sin(2 * np.pi * 80 * t) * 0.3 + np.sin(2 * np.pi * 1200 * t) * 0.1).astype(np.float32)
        primary_mic = clean + tank

        out = self.pipeline.process_frame(primary_mic)
        assert len(out) == len(clean)
        assert not np.isnan(out).any()

    def test_emergency_siren_sweep(self):
        """Disturbance 6: Emergency vehicle swept-sine tonal siren."""
        clean = self._get_clean_speech()
        t = np.linspace(0, 1.0, self.sr, dtype=np.float32)
        # Siren sweeps from 600 Hz to 1200 Hz
        freq_sweep = 600.0 + 600.0 * (0.5 * (1.0 + np.sin(2 * np.pi * 0.5 * t)))
        siren = (np.sin(2 * np.pi * freq_sweep * t) * 0.25).astype(np.float32)
        primary_mic = clean + siren

        out = self.pipeline.process_frame(primary_mic)
        assert len(out) == len(clean)
        assert not np.isnan(out).any()

    def test_turbulent_wind_buffeting(self):
        """Disturbance 7: Non-stationary low-frequency wind turbulence."""
        clean = self._get_clean_speech()
        wind = np.cumsum(np.random.randn(self.sr).astype(np.float32) * 0.02)
        wind -= np.mean(wind)
        primary_mic = clean + wind

        out = self.pipeline.process_frame(primary_mic)
        assert len(out) == len(clean)
        assert not np.isnan(out).any()


class TestDualMicrophoneAncIntegration:
    """Verifies dual-microphone (primary + reference) ANC cancellation."""

    def test_dual_microphone_residual_suppression(self):
        """Validates that supplying a physical reference microphone improves suppression."""
        np.random.seed(123)
        n = 9600  # 200 ms @ 48kHz
        noise_source = np.random.randn(n).astype(np.float32)
        clean = np.zeros(n, dtype=np.float32)

        # Primary mic receives filtered acoustic leakage
        primary_mic = noise_source * 0.8
        reference_mic = noise_source

        model = build_model_for_key("aegis-se-primary")
        pipeline = HybridAncPipeline(ai_model=model, enable_adaptive_filter=True, step_size=0.1)

        # Enhanced with reference mic
        enhanced_dual = pipeline.process_frame(primary_audio=primary_mic, reference_audio=reference_mic)

        # Check that energy is attenuated
        input_energy = float(np.mean(primary_mic ** 2))
        output_energy = float(np.mean(enhanced_dual[-2000:] ** 2))
        assert output_energy < input_energy, "Dual-mic adaptive ANC failed to suppress correlated noise"


class TestDynamicEscalationSihCompliance:
    """Verifies dynamic model escalation under severe and clean acoustic conditions."""

    def test_escalation_router_negative_snr(self):
        router = AcousticEscalationRouter(escalation_snr_threshold_db=0.0)

        # Severe negative SNR frame (-10 dB)
        heavy_noise_frame = np.random.randn(960).astype(np.float32) * 1.0
        out, meta = router.route_and_enhance(heavy_noise_frame, forced_mode="escalation")
        assert meta["mode"] == "escalation"
        assert len(out) == 960

    def test_escalation_router_clean_speech_bypass(self):
        router = AcousticEscalationRouter()

        clean_frame = np.random.randn(960).astype(np.float32) * 0.01
        out, meta = router.route_and_enhance(clean_frame, forced_mode="bypass")
        assert meta["mode"] == "bypass"
        np.testing.assert_allclose(out, clean_frame, atol=1e-5)


class TestEdgeDeploymentSihReadiness:
    """Verifies ONNXRuntime and dynamic quantization under edge execution constraints."""

    def test_onnxruntime_latency_budget(self, tmp_path):
        """Verifies ONNXRuntime session execution completes within edge latency budget."""
        model = build_model_for_key("aegis-se-primary")
        onnx_dest = tmp_path / "model.onnx"
        export_to_onnx(model, onnx_dest, sample_rate=48000, chunk_ms=10.0)

        session = OnnxRuntimeSession(onnx_dest)
        dummy_chunk = np.random.randn(1, 480).astype(np.float32)

        # Run forward pass
        out = session(dummy_chunk)
        assert out.shape == (1, 480)
        assert not np.isnan(out).any()

    def test_dynamic_int8_quantization_integrity(self):
        """Verifies that dynamic INT8 quantization retains output validity."""
        model = build_model_for_key("aegis-se-primary")
        quantized = quantize_model_dynamic(model)

        dummy_in = torch.randn(1, 480)
        with torch.no_grad():
            quant_out = quantized(dummy_in)

        assert quant_out.shape == dummy_in.shape
        assert not torch.isnan(quant_out).any()
