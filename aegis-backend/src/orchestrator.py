"""orchestrator.py — Binds audio → DSP → AI → WS loop.

Audio I/O threads run at SCHED_RR (real-time priority on Pi).
WS sends go through client.enqueue() — audio threads NEVER await.

Processing chain per frame (10ms):
  primary_mic → LMS(reference) → harmonic_preproc → RNNoise VAD
             → DeepFilterNet3 (or SNR blend) via ThreadPoolExecutor
             → AEC gate
             → limiter
             → DAC (headset_output)

WS push:
  30fps: fft_stream (binary)
  10Hz:  anc_state (json)
  1Hz:   telemetry (json)
"""
from __future__ import annotations
import asyncio
import time
import logging
import numpy as np
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from .ws.client import AegisClient
from .ws.protocol import FftStream, AncState, Telemetry, HwStatus
from .audio.lms_filter import NLMSFilter
from .audio.rnnoise_vad import RNNoiseVAD
from .audio.harmonic_preproc import HarmonicPreprocessor
from .audio.limiter import PeakLimiter
from .audio.ringbuffer import SpscRingBuffer
from .audio.aec import GatedAEC
from .ai.deepfilternet3 import DeepFilterNet3
from .ai.snr_state_fusion import SnrStateFusion
from .ai.model_loader import ModelLoader
from .dsp.fft_exporter import FftExporter
from .dsp.spl_meter import SplMeter
from .dsp.telemetry import TelemetryCollector
from .hardware.thermal_guard import ThermalGuard

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000
FRAME_SIZE = 480  # 10ms
FFT_RATE = 30
ANC_RATE = 10
TELEMETRY_RATE = 1


class Orchestrator:
    """
    Manages the full real-time DSP pipeline and WS broadcast schedule.
    """

    def __init__(self, client: AegisClient, config: dict) -> None:
        self._client = client
        self._config = config

        # DSP chain (single node)
        self._lms = NLMSFilter(mu=0.05, filter_length=512)
        self._vad = RNNoiseVAD(SAMPLE_RATE, FRAME_SIZE)
        self._harmonic = HarmonicPreprocessor(sample_rate=SAMPLE_RATE)
        self._limiter = PeakLimiter(SAMPLE_RATE)
        self._aec = GatedAEC(SAMPLE_RATE, FRAME_SIZE)

        # FFT, SPL, telemetry
        self._fft = FftExporter(sample_rate=SAMPLE_RATE)
        self._spl = SplMeter(SAMPLE_RATE)
        self._telemetry = TelemetryCollector()

        # AI model
        self._model_loader = ModelLoader()
        self._model_name = "none"
        self._fusion: Optional[SnrStateFusion] = None
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Ring buffer
        self._ring = SpscRingBuffer.from_ms(500, SAMPLE_RATE, FRAME_SIZE)

        # Thermal guard
        self._thermal = ThermalGuard(on_tier_change=self._on_thermal_tier_change)

        self._alsa = None
        self._running = False
        self._last_infer_ms = 0.0

    async def _on_thermal_tier_change(self, model_name: str, tier: str) -> None:
        """Called when thermal guard triggers a model swap."""
        logger.warning(f"Thermal tier change → model={model_name}, tier={tier}")
        if tier == "full" or tier == "tier1":
            model_key = "primary"
        elif tier == "tier2":
            model_key = "cleanumamba"
        else:
            model_key = "noisereduce"
        self._model_name = self._model_loader.swap_model(model_key)
        std_model = DeepFilterNet3(self._model_loader.get_session(), SAMPLE_RATE)
        self._fusion = SnrStateFusion(std_model, None, SAMPLE_RATE, FRAME_SIZE)

    def load_model(self) -> None:
        self._model_name = self._model_loader.load()
        std_model = DeepFilterNet3(self._model_loader.get_session(), SAMPLE_RATE)
        self._fusion = SnrStateFusion(std_model, None, SAMPLE_RATE, FRAME_SIZE)
        logger.info(f"Loaded model: {self._model_name}")

    def _generate_synthetic_frame(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        freq = 440.0
        noise_level = 0.3 + 0.1 * np.sin(t * 0.7)
        voice_level = 0.5 * np.abs(np.sin(t * 0.3))

        t_arr = t + np.arange(FRAME_SIZE) / SAMPLE_RATE
        voice = (voice_level * np.sin(2 * np.pi * freq * t_arr)).astype(np.float32)
        noise = (noise_level * np.random.randn(FRAME_SIZE)).astype(np.float32)
        reference_noise = (noise_level * 0.8 * np.random.randn(FRAME_SIZE)).astype(np.float32)

        primary = voice + noise
        return primary, reference_noise

    async def run_forever(self) -> None:
        """Main DSP loop — processes one 10ms frame per iteration."""
        self._running = True
        logger.info("Orchestrator loop started")

        await asyncio.gather(
            self._dsp_loop(),
            self._thermal.monitor_forever(),
        )

    def _run_inference_sync(self, frame: np.ndarray, snr_state: str) -> np.ndarray:
        if self._fusion:
            try:
                return self._fusion.process_frame(frame, snr_state)
            except Exception as e:
                logger.error(f"Inference error, passing through frame: {e}")
                return frame
        return frame

    async def _dsp_loop(self) -> None:
        """Core 10ms DSP loop."""
        last_anc_emit = time.monotonic()
        last_telemetry_emit = time.monotonic()

        while self._running:
            t_frame = time.monotonic()
            t_wall = time.time()

            # Acquire frame
            raw_primary = self._ring.read()
            if raw_primary is None:
                raw_primary, raw_reference = self._generate_synthetic_frame(t_wall)
            else:
                raw_reference = np.zeros(FRAME_SIZE, dtype=np.float32)

            is_primary_muted = self._client.muted.get("primary_mic", False)
            if is_primary_muted:
                raw_primary = np.zeros(FRAME_SIZE, dtype=np.float32)

            # 1. LMS
            after_lms = self._lms.process_frame(raw_primary, raw_reference)

            # 2. VAD
            vad_result = self._vad.process(after_lms)

            # 3. Harmonic
            if self._thermal.harmonic_enabled:
                after_harmonic = self._harmonic.process(after_lms, vad_result["snr_state"])
            else:
                after_harmonic = after_lms

            # 4. Inference (P0: thread pool to unblock WS)
            t_infer_start = time.monotonic()
            enhanced = await asyncio.get_running_loop().run_in_executor(
                self._executor, self._run_inference_sync, after_harmonic, vad_result["snr_state"]
            )
            infer_ms = (time.monotonic() - t_infer_start) * 1000
            self._last_infer_ms = 0.8 * self._last_infer_ms + 0.2 * infer_ms

            # 4.5. AEC
            after_aec = self._aec.process_frame(enhanced, raw_reference, vad_result["vad_speech"])

            # 5. Limiter
            limited = self._limiter.process_frame(after_aec)

            # 6. Telemetry capture
            self._telemetry.record_frame(raw_primary, limited)

            # 7. SPL
            out_db, in_db = self._spl.update(raw_primary, limited)

            # 8. FFT export (P1: Binary framing)
            if self._fft.should_emit():
                raw_bins = self._fft.compute_bins(raw_primary)
                enhanced_bins = self._fft.compute_bins(limited)
                if is_primary_muted:
                    raw_bins = [0] * 64
                    enhanced_bins = [0] * 64

                self._client.push_fft_binary(raw_bins, enhanced_bins)

            # 9. ANC state
            now = time.monotonic()
            if now - last_anc_emit >= 1.0 / ANC_RATE:
                last_anc_emit = now
                anc_msg = AncState(
                    clientId=self._client.node_id,
                    anc_active=self._client.anc_active,
                    vad_speech=vad_result["vad_speech"],
                    ambient_out_db=round(out_db, 1),
                    ambient_in_ear_db=round(in_db, 1),
                    sidetone_on=False,
                )
                self._client.push_anc_state(anc_msg)

            # 10. Telemetry (P2: Split latency)
            if now - last_telemetry_emit >= 1.0 / TELEMETRY_RATE:
                last_telemetry_emit = now
                tel = self._telemetry.collect(self._model_name)
                tel["aec_active"] = self._aec.is_active
                
                # Split latency: report measured inference time and network/buffering time
                tel["inference_ms"] = round(self._last_infer_ms, 2)
                tel["network_ms"] = round(max(0.0, tel["latency_ms"] - tel["inference_ms"]), 2)

                # Link health and queue telemetry
                stats = self._client.get_stats()
                tel["node_rtt_ms"] = stats.get("node_rtt_ms")
                tel["dropped_frames"] = stats.get("dropped_frames")
                tel["queue_depth"] = stats.get("queue_depth")
                tel["link_state"] = stats.get("link_state")
                
                if self._fusion:
                    tel["snr_state"] = self._fusion.snr_state
                    tel["blend_weight"] = self._fusion.blend_weight
                
                tel_msg = Telemetry(**tel)
                self._client.push_telemetry(tel_msg)

            # Sleep
            elapsed = time.monotonic() - t_frame
            sleep = max(0.0, (FRAME_SIZE / SAMPLE_RATE) - elapsed)
            await asyncio.sleep(sleep)

    def stop(self) -> None:
        self._running = False
        self._thermal.stop()
        self._executor.shutdown(wait=False)
