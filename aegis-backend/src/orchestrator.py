"""orchestrator.py — Binds audio → DSP → AI → WS loop.

Audio I/O threads run at SCHED_RR (real-time priority on Pi).
WS sends go through server.enqueue() — audio threads NEVER await.

Processing chain per frame (10ms):
  primary_mic → LMS(reference) → harmonic_preproc → RNNoise VAD
             → DeepFilterNet3 (or SNR blend)
             → AEC gate
             → limiter
             → DAC (headset_output)

WS push:
  30fps: fft_stream (raw_bins + bins)
  10Hz:  anc_state
  1Hz:   telemetry
"""
from __future__ import annotations
import asyncio
import time
import logging
import numpy as np
from typing import Optional

from .ws.server import AegisServer
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
    On MacBook: generates synthetic test audio when no hardware is present.
    """

    def __init__(self, server: AegisServer, config: dict) -> None:
        self._server = server
        self._config = config

        # Per-client DSP chains
        self._lms: dict[str, NLMSFilter] = {
            "person-1": NLMSFilter(mu=0.05, filter_length=512),
            "person-2": NLMSFilter(mu=0.05, filter_length=512),
        }
        self._vad: dict[str, RNNoiseVAD] = {
            "person-1": RNNoiseVAD(SAMPLE_RATE, FRAME_SIZE),
            "person-2": RNNoiseVAD(SAMPLE_RATE, FRAME_SIZE),
        }
        self._harmonic: dict[str, HarmonicPreprocessor] = {
            "person-1": HarmonicPreprocessor(sample_rate=SAMPLE_RATE),
            "person-2": HarmonicPreprocessor(sample_rate=SAMPLE_RATE),
        }
        self._limiter: dict[str, PeakLimiter] = {
            "person-1": PeakLimiter(SAMPLE_RATE),
            "person-2": PeakLimiter(SAMPLE_RATE),
        }
        self._aec: dict[str, GatedAEC] = {
            "person-1": GatedAEC(SAMPLE_RATE, FRAME_SIZE),
            "person-2": GatedAEC(SAMPLE_RATE, FRAME_SIZE),
        }

        # FFT, SPL, telemetry
        self._fft = FftExporter(sample_rate=SAMPLE_RATE)
        self._spl: dict[str, SplMeter] = {
            "person-1": SplMeter(SAMPLE_RATE),
            "person-2": SplMeter(SAMPLE_RATE),
        }
        self._telemetry = TelemetryCollector()

        # AI model
        self._model_loader = ModelLoader()
        self._model_name = "none"
        self._fusion: Optional[SnrStateFusion] = None

        # Ring buffers (filled by ALSA thread, read by DSP loop)
        self._rings: dict[str, SpscRingBuffer] = {
            "person-1": SpscRingBuffer.from_ms(500, SAMPLE_RATE, FRAME_SIZE),
            "person-2": SpscRingBuffer.from_ms(500, SAMPLE_RATE, FRAME_SIZE),
        }

        # Thermal guard
        self._thermal = ThermalGuard(on_tier_change=self._on_thermal_tier_change)

        # ALSA bridge (optional, Pi-only)
        self._alsa = None

        self._running = False

    async def _on_thermal_tier_change(self, model_name: str, tier: str) -> None:
        """Called when thermal guard triggers a model swap."""
        logger.warning(f"Thermal tier change → model={model_name}, tier={tier}")
        if tier == "full":
            model_key = "primary"
        elif tier == "tier1":
            model_key = "primary"
        elif tier == "tier2":
            model_key = "cleanumamba"
        else:
            model_key = "noisereduce"
        self._model_name = self._model_loader.swap_model(model_key)
        std_model = DeepFilterNet3(self._model_loader.get_session(), SAMPLE_RATE)
        # Note: In a real environment, we would also swap/load the low-SNR checkpoint if applicable.
        # For the prototype, we fall back to a single model.
        self._fusion = SnrStateFusion(std_model, None, SAMPLE_RATE, FRAME_SIZE)

    def load_model(self) -> None:
        self._model_name = self._model_loader.load()
        std_model = DeepFilterNet3(self._model_loader.get_session(), SAMPLE_RATE)
        self._fusion = SnrStateFusion(std_model, None, SAMPLE_RATE, FRAME_SIZE)
        logger.info(f"Loaded model: {self._model_name}")

    def _generate_synthetic_frame(self, client_id: str, t: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic primary + reference frames for MacBook dev mode.
        Simulates noise with slight voice-like modulation.
        """
        freq = 440.0 if client_id == "person-1" else 660.0
        noise_level = 0.3 + 0.1 * np.sin(t * 0.7)
        voice_level = 0.5 * np.abs(np.sin(t * 0.3))  # slow voice gate

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

        frame_count = 0
        loop_start = time.monotonic()

        await asyncio.gather(
            self._dsp_loop(),
            self._thermal.monitor_forever(),
        )

    async def _dsp_loop(self) -> None:
        """Core 10ms DSP loop."""
        last_anc_emit = time.monotonic()
        last_telemetry_emit = time.monotonic()

        while self._running:
            t_frame = time.monotonic()
            t_wall = time.time()

            for client_id in ["person-1", "person-2"]:
                session = self._server.get_session(client_id)

                # Acquire frame (from ALSA ring or synthetic)
                raw_primary = self._rings[client_id].read()
                if raw_primary is None:
                    raw_primary, raw_reference = self._generate_synthetic_frame(client_id, t_wall)
                else:
                    raw_reference = np.zeros(FRAME_SIZE, dtype=np.float32)  # Pi: read from reference ring

                # Check mute state
                is_primary_muted = False
                if session:
                    is_primary_muted = session.muted.get("primary_mic", False)

                if is_primary_muted:
                    raw_primary = np.zeros(FRAME_SIZE, dtype=np.float32)

                # 1. LMS noise subtraction
                after_lms = self._lms[client_id].process_frame(raw_primary, raw_reference)

                # 2. VAD + SNR state
                vad_result = self._vad[client_id].process(after_lms)

                # 3. Harmonic preprocessing (gated on thermal tier and SNR)
                harmonic_enabled = self._thermal.harmonic_enabled
                if harmonic_enabled:
                    after_harmonic = self._harmonic[client_id].process(after_lms, vad_result["snr_state"])
                else:
                    after_harmonic = after_lms

                # 4. AI model inference (via SNR State Fusion)
                if self._fusion:
                    enhanced = self._fusion.process_frame(after_harmonic, vad_result["snr_state"])
                else:
                    enhanced = after_harmonic

                # 4.5. Gated AEC
                after_aec = self._aec[client_id].process_frame(enhanced, raw_reference, vad_result["vad_speech"])

                # 5. Limiter (safety-critical — always runs)
                limited = self._limiter[client_id].process_frame(after_aec)

                # 6. Record for telemetry
                # Note: passing self._aec[client_id].is_active into collect() happens at 1Hz
                self._telemetry.record_frame(raw_primary, limited)

                # 7. SPL readings
                out_db, in_db = self._spl[client_id].update(raw_primary, limited)

                # 8. FFT export (30fps, both streams)
                if self._fft.should_emit():
                    raw_bins = self._fft.compute_bins(raw_primary)
                    enhanced_bins = self._fft.compute_bins(limited)
                    if is_primary_muted:
                        raw_bins = [0] * 64
                        enhanced_bins = [0] * 64

                    fft_msg = FftStream(
                        clientId=client_id,
                        bins=enhanced_bins,
                        raw_bins=raw_bins,
                        sampleRate=SAMPLE_RATE,
                        ts=int(t_wall * 1000),
                    )
                    self._server.push_fft(fft_msg.model_dump_json(), client_id)

                # 9. ANC state (10Hz)
                now = time.monotonic()
                if now - last_anc_emit >= 1.0 / ANC_RATE:
                    anc_active = session.anc_active if session else False
                    anc_msg = AncState(
                        clientId=client_id,
                        anc_active=anc_active,
                        vad_speech=vad_result["vad_speech"],
                        ambient_out_db=round(out_db, 1),
                        ambient_in_ear_db=round(in_db, 1),
                        sidetone_on=False,
                    )
                    self._server.push_anc_state(anc_msg.model_dump_json(), client_id)

            # ANC emit timer reset (once per loop covering both clients)
            now = time.monotonic()
            if now - last_anc_emit >= 1.0 / ANC_RATE:
                last_anc_emit = now

            # 10. Telemetry (1Hz)
            if now - last_telemetry_emit >= 1.0 / TELEMETRY_RATE:
                last_telemetry_emit = now
                tel = self._telemetry.collect(self._model_name)
                tel["aec_active"] = any(self._aec[c].is_active for c in ["person-1", "person-2"])
                if self._fusion:
                    tel["snr_state"] = self._fusion.snr_state
                    tel["blend_weight"] = self._fusion.blend_weight
                tel_msg = Telemetry(**tel)
                self._server.push_telemetry(tel_msg.model_dump_json())

            # Sleep for remainder of 10ms frame
            elapsed = time.monotonic() - t_frame
            sleep = max(0.0, (FRAME_SIZE / SAMPLE_RATE) - elapsed)
            await asyncio.sleep(sleep)

    def stop(self) -> None:
        self._running = False
        self._thermal.stop()
