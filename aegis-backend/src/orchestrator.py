"""orchestrator.py — Bidirectional Mesh Node Audio Pipeline.

Phase 1 Mesh Refactor: manages two concurrent real-time audio paths.

Path A — Upstream (Local Mic → Hub):
  1. Capture: ring_buffer reads 10ms frames from local mic (or synthetic fallback)
  2. DSP/AI: LMS(network_reference) → HarmonicPreproc → RNNoise VAD
             → DeepFilterNet3 (ThreadPoolExecutor) → GatedAEC → Limiter
  3. Encode: AudioCodec compresses enhanced PCM
  4. Transmit: MeshNodeClient packs binary frame and enqueues for WS send

Path B — Downstream (Hub → Local Speaker):
  1. Receive: MeshNodeClient.recv_audio_frame() yields inbound binary frames
  2. Decode: AudioCodec decompresses payload → float32 PCM
  3. AEC Injection: decoded PCM → GatedAEC.update_reference() (echo prevention)
  4. Playback: MeshPlaybackBuffer.push_network_audio() → jitter-buffered DAC write
  5. Record: MeshRecorder.write() → logs/mesh_audio_<NODE_ID>.wav

Telemetry:
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

from .ws.node_client import MeshNodeClient
from .ws.protocol import FftStream, AncState, Telemetry, HwStatus
from .audio.lms_filter import NLMSFilter
from .audio.rnnoise_vad import RNNoiseVAD
from .audio.harmonic_preproc import HarmonicPreprocessor
from .audio.limiter import PeakLimiter
from .audio.ringbuffer import SpscRingBuffer
from .audio.aec import GatedAEC
from .audio.codec import AudioCodec
from .audio.playback import MeshPlaybackBuffer
from .audio.recorder import MeshRecorder
from .ai.deepfilternet3 import DeepFilterNet3
from .ai.snr_state_fusion import SnrStateFusion
from .ai.model_loader import ModelLoader
from .dsp.fft_exporter import FftExporter
from .dsp.spl_meter import SplMeter
from .dsp.telemetry import TelemetryCollector
from .hardware.thermal_guard import ThermalGuard

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000
FRAME_SIZE = 480   # 10ms
FFT_RATE = 30
ANC_RATE = 10
TELEMETRY_RATE = 1


class Orchestrator:
    """
    Bidirectional mesh node pipeline manager.

    Runs five concurrent asyncio tasks:
      _upstream_task()    — mic capture → DSP/AI → encode → send to Hub
      _downstream_task()  — recv from Hub → AEC inject → playback buffer → recorder
      _playback_task()    — drain playback buffer → DAC
      _telemetry_task()   — FFT / ANC state / telemetry WS broadcast
      _thermal.monitor_forever() — model swap on thermal event
    """

    def __init__(
        self,
        client: MeshNodeClient,
        config: dict,
        codec: Optional[AudioCodec] = None,
        playback_buffer: Optional[MeshPlaybackBuffer] = None,
        recorder: Optional[MeshRecorder] = None,
        alsa_bridge=None,
    ) -> None:
        self._client = client
        self._config = config

        # DSP chain
        self._lms = NLMSFilter(mu=0.05, filter_length=512)
        self._vad = RNNoiseVAD(SAMPLE_RATE, FRAME_SIZE)
        self._harmonic = HarmonicPreprocessor(sample_rate=SAMPLE_RATE)
        self._limiter = PeakLimiter(SAMPLE_RATE)
        self._aec = GatedAEC(SAMPLE_RATE, FRAME_SIZE)

        # Mesh-specific subsystems
        self._codec = codec or AudioCodec(SAMPLE_RATE, FRAME_SIZE)
        self._playback_buffer = playback_buffer or MeshPlaybackBuffer(FRAME_SIZE)
        self._recorder = recorder  # Optional; None = no disk logging

        # ALSA bridge for DAC output (injected or lazy-created)
        self._alsa = alsa_bridge

        # FFT, SPL, telemetry
        self._fft = FftExporter(sample_rate=SAMPLE_RATE)
        self._spl = SplMeter(SAMPLE_RATE)
        self._telemetry = TelemetryCollector()

        # AI model
        self._model_loader = ModelLoader()
        self._model_name = "none"
        self._fusion: Optional[SnrStateFusion] = None
        self._executor = ThreadPoolExecutor(max_workers=1)

        # Mic ring buffer
        self._ring = SpscRingBuffer.from_ms(500, SAMPLE_RATE, FRAME_SIZE)

        # Thermal guard
        self._thermal = ThermalGuard(on_tier_change=self._on_thermal_tier_change)

        self._running = False
        self._last_infer_ms = 0.0

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    async def _on_thermal_tier_change(self, model_name: str, tier: str) -> None:
        logger.warning(f"Thermal tier change → model={model_name}, tier={tier}")
        if tier in ("full", "tier1"):
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
        logger.info(f"Loaded model: {self._model_name}  codec: {self._codec.mode}")

    # ------------------------------------------------------------------
    # Synthetic frame fallback (used when no real mic is available)
    # ------------------------------------------------------------------

    def _generate_synthetic_frame(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        freq = 440.0
        noise_level = 0.3 + 0.1 * np.sin(t * 0.7)
        voice_level = 0.5 * abs(np.sin(t * 0.3))

        t_arr = t + np.arange(FRAME_SIZE) / SAMPLE_RATE
        voice = (voice_level * np.sin(2 * np.pi * freq * t_arr)).astype(np.float32)
        noise = (noise_level * np.random.randn(FRAME_SIZE)).astype(np.float32)
        reference_noise = (noise_level * 0.8 * np.random.randn(FRAME_SIZE)).astype(np.float32)

        primary = voice + noise
        return primary, reference_noise

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run_forever(self) -> None:
        """Launch all pipeline tasks concurrently."""
        self._running = True
        logger.info("Orchestrator: starting bidirectional mesh pipeline")
        logger.info(f"  Codec mode : {self._codec.mode}")
        logger.info(f"  Recorder   : {'enabled → ' + self._recorder.wav_path if self._recorder else 'disabled'}")

        await asyncio.gather(
            self._upstream_task(),
            self._downstream_task(),
            self._playback_task(),
            self._telemetry_task(),
            self._thermal.monitor_forever(),
        )

    # ------------------------------------------------------------------
    # Path A: Upstream — Mic → DSP/AI → Encode → Hub
    # ------------------------------------------------------------------

    def _run_inference_sync(self, frame: np.ndarray, snr_state: str) -> np.ndarray:
        if self._fusion:
            try:
                return self._fusion.process_frame(frame, snr_state)
            except Exception as e:
                logger.error(f"Inference error, passing through frame: {e}")
                return frame
        return frame

    async def _upstream_task(self) -> None:
        """
        Continuously capture mic frames, run the full DSP/AI pipeline,
        encode, and send to Hub.
        """
        logger.info("Upstream task started")
        while self._running:
            t_frame = time.monotonic()
            t_wall = time.time()

            # 1. Acquire mic frame
            raw_primary = self._ring.read()
            if raw_primary is None:
                raw_primary, _ = self._generate_synthetic_frame(t_wall)

            is_muted = self._client.muted.get("primary_mic", False)
            if is_muted:
                raw_primary = np.zeros(FRAME_SIZE, dtype=np.float32)

            # 2. Get echo reference from network (set by downstream_task)
            #    This is the core of mesh AEC: the reference is what the speaker
            #    is playing (downstream network audio), not a local reference mic.
            network_reference = self._aec.get_network_reference()

            # 3. LMS adaptive filter (subtract correlated reference noise)
            after_lms = self._lms.process_frame(raw_primary, network_reference)

            # 4. VAD
            vad_result = self._vad.process(after_lms)

            # 5. Harmonic preprocessor
            if self._thermal.harmonic_enabled:
                after_harmonic = self._harmonic.process(after_lms, vad_result["snr_state"])
            else:
                after_harmonic = after_lms

            # 6. AI inference (thread pool to unblock WS)
            t_infer_start = time.monotonic()
            enhanced = await asyncio.get_running_loop().run_in_executor(
                self._executor,
                self._run_inference_sync,
                after_harmonic,
                vad_result["snr_state"],
            )
            infer_ms = (time.monotonic() - t_infer_start) * 1000
            self._last_infer_ms = 0.8 * self._last_infer_ms + 0.2 * infer_ms

            # 7. AEC gate (crossfade gated by VAD)
            after_aec = self._aec.process_frame(enhanced, network_reference, vad_result["vad_speech"])

            # 8. Peak limiter (SAFETY-CRITICAL — must always run)
            limited = self._limiter.process_frame(after_aec)

            # 9. Telemetry capture (used by _telemetry_task)
            self._telemetry.record_frame(raw_primary, limited)
            self._spl.update(raw_primary, limited)

            # Store last VAD result for telemetry task
            self._last_vad = vad_result
            self._last_raw_primary = raw_primary
            self._last_limited = limited
            self._last_is_muted = is_muted

            # 10. Encode and send to Hub
            if self._client.connected:
                try:
                    encoded = self._codec.encode(limited)
                    await self._client.send_audio_frame(encoded)
                except Exception as e:
                    logger.debug(f"Upstream encode/send error: {e}")

            # Frame timing: target 10ms per frame
            elapsed = time.monotonic() - t_frame
            sleep = max(0.0, (FRAME_SIZE / SAMPLE_RATE) - elapsed)
            await asyncio.sleep(sleep)

    # ------------------------------------------------------------------
    # Path B: Downstream — Hub → Decode → AEC inject → Playback → Recorder
    # ------------------------------------------------------------------

    async def _downstream_task(self) -> None:
        """
        Receive binary audio frames from Hub, decode them,
        inject into AEC as echo reference, buffer for playback, and log.
        """
        logger.info("Downstream task started")
        while self._running:
            try:
                # Wait for next inbound frame from Hub
                network_frame = await self._client.recv_audio_frame()

                # 1. Decode compressed payload → float32 PCM
                pcm = self._codec.decode(network_frame.payload)
                network_frame.pcm = pcm  # attach decoded PCM to frame

                # 2. AEC reference injection — CRITICAL for echo prevention
                #    This tells the AEC what audio the speaker is playing,
                #    so the upstream mic path can cancel it out.
                self._aec.update_reference(pcm)

                # 3. Push to jitter buffer for DAC playback
                await self._playback_buffer.push_network_audio(pcm)

                # 4. Log to disk (async, non-blocking)
                if self._recorder is not None:
                    self._recorder.write(pcm)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Downstream task error: {e}")
                await asyncio.sleep(0.001)

    # ------------------------------------------------------------------
    # Playback: drain jitter buffer → DAC
    # ------------------------------------------------------------------

    async def _playback_task(self) -> None:
        """
        Drain the playback jitter buffer and write PCM to the local DAC.
        Runs at 10ms frame cadence. Outputs silence during network gaps.
        """
        logger.info("Playback task started")
        while self._running:
            t_frame = time.monotonic()

            # Get next frame from jitter buffer (or silence if empty)
            pcm = await self._playback_buffer.pop_for_dac()

            # Write to DAC via ALSA bridge (if hardware is available)
            if self._alsa is not None:
                try:
                    self._alsa.write_frame(pcm)
                except Exception as e:
                    logger.debug(f"DAC write error: {e}")
            # If no ALSA bridge (dev mode), frames are silently consumed from the buffer

            elapsed = time.monotonic() - t_frame
            sleep = max(0.0, (FRAME_SIZE / SAMPLE_RATE) - elapsed)
            await asyncio.sleep(sleep)

    # ------------------------------------------------------------------
    # Telemetry: FFT / ANC state / system telemetry at fixed rates
    # ------------------------------------------------------------------

    async def _telemetry_task(self) -> None:
        """Emit FFT, ANC state, and telemetry to Hub at configured rates."""
        logger.info("Telemetry task started")
        last_anc_emit = time.monotonic()
        last_telemetry_emit = time.monotonic()

        # Initialise shared state (set by _upstream_task each frame)
        self._last_vad = {"vad_speech": False, "snr_state": "unknown"}
        self._last_raw_primary = np.zeros(FRAME_SIZE, dtype=np.float32)
        self._last_limited = np.zeros(FRAME_SIZE, dtype=np.float32)
        self._last_is_muted = False

        while self._running:
            await asyncio.sleep(1.0 / FFT_RATE)  # ~33ms tick

            raw_primary = self._last_raw_primary
            limited = self._last_limited
            vad_result = self._last_vad
            is_muted = self._last_is_muted

            now = time.monotonic()

            # FFT export (30fps binary)
            if self._fft.should_emit():
                raw_bins = self._fft.compute_bins(raw_primary)
                enhanced_bins = self._fft.compute_bins(limited)
                if is_muted:
                    raw_bins = [0] * 64
                    enhanced_bins = [0] * 64
                self._client.push_fft_binary(raw_bins, enhanced_bins)

            # ANC state (10Hz JSON)
            out_db, in_db = self._spl.update(raw_primary, limited)
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

            # System telemetry (1Hz JSON)
            if now - last_telemetry_emit >= 1.0 / TELEMETRY_RATE:
                last_telemetry_emit = now
                tel = self._telemetry.collect(self._model_name)
                tel["aec_active"] = self._aec.is_active
                tel["inference_ms"] = round(self._last_infer_ms, 2)
                tel["network_ms"] = round(
                    max(0.0, tel["latency_ms"] - tel["inference_ms"]), 2
                )

                stats = self._client.get_stats()
                tel["node_rtt_ms"] = stats["node_rtt_ms"]
                tel["dropped_frames"] = stats["dropped_frames"]
                tel["queue_depth"] = stats["queue_depth"]
                tel["link_quality"] = stats["link_quality"]

                if self._fusion:
                    tel["snr_state"] = self._fusion.snr_state
                    tel["blend_weight"] = self._fusion.blend_weight

                # Playback buffer stats
                pb_stats = self._playback_buffer.stats()
                tel["playback_buffer_ms"] = pb_stats["buffer_ms"]
                tel["playback_dropped"] = pb_stats["total_dropped"]

                tel_msg = Telemetry(**tel)
                self._client.push_telemetry(tel_msg)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def stop(self) -> None:
        self._running = False
        self._thermal.stop()
        self._executor.shutdown(wait=False)
        if self._recorder is not None:
            self._recorder.close()
        logger.info("Orchestrator: stopped")
