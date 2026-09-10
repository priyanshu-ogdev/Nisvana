"""audio/rnnoise_vad.py — VAD + SNR state classifier using noisereduce.

On Pi: ideally wraps libpyrnnoise for per-10ms frame VAD.
On MacBook dev: uses noisereduce spectral approach as placeholder.

Outputs per frame: {vad_speech: bool, snr_state: str, estimated_snr_db: float}
SNR states: "high" (>0dB), "mid" (-10–0dB), "low" (-15–-10dB), "severe" (<-15dB)
"""
from __future__ import annotations
import numpy as np
import logging

logger = logging.getLogger(__name__)


class RNNoiseVAD:
    """
    Voice Activity Detection + SNR state classifier.
    Frame size: 480 samples (10ms @ 48kHz).
    """

    def __init__(self, sample_rate: int = 48000, frame_size: int = 480) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self._voice_hold_frames = 8   # hold VAD active for 80ms after last voice frame
        self._hold_counter = 0
        self._noise_floor = 1e-6
        self._smoothed_snr = 0.0

    def process(self, frame: np.ndarray) -> dict:
        """
        Process one 10ms frame.
        Returns: {"vad_speech": bool, "snr_state": str, "estimated_snr_db": float}
        """
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)) + 1e-10)
        rms_db = 20.0 * np.log10(rms)

        # Slow-track noise floor (update only when signal is quiet)
        if rms < self._noise_floor * 3:
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * rms

        noise_db = 20.0 * np.log10(self._noise_floor + 1e-10)
        snr_db = rms_db - noise_db

        # Smooth SNR estimate
        self._smoothed_snr = 0.8 * self._smoothed_snr + 0.2 * snr_db

        # VAD decision: voice if SNR > 6dB
        is_voice = self._smoothed_snr > 6.0
        if is_voice:
            self._hold_counter = self._voice_hold_frames
        elif self._hold_counter > 0:
            self._hold_counter -= 1
            is_voice = True  # hold

        # SNR state bucketing
        snr = self._smoothed_snr
        if snr > 0.0:
            snr_state = "high"
        elif snr > -10.0:
            snr_state = "mid"
        elif snr > -15.0:
            snr_state = "low"
        else:
            snr_state = "severe"

        return {
            "vad_speech": bool(is_voice),
            "snr_state": snr_state,
            "estimated_snr_db": round(float(self._smoothed_snr), 1),
        }

    def reset(self) -> None:
        self._hold_counter = 0
        self._noise_floor = 1e-6
        self._smoothed_snr = 0.0
