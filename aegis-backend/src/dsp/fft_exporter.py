"""dsp/fft_exporter.py — 64-bin log-mel FFT at 30fps, dual stream (raw + enhanced)."""
from __future__ import annotations
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

NUM_BINS = 64
FPS = 30
FRAME_INTERVAL_S = 1.0 / FPS


class FftExporter:
    """
    Computes 64-bin log-mel magnitude spectrum from a frame of audio.
    Produces both raw_bins (pre-model) and bins (enhanced output).
    Both are uint8 [0-255].

    Output shape matches protocol.FftStream exactly.
    """

    def __init__(self, sample_rate: int = 48000, fft_size: int = 2048) -> None:
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self._mel_filters = self._build_mel_filters()
        self._last_emit: float = 0.0

    def _build_mel_filters(self) -> np.ndarray:
        """Build log-mel filterbank matrix [NUM_BINS, fft_size//2+1]."""
        freqs = np.fft.rfftfreq(self.fft_size, 1.0 / self.sample_rate)
        mel_min = self._hz_to_mel(20.0)
        mel_max = self._hz_to_mel(self.sample_rate / 2)
        mel_points = np.linspace(mel_min, mel_max, NUM_BINS + 2)
        hz_points = self._mel_to_hz(mel_points)

        filters = np.zeros((NUM_BINS, len(freqs)))
        for i in range(NUM_BINS):
            f_low, f_center, f_high = hz_points[i], hz_points[i + 1], hz_points[i + 2]
            for j, f in enumerate(freqs):
                if f_low <= f <= f_center:
                    filters[i, j] = (f - f_low) / (f_center - f_low + 1e-10)
                elif f_center < f <= f_high:
                    filters[i, j] = (f_high - f) / (f_high - f_center + 1e-10)
        return filters

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def compute_bins(self, frame: np.ndarray) -> list[int]:
        """
        Compute 64 log-mel bins for one audio frame.
        Returns list of ints in [0, 255].
        """
        # Zero-pad frame to fft_size
        padded = np.zeros(self.fft_size, dtype=np.float32)
        n = min(len(frame), self.fft_size)
        padded[:n] = frame[:n]

        # Window + FFT
        window = np.hanning(self.fft_size)
        spec = np.abs(np.fft.rfft(padded * window))

        # Apply mel filters
        mel_energy = self._mel_filters @ spec
        mel_db = 20.0 * np.log10(np.maximum(mel_energy, 1e-10))

        # Normalize to [0, 255]
        mel_min, mel_max = mel_db.min(), mel_db.max()
        if mel_max - mel_min < 1e-3:
            return [0] * NUM_BINS
        normalized = (mel_db - mel_min) / (mel_max - mel_min) * 255.0
        return [int(np.clip(v, 0, 255)) for v in normalized]

    def should_emit(self) -> bool:
        """Returns True if enough time has passed for the next 30fps frame."""
        now = time.monotonic()
        if now - self._last_emit >= FRAME_INTERVAL_S:
            self._last_emit = now
            return True
        return False
