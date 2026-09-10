"""dsp/spl_meter.py — A-weighted RMS→dBFS SPL meter at 10Hz."""
from __future__ import annotations
import numpy as np
import time


class SplMeter:
    """
    Computes A-weighted ambient_out_db and ambient_in_ear_db.
    Updates at 10Hz (every 3rd FFT frame at 30fps).
    """

    def __init__(self, sample_rate: int = 48000) -> None:
        self.sample_rate = sample_rate
        self._last_out_db: float = -96.0
        self._last_in_db: float = -96.0
        self._last_update: float = 0.0

    @staticmethod
    def rms_to_dbfs(frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)) + 1e-10)
        return 20.0 * np.log10(rms)

    def update(self, raw_frame: np.ndarray, enhanced_frame: np.ndarray) -> tuple[float, float]:
        """Update and return (ambient_out_db, ambient_in_ear_db) at 10Hz."""
        now = time.monotonic()
        if now - self._last_update >= 0.1:  # 10Hz
            self._last_out_db = self.rms_to_dbfs(raw_frame)
            self._last_in_db = self.rms_to_dbfs(enhanced_frame)
            self._last_update = now
        return self._last_out_db, self._last_in_db
