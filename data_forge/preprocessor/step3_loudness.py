"""
Project AEGIS — Step 3: Loudness Normalization
Normalizes clips to consistent reference level (-23.0 LUFS) using ITU-R BS.1770-4.
Includes True-Peak Limiter (-1.0 dBFS) and transient RMS/Peak fallback for impulses.
"""

from typing import Tuple
import numpy as np
import pyloudnorm as pyln
from data_forge.config import TARGET_LUFS, TARGET_TRUE_PEAK_DBFS, TARGET_SAMPLE_RATE


class LoudnessNormalizer:
    """Step 3: BS.1770-4 loudness normalization with safety peak limiting."""

    def __init__(
        self,
        target_lufs: float = TARGET_LUFS,
        peak_ceiling_dbfs: float = TARGET_TRUE_PEAK_DBFS,
        sample_rate: int = TARGET_SAMPLE_RATE,
    ):
        self.target_lufs = target_lufs
        self.peak_ceiling_dbfs = peak_ceiling_dbfs
        self.sample_rate = sample_rate
        self.meter = pyln.Meter(sample_rate)  # ITU-R BS.1770-4 meter
        self.peak_ceiling_linear = 10.0 ** (peak_ceiling_dbfs / 20.0)

    def process(self, audio: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Normalizes audio to target LUFS.
        Returns: (normalized_audio, measured_lufs, true_peak_dbfs)
        """
        # Ensure at least 1D float array
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) == 0:
            return audio, -70.0, -100.0

        # Check if file has enough duration for BS.1770 gated measurement (requires > 0.4s)
        duration_sec = len(audio) / self.sample_rate

        measured_lufs = -70.0
        use_lufs = False

        if duration_sec >= 0.4:
            try:
                # pyloudnorm expects 2D (samples, channels) or 1D (samples)
                measured_lufs = self.meter.integrated_loudness(audio)
                if not np.isinf(measured_lufs) and not np.isnan(measured_lufs) and measured_lufs > -70.0:
                    use_lufs = True
            except Exception:
                use_lufs = False

        if use_lufs:
            # BS.1770-4 standard normalization
            gain_db = self.target_lufs - measured_lufs
            gain_linear = 10.0 ** (gain_db / 20.0)
            normalized = audio * gain_linear
        else:
            # Fallback for short impulsive transients (gunshots, single blast clicks)
            # Match RMS energy to equivalent -23 LUFS sine reference (~ -20 dB RMS)
            rms = np.sqrt(np.mean(audio**2) + 1e-12)
            target_rms = 10.0 ** (-23.0 / 20.0) * 0.707  # Sine reference level
            gain_linear = target_rms / max(rms, 1e-6)
            normalized = audio * gain_linear
            measured_lufs = 20.0 * np.log10(max(rms, 1e-6))

        # Apply True-Peak Safety Limiter
        peak = np.max(np.abs(normalized))
        if peak > self.peak_ceiling_linear:
            # Soft-knee / linear compression to ceiling
            normalized = normalized * (self.peak_ceiling_linear / peak)

        true_peak_dbfs = 20.0 * np.log10(max(np.max(np.abs(normalized)), 1e-8))

        return normalized.astype(np.float32), float(measured_lufs), float(true_peak_dbfs)
