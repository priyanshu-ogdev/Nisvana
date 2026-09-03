"""
Project AEGIS — Step 6: Integrity and Corruption Check
Scans waveforms for clipping, all-zero files, NaN/Inf values, and truncated audio.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class IntegrityReport:
    is_clean: bool
    has_nan: bool
    has_inf: bool
    is_all_zero: bool
    clipped_sample_count: int
    clipped_ratio: float
    max_abs_amplitude: float
    rejection_reason: Optional[str] = None


class IntegrityChecker:
    """Step 6: Waveform integrity and deep audio corruption auditing."""

    def __init__(self, clip_threshold: float = 0.999, max_allowed_clip_ratio: float = 0.005):
        self.clip_threshold = clip_threshold
        self.max_allowed_clip_ratio = max_allowed_clip_ratio

    def process(self, audio: np.ndarray) -> IntegrityReport:
        """
        Audits waveform integrity.
        """
        if audio is None or len(audio) == 0:
            return IntegrityReport(
                is_clean=False,
                has_nan=False,
                has_inf=False,
                is_all_zero=True,
                clipped_sample_count=0,
                clipped_ratio=0.0,
                max_abs_amplitude=0.0,
                rejection_reason="Empty or null waveform",
            )

        # 1. NaN and Inf check
        has_nan = bool(np.isnan(audio).any())
        has_inf = bool(np.isinf(audio).any())

        if has_nan or has_inf:
            return IntegrityReport(
                is_clean=False,
                has_nan=has_nan,
                has_inf=has_inf,
                is_all_zero=False,
                clipped_sample_count=0,
                clipped_ratio=0.0,
                max_abs_amplitude=float("nan"),
                rejection_reason="Waveform contains NaN or Inf values",
            )

        # 2. All-zero check
        abs_audio = np.abs(audio)
        max_abs = float(np.max(abs_audio))
        if max_abs < 1e-7:
            return IntegrityReport(
                is_clean=False,
                has_nan=False,
                has_inf=False,
                is_all_zero=True,
                clipped_sample_count=0,
                clipped_ratio=0.0,
                max_abs_amplitude=max_abs,
                rejection_reason="All-zero or imperceptible waveform",
            )

        # 3. Hard-clipping check
        clipped_samples = int(np.sum(abs_audio >= self.clip_threshold))
        clip_ratio = clipped_samples / float(len(audio))

        if clip_ratio > self.max_allowed_clip_ratio:
            return IntegrityReport(
                is_clean=False,
                has_nan=False,
                has_inf=False,
                is_all_zero=False,
                clipped_sample_count=clipped_samples,
                clipped_ratio=clip_ratio,
                max_abs_amplitude=max_abs,
                rejection_reason=f"Severe hard-clipping detected ({clip_ratio*100:.2f}% of samples clipped)",
            )

        return IntegrityReport(
            is_clean=True,
            has_nan=False,
            has_inf=False,
            is_all_zero=False,
            clipped_sample_count=clipped_samples,
            clipped_ratio=clip_ratio,
            max_abs_amplitude=max_abs,
            rejection_reason=None,
        )
