"""
Project AEGIS — Blast Onset Windowing (Explosions & Gunshots)
Controlled time-domain windowing around shockwave arrival.
Preserves initial transient rise time while creating varied temporal alignments.
"""

from typing import Tuple
import numpy as np


class BlastOnsetWindow:
    """
    Detects blast shockwave onset and creates temporal window variations
    without altering physical explosion dynamics.
    """

    def __init__(self, sample_rate: int = 48000, onset_threshold_ratio: float = 0.25):
        self.sample_rate = sample_rate
        self.onset_threshold_ratio = onset_threshold_ratio

    def find_onset(self, audio: np.ndarray) -> int:
        """Finds the sharp rise point of the blast shockwave."""
        abs_wf = np.abs(audio)
        peak = np.max(abs_wf)
        threshold = peak * self.onset_threshold_ratio
        onset_indices = np.where(abs_wf >= threshold)[0]
        return int(onset_indices[0]) if len(onset_indices) > 0 else 0

    def process(
        self,
        audio: np.ndarray,
        target_length_samples: int,
        onset_offset_samples: int = 0,
    ) -> np.ndarray:
        """
        Windows audio so that the blast onset arrives at a controlled or jittered offset.
        """
        onset_idx = self.find_onset(audio)
        start = max(0, onset_idx - onset_offset_samples)
        end = start + target_length_samples

        if end <= len(audio):
            windowed = audio[start:end]
        else:
            windowed = audio[start:]
            # Pad with decaying ambient silence
            pad_len = target_length_samples - len(windowed)
            windowed = np.pad(windowed, (0, pad_len))

        return windowed.astype(np.float32)
