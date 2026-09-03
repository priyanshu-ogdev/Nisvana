"""
Project AEGIS — Step 4: Silence and VAD-based Validation
Trims leading and trailing silence and discards silent or corrupted clips.
"""

from typing import Optional, Tuple
import numpy as np
from data_forge.config import (
    MIN_ACTIVE_DURATION_SEC,
    SILENCE_ENERGY_THRESHOLD_DB,
    TARGET_SAMPLE_RATE,
)


class VadValidator:
    """Step 4: VAD speech trimming and dead clip removal."""

    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        min_active_sec: float = MIN_ACTIVE_DURATION_SEC,
        silence_threshold_db: float = SILENCE_ENERGY_THRESHOLD_DB,
        frame_len_ms: float = 30.0,
    ):
        self.sample_rate = sample_rate
        self.min_active_sec = min_active_sec
        self.silence_threshold_db = silence_threshold_db
        self.frame_len = int(sample_rate * (frame_len_ms / 1000.0))

    def process(self, audio: np.ndarray) -> Tuple[bool, np.ndarray, str]:
        """
        Evaluates audio clip, trims leading/trailing silence, and flags invalid clips.
        Returns: (is_valid, trimmed_audio, status_reason)
        """
        if audio is None or len(audio) == 0:
            return False, np.array([], dtype=np.float32), "Empty audio array"

        # If multichannel, compute energy across all channels
        if audio.ndim > 1:
            mono_ref = np.mean(audio, axis=1)
        else:
            mono_ref = audio

        # Check total duration
        total_sec = len(mono_ref) / self.sample_rate
        if total_sec < 0.1:
            return False, audio, f"Too short total duration: {total_sec:.3f}s"

        # Compute frame energies
        num_frames = len(mono_ref) // self.frame_len
        if num_frames == 0:
            return False, audio, "Insufficient samples for framing"

        frames = mono_ref[: num_frames * self.frame_len].reshape(num_frames, self.frame_len)
        frame_energy = np.mean(frames**2, axis=1)
        frame_db = 10.0 * np.log10(np.maximum(frame_energy, 1e-10))

        # Identify active speech/sound frames
        active_frames = np.where(frame_db > self.silence_threshold_db)[0]

        if len(active_frames) == 0:
            return False, audio, "All frames below silence threshold (dead recording)"

        # Trim leading and trailing silence with safety margin (3 frames = ~90ms padding)
        pad = 3
        start_frame = max(0, active_frames[0] - pad)
        end_frame = min(num_frames, active_frames[-1] + 1 + pad)

        start_sample = start_frame * self.frame_len
        end_sample = min(len(audio), end_frame * self.frame_len)

        trimmed_audio = audio[start_sample:end_sample]
        active_duration = len(trimmed_audio) / self.sample_rate

        if active_duration < self.min_active_sec:
            return False, trimmed_audio, f"Active duration {active_duration:.2f}s < minimum {self.min_active_sec:.2f}s"

        return True, trimmed_audio, f"Valid active audio ({active_duration:.2f}s)"
