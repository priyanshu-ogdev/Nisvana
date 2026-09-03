"""
Project AEGIS — Waveform Similarity Overlap-Add (WSOLA) Time-Stretching
Modifies playback rate within strictly bounded +/- 5-10% without altering pitch or formant structures.
"""

from typing import Optional
import numpy as np
import scipy.signal as sps
from .policy import AugmentationPolicyEngine


class WsolaTimeStretcher:
    """
    Time-domain WSOLA implementation that preserves vehicle engine harmonics
    and eliminates pitch/timbral shifts.
    """

    def __init__(self, win_size_ms: float = 40.0, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.win_size = int(sample_rate * (win_size_ms / 1000.0))
        self.hop_syn = self.win_size // 2
        self.window = np.hanning(self.win_size)

    def process(self, audio: np.ndarray, rate: float = 1.05) -> np.ndarray:
        """
        Stretches audio in time by factor `rate` (rate > 1 means faster/shorter, rate < 1 means slower/longer).
        Enforces 0.90 <= rate <= 1.10.
        """
        AugmentationPolicyEngine.validate_time_stretch_rate(rate)

        if abs(rate - 1.0) < 0.001:
            return audio

        audio = np.asarray(audio, dtype=np.float32)
        hop_ana = int(self.hop_syn * rate)
        max_delta = self.hop_syn // 2

        num_output_frames = int(len(audio) / (self.hop_syn * rate))
        output_len = num_output_frames * self.hop_syn + self.win_size
        output = np.zeros(output_len, dtype=np.float32)
        norm_weights = np.zeros(output_len, dtype=np.float32)

        input_pos = 0
        output_pos = 0

        for frame_idx in range(num_output_frames):
            target_pos = int(frame_idx * hop_ana)
            if target_pos + self.win_size + max_delta >= len(audio):
                break

            # Search around target_pos for maximum cross-correlation with previous frame
            best_pos = target_pos
            if frame_idx > 0 and (target_pos - max_delta) >= 0:
                search_region = audio[target_pos - max_delta : target_pos + max_delta + self.win_size]
                ref_frame = audio[input_pos : input_pos + self.win_size]
                corr = np.correlate(search_region, ref_frame, mode="valid")
                if len(corr) > 0:
                    best_offset = int(np.argmax(corr)) - max_delta
                    best_pos = target_pos + best_offset

            # Overlap-add windowed segment
            segment = audio[best_pos : best_pos + self.win_size] * self.window
            output[output_pos : output_pos + self.win_size] += segment
            norm_weights[output_pos : output_pos + self.win_size] += self.window**2

            input_pos = best_pos
            output_pos += self.hop_syn

        # Normalize by window weights
        valid_idx = norm_weights > 1e-4
        output[valid_idx] = output[valid_idx] / norm_weights[valid_idx]

        # Trim tail
        last_valid = np.where(valid_idx)[0]
        if len(last_valid) > 0:
            output = output[: last_valid[-1] + 1]

        return output.astype(np.float32)
