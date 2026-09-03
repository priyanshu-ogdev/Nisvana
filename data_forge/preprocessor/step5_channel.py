"""
Project AEGIS — Step 5: Channel Standardization
Applies consistent downmixing for monaural models while supporting multichannel preservation.
"""

from typing import Tuple
import numpy as np


class ChannelStandardizer:
    """Step 5: Channel standardization and energy-preserving downmix."""

    def __init__(self, mode: str = "mono_mean"):
        """
        :param mode: 'mono_mean', 'mono_first_channel', or 'multichannel_preserve'
        """
        self.mode = mode

    def process(self, audio: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Standardizes channel configuration.
        Returns: (processed_audio, num_channels)
        """
        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim == 1:
            # Already 1D mono
            return audio, 1

        channels = audio.shape[1]
        if channels == 1:
            return audio[:, 0], 1

        if self.mode == "multichannel_preserve":
            return audio, channels

        if self.mode == "mono_first_channel":
            # Select channel 0 (primary microphone)
            return audio[:, 0], 1

        # Default: mono_mean with energy preservation
        mono = np.mean(audio, axis=1)
        # Energy compensation to match average channel power
        orig_power = np.mean(np.mean(audio**2, axis=0))
        mono_power = np.mean(mono**2) + 1e-12
        scaling = np.sqrt(orig_power / mono_power)
        mono = mono * scaling

        return mono.astype(np.float32), 1
