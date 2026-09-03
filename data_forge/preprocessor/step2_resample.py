"""
Project AEGIS — Step 2: Sample-Rate Standardization
Resamples to 48kHz using high-quality polyphase resampler with Sync-Tier Tagging.
Tier 1: Native 48kHz (passes through unchanged)
Tier 2: 44.1kHz (resamples cleanly: up=160, down=147)
Tier 3: 16kHz-native (upsampled: up=3, down=1; tagged in metadata)
"""

import math
from typing import Tuple
import numpy as np
import scipy.signal as sps
from data_forge.config import TARGET_SAMPLE_RATE, SyncTier


class Resampler:
    """Step 2: High-quality polyphase resampler with sync tier attribution."""

    def __init__(self, target_sr: int = TARGET_SAMPLE_RATE):
        self.target_sr = target_sr

    def process(self, audio: np.ndarray, orig_sr: int) -> Tuple[np.ndarray, int, SyncTier]:
        """
        Resamples audio to target sample rate (48,000 Hz) using polyphase filtering.
        Returns: (resampled_audio, target_sr, sync_tier)
        """
        if orig_sr == self.target_sr:
            # Tier 1: Native 48kHz passes through completely untouched
            return audio, self.target_sr, SyncTier.TIER_1_NATIVE_48K

        # Determine sync tier
        if abs(orig_sr - 44100) < 100:
            sync_tier = SyncTier.TIER_2_RESAMPLED_44K
        elif orig_sr <= 24000:
            sync_tier = SyncTier.TIER_3_UPSAMPLED_16K
        else:
            sync_tier = SyncTier.TIER_2_RESAMPLED_44K

        # Calculate rational resampling factors
        gcd = math.gcd(orig_sr, self.target_sr)
        up = self.target_sr // gcd
        down = orig_sr // gcd

        # Perform polyphase resampling along the time axis (axis 0)
        resampled = sps.resample_poly(audio, up, down, axis=0).astype(np.float32)

        return resampled, self.target_sr, sync_tier
