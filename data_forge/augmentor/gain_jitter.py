"""
Project AEGIS — Transparent Gain / Level Jitter
Applies calibrated level jitter within perceptual transparency bounds (+/- 1.5 to 2.5 dB).
"""

import numpy as np
from .policy import AugmentationPolicyEngine


class GainJitter:
    """Applies randomized or specified gain variation within safe acoustic limits."""

    def __init__(self, max_jitter_db: float = 2.0):
        self.max_jitter_db = AugmentationPolicyEngine.validate_gain_jitter_db(max_jitter_db)

    def process(self, audio: np.ndarray, jitter_db: float = 0.0) -> np.ndarray:
        """
        Applies gain jitter to audio.
        """
        AugmentationPolicyEngine.validate_gain_jitter_db(jitter_db)
        linear_gain = 10.0 ** (jitter_db / 20.0)
        jittered = audio * linear_gain

        # Peak safety check
        peak = np.max(np.abs(jittered))
        if peak > 0.99:
            jittered = jittered * (0.99 / peak)

        return jittered.astype(np.float32)
