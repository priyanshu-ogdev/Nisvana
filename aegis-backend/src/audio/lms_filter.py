"""audio/lms_filter.py — NLMS Adaptive Filter on reference mic.

Subtracts correlated reference-mic content from primary-mic stream.
Runs BEFORE DeepFilterNet3.

NLMS: μ=0.05, 512 taps @ 48kHz (~10.7ms).
Pure numpy — Cython wrapper can be added for further speedup.
"""
from __future__ import annotations
import numpy as np


class NLMSFilter:
    """
    Normalized Least Mean Squares adaptive filter.
    Primary mic = signal + noise.
    Reference mic = correlated noise only.
    Output: primary - estimated_noise.
    """

    def __init__(
        self,
        mu: float = 0.05,
        filter_length: int = 512,
        eps: float = 1e-6,
    ) -> None:
        self.mu = mu
        self.filter_length = filter_length
        self.eps = eps
        self._weights = np.zeros(filter_length, dtype=np.float32)
        self._ref_buffer = np.zeros(filter_length, dtype=np.float32)

    def process_frame(
        self,
        primary: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """
        Process one frame (1D float32 arrays of equal length).
        Returns the noise-subtracted primary signal.
        """
        n = len(primary)
        output = np.empty(n, dtype=np.float32)

        for i in range(n):
            # Shift reference into buffer
            self._ref_buffer = np.roll(self._ref_buffer, 1)
            self._ref_buffer[0] = reference[i]

            # Estimated noise
            y = np.dot(self._weights, self._ref_buffer)

            # Error signal (what we keep)
            e = primary[i] - y
            output[i] = e

            # NLMS weight update
            power = np.dot(self._ref_buffer, self._ref_buffer) + self.eps
            self._weights += (self.mu / power) * e * self._ref_buffer

        return output

    def reset(self) -> None:
        self._weights[:] = 0.0
        self._ref_buffer[:] = 0.0
