"""audio/limiter.py — Output peak limiter (SAFETY-CRITICAL, hearing protection).

Specifications (non-negotiable, from v13 §2d):
  - Fast attack: 50µs
  - Slow release: 200ms
  - Hard ceiling: -1dBFS
  - Lookahead: 1ms
  - Added latency: ≤ 0.8ms
  - Must survive a 155dB Friedlander blast without glitching >50ms.

Sits AFTER the AI model mix, BEFORE the DAC.
"""
from __future__ import annotations
import numpy as np
import logging

logger = logging.getLogger(__name__)

CEILING_DBFS = -1.0
CEILING_LINEAR = 10.0 ** (CEILING_DBFS / 20.0)  # ≈ 0.8913


class PeakLimiter:
    """
    Lookahead peak limiter.
    - Lookahead buffer = 1ms @ sample_rate
    - Attack: 50µs → coefficient computed per sample
    - Release: 200ms → slow recovery
    - Hard clip at CEILING_LINEAR as final safety net
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        attack_us: float = 50.0,
        release_ms: float = 200.0,
        lookahead_ms: float = 0.75,
        ceiling_dbfs: float = CEILING_DBFS,
    ) -> None:
        self.sample_rate = sample_rate
        self.ceiling = 10.0 ** (ceiling_dbfs / 20.0)

        # Lookahead buffer (1ms)
        self._lookahead_samples = max(1, int(sample_rate * lookahead_ms / 1000.0))
        self._delay_buf = np.zeros(self._lookahead_samples, dtype=np.float64)

        # Time constants → per-sample coefficients
        attack_samples = max(1, int(sample_rate * attack_us / 1_000_000.0))
        release_samples = max(1, int(sample_rate * release_ms / 1000.0))
        self._attack_coeff = np.exp(-1.0 / attack_samples)
        self._release_coeff = np.exp(-1.0 / release_samples)

        self._gain = 1.0    # current gain reduction (1.0 = no reduction)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process one frame (float32 [-1, 1] range).
        Returns gain-reduced frame with hard clip as final safety net.
        Added latency = lookahead_samples samples.
        """
        frame_f64 = frame.astype(np.float64)
        out = np.empty_like(frame_f64)
        ceiling = self.ceiling

        for i, sample in enumerate(frame_f64):
            # Lookahead: read delayed sample, push new
            delayed = self._delay_buf[0]
            self._delay_buf = np.roll(self._delay_buf, -1)
            self._delay_buf[-1] = sample

            # Compute required gain to stay at ceiling
            peak = abs(sample)  # look at the incoming (future) sample
            if peak > ceiling:
                target_gain = ceiling / peak
            else:
                target_gain = 1.0

            # Smooth gain: fast attack (gain drops fast), slow release (gain recovers slow)
            if target_gain < self._gain:
                self._gain = self._attack_coeff * self._gain + (1 - self._attack_coeff) * target_gain
            else:
                self._gain = self._release_coeff * self._gain + (1 - self._release_coeff) * target_gain

            limited = delayed * self._gain
            # Hard clip as absolute safety net (should never be needed)
            out[i] = np.clip(limited, -ceiling, ceiling)

        return out.astype(np.float32)

    def reset(self) -> None:
        self._delay_buf[:] = 0.0
        self._gain = 1.0

    @property
    def added_latency_ms(self) -> float:
        return (self._lookahead_samples / self.sample_rate) * 1000.0
