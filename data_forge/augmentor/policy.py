"""
Project AEGIS — Augmentation Policy and Literature-Grounded Constraints
Enforces:
1. Zero pitch/formant-altering augmentation on clean speech targets (arXiv:2407.05471).
2. Zero pitch-shifting on defence vehicle noise (tanks, howitzers, aircraft, blast physics).
3. Strictly bounded time-stretch (+/- 5-10%) and transparent gain jitter for thin datasets only (UTMOS, arXiv:2204.02152).
"""

from data_forge.config import (
    AugmentationPolicy,
    FORBIDDEN_PITCH_CLASSES,
    UnifiedClass,
)


class ForbiddenPitchShiftError(Exception):
    """Raised whenever pitch shifting is attempted on forbidden acoustic classes."""
    pass


class AugmentationPolicyEngine:
    """Validates and enforces per-source augmentation policies."""

    @staticmethod
    def assert_pitch_shift_permitted(unified_class: UnifiedClass) -> None:
        """
        Enforces strict zero pitch-shift rule on clean speech targets,
        tanks, howitzers, aircraft, naval engines, explosions, and gunshots.
        """
        if unified_class in FORBIDDEN_PITCH_CLASSES:
            raise ForbiddenPitchShiftError(
                f"PITCH SHIFTING STRICTLY FORBIDDEN on class '{unified_class.value}'. "
                f"Pitch-shifting alters vocal tract formants (arXiv:2407.05471) "
                f"or teaches physically invalid engine RPM/bore signatures (Part 1 & 2)."
            )

    @staticmethod
    def validate_time_stretch_rate(rate: float) -> float:
        """
        Enforces narrow-band time stretch [0.90, 1.10] (perceptually transparent range).
        """
        if rate < 0.90 or rate > 1.10:
            raise ValueError(
                f"Time-stretch rate {rate} exceeds grounded perceptual bounds [0.90, 1.10] (UTMOS standard)."
            )
        return rate

    @staticmethod
    def validate_gain_jitter_db(gain_db: float) -> float:
        """
        Enforces transparent level jitter bounds [-3.0 dB, +3.0 dB].
        """
        if abs(gain_db) > 3.0:
            raise ValueError(
                f"Gain jitter {gain_db} dB exceeds safe transparent level range +/- 3.0 dB."
            )
        return gain_db
