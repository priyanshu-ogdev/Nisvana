"""
Project AEGIS — Augmentor Package
Grounded per-source augmentation engine strictly adhering to Part 1 and Part 2 policies.
"""

from .policy import AugmentationPolicyEngine, ForbiddenPitchShiftError
from .time_stretch import WsolaTimeStretcher
from .gain_jitter import GainJitter
from .blast_window import BlastOnsetWindow
from .engine import AugmentationEngine

__all__ = [
    "AugmentationPolicyEngine",
    "ForbiddenPitchShiftError",
    "WsolaTimeStretcher",
    "GainJitter",
    "BlastOnsetWindow",
    "AugmentationEngine",
]
