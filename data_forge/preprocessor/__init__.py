"""
Project AEGIS — Preprocessor Package
Implements the 10-Step Sequential Preprocessing Pipeline
"""

from .pipeline import PreprocessingPipeline
from .step1_format import FormatStandardizer
from .step2_resample import Resampler
from .step3_loudness import LoudnessNormalizer
from .step4_vad import VadValidator
from .step5_channel import ChannelStandardizer
from .step6_integrity import IntegrityChecker
from .step7_metadata import MetadataTagger
from .step8_dedup import Deduplicator
from .step9_license import LicenseFilter, LicenseComplianceMode
from .step10_split import Splitter

__all__ = [
    "PreprocessingPipeline",
    "FormatStandardizer",
    "Resampler",
    "LoudnessNormalizer",
    "VadValidator",
    "ChannelStandardizer",
    "IntegrityChecker",
    "MetadataTagger",
    "Deduplicator",
    "LicenseFilter",
    "LicenseComplianceMode",
    "Splitter",
]
