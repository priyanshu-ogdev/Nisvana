"""Project AEGIS — Inference Utilities"""

from .audio_io import load_audio_48k, save_audio_48k
from .transmission_prep import prepare_for_transmission, TransmissionPrepConfig
from .sih_metrics import (
    SIH_TARGET_SNR_DB,
    SIH_TARGET_DELTA_SNR_DB,
    SIH_TARGET_STOI,
    SIH_TARGET_PESQ,
    SIH_TARGET_MAX_RTF,
    SihEvaluationResult,
    evaluate_sih_compliance,
)

__all__ = [
    "load_audio_48k",
    "save_audio_48k",
    "prepare_for_transmission",
    "TransmissionPrepConfig",
    "SIH_TARGET_SNR_DB",
    "SIH_TARGET_DELTA_SNR_DB",
    "SIH_TARGET_STOI",
    "SIH_TARGET_PESQ",
    "SIH_TARGET_MAX_RTF",
    "SihEvaluationResult",
    "evaluate_sih_compliance",
]
