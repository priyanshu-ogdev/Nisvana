"""
Project AEGIS — Training Architecture, Models, and Pipelines.

This package provides the complete deep learning training infrastructure for
fullband (48 kHz) speech enhancement, classification gating, and acoustic
echo cancellation across 5 specialized model branches.
"""

from .trainers import (
    BaseTrainer,
    SePrimaryTrainer,
    SeEscalationTrainer,
    SeCrosscheckTrainer,
    ClassifierTrainer,
    AecGateTrainer,
)
from .configs import (
    BaseModelConfig,
    SePrimaryConfig,
    SeEscalationConfig,
    SeCrosscheckConfig,
    ClassifierConfig,
    AecGateConfig,
    DfLossConfig,
    EmaConfig,
    WorstClassCheckpointConfig,
    SpecMixConfig,
    SnrCurriculumConfig,
)
from .losses import (
    build_se_loss,
    ResolvedLossConfig,
    MultiResSpectralLoss,
    LocalSnrLoss,
    SISnrLoss,
    StftConsistencyLoss,
)
from .models import (
    build_model_for_key,
    DeepFilterNet3Wrapper,
    CleanUMambaWrapper,
    AudioClassifierNet,
    AecFilterNet,
)

__all__ = [
    # Trainers
    "BaseTrainer",
    "SePrimaryTrainer",
    "SeEscalationTrainer",
    "SeCrosscheckTrainer",
    "ClassifierTrainer",
    "AecGateTrainer",
    # Configs
    "BaseModelConfig",
    "SePrimaryConfig",
    "SeEscalationConfig",
    "SeCrosscheckConfig",
    "ClassifierConfig",
    "AecGateConfig",
    "DfLossConfig",
    "EmaConfig",
    "WorstClassCheckpointConfig",
    "SpecMixConfig",
    "SnrCurriculumConfig",
    # Losses
    "build_se_loss",
    "ResolvedLossConfig",
    "MultiResSpectralLoss",
    "LocalSnrLoss",
    "SISnrLoss",
    "StftConsistencyLoss",
    # Models
    "build_model_for_key",
    "DeepFilterNet3Wrapper",
    "CleanUMambaWrapper",
    "AudioClassifierNet",
    "AecFilterNet",
]
