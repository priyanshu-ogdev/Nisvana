"""Project AEGIS — Model Architectures & Loader"""

from .model_loader import (
    DeepFilterNet3Wrapper,
    CleanUMambaWrapper,
    AudioClassifierNet,
    AecFilterNet,
    build_model_for_key,
)

from .gated_inference import (
    ConfidenceGatedEnhancer,
    GatedEnhancementConfig,
)

__all__ = [
    "DeepFilterNet3Wrapper",
    "CleanUMambaWrapper",
    "AudioClassifierNet",
    "AecFilterNet",
    "build_model_for_key",
    "ConfidenceGatedEnhancer",
    "GatedEnhancementConfig",
]
