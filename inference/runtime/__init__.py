"""Project AEGIS — Real-Time Audio Runtime"""

from .hybrid_anc import NormalizedLMSFilter, HybridAncPipeline
from .audio_stream import AudioRingBuffer, StreamingAudioProcessor
from .escalation_router import AcousticEscalationRouter
from .multichannel_frontend import MultichannelHardwareFrontend, HardwareFrontendConfig

# NOTE: this __init__ eagerly imports every submodule above, including
# hybrid_anc/escalation_router which require torch. MultichannelHardwareFrontend
# itself has ZERO torch dependency (pure numpy) -- but importing it through
# this package init still requires torch to be installed, because the
# eager imports above run first. If a pure-DSP-only deployment (no AI
# models, e.g. a lightweight preprocessing stage on a device that doesn't
# run the ML models at all) is ever needed, import
# `inference.runtime.multichannel_frontend` directly by module path rather
# than through this package -- that bypasses the eager torch imports
# entirely. Not restructured to lazy imports in this pass: doing so safely
# needs to be verified against a real torch install, which wasn't
# available in the environment this fix was made in.

__all__ = [
    "NormalizedLMSFilter",
    "HybridAncPipeline",
    "AudioRingBuffer",
    "StreamingAudioProcessor",
    "AcousticEscalationRouter",
    "MultichannelHardwareFrontend",
    "HardwareFrontendConfig",
]
