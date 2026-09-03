"""Project AEGIS — Real-Time Edge Inference Suite & Hybrid ANC Engine"""

from .runtime.hybrid_anc import NormalizedLMSFilter, HybridAncPipeline
from .runtime.audio_stream import AudioRingBuffer, StreamingAudioProcessor
from .runtime.escalation_router import AcousticEscalationRouter
from .engines.onnx_engine import export_to_onnx, benchmark_edge_latency
from .engines.onnx_runtime_engine import OnnxRuntimeSession
from .engines.quantization import quantize_model_dynamic
from .utils.audio_io import load_audio_48k, save_audio_48k
from .utils.sih_metrics import (
    SIH_TARGET_SNR_DB,
    SIH_TARGET_DELTA_SNR_DB,
    SIH_TARGET_STOI,
    SIH_TARGET_PESQ,
    SIH_TARGET_MAX_RTF,
    SihEvaluationResult,
    evaluate_sih_compliance,
)

__all__ = [
    "NormalizedLMSFilter",
    "HybridAncPipeline",
    "AudioRingBuffer",
    "StreamingAudioProcessor",
    "AcousticEscalationRouter",
    "export_to_onnx",
    "benchmark_edge_latency",
    "OnnxRuntimeSession",
    "quantize_model_dynamic",
    "load_audio_48k",
    "save_audio_48k",
    "SIH_TARGET_SNR_DB",
    "SIH_TARGET_DELTA_SNR_DB",
    "SIH_TARGET_STOI",
    "SIH_TARGET_PESQ",
    "SIH_TARGET_MAX_RTF",
    "SihEvaluationResult",
    "evaluate_sih_compliance",
]
