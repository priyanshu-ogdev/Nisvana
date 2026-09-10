"""Project AEGIS — Edge Inference Engines"""

from .onnx_engine import export_to_onnx, benchmark_edge_latency
from .onnx_runtime_engine import OnnxRuntimeSession
from .onnx_model_adapter import OnnxModelAdapter, build_onnx_backed_router
from .quantization import quantize_model_dynamic, report_quantization_coverage

__all__ = [
    "export_to_onnx",
    "benchmark_edge_latency",
    "OnnxRuntimeSession",
    "OnnxModelAdapter",
    "build_onnx_backed_router",
    "quantize_model_dynamic",
    "report_quantization_coverage",
]
