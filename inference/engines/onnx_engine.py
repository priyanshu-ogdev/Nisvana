"""
inference/engines/onnx_engine.py — ONNX Edge Export & Latency Profiler

Provides:
- ONNX model export with dynamic axes for NVIDIA Jetson AGX Orin & DSPs
- Precision calibration (FP32, FP16, INT8 dynamic quantization)
- Real-Time Factor (RTF) and latency budget benchmarking
"""

import os
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


# Authoritative algorithmic buffering / lookahead delay benchmarks established for AEGIS
MODEL_ALGORITHMIC_DELAY_MS: Dict[str, float] = {
    "aegis-se-primary": 0.0,       # DeepFilterNet3 zero-lookahead (strictly causal streaming)
    "aegis-se-escalation": 40.0,   # DeepFilterNet3 standard 40ms lookahead (severe SNR escalation)
    "aegis-se-crosscheck": 25.0,   # RT-SEMamba / CleanUMamba 25ms framing window
    "aegis-clf-gate": 0.0,         # Instantaneous frame gating classifier
    "aegis-aec-gate": 20.0,        # TaylorBeamformer ~20ms delay buffer
}


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    sample_rate: int = 48000,
    chunk_ms: float = 10.0,
    opset_version: int = 17,
) -> Path:
    """
    Exports a PyTorch audio model to ONNX format.
    Args:
        model: PyTorch model instance (e.g. DeepFilterNet3Wrapper, AudioClassifierNet).
        output_path: Destination .onnx file path.
        sample_rate: Sampling rate (default: 48000 Hz).
        chunk_ms: Frame duration in milliseconds (default: 10 ms = 480 samples).
        opset_version: ONNX operator set version (default: 17).
    Returns:
        Path to generated ONNX model.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    frame_len = int(sample_rate * chunk_ms / 1000.0)
    dummy_input = torch.randn(1, frame_len, dtype=torch.float32)

    dynamic_axes = {
        "input": {0: "batch_size", 1: "time_steps"},
        "output": {0: "batch_size", 1: "time_steps"},
    }

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    except TypeError:
        # Older PyTorch versions without dynamo argument
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
        )

    return output_path


def benchmark_edge_latency(
    model: nn.Module,
    sample_rate: int = 48000,
    chunk_ms: float = 10.0,
    algorithmic_delay_ms: Optional[float] = None,
    model_key: Optional[str] = None,
    num_runs: int = 50,
    warmup_runs: int = 10,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Benchmarks real-time processing latency and Real-Time Factor (RTF).
    Args:
        model: Neural speech enhancement or classifier model.
        sample_rate: 48000 Hz.
        chunk_ms: Audio chunk duration (e.g. 10 ms = 480 samples).
        algorithmic_delay_ms: Explicit buffering or lookahead delay (e.g. 0.0 ms for primary, 40.0 ms for escalation).
        model_key: Optional model key to lookup default known algorithmic delay from MODEL_ALGORITHMIC_DELAY_MS.
        num_runs: Number of timing iterations.
        warmup_runs: Number of warm-up iterations.
        device: CPU or CUDA device.
    Returns:
        Dictionary with compute latency, algorithmic delay, total latency, and RTF metrics.
    """
    if algorithmic_delay_ms is None:
        algorithmic_delay_ms = MODEL_ALGORITHMIC_DELAY_MS.get(model_key or "", 0.0)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    chunk_samples = int(sample_rate * chunk_ms / 1000.0)
    dummy_input = torch.randn(1, chunk_samples, device=device)

    # Warm-up phase
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    latencies_ms = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model(dummy_input)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            latencies_ms.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(latencies_ms))
    p95_ms = float(np.percentile(latencies_ms, 95))
    rtf = float(mean_ms / chunk_ms)
    total_latency_ms = mean_ms + algorithmic_delay_ms
    total_rtf = float(total_latency_ms / chunk_ms)

    return {
        "chunk_ms": chunk_ms,
        "chunk_samples": chunk_samples,
        "latency_mean_ms": round(mean_ms, 3),
        "compute_latency_mean_ms": round(mean_ms, 3),
        "latency_p95_ms": round(p95_ms, 3),
        "algorithmic_delay_ms": round(algorithmic_delay_ms, 3),
        "total_latency_ms": round(total_latency_ms, 3),
        "real_time_factor": round(rtf, 4),
        "compute_real_time_factor": round(rtf, 4),
        "total_real_time_factor": round(total_rtf, 4),
        "meets_realtime": rtf < 1.0,
        "device": str(device),
    }
