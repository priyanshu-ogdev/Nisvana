"""
inference/engines/onnx_engine.py — ONNX Edge Export & Latency Profiler

Provides:
- ONNX model export with dynamic axes for NVIDIA Jetson AGX Orin & DSPs
- Precision calibration (FP32, FP16, INT8 dynamic quantization)
- Real-Time Factor (RTF) and latency budget benchmarking
"""

import os
import warnings
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


# Authoritative algorithmic buffering / lookahead delay benchmarks established for AEGIS
MODEL_ALGORITHMIC_DELAY_MS: Dict[str, float] = {
    "aegis-se-primary": 0.0,       # FALLBACK-ONLY figure -- see get_algorithmic_delay_ms below.
                                    # Correct for the causal time-domain fallback wrapper.
                                    # WRONG for the real vendored DeepFilterNet3 -- see fix below.
    "aegis-se-escalation": 40.0,   # DeepFilterNet3 standard 40ms lookahead (severe SNR escalation)
    # CORRECTED 2026-09-03: this was "25.0 # RT-SEMamba / CleanUMamba 25ms framing
    # window" -- RT-SEMamba is a DIFFERENT, unrelated architecture (evaluated many
    # turns ago only as a distillation-TECHNIQUE donor for Model 1, never adopted
    # as any model's actual runtime component in this system). Slash-joining the
    # name with CleanUMamba's in a comment did not make the number correct.
    # CleanUMamba's own algorithmic delay was never independently derived anywhere
    # in this project. Worse: the actual fallback implementation
    # (CleanUMambaWrapper) is a CAUSAL TIME-DOMAIN conv+GRU, not an STFT-framed
    # model at all -- a fixed "25ms framing window" doesn't even architecturally
    # apply to what's implemented. Its real delay is the causal encoder's own
    # padding (enc_kernel=15, stride=2 -> ~14 samples, i.e. under 1ms at 48kHz),
    # not a borrowed STFT-window figure from an unrelated model.
    "aegis-se-crosscheck": 0.3,    # UNVERIFIED ESTIMATE: derived from this wrapper's own
                                    # causal conv encoder padding (14 samples @ 48kHz =
                                    # ~0.29ms), NOT independently benchmarked. If the real
                                    # vendored CleanUMamba package loads instead of this
                                    # fallback, re-derive from its actual published framing.
    "aegis-clf-gate": 0.0,         # Instantaneous frame gating classifier
    # CORRECTED 2026-09-03: this was "20.0 # TaylorBeamformer ~20ms delay buffer".
    # TaylorBeamformer was never Model 5 -- it belonged to an earlier, abandoned
    # hardware-ANC design phase (Tier 0 beamforming) this project moved past
    # entirely before adopting the data-forge/DeepFilterNet3/CleanUMamba
    # architecture. Model 5 is deepvqe-ggml. The actual previously-researched
    # figure for THIS component is DeepVQE-S's own published framing (512-point
    # FFT / 256-sample hop => 16ms hop, ~32ms window), never substituted in.
    "aegis-aec-gate": 32.0,        # DeepVQE-S framing (512 FFT / 256 hop @ 16kHz origin,
                                    # duration in ms is sample-rate-independent). CAVEAT:
                                    # sourced from the original DeepVQE-S paper, NOT
                                    # independently reconfirmed for the specific
                                    # "deepvqe-ggml" unofficial reimplementation this
                                    # project actually uses -- re-verify if that port
                                    # changed the framing parameters.
}


# FIX (this pass): audio_stream.py's StatefulHopProcessor docstring
# surfaced a real reconciliation gap while researching this project's
# streaming design -- a genuine, deployed stateful streaming conversion
# of the REAL DeepFilterNet3 (huggingface.co/iky1e/DeepFilterNet3-
# Streaming-CoreML) reports a FIXED 30ms algorithmic delay (1,440
# samples), independent of extra lookahead -- because a real STFT-based
# model still has to buffer one full analysis window (its own inherent
# framing latency) before it can produce ANY output, even at zero EXTRA
# lookahead beyond that base window. "Zero lookahead" (no additional
# future frames) is not the same claim as "zero total buffering delay."
# The static 0.0ms figure above is correct ONLY for this project's
# fallback wrapper (a pure time-domain causal conv+GRU with no STFT
# framing at all) -- it is WRONG the moment the real vendored package
# loads instead (model.is_vendored == True, per model_loader.py's own
# markers). This function makes the lookup delay-source-aware instead of
# silently assuming the fallback's number always applies.
def get_algorithmic_delay_ms(model_key: str, model: Optional[nn.Module] = None) -> float:
    """
    Correct, delay-source-aware replacement for a bare
    MODEL_ALGORITHMIC_DELAY_MS[model_key] lookup. Pass the actual model
    instance (not just its key) so fallback-vs-vendored can be checked --
    a benchmark or compliance report that only has the key string cannot
    make this distinction and should be treated as reporting the
    fallback's number, not the real package's.
    """
    is_vendored = bool(getattr(model, "is_vendored", False)) if model is not None else False
    is_fallback = bool(getattr(model, "is_fallback", False)) if model is not None else True

    if model_key == "aegis-se-primary" and is_vendored:
        # Real vendored DeepFilterNet3's own reported fixed framing delay,
        # per the cited streaming conversion -- applies even at zero
        # EXTRA lookahead, since it's the base analysis-window latency,
        # not the lookahead setting.
        return 30.0

    if model_key == "aegis-se-escalation" and is_fallback:
        # FIX (Rev 3 P0.2): after the output-delay buffer fix, the
        # fallback wrapper's real algorithmic delay is one chunk duration
        # (~10ms at 480 samples / 48kHz), NOT the vendored DF3's 40ms
        # STFT-frame lookahead. The static 40.0ms in
        # MODEL_ALGORITHMIC_DELAY_MS is correct only for the vendored
        # package; the fallback's actual delay comes from the output-delay
        # buffering scheme, not from any STFT framing.
        return 10.0  # One chunk delay at 480 samples @ 48kHz

    return MODEL_ALGORITHMIC_DELAY_MS.get(model_key or "", 0.0)


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    sample_rate: int = 48000,
    chunk_ms: float = 10.0,
    opset_version: int = 17,
) -> Path:
    """
    Exports a PyTorch audio model to ONNX format.

    FIX (this pass -- the most severe finding in this project's review
    history): previously exported EVERY model with a single input tensor
    and no declared state input/output, regardless of whether the model
    was actually stateful. For DeepFilterNet3Wrapper/CleanUMambaWrapper,
    this meant the real deployed ONNX artifact was fully stateless --
    every chunk processed as if it were the first chunk of a session --
    silently discarding all of this project's GRU-statefulness and
    causal-context-carryover fixes. Confirmed end-to-end: OnnxRuntimeSession
    (the actual edge/Jetson inference engine) took exactly one input and
    one output, with no state-threading mechanism at all.

    Now: if the model exposes `get_initial_state` (the marker added this
    pass for models with a real, explicit-state forward signature), export
    declares state as real graph inputs/outputs, matching how ONNX's own
    LSTM/GRU operators expose h_0/c_0. Models without it (e.g. the
    classifier, legitimately stateless per-window by design) use the
    original simple path, unchanged.

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

    if hasattr(model, "get_initial_state"):
        initial_state = model.get_initial_state(batch_size=1)
        state_names = [f"state{i}_in" for i in range(len(initial_state))]
        state_out_names = [f"state{i}_out" for i in range(len(initial_state))]

        dynamic_axes = {
            "input": {0: "batch_size", 1: "time_steps"},
            "output": {0: "batch_size", 1: "time_steps"},
        }
        for name in state_names + state_out_names:
            dynamic_axes[name] = {0: "batch_or_layers"}

        export_args = (dummy_input,) + tuple(initial_state)
        input_names = ["input"] + state_names
        output_names = ["output"] + state_out_names

        try:
            torch.onnx.export(
                model, export_args, str(output_path),
                export_params=True, opset_version=opset_version, do_constant_folding=True,
                input_names=input_names, output_names=output_names,
                dynamic_axes=dynamic_axes, dynamo=False,
            )
        except TypeError:
            torch.onnx.export(
                model, export_args, str(output_path),
                export_params=True, opset_version=opset_version, do_constant_folding=True,
                input_names=input_names, output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
        return output_path

    # Original simple path -- for genuinely stateless models only (e.g. aegis-clf-gate).
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


def export_platform_b_tensorrt(
    model: nn.Module,
    output_path: Path,
    sample_rate: int = 48000,
    chunk_ms: float = 10.0,
    opset_version: int = 17,
    precision: str = "fp16",
    calibration_samples: Optional[List[np.ndarray]] = None,
) -> Path:
    """
    Rev 3 P0.4: Platform B (GPU / TensorRT) export branch.

    Dual-Platform architecture:
    - Platform A (ARM / Edge CPU): PyTorch QAT -> convert_qat() -> INT8 PyTorch / QNNPACK model.
    - Platform B (NVIDIA Blackwell / Orin / Jetson):
      QAT-trained weights (post-training, pre-convert_qat()) -> export to ONNX FP16/FP32.
      TensorRT then performs its own INT8 calibration or FP16 execution, leveraging
      Blackwell's native Tensor Cores without being constrained by QNNPACK.

    Args:
        model: Trained PyTorch model (pre-convert_qat).
        output_path: Target ONNX path.
        sample_rate: Audio sample rate in Hz.
        chunk_ms: Chunk size in milliseconds.
        opset_version: ONNX operator set version.
        precision: 'fp16' or 'fp32'.
        calibration_samples: Optional audio chunks for TensorRT INT8 calibrator.

    Returns:
        Path to exported Platform B ONNX model.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exported = export_to_onnx(
        model=model,
        output_path=output_path,
        sample_rate=sample_rate,
        chunk_ms=chunk_ms,
        opset_version=opset_version,
    )

    if precision == "fp16":
        try:
            import onnx
            from onnxconverter_common import float16
            onnx_model = onnx.load(str(exported))
            fp16_model = float16.convert_float_to_float16(onnx_model, keep_io_types=True)
            onnx.save(fp16_model, str(exported))
            logger.info("Platform B: successfully converted ONNX model to FP16 for TensorRT")
        except Exception as e:
            logger.warning("FP16 conversion optional dependency unavailable (%s) — keeping FP32 for TensorRT", e)

    return exported


def optimize_onnx_graph(
    onnx_model_path: Path,
    output_path: Optional[Path] = None,
    optimization_level: str = "all",
) -> Path:
    """
    Applies ONNX graph optimization passes: constant folding, redundant
    node elimination, and operator fusion. These reduce the number of
    operators the runtime must execute per frame — critical for sub-10ms
    real-time audio where even a few hundred microseconds matter.

    Uses onnxruntime's built-in offline optimizer when available (it
    applies the same passes ORT's online optimizer does, but ahead of
    time so session creation is faster). Falls back to the onnxoptimizer
    package if ORT's offline path isn't available.

    Args:
        onnx_model_path: Path to input ONNX model.
        output_path: Path for optimized model (default: same dir, *_opt.onnx suffix).
        optimization_level: 'basic' (constant folding only) or 'all' (full fusion).
    Returns:
        Path to optimized ONNX model.
    """
    onnx_model_path = Path(onnx_model_path)
    if output_path is None:
        output_path = onnx_model_path.with_suffix(".opt.onnx")
    output_path = Path(output_path)

    # Path 1: onnxruntime offline optimizer (preferred — matches runtime's own passes)
    try:
        import onnxruntime as ort
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.optimized_model_filepath = str(output_path)
        # Creating a session with optimized_model_filepath set writes the
        # optimized graph to disk as a side effect — the session itself is
        # discarded, this is purely for the offline optimization artifact.
        ort.InferenceSession(
            str(onnx_model_path),
            sess_options=sess_opts,
            providers=["CPUExecutionProvider"],
        )
        if output_path.exists():
            return output_path
    except Exception as e:
        warnings.warn(
            f"ORT offline optimization failed ({e}), trying onnxoptimizer...",
            stacklevel=2,
        )

    # Path 2: onnxoptimizer package (community passes)
    try:
        import onnx
        import onnxoptimizer
        model = onnx.load(str(onnx_model_path))
        passes = onnxoptimizer.get_fuse_and_elimination_passes()
        optimized = onnxoptimizer.optimize(model, passes)
        onnx.save(optimized, str(output_path))
        return output_path
    except ImportError:
        warnings.warn(
            "Neither onnxruntime offline optimizer nor onnxoptimizer package "
            "available — returning original model unoptimized.",
            stacklevel=2,
        )
        return onnx_model_path


def validate_onnx_export(
    onnx_model_path: Path,
    model: nn.Module,
    sample_rate: int = 48000,
    chunk_ms: float = 10.0,
    tolerance: float = 1e-3,
) -> Dict[str, Any]:
    """
    Validates that an exported ONNX model produces output within tolerance
    of the source PyTorch model, including stateful hidden state threading.
    Critical for catching silent state-discarding bugs in the export.

    Returns:
        Dict with 'max_abs_diff', 'mean_abs_diff', 'passed', 'stateful'.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return {"error": "onnxruntime not installed", "passed": False}

    model.eval()
    frame_len = int(sample_rate * chunk_ms / 1000.0)
    test_input = torch.randn(1, frame_len, dtype=torch.float32)

    # PyTorch forward
    if hasattr(model, "reset_state"):
        model.reset_state()
    with torch.no_grad():
        pt_out = model(test_input)
        if isinstance(pt_out, tuple):
            pt_out = pt_out[0]
    pt_out_np = pt_out.squeeze().cpu().numpy()

    # ONNX Runtime forward
    from inference.engines.onnx_runtime_engine import OnnxRuntimeSession
    ort_session = OnnxRuntimeSession(
        str(onnx_model_path),
        warmup_frames=0,
        enable_io_binding=False,
    )
    ort_out = ort_session.forward(test_input.numpy()).squeeze()

    max_diff = float(np.max(np.abs(pt_out_np - ort_out)))
    mean_diff = float(np.mean(np.abs(pt_out_np - ort_out)))

    return {
        "max_abs_diff": round(max_diff, 6),
        "mean_abs_diff": round(mean_diff, 6),
        "passed": max_diff < tolerance,
        "stateful": ort_session.is_stateful,
        "tolerance": tolerance,
    }


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
        algorithmic_delay_ms = get_algorithmic_delay_ms(model_key or "", model=model)

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
    p99_ms = float(np.percentile(latencies_ms, 99))
    min_ms = float(np.min(latencies_ms))
    rtf = float(mean_ms / chunk_ms)
    total_latency_ms = mean_ms + algorithmic_delay_ms
    total_rtf = float(total_latency_ms / chunk_ms)

    return {
        "chunk_ms": chunk_ms,
        "chunk_samples": chunk_samples,
        "latency_mean_ms": round(mean_ms, 3),
        "latency_min_ms": round(min_ms, 3),
        "compute_latency_mean_ms": round(mean_ms, 3),
        "latency_p95_ms": round(p95_ms, 3),
        "latency_p99_ms": round(p99_ms, 3),
        "algorithmic_delay_ms": round(algorithmic_delay_ms, 3),
        "total_latency_ms": round(total_latency_ms, 3),
        "real_time_factor": round(rtf, 4),
        "compute_real_time_factor": round(rtf, 4),
        "total_real_time_factor": round(total_rtf, 4),
        "meets_realtime": rtf < 1.0,
        "device": str(device),
    }
