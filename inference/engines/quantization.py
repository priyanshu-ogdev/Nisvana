"""
inference/engines/quantization.py — Dynamic INT8 Quantization for Edge Hardware

Applies post-training dynamic quantization to convert FP32 weights to INT8,
reducing edge RAM and accelerating CPU inference on ARM Cortex and x86.

FIX (this pass): the docstring previously claimed "~75% edge RAM reduction"
unconditionally. That number is only true for the layer types PyTorch's
dynamic quantization actually supports: nn.Linear and the RNN family
(nn.GRU, nn.LSTM, nn.RNN, their Cell variants) and nn.EmbeddingBag.
**nn.Conv1d/Conv2d are NOT supported by dynamic quantization at all** --
PyTorch silently leaves unsupported module types at FP32 rather than
raising an error, so passing an unsupported type in `target_layers`
doesn't fail, it just quietly does nothing for those layers.

This matters directly for THIS project: DeepFilterNet3Wrapper and
CleanUMambaWrapper (model_loader.py) are Conv1d-dominant architectures --
the GRU is a comparatively small fraction of their parameters. Calling
this function on them quantizes the GRU only; the Conv1d layers doing
most of the actual signal processing stay FP32, so the real achieved RAM/
speed reduction is far less than "~75%" for these specific models. Added
`report_quantization_coverage` below so this is measured and reported
honestly, not assumed. The correct path to actually quantize the Conv1d
layers is STATIC quantization (requires calibration data) or
quantization-aware training (QAT, trains the model to tolerate
quantization from the start) -- both larger undertakings than dynamic
quantization, flagged here rather than attempted, matching how this
project has handled other correctly-scoped-out-of-a-pass items.
"""

from typing import Optional, Set, Dict
import torch
import torch.nn as nn

# The complete, real set of module types PyTorch's dynamic quantization
# actually supports -- not this project's previous default, which named
# only nn.Linear and nn.GRU as if that were an exhaustive, deliberate
# choice rather than an incomplete guess.
DYNAMIC_QUANTIZATION_SUPPORTED_TYPES = {
    nn.Linear, nn.GRU, nn.LSTM, nn.RNN,
    nn.GRUCell, nn.LSTMCell, nn.RNNCell,
    nn.EmbeddingBag,
}


def quantize_model_mixed_precision(model: nn.Module) -> nn.Module:
    """
    Mixed FP16-INT8 post-training quantization for this project's actual
    architecture family (Conv1d + GRU), grounded in Rusci et al.,
    "Accelerating RNN-based Speech Enhancement on a Multi-Core MCU with
    Mixed FP16-INT8 Post-Training Quantization" (arXiv:2210.07692) --
    which validates exactly this split for GRU/LSTM-based speech
    enhancement models: INT8 for the recurrent layers (max-calibration
    range, per their finding that max quantization ranges specifically
    benefit RNN layers), FP16 for everything else. Their reported result
    on GRU-based SE models: PESQ drop of only 0.06, STOI drop of only
    0.007 -- "almost lossless," achieved with PTQ alone, no QAT required.

    WHY THIS IS THE RIGHT FIX FOR THIS PROJECT SPECIFICALLY (not just a
    generic technique): quantize_model_dynamic's default target set only
    quantizes nn.GRU (nn.Conv1d isn't supported by dynamic quantization
    at all, and DeepFilterNet3Wrapper/CleanUMambaWrapper are Conv1d-
    dominant -- see that function's docstring). Simply casting the
    Conv1d layers to FP16 alongside the already-correct INT8 GRU
    quantization is a real, achievable 2x compression on the parameters
    dynamic quantization cannot touch, without needing calibration data
    or QAT -- matching the low-effort profile PTQ is supposed to have,
    while actually covering the model's dominant layer type instead of
    silently skipping it.

    HONEST LIMITATION, stated rather than hidden: this project has not
    independently reproduced Rusci et al.'s PESQ/STOI degradation
    numbers on ITS OWN models -- those figures are from their paper's
    own GRU-based SE models on the Valentini dataset, not a result
    established for DeepFilterNet3Wrapper/CleanUMambaWrapper specifically.
    Treat "almost lossless" as the literature's finding for this
    technique on this architecture FAMILY, not a guarantee already
    confirmed for this project's exact checkpoints -- run the SIH metrics
    test (test_sih_compliance_real_data.py) before and after applying
    this to confirm it holds here too, once a real trained checkpoint
    exists to test.
    """
    model.eval()

    # INT8 for the RNN family -- the part dynamic quantization actually supports.
    quantized = torch.ao.quantization.quantize_dynamic(
        model, {nn.GRU, nn.LSTM, nn.RNN}, dtype=torch.qint8,
    )

    # FP16 for everything else (primarily Conv1d/ConvTranspose1d in this
    # project's models) -- real, achievable 2x compression PyTorch's
    # dynamic quantization API cannot provide for these layer types.
    # Applied via .half() on the non-RNN submodules only, so the already-
    # INT8-quantized GRU internals (which manage their own dtype) aren't
    # double-converted.
    for name, module in quantized.named_modules():
        if isinstance(module, (nn.GRU, nn.LSTM, nn.RNN)):
            continue
        if hasattr(module, "weight") and module.weight is not None and module.weight.dtype == torch.float32:
            module.half()

    return quantized


def quantize_model_dynamic(
    model: nn.Module,
    target_layers: Optional[Set] = None,
    dtype: torch.dtype = torch.qint8,
) -> nn.Module:
    """
    Applies PyTorch dynamic post-training quantization.
    Args:
        model: Floating point PyTorch model.
        target_layers: Layer types to quantize (default: Linear, GRU).
            NOTE: nn.Conv1d/Conv2d are not valid here -- see module
            docstring. Passing them is not an error, it's a silent no-op.
        dtype: Target quantized integer type (default: torch.qint8).
    Returns:
        Quantized PyTorch model.
    """
    if target_layers is None:
        target_layers = {nn.Linear, nn.GRU}

    unsupported = target_layers - DYNAMIC_QUANTIZATION_SUPPORTED_TYPES
    if unsupported:
        import warnings
        warnings.warn(
            f"quantize_model_dynamic: {unsupported} are not supported by PyTorch "
            f"dynamic quantization and will be silently left at FP32 -- this is not "
            f"an error, but it means requested RAM/speed savings for those layer "
            f"types will not materialize. Use static quantization or QAT for "
            f"Conv1d/Conv2d layers instead.",
            stacklevel=2,
        )

    model.eval()
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        target_layers,
        dtype=dtype,
    )
    return quantized_model


def report_quantization_coverage(model: nn.Module, target_layers: Optional[Set] = None) -> Dict[str, float]:
    """
    Reports what FRACTION of the model's parameters actually sit in
    quantizable-by-this-function layer types, vs. layers that will
    silently remain FP32. Added this pass so "I ran quantize_model_dynamic"
    and "I achieved meaningful RAM/speed savings" are not silently treated
    as the same claim -- for a Conv1d-dominant model, they usually are not.

    Returns a dict with 'quantizable_params', 'total_params', and
    'quantizable_fraction' -- report this alongside any quantization
    benchmark, not just the fact that quantization ran without error.
    """
    if target_layers is None:
        target_layers = {nn.Linear, nn.GRU}

    total_params = 0
    quantizable_params = 0
    for module in model.modules():
        n_params = sum(p.numel() for p in module.parameters(recurse=False))
        total_params += n_params
        if type(module) in target_layers:
            quantizable_params += n_params

    fraction = quantizable_params / total_params if total_params > 0 else 0.0
    return {
        "quantizable_params": quantizable_params,
        "total_params": total_params,
        "quantizable_fraction": fraction,
    }


def prepare_static_quantization(
    model: nn.Module,
    calibration_data: Optional[list] = None,
    n_calibration_steps: int = 100,
    backend: str = "qnnpack",
) -> nn.Module:
    """
    Prepares a model for static quantization with calibration data.
    Unlike dynamic quantization (which can only handle Linear/GRU),
    static quantization supports Conv1d — the dominant layer type
    in DeepFilterNet3Wrapper and CleanUMambaWrapper.

    IMPORTANT: Requires calibration data representative of the actual
    deployment distribution. Using random noise will produce poor
    quantization ranges and degrade quality. Use a subset of the
    validation shards.

    Args:
        model: PyTorch model to prepare for static quantization.
        calibration_data: List of input tensors for calibration.
            If None, uses random calibration (NOT recommended for
            production — use real validation data).
        n_calibration_steps: Number of calibration forward passes.
        backend: Quantization backend ('qnnpack' for ARM/mobile,
                 'fbgemm' for x86, 'x86' for newer PyTorch versions).
    Returns:
        Quantized model ready for inference.
    """
    import copy
    import warnings

    model.eval()
    model_copy = copy.deepcopy(model)

    # Set quantization backend
    try:
        torch.backends.quantized.engine = backend
    except Exception:
        warnings.warn(
            f"Quantization backend '{backend}' not available, using default.",
            stacklevel=2,
        )

    # Insert quantization stubs
    model_copy.qconfig = torch.ao.quantization.get_default_qconfig(backend)

    # Prepare the model (inserts observers)
    try:
        prepared = torch.ao.quantization.prepare(model_copy, inplace=False)
    except Exception as e:
        warnings.warn(
            f"Static quantization preparation failed ({e}). "
            f"Model may have unsupported layer configurations. "
            f"Consider QAT instead.",
            stacklevel=2,
        )
        return model

    # Run calibration
    prepared.eval()
    with torch.no_grad():
        if calibration_data is not None:
            for i, data in enumerate(calibration_data):
                if i >= n_calibration_steps:
                    break
                if not isinstance(data, torch.Tensor):
                    data = torch.tensor(data, dtype=torch.float32)
                prepared(data)
        else:
            warnings.warn(
                "Using random calibration data — NOT recommended for production. "
                "Use real validation audio samples for proper quantization ranges.",
                stacklevel=2,
            )
            for _ in range(n_calibration_steps):
                dummy = torch.randn(1, 480, dtype=torch.float32)  # 10ms @ 48kHz
                prepared(dummy)

    # Convert to quantized model
    try:
        quantized = torch.ao.quantization.convert(prepared, inplace=False)
        return quantized
    except Exception as e:
        warnings.warn(
            f"Static quantization conversion failed ({e}). "
            f"Returning original model.",
            stacklevel=2,
        )
        return model


def get_qat_config(backend: str = "qnnpack") -> dict:
    """
    Returns QAT (Quantization-Aware Training) configuration hooks for
    training-side integration. QAT produces the best quantized quality
    by teaching the model to tolerate quantization noise during training,
    but requires re-training (not just post-training conversion).

    Usage in a trainer:
        qat_cfg = get_qat_config()
        model.qconfig = qat_cfg['qconfig']
        model = torch.ao.quantization.prepare_qat(model)
        # ... train as normal ...
        model = torch.ao.quantization.convert(model)

    Returns:
        Dict with 'qconfig', 'prepare_fn', 'convert_fn' callables.
    """
    qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)

    def prepare_fn(model: nn.Module) -> nn.Module:
        model.train()
        model.qconfig = qconfig
        return torch.ao.quantization.prepare_qat(model, inplace=False)

    def convert_fn(model: nn.Module) -> nn.Module:
        model.eval()
        return torch.ao.quantization.convert(model, inplace=False)

    return {
        "qconfig": qconfig,
        "backend": backend,
        "prepare_fn": prepare_fn,
        "convert_fn": convert_fn,
    }

