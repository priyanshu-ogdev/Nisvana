"""
inference/engines/quantization.py — Dynamic INT8 Quantization for Edge Hardware

Applies post-training dynamic quantization to convert FP32 weights to INT8,
reducing edge RAM by ~75% and accelerating CPU inference on ARM Cortex and x86.
"""

from typing import Optional, Set
import torch
import torch.nn as nn


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
        dtype: Target quantized integer type (default: torch.qint8).
    Returns:
        Quantized PyTorch model.
    """
    if target_layers is None:
        target_layers = {nn.Linear, nn.GRU}

    model.eval()
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        target_layers,
        dtype=dtype,
    )
    return quantized_model
