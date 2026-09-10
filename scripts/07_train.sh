#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — [07] Master Unified Training Pipeline Runner (Rev 3)
# Connects all model training layers into a single, cohesive, production-grade script.
#
# Target Hardware:
#   - NVIDIA DGX Spark GB10 (Grace Blackwell, CUDA 13, 128GB Unified Memory)
#
# Features:
#   - Native bfloat16 AMP (--precision bf16, Grace Blackwell optimized)
#   - Platform-adaptive QAT from Epoch 1 (--qat)
#   - CleanUMamba (SSM) Knowledge Distillation into DeepFilterNet3 (--distillation)
#   - Model 2 Lookahead Output-Delay Buffering (10ms future context)
#   - Worst-Class Pareto Guard on fragile defence classes
#   - SNR Curriculum scheduling (+15 dB down to -5 dB)
#
# Usage:
#   ./scripts/07_train.sh [OPTIONS]
#
# Options:
#   --model [all|se_primary|se_escalation|se_crosscheck|classifier|aec]
#                     Target model to train (default: all)
#   --epochs N        Max epochs (default: model-specific)
#   --batch-size N    Batch size per step
#   --grad-accum N    Gradient accumulation steps (default: 4)
#   --lr FLOAT        Learning rate override
#   --precision [bf16|fp16|fp32]
#                     Precision mode (default: bf16)
#   --qat / --no-qat  Quantization-Aware Training (default: enabled)
#   --distillation / --no-distillation
#                     CleanUMamba distillation (default: enabled)
#   --distillation-factor FLOAT
#                     Distillation weight multiplier (default: 0.3)
#   --device [cuda|cpu]
#                     Compute device (default: cuda if available)
#   --resume PATH     Path to checkpoint to resume
#   --force           Force execution (required for Model 5 AEC)
#   --dry-run         Verify initialization without running training loop
#   --help            Show this help message and exit
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"
PY_BIN="$(command -v python3 || command -v python)"

# Detect compute device and hardware
GPU_INFO="CPU"
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "NVIDIA GPU")
fi

echo "=============================================================================="
echo "      PROJECT AEGIS: MASTER UNIFIED ML TRAINING PIPELINE (REV 3)"
echo "=============================================================================="
echo "Working directory : ${ROOT_DIR}"
echo "Python binary     : ${PY_BIN}"
echo "Hardware detected : ${GPU_INFO}"
echo "Started at        : $(date)"
echo "=============================================================================="

# Launch the unified Python training pipeline orchestrator
${PY_BIN} -m training.scripts.train_pipeline "$@"

echo "=============================================================================="
echo "      PROJECT AEGIS: TRAINING PIPELINE EXECUTION FINISHED"
echo "=============================================================================="
