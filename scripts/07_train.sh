#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — Master Unified ML Training Pipeline Runner (Rev 3)
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
source "${SCRIPT_DIR}/common_env.sh" 2>/dev/null || true

cd "${ROOT_DIR}"
if declare -f detect_python > /dev/null; then
    PY_BIN="$(detect_python)"
else
    PY_BIN="$(command -v python3 || command -v python)"
fi

# Stage 7 consumes already-exported WebDataset shards. It must never invoke
# data-forge or regenerate data on the training server.
SHARDS_ROOT="${DATA_FORGE_DATA_DIR:-${ROOT_DIR}/data}/shards"
if [ ! -d "${SHARDS_ROOT}" ]; then
    echo "ERROR: expected existing data-forge shards at ${SHARDS_ROOT}" >&2
    exit 1
fi

REQUESTED_DEVICE="cuda"
previous_arg=""
for arg in "$@"; do
    if [ "${previous_arg}" = "--device" ]; then
        REQUESTED_DEVICE="${arg}"
    fi
    previous_arg="${arg}"
done
if [ "${REQUESTED_DEVICE}" != "cpu" ] && ! "${PY_BIN}" -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'" 2>/dev/null; then
    echo "ERROR: CUDA is unavailable in the selected Python environment." >&2
    echo "Activate the DGX/GB10 environment containing CUDA-enabled PyTorch, then retry." >&2
    exit 1
fi

echo "Existing shard inventory:"
find "${SHARDS_ROOT}" -type f -name '*.tar' -printf '  %P\n' | sort | head -20
SHARD_COUNT="$(find "${SHARDS_ROOT}" -type f -name '*.tar' | wc -l)"
if [ "${SHARD_COUNT}" -eq 0 ]; then
    echo "ERROR: no WebDataset .tar shards found under ${SHARDS_ROOT}" >&2
    exit 1
fi
echo "  Total .tar shards: ${SHARD_COUNT}"

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
