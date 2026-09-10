#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — [00] DGX Spark GB10 Environment Setup
# Target: NVIDIA DGX Spark (Grace Blackwell GB10, CUDA 13, Ubuntu)
# 
# Installs ALL dependencies including:
# - PyTorch with CUDA support
# - deepfilternet[train] (requires Rust toolchain)
# - mamba-ssm + causal-conv1d (requires CUDA toolkit headers)
# - Full data forge, training, and inference stack
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "============================================================"
echo "  PROJECT AEGIS — DGX Spark GB10 Environment Setup"
echo "  Target: Grace Blackwell / CUDA 13"
echo "============================================================"
echo "Working directory: ${ROOT_DIR}"
echo ""

# ==============================================================================
# 1. System Prerequisites Check
# ==============================================================================
echo ">>> [1/7] Checking system prerequisites..."

if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH."
    exit 1
fi

PY_BIN="$(command -v python3 || command -v python)"
PY_VERSION=$(${PY_BIN} --version 2>&1)
echo "  Python: ${PY_VERSION} at ${PY_BIN}"

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "unknown")
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo "unknown")
    echo "  GPU: ${GPU_NAME}"
    echo "  CUDA Driver Version: ${CUDA_VERSION}"
else
    echo "  WARNING: nvidia-smi not found. GPU/CUDA may not be available."
fi

# Check disk space
AVAIL_GB=$(df -BG "${ROOT_DIR}" | tail -1 | awk '{print $4}' | tr -d 'G' || echo "0")
echo "  Available disk: ${AVAIL_GB}GB"
if [ "${AVAIL_GB}" -lt 100 ]; then
    echo "  WARNING: Less than 100GB free. TB-scale data pipeline may need more space."
fi

# ==============================================================================
# 2. Rust Toolchain (required by deepfilternet[train])
# ==============================================================================
echo ""
echo ">>> [2/7] Checking/installing Rust toolchain for deepfilternet[train]..."

if command -v rustc &> /dev/null; then
    RUST_VERSION=$(rustc --version)
    echo "  Found Rust: ${RUST_VERSION}"
else
    echo "  Installing Rust toolchain via rustup (non-interactive)..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    source "${HOME}/.cargo/env" 2>/dev/null || true
    export PATH="${HOME}/.cargo/bin:${PATH}"
    echo "  Installed Rust: $(rustc --version)"
fi

# ==============================================================================
# 3. Python Environment & Pip Upgrade
# ==============================================================================
echo ""
echo ">>> [3/7] Upgrading pip and build tools..."

${PY_BIN} -m pip install --upgrade pip setuptools wheel build maturin 2>&1 | tail -3

# ==============================================================================
# 4. Core Dependencies from requirements.txt
# ==============================================================================
echo ""
echo ">>> [4/7] Installing Python dependencies from requirements.txt..."

${PY_BIN} -m pip install -r "${ROOT_DIR}/requirements.txt" 2>&1 | tail -10

# Verify critical packages
echo ""
echo "  Verifying critical installations..."

${PY_BIN} -c "import torch; print(f'  torch {torch.__version__} | CUDA available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" || echo "  WARNING: torch import failed"
${PY_BIN} -c "import torchaudio; print(f'  torchaudio {torchaudio.__version__}')" || echo "  WARNING: torchaudio import failed"

# deepfilternet verification
${PY_BIN} -c "
try:
    from df.model import ModelParams
    from df.loss import Loss as DfLoss
    print('  deepfilternet[train] ✓ (vendored loss available)')
except ImportError as e:
    print(f'  deepfilternet[train] ✗ ({e})')
    print('  NOTE: Models 1/2 will use fallback wrappers instead of real DeepFilterNet3.')
    print('  To fix: ensure Rust toolchain is installed, then pip install deepfilternet[train]')
" 2>&1

# mamba-ssm verification
${PY_BIN} -c "
try:
    from mamba_ssm import Mamba
    print('  mamba-ssm ✓ (Mamba SSM kernel available)')
except ImportError as e:
    print(f'  mamba-ssm ✗ ({e})')
    print('  NOTE: Model 3 will use fallback CleanUMambaWrapper instead of real Mamba blocks.')
    print('  To fix: ensure CUDA toolkit headers are installed, then pip install mamba-ssm')
" 2>&1

# ONNX Runtime verification
${PY_BIN} -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print(f'  onnxruntime {ort.__version__} | Providers: {providers}')
if 'CUDAExecutionProvider' in providers:
    print('  CUDA execution provider ✓')
elif 'TensorrtExecutionProvider' in providers:
    print('  TensorRT execution provider ✓')
else:
    print('  WARNING: No GPU execution provider. Edge inference will use CPU.')
" 2>&1 || echo "  WARNING: onnxruntime import failed"

# ==============================================================================
# 5. Directory Structure Initialization
# ==============================================================================
echo ""
echo ">>> [5/7] Initializing data/ storage hierarchy..."

DIRS=(
    "data/raw"
    "data/processed"
    "data/augmented"
    "data/splits/train"
    "data/splits/val"
    "data/splits/test"
    "data/forge/branch_speech_enhancement/noisy"
    "data/forge/branch_speech_enhancement/clean"
    "data/forge/branch_speech_enhancement/rir"
    "data/forge/branch_classifier/audio"
    "data/forge/branch_classifier/labels"
    "data/forge/branch_aec/mic"
    "data/forge/branch_aec/farend"
    "data/forge/branch_aec/nearend"
    "data/forge/branch_aec/echo"
    "data/shards"
    "data/manifests"
    "data/onnx_models"
    "training/checkpoints"
    "training/runs"
)

for d in "${DIRS[@]}"; do
    mkdir -p "${ROOT_DIR}/${d}"
done
echo "  Created $(echo ${#DIRS[@]}) directories."

# ==============================================================================
# 6. Load .env & Check API Credentials
# ==============================================================================
echo ""
echo ">>> [6/7] Checking external dataset API tokens..."

# Source .env if present
if [ -f "${ROOT_DIR}/data_forge/.env" ]; then
    set -a
    source "${ROOT_DIR}/data_forge/.env"
    set +a
    echo "  Loaded .env from data_forge/.env"
elif [ -f "${ROOT_DIR}/.env" ]; then
    set -a
    source "${ROOT_DIR}/.env"
    set +a
    echo "  Loaded .env from root .env"
fi

if [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; then
    if [ ! -f "${HOME}/.kaggle/kaggle.json" ]; then
        echo "  ⚠ Kaggle credentials not detected."
        echo "    To download MAD (~1.1GB): https://www.kaggle.com/settings -> API -> Create New Token"
    else
        echo "  ✓ Kaggle credentials found in ${HOME}/.kaggle/kaggle.json"
    fi
else
    echo "  ✓ Kaggle credentials found in environment."
fi

if [ -z "${DRYAD_API_TOKEN:-}" ]; then
    echo "  ⚠ DRYAD_API_TOKEN not set. Gunshot dataset via Dryad API unavailable."
    echo "    Drop raw WAVs into data/raw/gunshot_dryad/ or set DRYAD_API_TOKEN."
else
    echo "  ✓ DRYAD_API_TOKEN found in environment."
fi

# ==============================================================================
# 7. TB-Scale Download Settings Report
# ==============================================================================
echo ""
echo ">>> [7/7] Download resilience configuration:"
echo "  Timeout:          ${DATA_FORGE_FETCH_TIMEOUT:-300}s"
echo "  Max retries:      ${DATA_FORGE_FETCH_MAX_RETRIES:-10}"
echo "  Backoff base:     ${DATA_FORGE_FETCH_BACKOFF_BASE:-5}s (exponential)"
echo "  Chunk size:       $(echo ${DATA_FORGE_FETCH_CHUNK_BYTES:-4194304} | awk '{printf "%.0f MB", $1/1048576}') per write"
echo "  Concurrent DLs:   ${DATA_FORGE_FETCH_MAX_CONCURRENT:-4}"
echo "  Resume:           ${DATA_FORGE_FETCH_RESUME:-true}"
echo "  Disk safety:      ${DATA_FORGE_DISK_SAFETY_MARGIN_GB:-50}GB margin"
echo "  Max workers:      ${DATA_FORGE_MAX_WORKERS:-64}"

echo ""
echo "============================================================"
echo "  ENVIRONMENT SETUP COMPLETE"
echo ""
echo "  Next steps:"
echo "    1. Fill API tokens in data_forge/.env"
echo "    2. Run: bash scripts/01_data_pipeline_dry_run.sh"
echo "    3. Run: bash scripts/03_data_pipeline_full_run.sh"
echo "    4. Run: bash scripts/07_train_se_primary.sh"
echo "============================================================"
