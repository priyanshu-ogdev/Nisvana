#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — [00] DGX Spark GB10 Environment Setup
# Target: NVIDIA DGX Spark (Grace Blackwell GB10, CUDA 13, Ubuntu)
# 
# Installs ALL dependencies including:
# - PyTorch with CUDA support (or detects pre-installed PyTorch 2.7+)
# - Build tools (ninja, packaging, maturin, wheel)
# - deepfilternet[train] (requires Rust toolchain)
# - causal-conv1d + mamba-ssm (compiled with --no-build-isolation to detect torch)
# - Full data forge, training, and inference stack
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Source common Python detector
source "${SCRIPT_DIR}/common_env.sh" 2>/dev/null || true

echo "============================================================"
echo "  PROJECT AEGIS — DGX Spark GB10 Environment Setup"
echo "  Target: Grace Blackwell / CUDA 13"
echo "============================================================"
echo "Working directory: ${ROOT_DIR}"
echo ""

# ==============================================================================
# 1. System Prerequisites & Active Python Detection (Non-Sandboxed)
# ==============================================================================
echo ">>> [1/7] Detecting active Python environment..."

if declare -f detect_python > /dev/null; then
    PY_BIN="$(detect_python)"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PY_BIN="${VIRTUAL_ENV}/bin/python"
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PY_BIN="${CONDA_PREFIX}/bin/python"
elif command -v python &> /dev/null && python -c "import torch" &> /dev/null; then
    PY_BIN="$(command -v python)"
elif command -v python3 &> /dev/null && python3 -c "import torch" &> /dev/null; then
    PY_BIN="$(command -v python3)"
else
    PY_BIN="$(command -v python3 || command -v python)"
fi

if [ -z "${PY_BIN}" ] || ! command -v "${PY_BIN}" &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH."
    exit 1
fi

PY_VERSION=$(${PY_BIN} --version 2>&1)
echo "  Target Python: ${PY_VERSION} at ${PY_BIN}"

# Detect if PyTorch is already present
HAS_TORCH=false
if ${PY_BIN} -c "import torch" 2>/dev/null; then
    HAS_TORCH=true
    TORCH_INFO=$(${PY_BIN} -c "import torch; print(f'v{torch.__version__} | CUDA Available: {torch.cuda.is_available()}')")
    echo "  ✓ Active PyTorch found: ${TORCH_INFO}"
else
    echo "  PyTorch not found in target Python. Will be installed."
fi

# Check CUDA availability via nvidia-smi
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
# 3. Upgrading Build Tools (Ninja, Packaging, Maturin for C++/CUDA Compiles)
# ==============================================================================
echo ""
echo ">>> [3/7] Upgrading pip and build tools (ninja, packaging, maturin)..."

${PY_BIN} -m pip install --upgrade pip setuptools wheel build ninja packaging maturin 2>&1 | tail -5

# ==============================================================================
# 4. Core Dependencies & PyTorch Installation
# ==============================================================================
echo ""
if [ "${HAS_TORCH}" = "false" ]; then
    echo ">>> [4/7] Installing PyTorch with CUDA support..."
    ${PY_BIN} -m pip install torch torchaudio torchvision --extra-index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -5
else
    echo ">>> [4/7] Reusing active PyTorch installation (${TORCH_INFO})..."
fi

echo "  Installing base requirements from requirements.txt..."
${PY_BIN} -m pip install -r "${ROOT_DIR}/requirements.txt" 2>&1 | tail -10

# ==============================================================================
# 5. Non-Sandboxed Mamba SSM & Causal-Conv1D Installation (--no-build-isolation)
# ==============================================================================
echo ""
echo ">>> [5/7] Installing Mamba SSM and Causal-Conv1D (NO BUILD ISOLATION)..."
echo "  Note: Compiling directly against active PyTorch (${PY_BIN}) to avoid sandbox isolation."

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0;10.0;12.0}"
export MAX_JOBS="${MAX_JOBS:-8}"

# causal-conv1d MUST be installed first without build isolation
echo "  Installing causal-conv1d (--no-build-isolation)..."
if ${PY_BIN} -m pip install --no-build-isolation "causal-conv1d>=1.4.0" 2>&1 | tail -5; then
    echo "  ✓ causal-conv1d installed successfully."
else
    echo "  ⚠ Building causal-conv1d from source with nvcc failed or compiler not present."
    echo "    Trying binary distribution fallback..."
    ${PY_BIN} -m pip install --prefer-binary "causal-conv1d>=1.4.0" || true
fi

# mamba-ssm MUST be installed after causal-conv1d without build isolation
echo "  Installing mamba-ssm (--no-build-isolation)..."
if ${PY_BIN} -m pip install --no-build-isolation "mamba-ssm>=2.2.0" 2>&1 | tail -5; then
    echo "  ✓ mamba-ssm installed successfully."
else
    echo "  ⚠ Building mamba-ssm from source failed (requires CUDA toolkit nvcc compiler)."
    echo "    Trying binary distribution fallback..."
    ${PY_BIN} -m pip install --prefer-binary "mamba-ssm>=2.2.0" || true
fi

# DeepFilterNet3 check
echo "  Verifying deepfilternet[train]..."
if ! ${PY_BIN} -c "from df.model import ModelParams" 2>/dev/null; then
    echo "  Installing deepfilternet[train]..."
    ${PY_BIN} -m pip install "deepfilternet[train]>=0.5.6" || \
    ${PY_BIN} -m pip install --no-build-isolation "deepfilternet[train]>=0.5.6" || {
        echo "  ⚠ deepfilternet[train] build skipped. Models 1 & 2 will use fallback CleanUMamba/DF wrappers."
    }
fi

# Run verification of all critical packages
echo ""
echo "  Verifying installed neural stack:"
${PY_BIN} -c "import torch; print(f'  torch {torch.__version__} | CUDA available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" || echo "  WARNING: torch import failed"
${PY_BIN} -c "import torchaudio; print(f'  torchaudio {torchaudio.__version__}')" || echo "  WARNING: torchaudio import failed"

${PY_BIN} -c "
try:
    from df.model import ModelParams
    from df.loss import Loss as DfLoss
    print('  deepfilternet[train] ✓ (vendored loss available)')
except ImportError as e:
    print(f'  deepfilternet[train] ✗ ({e})')
    print('  NOTE: Models 1/2 will use fallback wrappers instead of real DeepFilterNet3.')
" 2>&1

${PY_BIN} -c "
try:
    from mamba_ssm import Mamba
    print('  mamba-ssm ✓ (Mamba SSM kernel available)')
except ImportError as e:
    print(f'  mamba-ssm ✗ ({e})')
    print('  NOTE: Model 3 will use verified CleanUMambaWrapper architectural fallback.')
" 2>&1

${PY_BIN} -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print(f'  onnxruntime {ort.__version__} | Providers: {providers}')
if 'CUDAExecutionProvider' in providers:
    print('  CUDA execution provider ✓')
elif 'TensorrtExecutionProvider' in providers:
    print('  TensorRT execution provider ✓')
else:
    print('  NOTE: CPU provider active for edge ONNX inference.')
" 2>&1 || echo "  WARNING: onnxruntime import failed"

# ==============================================================================
# 6. Directory Structure Initialization
# ==============================================================================
echo ""
echo ">>> [6/7] Initializing data/ storage hierarchy..."

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
# 7. Load .env & Check API Credentials
# ==============================================================================
echo ""
echo ">>> [7/7] Checking external dataset API tokens..."

# Load both .env files if present
if [ -f "${ROOT_DIR}/.env" ]; then
    set -a
    source "${ROOT_DIR}/.env" 2>/dev/null || true
    set +a
    echo "  Loaded credentials from root .env"
fi
if [ -f "${ROOT_DIR}/data_forge/.env" ]; then
    set -a
    source "${ROOT_DIR}/data_forge/.env" 2>/dev/null || true
    set +a
    echo "  Loaded credentials from data_forge/.env"
fi

if [ -n "${KAGGLE_ACCESS_TOKEN:-}" ]; then
    echo "  ✓ Kaggle Bearer access token found in environment."
elif [ -n "${KAGGLE_USERNAME:-}" ] && [ -n "${KAGGLE_KEY:-}" ]; then
    echo "  ✓ Kaggle credentials found in environment."
elif [ -f "${HOME}/.kaggle/kaggle.json" ]; then
    echo "  ✓ Kaggle credentials found in ${HOME}/.kaggle/kaggle.json"
else
    echo "  ⚠ Kaggle credentials not detected."
    echo "    Set KAGGLE_ACCESS_TOKEN in .env (https://www.kaggle.com/settings -> API -> Create New Token)"
fi

if [ -n "${DRYAD_API_TOKEN:-}" ] && [ "${DRYAD_API_TOKEN}" != "your_dryad_bearer_token" ]; then
    echo "  ✓ DRYAD_API_TOKEN found in environment."
else
    echo "  ⚠ DRYAD_API_TOKEN not set. Gunshot dataset via Dryad API unavailable."
    echo "    Drop raw WAVs into data/raw/gunshot_dryad/ or set DRYAD_API_TOKEN."
fi

if [ -n "${GITHUB_TOKEN:-}" ] || [ -n "${DATA_FORGE_GITHUB_TOKEN:-}" ]; then
    echo "  ✓ GitHub PAT found in environment."
fi

if [ -n "${HF_TOKEN:-}" ] || [ -n "${DATA_FORGE_HF_TOKEN:-}" ]; then
    echo "  ✓ Hugging Face token found in environment."
fi

echo ""
echo "============================================================"
echo "  ENVIRONMENT SETUP COMPLETE"
echo ""
echo "  Next steps:"
echo "    1. Run: bash scripts/01_data_pipeline_dry_run.sh"
echo "    2. Run: bash scripts/02_data_pipeline_sample_test.sh"
echo "    3. Run: bash scripts/07_train.sh --model all"
echo "============================================================"
