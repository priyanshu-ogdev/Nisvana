#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — [00] DGX Spark GB10 Environment Setup
# Target: NVIDIA DGX Spark (Grace Blackwell GB10, CUDA 13, Ubuntu)
# 
# Installs ALL dependencies including:
# - PyTorch with CUDA support (or detects pre-installed PyTorch 2.7+)
# - Build tools (ninja, packaging, maturin, wheel)
# - deepfilternet (pre-built runtime library, avoiding deepfilterdataloader build errors)
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

# Detect host GPU and CUDA version via nvidia-smi
CUDA_MAJOR="13"
CUDA_VERSION="unknown"
GPU_NAME="unknown"
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' || echo "unknown")
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown")
    echo "  GPU: ${GPU_NAME}"
    echo "  CUDA Driver Version: ${CUDA_VERSION}"
    if [[ "${CUDA_VERSION}" =~ ^([0-9]+) ]]; then
        CUDA_MAJOR="${BASH_REMATCH[1]}"
    fi
else
    echo "  WARNING: nvidia-smi not found. GPU/CUDA may not be available."
fi

# Determine target PyTorch CUDA wheel repository
if [ "${CUDA_MAJOR}" -ge 13 ] 2>/dev/null; then
    TARGET_PYTORCH_TAG="cu130"
else
    TARGET_PYTORCH_TAG="cu126"
fi

# Detect if PyTorch with matching CUDA architecture is already present
NEED_TORCH_REINSTALL=false
HAS_TORCH_CUDA=false
TORCH_VER="none"
if ${PY_BIN} -c "import torch" 2>/dev/null; then
    CUDA_AVAIL=$(${PY_BIN} -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    TORCH_VER=$(${PY_BIN} -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
    TORCH_CUDA_BUILD=$(${PY_BIN} -c "import torch; print(getattr(torch.version, 'cuda', 'none') or 'none')" 2>/dev/null || echo "none")
    echo "  ✓ Active PyTorch found: v${TORCH_VER} (CUDA build: ${TORCH_CUDA_BUILD} | CUDA Available: ${CUDA_AVAIL})"

    if [ "${CUDA_AVAIL}" != "True" ]; then
        echo "  ⚠ PyTorch is CPU-only. Upgrade to CUDA ${TARGET_PYTORCH_TAG} required."
        NEED_TORCH_REINSTALL=true
    elif [ "${CUDA_MAJOR}" -ge 13 ] 2>/dev/null && [[ ! "${TORCH_CUDA_BUILD}" =~ ^13 ]]; then
        echo "  ⚠ CUDA VERSION MISMATCH DETECTED:"
        echo "    System has CUDA ${CUDA_VERSION} / GPU ${GPU_NAME} (sm_121),"
        echo "    but active PyTorch was built for CUDA ${TORCH_CUDA_BUILD} (${TORCH_VER})."
        echo "    This mismatch causes runtime sm_121 kernel incompatibility and nvcc C++ build crashes."
        echo "    Purging mismatched PyTorch and installing CUDA 13.0 (${TARGET_PYTORCH_TAG}) build..."
        NEED_TORCH_REINSTALL=true
    else
        echo "  ✓ Active PyTorch CUDA build matches target system (${TARGET_PYTORCH_TAG})."
        HAS_TORCH_CUDA=true
    fi
else
    echo "  PyTorch not found in target Python. Will be installed."
    NEED_TORCH_REINSTALL=true
fi

# Check disk space
AVAIL_GB=$(df -BG "${ROOT_DIR}" | tail -1 | awk '{print $4}' | tr -d 'G' || echo "0")
echo "  Available disk: ${AVAIL_GB}GB"
if [ "${AVAIL_GB}" -lt 100 ]; then
    echo "  WARNING: Less than 100GB free. TB-scale data pipeline may need more space."
fi

# ==============================================================================
# 2. Rust Toolchain (optional, for custom Rust builds)
# ==============================================================================
echo ""
echo ">>> [2/7] Checking Rust toolchain..."

if command -v rustc &> /dev/null; then
    RUST_VERSION=$(rustc --version)
    echo "  Found Rust: ${RUST_VERSION}"
else
    echo "  Rust toolchain not detected in PATH. If required, install via:"
    echo "    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
fi

# ==============================================================================
# 3. Upgrading Build Tools (Ninja, Packaging, Maturin for C++/CUDA Compiles)
# ==============================================================================
echo ""
echo ">>> [3/7] Upgrading pip and build tools (ninja, packaging, maturin)..."

${PY_BIN} -m pip install --upgrade pip setuptools wheel build ninja packaging maturin 2>&1 | tail -5

# ==============================================================================
# 4. Core Dependencies & PyTorch Installation (CUDA 13 / cu130)
# ==============================================================================
echo ""
if [ "${NEED_TORCH_REINSTALL}" = "true" ]; then
    echo ">>> [4/7] Installing PyTorch with CUDA support (${TARGET_PYTORCH_TAG})..."
    echo "  Purging mismatched PyTorch, torchaudio, torchvision, C++ extensions, and old CUDA runtime libs..."
    ${PY_BIN} -m pip uninstall -y torch torchaudio torchvision causal-conv1d mamba-ssm 2>/dev/null || true
    ${PY_BIN} -m pip uninstall -y nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12 2>/dev/null || true

    echo "  Downloading PyTorch from ${TARGET_PYTORCH_TAG} index..."
    ${PY_BIN} -m pip install torch torchaudio torchvision \
        --index-url "https://download.pytorch.org/whl/${TARGET_PYTORCH_TAG}" \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
        --extra-index-url https://pypi.org/simple 2>&1 | tail -10
else
    echo ">>> [4/7] Reusing active verified CUDA PyTorch installation (v${TORCH_VER} | CUDA ${TARGET_PYTORCH_TAG})..."
fi

echo "  Installing base requirements from requirements.txt..."
${PY_BIN} -m pip install -r "${ROOT_DIR}/requirements.txt" 2>&1 | tail -10

# ==============================================================================
# 5. Non-Sandboxed Mamba SSM, Causal-Conv1D & DeepFilterNet (--no-build-isolation)
# ==============================================================================
echo ""
echo ">>> [5/7] Installing Mamba SSM, Causal-Conv1D, and DeepFilterNet..."
echo "  Note: Compiling directly against active PyTorch (${PY_BIN}) to avoid sandbox isolation."

# Auto-detect target CUDA architecture (e.g. sm_121 for GB10) if not specified
if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    echo "  Using specified TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
else
    DETECTED_CC=$(${PY_BIN} -c "import torch; print(f'{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else '')" 2>/dev/null || true)
    if [ -n "${DETECTED_CC}" ]; then
        export TORCH_CUDA_ARCH_LIST="${DETECTED_CC}"
        echo "  Targeting active GPU CUDA Architecture: sm_${DETECTED_CC//./} (${DETECTED_CC})"
    fi
fi
export MAX_JOBS="${MAX_JOBS:-8}"

# causal-conv1d MUST be installed first without build isolation
echo "  Installing causal-conv1d (--no-build-isolation)..."
if ${PY_BIN} -m pip install --no-build-isolation "causal-conv1d>=1.4.0" 2>&1 | tail -5; then
    echo "  ✓ causal-conv1d installed successfully."
else
    echo "  ⚠ Building causal-conv1d from source failed. Trying binary distribution fallback..."
    ${PY_BIN} -m pip install --no-build-isolation --prefer-binary "causal-conv1d>=1.4.0" || true
fi

# mamba-ssm MUST be installed after causal-conv1d without build isolation
echo "  Installing mamba-ssm (--no-build-isolation)..."
if ${PY_BIN} -m pip install --no-build-isolation "mamba-ssm>=2.2.0" 2>&1 | tail -5; then
    echo "  ✓ mamba-ssm installed successfully."
else
    echo "  ⚠ Building mamba-ssm CUDA C++ extension failed on host compiler."
    echo "    Installing mamba-ssm with Triton backend fallback (MAMBA_SKIP_CUDA_BUILD=TRUE)..."
    MAMBA_SKIP_CUDA_BUILD=TRUE ${PY_BIN} -m pip install --no-build-isolation "mamba-ssm>=2.2.0" 2>&1 | tail -5 || true
fi

# DeepFilterNet: install without packaging downgrade conflict, then patch torchaudio.backend
echo "  Installing and configuring DeepFilterNet..."
${PY_BIN} -m pip install "deepfilterlib==0.5.6" appdirs loguru 2>&1 | tail -3 || true
# Install deepfilternet with --no-deps to prevent it from downgrading packaging to 23.x
${PY_BIN} -m pip install --no-deps "deepfilternet>=0.5.6" 2>&1 | tail -3 || true
# Ensure packaging>=24.0 for build tools
${PY_BIN} -m pip install "packaging>=24.0" 2>&1 | tail -3 || true

# Apply torchaudio backwards compatibility patch for modern torchaudio (2.x / 2.9+ / 2.11 / 2.14)
${PY_BIN} -c "
import sys, os, importlib.util, re

# 1. Provide sitecustomize shim so any process in this Python env has torchaudio.backend & AudioMetaData
try:
    import site
    site_pkgs = site.getsitepackages()
    if site_pkgs:
        sc_path = os.path.join(site_pkgs[0], 'sitecustomize.py')
        shim = '''# Project AEGIS torchaudio backwards compatibility shim for DeepFilterNet
try:
    import sys, types, torchaudio
    from dataclasses import dataclass
    @dataclass
    class AudioMetaData:
        sample_rate: int = 0
        num_frames: int = 0
        num_channels: int = 0
        bits_per_sample: int = 0
        encoding: str = \"\"

    if not hasattr(torchaudio, \"AudioMetaData\"):
        torchaudio.AudioMetaData = AudioMetaData

    if not hasattr(torchaudio, \"backend\"):
        backend_mod = types.ModuleType(\"torchaudio.backend\")
        common_mod = types.ModuleType(\"torchaudio.backend.common\")
        common_mod.AudioMetaData = AudioMetaData
        backend_mod.common = common_mod
        sys.modules[\"torchaudio.backend\"] = backend_mod
        sys.modules[\"torchaudio.backend.common\"] = common_mod
        setattr(torchaudio, \"backend\", backend_mod)
    else:
        if not hasattr(torchaudio.backend, \"common\"):
            common_mod = types.ModuleType(\"torchaudio.backend.common\")
            common_mod.AudioMetaData = AudioMetaData
            torchaudio.backend.common = common_mod
            sys.modules[\"torchaudio.backend.common\"] = common_mod
        else:
            if not hasattr(torchaudio.backend.common, \"AudioMetaData\"):
                torchaudio.backend.common.AudioMetaData = AudioMetaData
except Exception:
    pass
'''
        existing = ''
        if os.path.isfile(sc_path):
            with open(sc_path, 'r', encoding='utf-8') as f:
                existing = f.read()
        if 'Project AEGIS torchaudio backwards compatibility shim' in existing:
            parts = existing.split('# Project AEGIS torchaudio backwards compatibility shim')
            existing = parts[0]
        with open(sc_path, 'w', encoding='utf-8') as f:
            f.write(existing.strip() + '\n' + shim)
        print('  ✓ Updated torchaudio.backend & AudioMetaData shim in sitecustomize.py')
except Exception as e:
    pass

# 2. Patch df/io.py if present with standalone AudioMetaData dataclass
try:
    spec = importlib.util.find_spec('df')
    if spec and spec.submodule_search_locations:
        df_dir = list(spec.submodule_search_locations)[0]
        io_file = os.path.join(df_dir, 'io.py')
        if os.path.isfile(io_file):
            with open(io_file, 'r', encoding='utf-8') as f:
                content = f.read()
            replacement_code = '''from dataclasses import dataclass
@dataclass
class AudioMetaData:
    sample_rate: int = 0
    num_frames: int = 0
    num_channels: int = 0
    bits_per_sample: int = 0
    encoding: str = \"\"
'''
            # Replace any previous torchaudio AudioMetaData import variants
            new_content = re.sub(
                r'try:\s+from torchaudio\.backend\.common import AudioMetaData\s+except [^:]+:\s+from torchaudio import AudioMetaData',
                replacement_code,
                content
            )
            new_content = re.sub(
                r'from torchaudio(?:\.backend\.common)? import AudioMetaData',
                replacement_code,
                new_content
            )
            if new_content != content:
                with open(io_file, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print('  ✓ Patched df/io.py with standalone AudioMetaData dataclass')
            else:
                print('  ✓ df/io.py already patched with standalone AudioMetaData')
except Exception as e:
    pass
" 2>&1

# Run verification of all critical packages
echo ""
echo "  Verifying installed neural stack:"
${PY_BIN} -c "
import torch
cuda_ok = torch.cuda.is_available()
dev = torch.cuda.get_device_name(0) if cuda_ok else 'CPU'
cc = f'sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}' if cuda_ok else 'N/A'
cuda_ver = getattr(torch.version, 'cuda', 'none')
print(f'  torch {torch.__version__} (CUDA {cuda_ver}) | CUDA available: {cuda_ok} | Device: {dev} ({cc})')
" || echo "  WARNING: torch import failed"
${PY_BIN} -c "import torchaudio; print(f'  torchaudio {torchaudio.__version__}')" || echo "  WARNING: torchaudio import failed"

${PY_BIN} -c "
try:
    from df.model import ModelParams
    print('  deepfilternet ✓ (DeepFilterNet modules available)')
except ImportError as e:
    print(f'  deepfilternet ✗ ({e})')
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
