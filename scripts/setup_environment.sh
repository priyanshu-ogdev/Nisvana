#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — Environment Setup & Directory Initialization
# Prepares the server environment for multi-terabyte Data-Forge pipeline runs.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== PROJECT AEGIS: ENVIRONMENT SETUP ==="
echo "Working directory: ${ROOT_DIR}"

# 1. Python Environment Check
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python 3 is not installed or not in PATH."
    exit 1
fi

PY_BIN="$(command -v python3 || command -v python)"
echo "Using Python: $(${PY_BIN} --version) at ${PY_BIN}"

# 2. Dependency Installation
echo ">>> Installing dependencies from requirements.txt..."
${PY_BIN} -m pip install --upgrade pip
${PY_BIN} -m pip install -r "${ROOT_DIR}/requirements.txt"
${PY_BIN} -m pip install pytest webdataset torch --extra-index-url https://download.pytorch.org/whl/cpu || true

# 3. Directory Structure Initialization
echo ">>> Initializing complete data/ storage hierarchy..."
DATA_DIR="${ROOT_DIR}/data"
mkdir -p "${DATA_DIR}/raw/noisex92"
mkdir -p "${DATA_DIR}/raw/shared_explosions"
mkdir -p "${DATA_DIR}/raw/gunshot_dryad"
mkdir -p "${DATA_DIR}/raw/drone_audioset"
mkdir -p "${DATA_DIR}/raw/mad"
mkdir -p "${DATA_DIR}/raw/vctk_demand"
mkdir -p "${DATA_DIR}/raw/dns_challenge/noise_fullband"
mkdir -p "${DATA_DIR}/raw/aec_challenge"
mkdir -p "${DATA_DIR}/raw/sirens_urban"
mkdir -p "${DATA_DIR}/raw/openslr_rirs/rir_wavs"

mkdir -p "${DATA_DIR}/processed"
mkdir -p "${DATA_DIR}/augmented"
mkdir -p "${DATA_DIR}/splits"
mkdir -p "${DATA_DIR}/manifests"

mkdir -p "${DATA_DIR}/forge/branch_speech_enhancement/noisy"
mkdir -p "${DATA_DIR}/forge/branch_speech_enhancement/clean"
mkdir -p "${DATA_DIR}/forge/branch_speech_enhancement/rir"
mkdir -p "${DATA_DIR}/forge/branch_classifier/audio"
mkdir -p "${DATA_DIR}/forge/branch_aec/mic"
mkdir -p "${DATA_DIR}/forge/branch_aec/farend"
mkdir -p "${DATA_DIR}/forge/branch_aec/nearend"
mkdir -p "${DATA_DIR}/forge/branch_aec/echo"

mkdir -p "${DATA_DIR}/shards/speech_enhancement"
mkdir -p "${DATA_DIR}/shards/classifier"
mkdir -p "${DATA_DIR}/shards/aec"

echo ">>> Directory hierarchy initialized."

# 4. Check Optional API Credentials
echo ">>> Checking external dataset API tokens..."
if [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; then
    if [ ! -f "${HOME}/.kaggle/kaggle.json" ]; then
        echo "NOTE: Kaggle credentials not detected (KAGGLE_USERNAME/KEY or ~/.kaggle/kaggle.json)."
        echo "      To download the full Military Audio Dataset (MAD, ~1.1GB), obtain an API token"
        echo "      from https://www.kaggle.com/settings -> API -> Create New Token."
    else
        echo "Found Kaggle credentials in ${HOME}/.kaggle/kaggle.json"
    fi
else
    echo "Found Kaggle credentials in environment variables."
fi

if [ -z "${DRYAD_API_TOKEN:-}" ]; then
    echo "NOTE: DRYAD_API_TOKEN not set in environment. If fetching gunshots via Dryad REST API,"
    echo "      provide DRYAD_API_TOKEN or drop raw WAVs into data/raw/gunshot_dryad/."
else
    echo "Found DRYAD_API_TOKEN in environment."
fi

echo "=== ENVIRONMENT SETUP COMPLETE ==="
