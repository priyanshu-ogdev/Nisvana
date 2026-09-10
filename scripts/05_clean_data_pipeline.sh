#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — Data Pipeline Clean & Reset Script (Blank Slate)
# Purges all downloaded, processed, augmented, mixed, and sharded data artifacts
# while restoring the clean directory structure for a fresh production run.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data"

echo "=== PROJECT AEGIS: CLEANING DATA PIPELINE ==="
echo "Target directory: ${DATA_DIR}"

PURGE_RAW=0
for arg in "$@"; do
    if [ "$arg" == "--purge-raw" ] || [ "$arg" == "--all" ]; then
        PURGE_RAW=1
    fi
done

# Remove generated data artifacts safely while protecting raw downloads
if [ -d "${DATA_DIR}" ]; then
    echo ">>> Purging generated pipeline artifacts..."
    rm -rf "${DATA_DIR}/processed"
    rm -rf "${DATA_DIR}/augmented"
    rm -rf "${DATA_DIR}/splits"
    rm -rf "${DATA_DIR}/forge"
    rm -rf "${DATA_DIR}/shards"
    rm -rf "${DATA_DIR}/manifests"
    
    if [ "${PURGE_RAW}" -eq 1 ]; then
        echo ">>> [--purge-raw specified] Purging data/raw download archives..."
        rm -rf "${DATA_DIR}/raw"
    else
        echo ">>> [PROTECTED] Preserving data/raw (manually placed archives & raw downloads intact)."
        echo ">>> To also purge raw downloads, run: $0 --purge-raw"
    fi
fi

echo ">>> Recreating empty directory skeleton for a fresh start..."
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

# Place .gitkeep in each directory to ensure clean tracking without files
find "${DATA_DIR}" -type d -exec touch {}/.gitkeep \; 2>/dev/null || true

echo ">>> data/ folder is 100% clean and ready for a fresh production run."
echo "=== DATA CLEANUP COMPLETE ==="
