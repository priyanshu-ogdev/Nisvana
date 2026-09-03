#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — [10] Train Model 4: Acoustic Environment & Gating Classifier
# 3-way taxonomy (harmonic / impulsive / speech_dominant) on classifier shards.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"
PY_BIN="$(command -v python3 || command -v python)"

echo "=== PROJECT AEGIS: TRAINING MODEL 4 (aegis-clf-gate) ==="
echo "Configuration: 3-way classification + SNR estimation + Model 1/2 gap validation"
${PY_BIN} -m training.scripts.train_classifier "$@"
echo "=== MODEL 4 TRAINING COMPLETED ==="
