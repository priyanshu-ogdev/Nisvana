#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — [11] Train Model 5: Gated Acoustic Echo Cancellation
# Safety-gated training script (requires explicit --force).
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"
PY_BIN="$(command -v python3 || command -v python)"

echo "=== PROJECT AEGIS: TRAINING MODEL 5 (aegis-aec-gate) ==="
echo "Note: Default deployment uses pretrained deepvqe-ggml checkpoint."
${PY_BIN} -m training.scripts.train_aec "$@"
echo "=== MODEL 5 SCRIPT FINISHED ==="
