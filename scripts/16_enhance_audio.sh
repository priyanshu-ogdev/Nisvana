#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — [16] Offline Audio File Enhancement Runner
# Enhances noisy audio files using DeepFilterNet3, CleanUMamba, or Hybrid ANC.
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

echo "=== PROJECT AEGIS: RUNNING AUDIO FILE ENHANCER ==="
${PY_BIN} -m inference.scripts.enhance_audio "$@"
echo "=== AUDIO ENHANCEMENT COMPLETED ==="
