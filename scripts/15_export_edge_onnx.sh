#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — [15] ONNX Edge Model Exporter & Latency Profiler
# Exports models using the canonical 48 kHz / 10 ms streaming contract.
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

echo "=== PROJECT AEGIS: EXPORTING MODEL TO ONNX FOR EDGE HARDWARE ==="
${PY_BIN} -m inference.scripts.export_onnx "$@"
echo "=== ONNX EXPORT COMPLETED ==="
