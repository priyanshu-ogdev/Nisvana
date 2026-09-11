#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — [13] Multi-Model Evaluation Suite
# Evaluates speech enhancement (PESQ, STOI, SI-SNR, SSNR), classifier (Accuracy, F1),
# and AEC (ERLE) across validation or generalization test splits.
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

echo "=== PROJECT AEGIS: RUNNING REAL-DATA AUDIO METRICS EVALUATION SUITE ==="
${PY_BIN} -m training.scripts.evaluate_models "$@"
echo "=== EVALUATION COMPLETED ==="
