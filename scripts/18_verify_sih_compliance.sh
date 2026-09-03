#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — [18] SIH Defence Benchmark & Inference Verification Runner
# Validates all official SIH metrics: SNR > 15 dB, STOI > 0.85, PESQ > 2.50,
# all 7 defence disturbances, Hybrid AI+NLMS ANC, and edge hardware execution.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"
PY_BIN="$(command -v python3 || command -v python)"

echo "=== PROJECT AEGIS: RUNNING OFFICIAL SIH DEFENCE BENCHMARK AUDIT ==="
${PY_BIN} -m pytest -v "${ROOT_DIR}/tests/test_sih_inference_metrics.py" "$@"
echo "=== SIH DEFENCE COMPLIANCE AUDIT PASSED ==="
