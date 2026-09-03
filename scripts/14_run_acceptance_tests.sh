#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — [14] Mission-Critical Defence Acceptance Test Runner
# Verifies all acceptance criteria: PESQ > 2.5, STOI > 0.85, SNR > 15 dB,
# defence disturbance scenarios, hybrid AI+NLMS ANC pipeline, and ONNX export.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"
PY_BIN="$(command -v python3 || command -v python)"

echo "=== PROJECT AEGIS: RUNNING MISSION-CRITICAL DEFENCE ACCEPTANCE SUITE ==="
${PY_BIN} -m pytest -v "${ROOT_DIR}/tests/test_defence_mission_critical_acceptance.py" "$@"
echo "=== DEFENCE ACCEPTANCE TESTS PASSED ==="
