#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — Sample Mode Pipeline Execution
# Runs a lightweight, fast end-to-end verification pass using sample subsets.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== PROJECT AEGIS: SAMPLE MODE EXECUTION START ==="
cd "${ROOT_DIR}"

PY_BIN="$(command -v python3 || command -v python)"
NUM_MIXTURES="${1:-50}"

echo ">>> Running end-to-end pipeline in sample mode with ${NUM_MIXTURES} mixtures..."
${PY_BIN} -m data_forge run-all --sample-mode --num-mixtures "${NUM_MIXTURES}"

echo "=== PROJECT AEGIS: SAMPLE MODE EXECUTION COMPLETE ==="
