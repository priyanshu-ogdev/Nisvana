#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — [01] Data Pipeline Dry-Run Probing Script
# Probes remote endpoints, HTTP headers, sizes, and verifiers without writing multi-GB data.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/common_env.sh" 2>/dev/null || true

echo "=== PROJECT AEGIS: DATA PIPELINE DRY-RUN PROBE START ==="
cd "${ROOT_DIR}"

if declare -f detect_python > /dev/null; then
    PY_BIN="$(detect_python)"
else
    PY_BIN="$(command -v python3 || command -v python)"
fi
${PY_BIN} -m data_forge run-all --dry-run "$@"

echo "=== PROJECT AEGIS: DATA PIPELINE DRY-RUN PROBE COMPLETE ==="
