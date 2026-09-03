#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — [06] Automated Test Runner
# Runs the full 49-test suite verifying all 6 subpackages.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"
PY_BIN="$(command -v python3 || command -v python)"

echo "=== PROJECT AEGIS: RUNNING ALL UNIT & INTEGRATION TESTS ==="
${PY_BIN} -m pytest -v "${ROOT_DIR}/tests" "$@"
echo "=== ALL TESTS PASSED ==="
