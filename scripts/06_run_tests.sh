#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — [06] Automated Test Runner
# Runs the full 286-test suite across 32 test files verifying all subpackages.
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

echo "=== PROJECT AEGIS: RUNNING ALL UNIT & INTEGRATION TESTS ==="
${PY_BIN} -m pytest -v "${ROOT_DIR}/tests" "$@"
echo "=== ALL TESTS PASSED ==="
