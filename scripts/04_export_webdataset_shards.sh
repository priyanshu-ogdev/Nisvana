#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — [04] Standalone WebDataset Sharding Script
# Repacks data/forge/ outputs into sequential .tar shards with auto-generated DATASET_CARD.md
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

echo "=== PROJECT AEGIS: WEBDATASET EXPORT INITIATED ==="
${PY_BIN} -m data_forge export "$@"
echo "=== PROJECT AEGIS: WEBDATASET EXPORT COMPLETED ==="
