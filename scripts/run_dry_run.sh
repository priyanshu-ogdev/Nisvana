#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — Pipeline Dry-Run Probing Script
# Probes remote endpoints, HTTP headers, sizes, and verifiers without writing multi-GB data.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== PROJECT AEGIS: DRY-RUN PROBE START ==="
cd "${ROOT_DIR}"

PY_BIN="$(command -v python3 || command -v python)"
${PY_BIN} -m data_forge run-all --dry-run "$@"

echo "=== PROJECT AEGIS: DRY-RUN PROBE COMPLETE ==="
