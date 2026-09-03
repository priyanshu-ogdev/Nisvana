#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — [07] Train Model 1: Primary Real-Time SE (DeepFilterNet3)
# Zero-lookahead (0 ms latency) streaming speech enhancement on 48kHz shards.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"
PY_BIN="$(command -v python3 || command -v python)"

echo "=== PROJECT AEGIS: TRAINING MODEL 1 (aegis-se-primary) ==="
echo "Configuration: 0 ms lookahead, multi-res spectral + local SNR loss, SpecMix, EMA"
${PY_BIN} -m training.scripts.train_se_primary "$@"
echo "=== MODEL 1 TRAINING COMPLETED ==="
