#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — [09] Train Model 3: State-Space SE Cross-Check (CleanUMamba)
# Independent architecture cross-check and GPU-fallback speech enhancement.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"
PY_BIN="$(command -v python3 || command -v python)"

echo "=== PROJECT AEGIS: TRAINING MODEL 3 (aegis-se-crosscheck) ==="
echo "Configuration: CleanUMamba 48kHz native fine-tune on AEGIS shards"
${PY_BIN} -m training.scripts.train_se_crosscheck "$@"
echo "=== MODEL 3 TRAINING COMPLETED ==="
