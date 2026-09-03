#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — [08] Train Model 2: Escalation SE (DeepFilterNet3)
# 40 ms lookahead (df_lookahead=2, conv_lookahead=2) for high-difficulty frames.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"
PY_BIN="$(command -v python3 || command -v python)"

echo "=== PROJECT AEGIS: TRAINING MODEL 2 (aegis-se-escalation) ==="
echo "Configuration: 40 ms lookahead, negative SNR emphasis ladder, dual-metric guard"
${PY_BIN} -m training.scripts.train_se_escalation "$@"
echo "=== MODEL 2 TRAINING COMPLETED ==="
