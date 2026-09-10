#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — Train Model 2: Escalation SE (DeepFilterNet3 Escalation)
# 1-chunk (10ms) lookahead output-delay streaming speech enhancement for extreme noise.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/07_train.sh" --model se_escalation "$@"
