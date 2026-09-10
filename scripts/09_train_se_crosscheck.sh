#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — Train Model 3: CleanUMamba (SSM Crosscheck & Distillation Teacher)
# Selective State Space Model for long-horizon temporal memory and distillation.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train.sh" --model se_crosscheck "$@"
