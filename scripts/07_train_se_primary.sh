#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — Train Model 1: Primary Real-Time SE (DeepFilterNet3 Base)
# Zero-lookahead (0 ms latency) streaming speech enhancement with QAT & Distillation.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train.sh" --model se_primary "$@"
