#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — Train Model 4: Gating Acoustic Classifier (3-Way Conformer/Conv1d)
# Fast acoustic environment threat classifier evaluated on 0.2s windows.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/07_train.sh" --model classifier "$@"
