#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — Model 5: Gated Acoustic Echo Cancellation (AEC Challenge)
# Dual-branch complex STFT post-filter for tactical radio echo suppression.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/07_train.sh" --model aec "$@"
