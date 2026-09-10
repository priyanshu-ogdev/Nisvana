#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — [17] Real-Time Live Microphone & Headset ANC Prototype
# Live tactical streaming demo with acoustic classification, dynamic escalation,
# and hybrid AI + Normalized LMS adaptive filtering.
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

echo "=== PROJECT AEGIS: RUNNING LIVE STREAMING ANC PROTOTYPE DEMONSTRATION ==="
${PY_BIN} -m inference.scripts.live_mic_anc "$@"
echo "=== PROTOTYPE DEMONSTRATION FINISHED ==="
