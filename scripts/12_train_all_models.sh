#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — [12] Master Pipeline: Train All Models Sequentially
# Executes complete sequential ensemble training in scientific dependency order:
# CleanUMamba (Teacher) -> DeepFilterNet3 Base (Student) -> Escalation -> Classifier -> AEC.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train.sh" --model all "$@"
