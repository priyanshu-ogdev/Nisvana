#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — [12] Master Pipeline: Train All Models Sequentially
# Trains Model 1 (Primary SE), Model 2 (Escalation SE), Model 3 (Crosscheck),
# and Model 4 (Classifier) in sequence.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"
PY_BIN="$(command -v python3 || command -v python)"

echo "========================================================================"
echo "PROJECT AEGIS — SEQUENTIAL MULTI-MODEL TRAINING PIPELINE"
echo "========================================================================"

echo "[1/4] Training Model 1 (aegis-se-primary)..."
bash "${SCRIPT_DIR}/07_train_se_primary.sh" "$@"

echo "[2/4] Training Model 2 (aegis-se-escalation)..."
bash "${SCRIPT_DIR}/08_train_se_escalation.sh" "$@"

echo "[3/4] Training Model 3 (aegis-se-crosscheck)..."
bash "${SCRIPT_DIR}/09_train_se_crosscheck.sh" "$@"

echo "[4/4] Training Model 4 (aegis-clf-gate)..."
bash "${SCRIPT_DIR}/10_train_classifier.sh" "$@"

echo "========================================================================"
echo "ALL AEGIS MODELS SUCCESSFULLY TRAINED AND VERIFIED"
echo "Checkpoints banked in: ${ROOT_DIR}/data/checkpoints/"
echo "========================================================================"
