#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — [15] ONNX Edge Model Exporter & Latency Profiler
# Exports models to ONNX format with dynamic shapes for NVIDIA Jetson AGX Orin & DSPs.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"
PY_BIN="$(command -v python3 || command -v python)"

echo "=== PROJECT AEGIS: EXPORTING MODEL TO ONNX FOR EDGE HARDWARE ==="
${PY_BIN} -m inference.scripts.export_onnx "$@"
echo "=== ONNX EXPORT COMPLETED ==="
