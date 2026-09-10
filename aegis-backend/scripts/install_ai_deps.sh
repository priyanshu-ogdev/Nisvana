#!/usr/bin/env bash
# install_ai_deps.sh — Install AI / ML dependencies on Raspberry Pi 5
# Run as: bash scripts/install_ai_deps.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Installing AEGIS AI dependencies ==="

# Create venv if needed
if [ ! -d "$ROOT/venv" ]; then
    python3 -m venv "$ROOT/venv"
fi

source "$ROOT/venv/bin/activate"
pip install --upgrade pip

# Core runtime (no GPU on Pi 5)
pip install -r "$ROOT/requirements.txt"

# ONNX Runtime (CPU, arm64 wheel)
pip install onnxruntime

# Optional: libDF Python bindings for native DeepFilterNet3 inference
# pip install deepfilternet   # PyPI — requires gcc, libstdc++

echo "=== AI dependencies installed. ==="
echo "Model checkpoint paths should be placed in $ROOT/models/ as configured in config/models.yaml"
