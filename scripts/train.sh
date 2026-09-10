#!/usr/bin/env bash
# Project AEGIS — Convenience wrapper for master training pipeline
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/07_train.sh" "$@"
