#!/usr/bin/env bash
# Project AEGIS — Convenience wrapper for environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/00_setup_environment.sh" "$@"
