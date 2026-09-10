#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — Common Python Environment Resolver
# Target: DGX Spark GB10 / Linux / Edge Systems
# 
# Resolves the exact active Python binary that contains PyTorch and project
# dependencies, preventing scripts from accidentally falling back to sandboxed
# system Python or empty virtual environments.
# ==============================================================================

detect_python() {
    # 1. Active Virtualenv (highest priority)
    if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
        echo "${VIRTUAL_ENV}/bin/python"
        return 0
    fi

    # 2. Active Conda environment (GB10 / DGX Spark standard)
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
        echo "${CONDA_PREFIX}/bin/python"
        return 0
    fi

    # 3. 'python' command in PATH if it has PyTorch installed
    if command -v python &> /dev/null && python -c "import torch" &> /dev/null; then
        command -v python
        return 0
    fi

    # 4. 'python3' command in PATH if it has PyTorch installed
    if command -v python3 &> /dev/null && python3 -c "import torch" &> /dev/null; then
        command -v python3
        return 0
    fi

    # 5. Standard fallbacks in PATH
    if command -v python3 &> /dev/null; then
        command -v python3
    elif command -v python &> /dev/null; then
        command -v python
    else
        echo "python3"
    fi
}
