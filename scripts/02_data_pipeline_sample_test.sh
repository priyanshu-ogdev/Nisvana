#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — [02] Sample Data Pipeline Test & Clean Slate Orchestrator
# Runs an end-to-end integration test on a sample subset, verifies compliance,
# and automatically purges data/ afterwards to leave a 100% blank slate for
# the real production server pipeline.
#
# Flags:
#   --keep-data     Skip post-test cleanup to inspect sample outputs
#   --mixtures N    Number of sample mixtures to synthesize (default: 20)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"
PY_BIN="$(command -v python3 || command -v python)"

NUM_MIXTURES=20
CLEAN_AFTER=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --keep-data)
            CLEAN_AFTER=0
            shift
            ;;
        --mixtures)
            NUM_MIXTURES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--keep-data] [--mixtures N]"
            exit 1
            ;;
    esac
done

echo "=============================================================================="
echo "      PROJECT AEGIS: SAMPLE DATA PIPELINE INTEGRATION TEST & RESET"
echo "=============================================================================="
echo "Working directory : ${ROOT_DIR}"
echo "Sample mixtures   : ${NUM_MIXTURES}"
echo "Auto clean after  : ${CLEAN_AFTER}"
echo "Started at        : $(date)"
echo "=============================================================================="

# 1. Reset directory structure to ensure clean starting state
echo ">>> [STAGE 1/4] Ensuring clean starting directory structure..."
bash "${SCRIPT_DIR}/05_clean_data_pipeline.sh"

# 2. Run the complete pipeline in sample mode
echo ">>> [STAGE 2/4] Executing sample data pipeline end-to-end..."
${PY_BIN} -m data_forge run-all --sample-mode --num-mixtures "${NUM_MIXTURES}"

# 3. Verify that the pipeline passed audit
echo ">>> [STAGE 3/4] Validating sample pipeline audit..."
AUDIT_REPORT="${ROOT_DIR}/data/manifests/pipeline_audit_report.json"
if [ ! -f "${AUDIT_REPORT}" ]; then
    echo "ERROR: Pipeline audit report was not generated."
    exit 1
fi

echo "Sample integration test passed successfully."

# 4. Cleanup to leave 100% blank slate for the real server pipeline
if [ "${CLEAN_AFTER}" -eq 1 ]; then
    echo ">>> [STAGE 4/4] Purging sample artifacts to provide a 100% blank slate..."
    bash "${SCRIPT_DIR}/05_clean_data_pipeline.sh"
    echo ">>> data/ folder has been completely cleaned and reset."
else
    echo ">>> [STAGE 4/4] --keep-data specified: Sample artifacts preserved in data/."
fi

echo "=============================================================================="
echo "      PROJECT AEGIS: SAMPLE TEST COMPLETED & SLATE IS 100% BLANK"
echo "=============================================================================="
