#!/usr/bin/env bash
# ==============================================================================
# Project AEGIS — [03] Full Production Data Pipeline Orchestrator
# Runs the complete multi-terabyte download, 10-step preprocessing, augmentation,
# multi-branch forge mixing, WebDataset shard export, and pipeline verification
# on the high-capacity ML machine (4 TB storage).
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"
PY_BIN="$(command -v python3 || command -v python)"

# Default configurations
WORKERS="${WORKERS:-16}"
NUM_MIXTURES="${NUM_MIXTURES:-200000}"
COMMERCIAL_STRICT="${COMMERCIAL_STRICT:-0}"

# Parse optional command line flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --mixtures)
            NUM_MIXTURES="$2"
            shift 2
            ;;
        --commercial-strict)
            COMMERCIAL_STRICT=1
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--workers N] [--mixtures M] [--commercial-strict]"
            exit 1
            ;;
    esac
done

echo "=============================================================================="
echo "          PROJECT AEGIS: FULL PRODUCTION DATA-FORGE EXECUTION"
echo "=============================================================================="
echo "Working Directory      : ${ROOT_DIR}"
echo "Max Worker Threads     : ${WORKERS}"
echo "Target SE Mixtures     : ${NUM_MIXTURES}"
echo "Commercial Strict Mode : ${COMMERCIAL_STRICT}"
echo "Started at             : $(date)"
echo "=============================================================================="

START_TIME=$(date +%s)

# Step 0: Ensure directory tree exists
bash "${SCRIPT_DIR}/00_setup_environment.sh"

# Step 1: Multi-Source Download in Full Mode
echo ""
echo ">>> [STAGE 1/6] Downloading verified raw datasets in FULL PRODUCTION MODE..."
${PY_BIN} -m data_forge fetch --source all --full-mode

# Step 2: 10-Step Sequential Preprocessing Pipeline
echo ""
echo ">>> [STAGE 2/6] Executing 10-step sequential preprocessing pipeline..."
PREPROCESS_ARGS=(--max-workers "${WORKERS}")
if [ "${COMMERCIAL_STRICT}" -eq 1 ]; then
    PREPROCESS_ARGS+=(--commercial-strict)
fi
${PY_BIN} -m data_forge preprocess "${PREPROCESS_ARGS[@]}"

# Step 3: Grounded Per-Source Augmentation (Zero pitch-shift invariant enforced)
echo ""
echo ">>> [STAGE 3/6] Applying grounded per-source augmentations..."
${PY_BIN} -m data_forge augment

# Step 4: Multi-Branch Training Mixing (Models 1-5)
echo ""
echo ">>> [STAGE 4/6] Generating multi-branch model training corpora (${NUM_MIXTURES} mixtures)..."
${PY_BIN} -m data_forge mix --num-mixtures "${NUM_MIXTURES}" --min-snr -5.0 --max-snr 20.0

# Step 5: WebDataset Shard Export & HuggingFace Dataset Card
echo ""
echo ">>> [STAGE 5/6] Packing training branches into WebDataset sequential tar-shards..."
${PY_BIN} -m data_forge export

# Step 6: Complete Pipeline Audit & Verification
echo ""
echo ">>> [STAGE 6/6] Executing full pipeline audit and compliance verification..."
${PY_BIN} -m data_forge verify

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "=============================================================================="
echo "          PROJECT AEGIS: FULL PIPELINE COMPLETED SUCCESSFULLY"
echo "=============================================================================="
echo "Total Elapsed Time: $((ELAPSED / 3600))h $(((ELAPSED % 3600) / 60))m $((ELAPSED % 60))s"
echo "Output Data Tree  : ${ROOT_DIR}/data"
echo "WebDataset Shards : ${ROOT_DIR}/data/shards"
echo "Dataset Card      : ${ROOT_DIR}/data/shards/DATASET_CARD.md"
echo "Audit Report      : ${ROOT_DIR}/data/manifests/pipeline_audit_report.md"
echo "=============================================================================="
