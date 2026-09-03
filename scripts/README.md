# Project AEGIS — Linux Data Pipeline Orchestration Scripts

This directory contains POSIX-compliant, hardened Bash scripts (`set -euo pipefail`) designed to orchestrate the Project AEGIS Data-Forge pipeline on Linux environments.

The scripts are numbered in their recommended chronological order of execution:

```
scripts/
├── 00_setup_environment.sh               # [Step 0] Python dependencies & directory tree initialization
├── 01_data_pipeline_dry_run.sh           # [Step 1] Zero-disk-write endpoint & HTTP probe
├── 02_data_pipeline_sample_test.sh       # [Step 2] End-to-end integration test + auto blank slate reset
├── 03_data_pipeline_server_production.sh # [Step 3] Full-scale multi-worker 4 TB server pipeline runner
├── 04_export_webdataset_shards.sh        # [Step 4] Standalone WebDataset tar packer & dataset card generator
├── 05_clean_data_pipeline.sh             # [Utility] Complete data/ purge and blank slate reset
└── 06_run_tests.sh                       # [Testing] Automated 49-test suite verification
```

---

## Script Reference

### `00_setup_environment.sh`
- **Purpose**: Prepares a fresh server or development machine.
- **Actions**:
  - Verifies Python 3.10+ installation.
  - Upgrades `pip` and installs all dependencies from `requirements.txt`.
  - Installs PyTorch and `webdataset`.
  - Initializes the complete `data/` directory hierarchy (raw, processed, augmented, splits, forge, shards).
  - Checks for optional API tokens (`KAGGLE_USERNAME`/`KEY` and `DRYAD_API_TOKEN`).
- **Usage**:
  ```bash
  bash scripts/00_setup_environment.sh
  ```

---

### `01_data_pipeline_dry_run.sh`
- **Purpose**: Validates all remote dataset URLs, Azure Blob containers, and format verifiers with **zero data written to disk**.
- **Actions**:
  - Sends HTTP `HEAD` (or streamed `GET`) requests to all 10 authoritative sources.
  - Probes multi-gigabyte DNS-5 training blobs on Azure Storage.
  - Runs the compliance auditor on metadata and configuration profiles.
- **Usage**:
  ```bash
  bash scripts/01_data_pipeline_dry_run.sh
  ```

---

### `02_data_pipeline_sample_test.sh`
- **Purpose**: Executes a fast end-to-end integration test of the entire pipeline and **automatically cleans the `data/` folder afterwards** so you can start the real server pipeline with a 100% blank slate.
- **Workflow**:
  1. Calls `05_clean_data_pipeline.sh` to ensure a pristine starting directory.
  2. Runs fetch (sample mode), 10-step preprocessor, augmentor, mixer (20 mixtures), WebDataset shard exporter, and verifier.
  3. Verifies that the audit report passes.
  4. Calls `05_clean_data_pipeline.sh` to purge sample artifacts, leaving the `data/` folder 100% clean.
- **Options**:
  - `--keep-data`: Disables post-test cleanup to let you inspect sample audio and manifest outputs.
  - `--mixtures N`: Number of sample triplets to generate (default: 20).
- **Usage**:
  ```bash
  # Standard test + auto clean slate
  bash scripts/02_data_pipeline_sample_test.sh

  # Test and retain sample files for manual inspection
  bash scripts/02_data_pipeline_sample_test.sh --keep-data --mixtures 50
  ```

---

### `03_data_pipeline_server_production.sh`
- **Purpose**: **Primary production orchestrator for the 4 TB server**. Runs the entire multi-source download, 10-step preprocessing, grounded augmentation, multi-branch mixing, WebDataset sharding, and pipeline audit.
- **Workflow**:
  1. `data_forge fetch --source all --server-mode`: Downloads complete training sets (including multi-part Azure Blobs for DNS-5 noise & clean speech, Kaggle MAD, Dryad gunshots, SHAReD, NOISEX-92, etc.).
  2. `data_forge preprocess --max-workers N`: Parallel 10-step standardization to 48kHz, -23 LUFS, mono.
  3. `data_forge augment`: Grounded WSOLA time-stretch [0.90, 1.10] and gain jitter (zero pitch-shifting).
  4. `data_forge mix --num-mixtures M`: Synthesizes Model 1-3 SE triplets, Model 4 classifier, and Model 5 AEC samples.
  5. `data_forge export`: Packs samples into sequential `se-*.tar`, `clf-*.tar`, `aec-*.tar` shards and writes `DATASET_CARD.md`.
  6. `data_forge verify`: Generates final `pipeline_audit_report.md` and audit JSON.
- **Options**:
  - `--workers N`: Number of CPU worker threads for polyphase resampling (default: 16).
  - `--mixtures M`: Number of speech enhancement triplets to generate (default: 200,000).
  - `--commercial-strict`: Excludes CC-BY-NC non-commercial datasets.
- **Usage**:
  ```bash
  # Full production run (recommend inside tmux)
  bash scripts/03_data_pipeline_server_production.sh --workers 16 --mixtures 200000
  ```

---

### `04_export_webdataset_shards.sh`
- **Purpose**: Standalone tool to pack existing `data/forge/` branches into WebDataset `.tar` shards without re-running preprocessing or mixing.
- **Actions**:
  - Organizes samples into ~2048 samples/shard.
  - Computes shard checksums and writes Croissant/HuggingFace-compliant `DATASET_CARD.md`.
- **Usage**:
  ```bash
  bash scripts/04_export_webdataset_shards.sh
  ```

---

### `05_clean_data_pipeline.sh`
- **Purpose**: Completely purges all raw, processed, augmented, forged, and sharded files from `data/`, restoring an empty directory skeleton.
- **Safety**: Recreates all required subdirectories and adds empty `.gitkeep` markers so git tracking remains clean.
- **Usage**:
  ```bash
  bash scripts/05_clean_data_pipeline.sh
  ```

---

### `06_run_tests.sh`
- **Purpose**: Executes the complete 49-test unit and integration test suite across all 6 packages using `pytest`.
- **Usage**:
  ```bash
  bash scripts/06_run_tests.sh
  ```
