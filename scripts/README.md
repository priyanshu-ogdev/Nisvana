# Project AEGIS — Linux Data Pipeline Orchestration Scripts

This directory contains POSIX-compliant, hardened Bash scripts (`set -euo pipefail`) designed to orchestrate the Project AEGIS Data-Forge pipeline on Linux environments.

The scripts are numbered in their recommended chronological order of execution:

```
scripts/
├── 00_setup_environment.sh         # [Step 0] Python dependencies & directory tree initialization
├── 01_data_pipeline_dry_run.sh     # [Step 1] Zero-disk-write endpoint & HTTP probe
├── 02_data_pipeline_sample_test.sh # [Step 2] End-to-end integration test + auto blank slate reset
├── 03_data_pipeline_full_run.sh    # [Step 3] Full production multi-worker 4TB pipeline runner
├── 04_export_webdataset_shards.sh  # [Step 4] Standalone WebDataset tar packer & dataset card generator
├── 05_clean_data_pipeline.sh       # [Utility] Complete data/ purge and blank slate reset
├── 06_run_tests.sh                 # [Testing] Automated 108-test suite verification
├── 07_train_se_primary.sh/.ps1     # [Train] Model 1: Primary Real-Time SE (0ms lookahead)
├── 08_train_se_escalation.sh/.ps1  # [Train] Model 2: Escalation SE (40ms lookahead)
├── 09_train_se_crosscheck.sh/.ps1  # [Train] Model 3: State-Space SE (CleanUMamba)
├── 10_train_classifier.sh/.ps1     # [Train] Model 4: Acoustic Environment & Gate Classifier
├── 11_train_aec.sh/.ps1            # [Train] Model 5: Acoustic Echo Cancellation (--force)
├── 12_train_all_models.sh/.ps1     # [Train] Sequential Multi-Model Master Orchestrator
├── 13_evaluate_models.sh/.ps1      # [Eval] Multi-Model Audio Evaluation Suite (PESQ, STOI, SI-SNR, SSNR)
├── 14_run_acceptance_tests.sh/.ps1 # [Acceptance] Mission-Critical Defence Acceptance Test Suite
└── 15_export_edge_onnx.sh/.ps1     # [Edge] ONNX Edge Model Exporter & Latency Profiler
```

> **Cross-Platform**: All scripts are provided in dual implementations: hardened Bash (`.sh`) for Linux environments and PowerShell (`.ps1`) for Windows ML workstations.

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

### `03_data_pipeline_full_run.sh`
- **Purpose**: **Primary production orchestrator for the high-capacity ML machine (4 TB storage)**. Runs the entire multi-source download, 10-step preprocessing, grounded augmentation, multi-branch mixing, WebDataset sharding, and pipeline audit.
- **Workflow**:
  1. `data_forge fetch --source all --full-mode`: Downloads complete training sets (including multi-part Azure Blobs for DNS-5 noise & clean speech, Kaggle MAD, Dryad gunshots, SHAReD, NOISEX-92, etc.).
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
  bash scripts/03_data_pipeline_full_run.sh --workers 16 --mixtures 200000
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
- **Purpose**: Executes the complete 108-test unit and integration test suite across all subpackages using `pytest`.
- **Usage**:
  ```bash
  bash scripts/06_run_tests.sh
  ```

---

### `07_train_se_primary.sh` / `.ps1`
- **Purpose**: Trains **Model 1 (`aegis-se-primary`)**, the low-latency streaming speech enhancement engine.
- **Specs**: DeepFilterNet3 architecture, 0 ms lookahead, multi-res spectral + local SNR loss, SpecMix augmentation, EMA shadow weights.
- **Usage**:
  ```bash
  bash scripts/07_train_se_primary.sh [--resume PATH]
  # PowerShell: .\scripts\07_train_se_primary.ps1
  ```

---

### `08_train_se_escalation.sh` / `.ps1`
- **Purpose**: Trains **Model 2 (`aegis-se-escalation`)**, the high-capacity semi-causal speech enhancement model.
- **Specs**: DeepFilterNet3 architecture, 40 ms lookahead (`df_lookahead=2`, `conv_lookahead=2`), emphasis on negative SNR segments, dual-metric worst-class checkpoint selector.
- **Usage**:
  ```bash
  bash scripts/08_train_se_escalation.sh [--resume PATH]
  # PowerShell: .\scripts\08_train_se_escalation.ps1
  ```

---

### `09_train_se_crosscheck.sh` / `.ps1`
- **Purpose**: Trains **Model 3 (`aegis-se-crosscheck`)**, the CleanUMamba state-space U-Net model.
- **Specs**: Native 48kHz training on AEGIS shards, linear $\mathcal{O}(L)$ state-space representation, serves as architectural cross-validation and GPU fallback.
- **Usage**:
  ```bash
  bash scripts/09_train_se_crosscheck.sh [--resume PATH]
  # PowerShell: .\scripts\09_train_se_crosscheck.ps1
  ```

---

### `10_train_classifier.sh` / `.ps1`
- **Purpose**: Trains **Model 4 (`aegis-clf-gate`)**, the acoustic environment and escalation classifier.
- **Specs**: 3-way taxonomy (`harmonic`, `impulsive`, `speech_dominant`), class-weighted cross-entropy, validates escalation boundary against Model 1/2 gap.
- **Usage**:
  ```bash
  bash scripts/10_train_classifier.sh [--resume PATH]
  # PowerShell: .\scripts\10_train_classifier.ps1
  ```

---

### `11_train_aec.sh` / `.ps1`
- **Purpose**: Fine-tunes **Model 5 (`aegis-aec-gate`)**, the acoustic echo cancellation model.
- **Safety**: Requires `--force` argument because production deployment defaults to the proven `deepvqe-ggml` checkpoint.
- **Usage**:
  ```bash
  bash scripts/11_train_aec.sh --force
  # PowerShell: .\scripts\11_train_aec.ps1 -force
  ```

---

### `12_train_all_models.sh` / `.ps1`
- **Purpose**: **Master Training Orchestrator**. Sequentially trains Model 1, Model 2, Model 3, and Model 4, logging checkpoints into `data/checkpoints/`.
- **Usage**:
  ```bash
  bash scripts/12_train_all_models.sh
  # PowerShell: .\scripts\12_train_all_models.ps1
  ```

---

### `13_evaluate_models.sh` / `.ps1`
- **Purpose**: **Multi-Model Audio Evaluation Suite**. Runs comprehensive objective metrics across validation (`val`) or generalization (`gentest`) splits:
  - **Speech Enhancement**: PESQ (Wideband MOS [1.0, 4.5]), STOI ([0.0, 1.0]), SI-SNR (dB), Segmental SNR (dB), DNSMOS P.835 (SIG, BAK, OVRL).
  - **Environment Classifier**: 3-Way Categorical Accuracy, Macro-F1, per-class sensitivity (`harmonic`, `impulsive`, `speech_dominant`).
  - **AEC**: Echo Return Loss Enhancement (ERLE dB).
- **Per-Class Breakdown**: Evaluates fragile classes (`wind`, `rotor_vehicle_drone`, `tank_tracked`, `artillery_howitzer`) individually.
- **Reporting**: Logs a structured Markdown table to stdout and writes JSON reports to `data/eval_reports/`.
- **Usage**:
  ```bash
  # Evaluate Model 1 on validation split (50 samples)
  bash scripts/13_evaluate_models.sh --model aegis-se-primary --num-samples 50 --split val
  # PowerShell: .\scripts\13_evaluate_models.ps1 --model aegis-se-primary --num-samples 50 --split val

---

### `14_run_acceptance_tests.sh` / `.ps1`
- **Purpose**: **Mission-Critical Defence Acceptance Test Suite**. Runs strict verification against all operational specifications:
  - **Threshold Criteria**: Validates `PESQ > 2.5 MOS`, `STOI > 0.85 Intelligibility`, `SNR > 15.0 dB`.
  - **Defence Disturbances**: Validates handling of impulsive gunshots & artillery, periodic drone UAV & helicopter rotors, low-frequency armored tank rumble, sirens, and turbulent wind.
  - **Hybrid AI + ANC Pipeline**: Validates end-to-end integration of deep enhancement with the Normalized LMS adaptive filter.
  - **Edge Hardware Readiness**: Validates ONNX exportability and frame latency profiling.
- **Usage**:
  ```bash
  bash scripts/14_run_acceptance_tests.sh
  # PowerShell: .\scripts\14_run_acceptance_tests.ps1
  ```

---

### `15_export_edge_onnx.sh` / `.ps1`
- **Purpose**: **Edge Model Exporter & Latency Profiler**. Exports models to standard ONNX format with dynamic batch and time dimensions for embedded deployment (e.g. NVIDIA Jetson AGX Orin, edge SoCs, DSPs).
- **Actions**:
  - Exports PyTorch model to `data/onnx_models/<model_key>.onnx`.
  - Benchmarks execution latency across 10ms, 20ms, or 40ms frame chunks and calculates Real-Time Factor (RTF).
- **Usage**:
  ```bash
  # Export Model 1 for real-time edge streaming
  bash scripts/15_export_edge_onnx.sh --model aegis-se-primary --chunk-ms 20.0
  # PowerShell: .\scripts\15_export_edge_onnx.ps1 --model aegis-se-primary --chunk-ms 20.0
  ```



