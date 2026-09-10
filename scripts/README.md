# Project AEGIS — Linux Orchestration Scripts (Rev 3)

This directory contains POSIX-compliant, hardened Bash scripts (`set -euo pipefail`) designed to orchestrate the complete Project AEGIS stack on Linux (NVIDIA DGX Spark GB10 / Ubuntu 22.04 / 24.04 LTS).

---

## 1. Script Inventory

```
scripts/
├── 00_setup_environment.sh         # [Step 0] Full hardware, Rust, CUDA 13, and PyTorch environment setup
├── 00_setup_env.sh                 # [Step 0] Convenience alias for 00_setup_environment.sh
├── common_env.sh                   # [Helper] Shared Python/PyTorch non-sandboxed environment resolver
├── 01_data_pipeline_dry_run.sh     # [Step 1] Zero-disk-write endpoint & Azure Blob reachability probe
├── 02_data_pipeline_sample_test.sh # [Step 2] End-to-end integration test with auto-clean blank slate
├── 03_data_pipeline_full_run.sh    # [Step 3] Full production multi-worker 4TB pipeline runner
├── 04_export_webdataset_shards.sh  # [Step 4] Standalone WebDataset tar packer & dataset card generator
├── 05_clean_data_pipeline.sh       # [Utility] Complete data/ purge and blank slate reset
├── 06_run_tests.sh                 # [Testing] Automated 286-test suite verification across all 32 suites
│
├── 07_train.sh                     # [MASTER TRAIN] Unified Master ML Training Pipeline Runner (Rev 3)
├── train.sh                        # [MASTER TRAIN] Convenience alias for 07_train.sh
├── 07_train_se_primary.sh          # [Train Alias] Model 1: DeepFilterNet3 Base (0ms lookahead, QAT, Distillation)
├── 08_train_se_escalation.sh       # [Train Alias] Model 2: DeepFilterNet3 Escalation (10ms lookahead delay buffer)
├── 09_train_se_crosscheck.sh       # [Train Alias] Model 3: CleanUMamba SSM (Distillation Teacher)
├── 10_train_classifier.sh          # [Train Alias] Model 4: Acoustic Gating Classifier (0.2s windows)
├── 11_train_aec.sh                 # [Train Alias] Model 5: Gated Acoustic Echo Cancellation (--force)
├── 12_train_all_models.sh          # [Train Alias] Full sequential training in scientific dependency order
│
├── 13_evaluate_models.sh           # [Eval] Multi-Model Audio Evaluation Suite (PESQ, STOI, SI-SNR, SSNR)
├── 14_run_acceptance_tests.sh      # [Acceptance] Mission-Critical Defence Acceptance Test Suite
├── 15_export_edge_onnx.sh          # [Edge] Dual-Platform ONNX / TensorRT Exporter (Platform A / Platform B)
├── 16_enhance_audio.sh             # [Inference] Offline Audio File Enhancement Runner with Escalation Router
├── 17_live_stream_prototype.sh     # [Inference] Real-Time Live Microphone & Hybrid ANC Prototype
└── 18_verify_sih_compliance.sh     # [Audit] Official SIH Defence Benchmark & Compliance Runner
```

---

## 2. Master Training Script Reference (`07_train.sh` / `train.sh`)

The master training script connects all model training layers into a single, cohesive, production-grade CLI.

### Features
- **Grace Blackwell GB10 Native bfloat16 AMP**: Runs native `torch.autocast('cuda', dtype=torch.bfloat16)` across 128GB unified RAM.
- **Quantization-Aware Training (QAT)**: Injects fake-quantization observers exclusively into `Conv1d`, `Conv2d`, and `Linear` layers from epoch 1, leaving recurrent modules unquantized.
- **CleanUMamba Spectrogram Distillation**: Automatically distills long-horizon state representations from frozen Model 3 teacher into DeepFilterNet3 students ($\lambda_{\text{distill}} = 0.3$).
- **Model 2 Lookahead Output Delay**: Configures streaming 1-chunk (10ms / 480-sample) future context buffer.
- **Worst-Class Pareto Checkpoint Guard**: Rejects checkpoints if any of the 6 fragile defence classes regress.
- **SNR Curriculum**: Gradually scales difficulty from $+15\text{ dB}$ down to $-5\text{ dB}$.

### Usage Examples
```bash
# Train complete 5-model ensemble in scientific dependency order:
./scripts/07_train.sh --model all

# Train Model 1 with native Blackwell bfloat16 AMP and QAT from epoch 1:
./scripts/07_train.sh --model se_primary --precision bf16 --qat

# Train Model 2 with 10ms lookahead output delay:
./scripts/07_train.sh --model se_escalation --epochs 80

# Train Model 3 (CleanUMamba teacher):
./scripts/07_train.sh --model se_crosscheck --epochs 100

# Quick verification dry run (tests initialization of all 5 architectures):
./scripts/07_train.sh --dry-run
```

---

## 3. Data Pipeline & Verification Scripts

### `00_setup_environment.sh` (or `00_setup_env.sh`)
- Detects the active Python environment (virtualenv / Conda / system) using `common_env.sh` to prevent accidental sandboxed system Python execution.
- Checks if PyTorch (v2.7+) is already active; installs CUDA 12.6/13 PyTorch wheels only if absent.
- Upgrades build tools (`ninja`, `packaging`, `maturin`, `wheel`) for parallel CUDA extension builds.
- Installs the stable Rust toolchain (`rustup`) for `deepfilternet[train]`.
- Compiles and installs `causal-conv1d>=1.4.0` and `mamba-ssm>=2.2.0` with `--no-build-isolation`, directly linking against your active PyTorch without pip sandbox isolation.
- Loads `.env` and `data_forge/.env` and validates external API tokens (including `KAGGLE_ACCESS_TOKEN`).

### `01_data_pipeline_dry_run.sh`
- Probes all 10 authoritative datasets via HTTP HEAD/GET range requests without writing multi-GB files.
- Generates pipeline audit report confirming zero invariant violations.

### `02_data_pipeline_sample_test.sh`
- Executes a fast end-to-end integration test on sample data.
- Purges sample artifacts upon completion to leave a 100% blank slate for production training.

### `03_data_pipeline_full_run.sh`
- Primary production orchestrator for downloading, 10-step DSP preprocessing, multi-branch mixing, and WebDataset sharding across 4TB storage.

### `05_clean_data_pipeline.sh`
- Safely resets `data/` subdirectories (`raw`, `processed`, `augmented`, `splits`, `forge`, `shards`, `manifests`) while restoring the directory skeleton.

### `06_run_tests.sh`
- Runs the comprehensive 286-test automated pytest suite across all 32 test suites.

---

## 4. Edge Deployment & Inference Scripts

### `15_export_edge_onnx.sh`
Exports PyTorch checkpoints to optimized edge runtimes:
```bash
# Platform A (Raspberry Pi / CPU INT8 quantization)
./scripts/15_export_edge_onnx.sh --model se_primary --platform platform_a --quantize

# Platform B (GPU Laptop / Jetson AGX Orin TensorRT FP16)
./scripts/15_export_edge_onnx.sh --model se_primary --platform platform_b --fp16
```

### `16_enhance_audio.sh`
Offline audio enhancement with dynamic escalation routing:
```bash
./scripts/16_enhance_audio.sh --input noisy_cockpit.wav --output clean_cockpit.wav --router
```

### `17_live_stream_prototype.sh`
Real-time microphone streaming with Hybrid Active Noise Cancellation (dual-buffered NLMS filter):
```bash
./scripts/17_live_stream_prototype.sh --engine onnx --provider cuda --anc-enable
```

### `18_verify_sih_compliance.sh`
Runs the official SIH defence benchmark suite verifying SNR $>15\text{ dB}$, STOI $>0.85$, and PESQ $>2.5$.
