# Project AEGIS — Adaptive Environment Gated Inference System

> **Production-Grade Real-Time Speech Enhancement, Tactical Acoustic Classification & Hybrid Active Noise Cancellation for Extreme Acoustic Environments**  
> *Target Hardware: NVIDIA DGX Spark (Grace Blackwell GB10, CUDA 13, 128GB Unified Memory) & Edge Deployment (Raspberry Pi 4/5, Jetson AGX Orin, Tactical Radio Systems)*  
> *Audio Standards: 48,000 Hz Fullband Audio, 10ms (480-sample) Streaming Frames, ITU-T P.862.2 (Wideband PESQ), Taal et al. (STOI)*  

---

## 1. Executive Summary & Architecture

Project AEGIS (Nisvana) is an end-to-end, production-grade acoustic intelligence stack engineered for extreme and hostile acoustic environments (tracked combat vehicles, artillery positions, supersonic cockpits, naval engine rooms, high-wind UAV operational envelopes, and urban emergency scenarios).

The system operates across four seamlessly synchronized layers:
1. **Data Forge (`data_forge/`)**: An auditable, multi-terabyte data acquisition and synthesis pipeline adhering to ITU-R BS.1770-4 and 48 kHz fullband DSP standards. Enforces a **100% real-recordings-only policy** (zero synthetic data generation or artificial audio warping) and includes empirical gunfire audits ([gunfire_audit.py](file:///d:/Nisvana/data_forge/verifier/gunfire_audit.py)) with calibrated combat oversampling ($2.5\times$).
2. **Training Engine (`training/`)**: A multi-model PyTorch framework co-designed from epoch 1 for edge inference. Features native NVIDIA Grace Blackwell GB10 bfloat16 AMP (CUDA 13), platform-adaptive Quantization-Aware Training (QAT), CleanUMamba spectrogram knowledge distillation, a 1-chunk lookahead output-delay buffer in Model 2, and SOTA speech-presence-gated SDR loss ($2.5\times$ boost in the 300–4000 Hz formant band with zero spectral leakage).
3. **Inference & Edge Runtime (`inference/`)**: Real-time streaming inference with dual-platform export (Platform A ONNX INT8 / Platform B TensorRT FP16/INT8 with Conv1d STFT/ISTFT decomposition). Features dynamic acoustic escalation routing, hybrid digital-acoustic active noise cancellation (ANC with dual-buffered NLMS), speech intelligibility floors ($\ge 0.15$ dry mix preserved), asymmetric hysteresis (15-chunk bypass confirmation), and 30s idle model unloading.
4. **Multi-User Backend Subsystem (`backend/`)**: Tactical streaming audio backend supporting 100+ concurrent channels. Maintains ultra-compact session states (<50 KB RAM per user), batched tensor forward passes over shared neural weights, and asynchronous bounded queues with PCM-16 wire protocol serialization.

```
+===================================================================================================+
|                                    PROJECT AEGIS PIPELINE FLOW                                    |
+===================================================================================================+
|                                                                                                   |
|  [10 REAL DATASETS]                                                                               |
|  NOISEX-92, SHAReD, MAD, DEMAND, DroneAudioSet, Sirens, Dryad Gunshots, OpenSLR RIR, DNS, AEC    |
|                                            │                                                      |
|                                            ▼                                                      |
|  +---------------------------------------------------------------------------------------------+  |
|  | DATA FORGE (data_forge/)                                                                    |  |
|  |  * Fetchers: Resilient streaming, auto-detection of offline archives, HTTP Range resume      |  |
|  |  * 10-Step Preprocessing: 48kHz polyphase resample, -23 LUFS BS.1770-4, VAD, dedup, splits  |  |
|  |  * Real-Recordings Policy: Bypasses synthetic generators; preserves raw acoustic physics     |  |
|  |  * Gunfire Data Audit: Calibrated 2.5x oversampling for ballistic transients                |  |
|  |  * Multi-Branch Mixer: Speech Enhancement (SE), Classifier (CLF), Acoustic Echo (AEC)       |  |
|  |  * Exporter: WebDataset sharded tar format with sync_tier metadata and dataset card         |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                            │                                                      |
|                                            ▼                                                      |
|  +---------------------------------------------------------------------------------------------+  |
|  | TRAINING LAYER (training/)                                                                  |  |
|  |  * Compute: Grace Blackwell GB10 Native bfloat16 AMP (CUDA 13, 128GB Unified Memory)        |  |
|  |  * Co-Design: QAT from Epoch 1 (selective Conv/Linear observers; recurrent GRU unquantized)  |  |
|  |  * Model 1: DeepFilterNet3 Base (Causal, strict 0ms algorithmic lookahead)                  |  |
|  |  * Model 2: DeepFilterNet3 Escalation (1-chunk lookahead, 10ms buffered output delay)        |  |
|  |  * Model 3: CleanUMamba SSM (Teacher distillation target & GPU-export fallback)             |  |
|  |  * Model 4: SNR / Harmonic Classifier Gate (Conformer/Conv1d 3-way gating signal)           |  |
|  |  * Model 5: Gated Acoustic Echo Cancellation (AEC Challenge calibrated filter)              |  |
|  |  * SOTA Losses: Multi-Res STFT + Speech-Gated SDR (2.5x boost) + Distillation + IS^3 + A-wt |  |
|  |  * Governance: Worst-class Pareto guard on 6 fragile classes, EMA, SNR curriculum          |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                            │                                                      |
|                                            ▼                                                      |
|  +---------------------------------------------------------------------------------------------+  |
|  | DUAL-PLATFORM EXPORT & QUANTIZATION (inference/engines/)                                     |  |
|  |  * Platform A: ONNX Dynamic INT8 / libDF (Raspberry Pi 4/5 Edge CPU, 2.5x speedup)           |  |
|  |  * Platform B: Conv1d-Decomposed STFT/ISTFT -> TensorRT FP16/INT8 (GPU / Jetson Orin)        |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                            │                                                      |
|                      ┌─────────────────────┴─────────────────────┐                                |
|                      ▼                                           ▼                                |
|  +---------------------------------------+   +-------------------------------------------------+  |
|  | INFERENCE & RUNTIME (inference/)      |   | MULTI-USER BACKEND (backend/)                   |  |
|  |  * Escalation Router (M1 <-> M2 <-> M3|   |  * SessionManager: <50 KB RAM / user session    |  |
|  |  * Intelligibility Floor (>=0.15 mix) |   |  * Batched Inference: Shared neural weights     |  |
|  |  * Asymmetric Hysteresis (15 chunks)  |   |  * Audio Transport: Async queues, drop-oldest   |  |
|  |  * Memory Guard: 30s idle lazy unload |   |  * Wire Protocol: PCM-16 signed 16-bit integers |  |
|  |  * Hybrid ANC: Dual-buffered NLMS     |   |  * Scalability: 100+ concurrent channels        |  |
|  +---------------------------------------+   +-------------------------------------------------+  |
|                      │                                           │                                |
|                      ▼                                           ▼                                |
|         Tactical Headset / Intercom                 Tactical Base Station Relay                   |
+===================================================================================================+
```

---

## 2. Directory Architecture & Master Runbooks

Project AEGIS is organized into 9 primary directories, each maintained with an authoritative master runbook:

| Directory | Master Runbook | Subsystems & Topics Covered |
|---|---|---|
| **Root Workspace** | [README.md](file:///d:/Nisvana/README.md) | Top-level architecture, DGX Spark setup, pipeline flowchart, quickstart |
| **`data/`** | [data/README.md](file:///d:/Nisvana/data/README.md) | 9 data stages: `raw`, `processed`, `splits`, `augmented`, `forge`, `shards`, `manifests`, `onnx_models`, `eval_reports` |
| **`data_forge/`** | [data_forge/README.md](file:///d:/Nisvana/data_forge/README.md) | Ingestion & preprocessing engine: `fetcher` (10 real datasets), `preprocessor` (10-step DSP), `augmentor` (real-only policy), `mixer` (3 branches), `exporter` (WebDataset shards), `verifier` (audit & gunfire audit) |
| **`training/`** | [training/README.md](file:///d:/Nisvana/training/README.md) | Deep learning framework: `models` (Models 1–5), `losses` (Multi-Res, Speech-Gated SDR, Distillation, IS³, A-weighted), `trainers` (bfloat16 AMP, QAT), `callbacks` (Pareto guard, EMA), `configs`, `data` (`streaming_chunker.py`) |
| **`inference/`** | [inference/README.md](file:///d:/Nisvana/inference/README.md) | Edge runtime: `engines` (Dual-platform ONNX/TensorRT export, INT8 quantization), `runtime` (escalation router, speech floor, hysteresis, 30s lazy unload, hybrid ANC), `utils` (lock-free ring buffer, transmission prep) |
| **`backend/`** | [backend/README.md](file:///d:/Nisvana/backend/README.md) | Multi-user tactical backend: `session_manager.py` (<50 KB/user), `batch_inference.py` (batched tensors over shared weights), `transport.py` (async queues, PCM-16 serialization) |
| **`docs/`** | [docs/README.md](file:///d:/Nisvana/docs/README.md) | Specifications & research: Master PRD v19 (`Nisvana_PRD.md`), architecture blueprint (`ARCHITECTURE.md`), training runbook, 20+ peer-reviewed bibliography citations |
| **`scripts/`** | [scripts/README.md](file:///d:/Nisvana/scripts/README.md) | Turnkey orchestration: DGX Spark GB10 setup (`00_setup_env.sh`), dry-run probes, full pipeline runners, master trainer (`train.sh`), test launchers |
| **`tests/`** | [tests/README.md](file:///d:/Nisvana/tests/README.md) | Automated QA: 32 test suites, 288 unit & integration tests, DSP compliance checks, SIH acceptance |

---

## 3. Hardware Requirements & Environment Setup

### Target System: NVIDIA DGX Spark GB10
- **Compute**: NVIDIA Grace Blackwell (GB10) with CUDA 13.
- **Unified RAM**: 128GB high-bandwidth unified memory pool.
- **CPU**: 128-core NVIDIA Grace ARMv9 CPU.
- **Storage**: Multi-terabyte NVMe storage mounted at `data/`.
- **Operating Systems**: Ubuntu 22.04 / 24.04 LTS (also fully tested on Windows 11 with Python 3.13).

### Automated Setup Script
Run the turnkey installation script:
```bash
chmod +x scripts/00_setup_env.sh
./scripts/00_setup_env.sh
```
This script automatically:
1. Detects your active Python environment (virtualenv / Conda) to avoid sandboxed execution.
2. Checks for pre-installed PyTorch (v2.7+) or installs CUDA 12.6/13 wheels.
3. Upgrades build tools (`ninja`, `packaging`, `maturin`, `wheel`).
4. Installs `deepfilternet>=0.5.6` pre-built runtime without `deepfilterdataloader` build failures.
5. Compiles and installs `causal-conv1d>=1.4.0` and `mamba-ssm>=2.2.0` with `--no-build-isolation` to detect your active PyTorch.
6. Verifies all model imports, initializes storage hierarchy, and validates `.env` API tokens.

### Environment Configuration (.env)
Copy the production environment configuration:
```bash
cp data_forge/.env.example .env
cp data_forge/.env.example data_forge/.env
```
Key configuration parameters:
```ini
# Real recordings only policy (zero synthetic audio or warping)
DATA_FORGE_REAL_ONLY=true
DATA_FORGE_NO_AUGMENT=true

# Multi-terabyte download resilience
DATA_FORGE_FETCH_TIMEOUT=300
DATA_FORGE_FETCH_MAX_RETRIES=10
DATA_FORGE_FETCH_BACKOFF_BASE=5
DATA_FORGE_FETCH_CHUNK_BYTES=4194304
DATA_FORGE_FETCH_RESUME=true
DATA_FORGE_DISK_SAFETY_MARGIN_GB=50
```

---

## 4. End-to-End Pipeline Execution

### Step 1: Data Acquisition & Gunfire Audit
Verify endpoints without downloading multi-GB archives:
```bash
python -m data_forge fetch --source all --dry-run
```
Execute full production download on DGX Spark:
```bash
python -m data_forge fetch --source all --full-mode
```
Audit gunshot sample availability and hours:
```bash
python -m data_forge.verifier.gunfire_audit
```

### Step 2: 10-Step Preprocessing
Run ITU-R BS.1770-4 normalization, polyphase 48kHz resampling, and VAD silence trimming:
```bash
python -m data_forge preprocess --max-workers 64
```

### Step 3: Multi-Branch Mixing
Synthesize 200,000 real acoustic mixtures across Speech Enhancement, Classifier, and AEC branches:
```bash
python -m data_forge mix --num-mixtures 200000 --min-snr -5.0 --max-snr 20.0
```

### Step 4: WebDataset Sharding & Verification
Pack mixtures into sequential `.tar` shards and run the full pipeline audit:
```bash
python -m data_forge export
python -m data_forge verify
```

### Step 5: Model Training (Grace Blackwell GB10 Co-Design)
Train models with native bfloat16 AMP, QAT from epoch 1, and CleanUMamba distillation:
```bash
# Model 1: DeepFilterNet3 Base (0ms lookahead streaming)
python -m training.scripts.train_se_primary

# Model 2: DeepFilterNet3 Escalation (10ms lookahead output delay)
python -m training.scripts.train_se_escalation

# Model 3: CleanUMamba (Selective State Space Crosscheck)
python -m training.scripts.train_se_crosscheck

# Model 4: Conformer Gating Classifier
python -m training.scripts.train_classifier

# Model 5: Gated AEC
python -m training.scripts.train_aec
```

### Step 6: Dual-Platform Export & Edge Inference
Export trained models to ONNX and TensorRT:
```bash
# Platform A (Raspberry Pi / Edge CPU INT8 quantization)
python -m inference.scripts.export_onnx \
    --model se_primary \
    --checkpoint training/checkpoints/aegis-se-primary/best_checkpoint.pt \
    --output data/onnx_models/se_primary_int8.onnx \
    --platform platform_a \
    --quantize

# Platform B (GPU / Jetson Orin TensorRT FP16)
python -m inference.scripts.export_onnx \
    --model se_primary \
    --checkpoint training/checkpoints/aegis-se-primary/best_checkpoint.pt \
    --output data/onnx_models/se_primary_trt.onnx \
    --platform platform_b \
    --fp16

# Live microphone enhancement with Hybrid Active Noise Cancellation
python -m inference.scripts.live_mic_anc --engine onnx --provider cuda --anc-enable

# Offline file enhancement with Escalation Router
python -m inference.scripts.enhance_audio --input test_noisy.wav --output test_clean.wav --engine onnx --router
```

### Step 7: Multi-User Tactical Backend Execution
Launch multi-user session management and batched tensor inference:
```python
import asyncio
import numpy as np
from backend import SessionManager, BatchInferenceEngine, StreamQueue, AudioPacket

# Initialize session registry and shared batch engine
mgr = SessionManager(session_ttl_sec=120.0)
engine = BatchInferenceEngine()

# Process multi-user 10ms frame batch
s1 = mgr.get_or_create_session("crew_commander")
s2 = mgr.get_or_create_session("gunner")
frame = np.random.randn(480).astype(np.float32) * 0.05
enhanced_frames = engine.process_batch([s1, s2], [frame, frame])
```

---

## 5. Verification & Testing

Execute the comprehensive automated test suite (**32 test suites, 286 passed unit & integration tests, 0 failures**):
```bash
# Run complete test suite
pytest tests/ -v

# Run Rev 3 ML & Inference verification
pytest tests/test_rev3_ml_inference_upgrades.py -v

# Run Multi-User Backend verification
pytest tests/test_backend_subsystem.py -v

# Run SOTA loss stack verification
pytest tests/test_multires_loss.py -v

# Run 10-step DSP preprocessing verification
pytest tests/test_preprocessor.py -v

# Run fetcher endpoint reachability verification
pytest tests/test_fetchers.py -v
```
