# Project AEGIS — System Architecture Blueprint

> **System:** Project AEGIS (Nisvana)  
> **Mission:** Fullband Real-Time Speech Enhancement, Tactical Classification, and Hybrid Active Noise Cancellation for Hostile Military Environments  
> **Target Hardware:**  
> - **Training:** NVIDIA DGX Spark GB10 (Grace Blackwell, CUDA 13, 128GB Unified Memory)  
> - **Edge Inference:** Raspberry Pi 4/5 (Platform A: CPU INT8 / libDF) & GPU Laptop / Jetson AGX Orin (Platform B: TensorRT FP16/INT8)  
> - **Tactical Node:** Multi-User Tactical Intercom & Base Station Relay (100+ concurrent channels, <50 KB state/user)  

---

## 1. Executive Summary & Core Architectural Invariants

Project AEGIS is an end-to-end, mission-critical audio intelligence stack designed to maintain uninterrupted voice intelligibility across extreme acoustic environments (tracked combat vehicles, artillery firing positions, supersonic cockpits, naval engine bays, and high-velocity UAV envelopes).

The architecture is governed by four strict, non-negotiable invariants:
1. **100% Real-Recordings-Only Policy**: Zero synthetic, generative, or procedurally warped audio. All training data is fetched from authenticated research repositories (NATO RSG.10, Harvard Dataverse, Dryad, Edinburgh DataShare, Nature Scientific Data, Microsoft Research).
2. **Zero Pitch-Shift Invariant**: Mechanical vehicles and clean speech targets are never pitch-shifted, preventing corruption of human vocal tract formants and preserving physically valid acoustic signatures.
3. **Sub-10ms Streaming Frame Granularity**: All edge streaming pipelines operate in contiguous 480-sample (10ms @ 48 kHz fullband) chunks with stateful context carryover, preventing boundary discontinuity clicks.
4. **Zero Speech Truncation Guarantee**: The runtime escalation router and multi-user batch inference engine enforce an **intelligibility floor** ($\ge 0.15$ dry mix preserved during speech dominance) combined with asymmetric hysteresis (15-chunk confirmation for bypass), eliminating neural syllable dropouts.

---

## 2. End-to-End System Flow

```
+===================================================================================================+
|                                    PROJECT AEGIS SYSTEM TOPOLOGY                                  |
+===================================================================================================+
|                                                                                                   |
|  [10 Real Datasets] ──► DATA FORGE (data_forge/)                                                  |
|                           * 10-Step DSP: 48kHz Polyphase, -23 LUFS BS.1770-4, VAD, Mono           |
|                           * Gunfire Audit: Cooper & Shaw (Dryad) + MAD Gunshot (2.5x oversampling)|
|                           * 3-Branch Mixer: Speech Enhancement, 3-Way Classifier, AEC Quadruplets |
|                           * WebDataset Sharding: Sequential .tar archives with sync_tier metadata |
|                                     │                                                             |
|                                     ▼                                                             |
|                         TRAINING ENGINE (training/)                                               |
|                           * Hardware: Grace Blackwell GB10 Native bfloat16 AMP (CUDA 13)         |
|                           * Co-Design: QAT from Epoch 1 (Conv/Linear only; GRU unquantized)       |
|                           * Loss Stack: Multi-Res STFT + Speech-Gated SDR (2.5x boost) + Distill  |
|                           * Teacher Distillation: CleanUMamba (SSM) -> DeepFilterNet3 Student     |
|                           * Lookahead Buffering: 1-chunk output delay (10ms future context)       |
|                           * Pareto Guard: Watches 6 fragile classes (PESQ > 0.05, SNR > 0.5dB)    |
|                                     │                                                             |
|                                     ▼                                                             |
|                         DUAL-PLATFORM EXPORT & QUANTIZATION                                       |
|                           * Platform A: ONNX Dynamic INT8 / libDF (Raspberry Pi 4/5)              |
|                           * Platform B: Conv1d STFT/ISTFT Decomposition -> TensorRT FP16/INT8     |
|                                     │                                                             |
|                   ┌─────────────────┴──────────────────┐                                          |
|                   ▼                                    ▼                                          |
|         EDGE RUNTIME (inference/)           MULTI-USER BACKEND (backend/)                         |
|         * Escalation Router (M1/M2/M3)       * SessionManager (<50 KB RAM/session)                |
|         * Intelligibility Floor (>=0.15)     * BatchInferenceEngine (shared neural weights)       |
|         * Asymmetric Hysteresis (15 chunks)  * Async StreamQueue (bounded drop-oldest)            |
|         * Lazy Unload (30s idle reap)        * PCM-16 Wire Protocol Serialization                 |
|         * Hybrid ANC (Dual-buffered NLMS)    * 100+ Concurrent Audio Streams                      |
|                   │                                    │                                          |
|                   ▼                                    ▼                                          |
|         Tactical Headset / Intercom          Base Station Relay / Radio Network                   |
+===================================================================================================+
```

---

## 3. Data Forge Subsystem (`data_forge/`)

### 3.1 Five-Tier Storage Hierarchy
To eliminate the inode bottleneck of millions of loose audio files while maintaining complete traceability:
- **Tier 0 (`data/raw/`)**: Unaltered downloads from 10 open research datasets with automatic fallback detection for manual archives.
- **Tier 1 (`data/processed/`)**: 10-step standardized pool: 48 kHz polyphase resampled, $-23.0 \pm 0.5$ LUFS normalized (ITU-R BS.1770-4), $-1.0$ dBFS peak limited, mono downmixed.
- **Tier 2 (`data/augmented/`)**: Bounded WSOLA time-stretch $[0.90, 1.10]$, gain jitter $[-3, +3]$ dB, and blast onset preserving windowing.
- **Tier 3 (`data/splits/`)**: Leak-free split manifests (`train`, `val`, `test_generalization`) strictly isolating speakers and recording sessions.
- **Tier 4 (`data/forge/`)**: Multi-branch mixtures (Speech Enhancement triplets, Classifier 0.2s windows, AEC quadruplets).
- **Tier 5 (`data/shards/`)**: WebDataset sharded `.tar` archives (2,048 samples/shard) with JSON metadata sidecars.

### 3.2 Real Acoustic Datasets & Gunfire Audit
- **NOISEX-92**: NATO RSG.10 combat vehicles (Leopard 1 tank, M109 howitzer, F-16 cockpit, naval destroyer).
- **SHAReD**: Harvard Dataverse 326 high-explosive detonation waveforms (C-4, TNT, ANFO).
- **Cooper & Shaw (Dryad)**: Multi-microphone ballistic gunshot shockwaves and muzzle blasts.
- **Military Audio Dataset (MAD)**: Nature Scientific Data combat machines and firearm recordings.
- **DroneAudioSet**: Multi-rotor UAV ego-noise and flybys.
- **VCTK + DEMAND**: Fullband clean speech targets and diverse environmental noise backgrounds.
- **DNS-5 & AEC-Challenge**: Microsoft fullband clean speech, noise, RIRs, and echo quadruplets.
- **Gunfire Audit (`gunfire_audit.py`)**: Quantifies exact hours and clips on disk, establishing calibrated $2.5\times$ oversampling for gunshot classes.

---

## 4. Machine Learning Training Architecture (`training/`)

### 4.1 Heterogeneous Model Ensemble
1. **Model 1 (`aegis-se-primary`) — DeepFilterNet3 Base**:
   - Causal 32-band ERB filterbank + Order-5 deep complex filtering.
   - Strict 0ms algorithmic lookahead, $<10\text{ ms}$ processing time.
2. **Model 2 (`aegis-se-escalation`) — DeepFilterNet3 Escalation**:
   - 1-chunk lookahead output-delay buffer (`_output_delay_buffer` introducing 10ms / 480-sample future context).
   - Absorbs severe acoustic transients and negative SNR conditions ($\text{SNR} < 0\text{ dB}$).
3. **Model 3 (`aegis-se-crosscheck`) — CleanUMamba**:
   - Selective State Space Model (SSM) based on Mamba blocks.
   - Linear-time causal sequence modeling, serving as a time-domain teacher and GPU-export fallback.
4. **Model 4 (`aegis-clf-gate`) — 3-Way Acoustic Classifier**:
   - Processes 200ms audio windows into 3 logits: `stationary_harmonic`, `non_stationary_transient`, `speech_dominant`.
5. **Model 5 (`aegis-aec-gate`) — Gated Acoustic Echo Cancellation**:
   - Dual-branch complex STFT post-filter suppressing residual acoustic echo (ERLE $>30\text{ dB}$).

### 4.2 SOTA Loss Stack & Co-Design
- **Multi-Resolution Spectral Loss**: 4 STFT window lengths ($N_{\text{FFT}} \in \{256, 512, 1024, 2048\}$) with compressed magnitude ($\gamma = 0.3$).
- **Speech-Presence-Gated SDR Loss**: Signal-to-Distortion Ratio loss with a $2.5\times$ boost on active speech frames. Gated via a 1024-point Hann-windowed formant filter in the 300–4000 Hz band, eliminating rectangular spectral sidelobes and gradient conflicts.
- **CleanUMamba Knowledge Distillation**: Distills temporal context from frozen CleanUMamba teacher to student models ($\lambda_{\text{distill}} = 0.3$).
- **Impulse-Weighted $\text{IS}^3$ Loss**: Applies a $3.0\times$ onset penalty when frame energy ratio exceeds $2.0$.
- **Perceptual A-Weighted Loss**: Emphasizes the 1–6 kHz speech intelligibility band.

### 4.3 Training Governance & Grace Blackwell Co-Design
- **Grace Blackwell GB10 Native bfloat16 AMP**: Configured via `precision="bf16"`, running native `torch.autocast('cuda', dtype=torch.bfloat16)` across 128GB unified RAM without loss scaling.
- **Platform-Adaptive QAT**: Prepares quantization-aware observers on `Conv1d`, `Conv2d`, and `Linear` layers from epoch 1, leaving recurrent modules unquantized.
- **Worst-Class Pareto Checkpoint Guard**: Tracks individual high-water marks across 6 fragile defence classes, rejecting regressions $>0.05$ PESQ or $>0.5\text{ dB}$ SNR.

---

## 5. Real-Time Edge Inference Suite (`inference/`)

### 5.1 Dual Deployment Path
- **Platform A (Raspberry Pi / Edge CPU)**:
  - DeepFilterNet3 native Rust engine (`libDF`) or `deepfilter-stream` via ONNX Runtime CPU.
  - INT8 dynamic quantization compresses models by $\sim 70\%$ ($<6\text{ MB}$ footprint, $2.5\times$ CPU speedup).
- **Platform B (GPU Laptop / Jetson / Tactical Node)**:
  - Conv1d-decomposed STFT/ISTFT export (`export_platform_b_tensorrt`) bypasses dynamic FFT barriers.
  - Compiles to TensorRT FP16/INT8 with persistent contexts and CUDA Graphs.

### 5.2 Dynamic Routing & Runtime Safeguards
- **Escalation Router (`escalation_router.py`)**: Evaluates 200ms audio windows and selects the optimal path:
  - High SNR / Silence $\rightarrow$ Battery-saving bypass.
  - Moderate Noise $\rightarrow$ Model 1 (0ms lookahead).
  - Shockwaves / Gunfire $\rightarrow$ Model 2 (10ms lookahead output delay).
  - Dense Harmonics $\rightarrow$ Model 3 (CleanUMamba SSM).
- **Speech Intelligibility Floor**: Guarantees $\ge 0.15$ dry audio mix on speech-dominant frames across synchronous and pipelined routes.
- **Asymmetric Hysteresis**: 1-chunk immediate escalation vs. 15-chunk confirmation for bypass.
- **Lazy Loading & 30s Idle Unloading**: Unloads heavy escalation models after 30s of inactivity, keeping idle memory $\le 4\text{--}8\text{ GB}$.
- **Hybrid Active Noise Cancellation (`hybrid_anc.py`)**: Dual-buffered Normalized Least Mean Squares (NLMS) filter cancels acoustic whine before neural enhancement.

---

## 6. Multi-User Backend Subsystem (`backend/`)

### 6.1 Tactical Audio Infrastructure
Designed for multi-channel vehicle intercoms, squad radios, and base station relays:
- **`SessionManager`**:
  - Memory-efficient per-user state containment (<50 KB RAM per session).
  - Holds recurrent hidden states, lookahead buffers, and speech moving averages.
  - Automatic 120s TTL idle cleanup.
- **`BatchInferenceEngine`**:
  - Stacks active user frames into unified tensors $(B, 1, 480)$ for a single GPU forward pass over shared neural weights.
  - Demultiplexes outputs and preserves per-user causal state continuity.
  - Enforces session-level intelligibility floors ($\text{dry\_mix} \ge 0.15$).
- **`Async Audio Transport` (`transport.py`)**:
  - Non-blocking `StreamQueue` with drop-oldest backpressure control.
  - Zero-copy `AudioPacket` serialization for PCM-16 linear wire protocols.

---

## 7. Hardware Specifications & Bill of Materials

| Hardware Tier | Platform Specification | Primary Role | Memory / Latency Budget |
| :--- | :--- | :--- | :--- |
| **Training Workstation** | NVIDIA DGX Spark GB10 (Grace Blackwell) | Model training, QAT, and hyperparameter sweeps | 128GB Unified Memory, CUDA 13 |
| **Tactical Server / Relay** | NVIDIA Jetson AGX Orin / GPU Workstation | Multi-user backend (100+ channels) | $\le 8\text{ GB}$ VRAM, $<10\text{ ms}$ batch latency |
| **Edge Soldier Node** | Raspberry Pi 4 / 5 (ARMv8 / ARMv9) | Single-user wearable DSP & headset ANC | $<2\text{ GB}$ RAM, $<30\text{ ms}$ end-to-end |
| **Tactical Headset Transducers** | Passive earmuffs + 40mm drivers + 3 mics | Primary air mic, reference mic, throat piezo mic | Approx. ₹3,850–4,650 build total |

---

## 8. Verification & QA Matrix

All subsystems are validated across 32 automated test suites (**286 passed unit & integration tests**, 0 failures):
- **Data Layer QA**: `test_preprocessor.py`, `test_fetchers.py`, `test_sih_compliance_real_data.py`.
- **Training Engine QA**: `test_multires_loss.py`, `test_training_loop.py`, `test_training_sync.py`.
- **Edge Inference QA**: `test_inference_suite.py`, `test_rev3_ml_inference_upgrades.py`, `test_latency_regression.py`.
- **Multi-User Backend QA**: `test_backend_subsystem.py` (session footprint, batched tensors, async transport).
