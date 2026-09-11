# Project AEGIS — Real-Time Edge Inference Suite & Hybrid ANC Engine

> **Location:** `inference/`  
> **Target Hardware:** Edge AI Hardware, Tactical Headsets, NVIDIA Jetson AGX Orin, DGX Edge / Workstations  
> **Streaming Target:** Real-Time Full-Duplex Audio Processing at 48,000 Hz ($<10\text{ ms}$ algorithmic latency)

---

## 1. Executive Summary & Runtime Architecture

> **Implementation note:** The authoritative end-to-end workflow, backend
> selection, state contract, and SIH verification procedure are in
> [docs/ML_PIPELINE_RUNBOOK.md](../docs/ML_PIPELINE_RUNBOOK.md). This file
> describes the inference components; the runbook takes precedence where an
> older example or estimate differs from the current CLI.

The `inference` package is an ultra-low-latency streaming inference engine and active noise cancellation (ANC) suite. It transitions models trained in `training/` into mission-critical, real-time edge deployment pipelines.

The runtime coordinates four synchronized systems:
1. **Acoustic Escalation Router**: Dynamically switches between Model 1 (DeepFilterNet3 Base), Model 2 (Escalation), and Model 3 (CleanUMamba) based on real-time acoustic classification.
2. **Hybrid ANC Engine**: Integrates a dual-buffered Normalized Least Mean Squares (NLMS) adaptive filter with deep neural speech enhancement.
3. **Stateful Hop Processor**: Feeds contiguous 10ms (480-sample) audio chunks to causal recurrent states without windowing phase distortions.
4. **Hardware-Accelerated ONNX Runtime**: Delivers zero-copy GPU inference via TensorRT / CUDA execution providers and INT8 dynamic quantization.

```
+===================================================================================================+
|                                  INFERENCE & STREAMING ARCHITECTURE                                |
+===================================================================================================+
|                                                                                                   |
|  [Microphone Stream: 48,000 Hz, 10ms chunk = 480 samples]                                         |
|                                            │                                                      |
|                                            ▼                                                      |
|  +---------------------------------------------------------------------------------------------+  |
|  | 1. AUDIO RING BUFFER (inference/utils/audio_io.py)                                          |  |
|  |    Zero-allocation circular buffer bridging asynchronous driver threads                     |  |
|  |    Lock-free 'drop_oldest' policy prevents lag buildup if GPU momentarily slows             |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                            │                                                      |
|                                            ▼                                                      |
|  +---------------------------------------------------------------------------------------------+  |
|  | 2. ACOUSTIC ESCALATION ROUTER (inference/runtime/escalation_router.py)                       |  |
|  |    Runs Model 4 Gating Classifier on 200ms audio windows:                                   |  |
|  |    * SNR > 25 dB (Clean Speech)     ──> Bypass Mode (Battery-Saving Passthrough)            |  |
|  |    * Moderate / Stationary Noise   ──> Model 1: DeepFilterNet3 Base (0ms lookahead)         |  |
|  |    * Severe Noise / Sudden Blast   ──> Model 2: DeepFilterNet3 Escalation (10ms delay)      |  |
|  |    * Dense Harmonic Machinery      ──> Primary/escalation path; Model 3 is teacher/fallback   |  |
|  |    * Crossfade Smoother: 20ms Hanning crossfade eliminates handover clicks and pops         |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                            │                                                      |
|                                            ▼                                                      |
|  +---------------------------------------------------------------------------------------------+  |
|  | 3. HYBRID ACTIVE NOISE CANCELLATION (inference/runtime/hybrid_anc.py)                       |  |
|  |    Stage 1: Dual-Buffered Adaptive NLMS Filter (Cancels periodic engine/propeller whine)    |  |
|  |    Stage 2: Deep Complex Filtering (Suppresses non-linear speech-correlated hostile noise)  |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                            │                                                      |
|                                            ▼                                                      |
|  +---------------------------------------------------------------------------------------------+  |
|  | 4. TRANSMISSION PREPARATION (inference/utils/transmission_prep.py)                          |  |
|  |    80 Hz High-Pass Filter -> Dynamic Range Compressor -> -1.0 dBFS Limiter                  |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                            │                                                      |
|                                            ▼                                                      |
|  [Clean Enhanced Output: Headphone Driver / Tactical Military Radio Stream]                       |
+===================================================================================================+
```

---

## 2. Deep-Dive: Subsystem Architecture & Components

### 2.1 Inference Engines & Optimization (`inference/engines/`)
- **ONNX Export Engine (`onnx_engine.py`)**:
  - Exports PyTorch models to ONNX Opset 17/18 with constant folding and operator fusion.
  - **Explicit Stateful IO**: Exposes recurrent hidden states (`h_in`, `h_out`, `context_buffer`) as inputs/outputs of the ONNX graph, preserving causality across streaming frames without graph resets.
  - **Platform B TensorRT Export (`export_platform_b_tensorrt`)**: Decomposes STFT/ISTFT operations into equivalent causal Conv1d operators, bypassing TensorRT dynamic STFT limitations and compiling into FP16/INT8 TensorRT engines.
- **ONNX Runtime Session (`onnx_runtime_engine.py`)**:
  - Manages hardware execution via `TensorrtExecutionProvider`, `CUDAExecutionProvider`, or `CPUExecutionProvider`.
  - Recycles output state tensors into next-chunk input states with zero memory allocation.
  - Profiles compute latency and memory consumption per 10ms frame.
- **Dynamic INT8 Quantization (`quantization.py`)**:
  - Quantizes weights of Linear and recurrent layers to INT8 (`torch.qint8`).
  - Reduces model footprint by $\sim 60\text{--}75\%$ (e.g., DF3 Base compressed to $<6\text{ MB}$).
  - Delivers a $2.0\text{--}2.8\times$ CPU speedup with $<0.05$ PESQ perceptual degradation.
- **Model Adapter (`onnx_model_adapter.py`)**:
  - Provides a uniform callable interface (`enhanced = model(audio)`), allowing scripts to use PyTorch models or ONNX sessions interchangeably.

### 2.2 Streaming Runtime & Routing (`inference/runtime/`)
- **Acoustic Escalation Router (`escalation_router.py`)**:
  - Evaluates instantaneous acoustic conditions using Model 4 (`aegis-clf-gate`).
  - **Dynamic Branch Escalation**: Automatically routes to the optimal model based on detected threat profiles:
    - Normal background $\rightarrow$ Model 1 (zero lookahead).
    - Sudden blast or gunfire onset $\rightarrow$ Model 2 (absorbs transient via 10ms lookahead output-delay buffer).
    - Sustained tank engine / drone harmonics $\rightarrow$ Model 1 by default;
      Model 3 is the trained cross-check/teacher and can be deployed as an
      explicit fallback, but is not silently run as a third hot-path model.
  - **Intelligibility Floor Safeguard**: When speech dominance is detected (`speech_dominant`), an intelligibility floor preserves $\ge 15\%$ ($\text{dry\_mix} \ge 0.15$) of the uncorrupted input signal in both synchronous and pipelined routes, preventing neural over-suppression from truncating quiet consonant tails.
  - **Asymmetric Hysteresis**: Escalation transitions trigger immediately (1 chunk) to shield against hostile blasts, while bypass transitions require **15 consecutive clean chunks** (150ms) to confirm genuine silence, eliminating route flapping.
  - **Memory Safeguard (Lazy Loading & 30s Idle Unload)**: Heavy escalation models (Model 2, Model 3) are lazy-loaded only when triggered and automatically unloaded if idle for $>30.0$ seconds (`escalation_idle_unload_sec=30.0`), enforcing a strict $\le 4\text{--}8\text{ GB}$ VRAM/RAM ceiling.
  - **Click-Free Crossfade Handover**: Transitions between models over a 20ms Hanning window, eliminating phase discontinuity artifacts.
- **Stateful Hop Processor (`audio_stream.py`)**:
  - Solves the state-corruption issue of standard Overlap-Add (OLA) on causal recursive models.
  - Feeds contiguous 10ms chunks directly into recurrent state buffers, preserving time-domain phase alignment.
- **Hybrid ANC Processor (`hybrid_anc.py`)**:
  - Implements a Normalized Least Mean Squares (NLMS) filter running on double-buffered audio rings to prevent thread contention between soundcard I/O and weight adaptation:
    $$\mathbf{w}[n+1] = \mathbf{w}[n] + \frac{\mu}{\|\mathbf{x}[n]\|^2 + \epsilon} e[n] \mathbf{x}[n]$$
  - Delivers physical acoustic noise cancellation combined with neural speech restoration.
- **Multichannel Frontend (`multichannel_frontend.py`)**:
  - Energy-weighted array fusion and SNR-gated throat-mic blending for multi-transducer tactical setups.

### 2.3 Audio Utilities & Buffers (`inference/utils/`)
- **Circular Ring Buffer (`audio_io.py`)**: Thread-safe, lock-free ring buffer with an atomic `drop_oldest` policy that prevents latency drift during compute spikes.
- **Audio File I/O (`audio_io.py`)**: High-performance reading and writing of 48 kHz WAV/FLAC files.
- **Transmission Preprocessing (`transmission_prep.py`)**:
  - 80 Hz high-pass filter to remove structural rumble and vehicle vibration.
  - Soft-knee dynamic range compression maximizing voice intelligibility.
  - $-1.0$ dBFS true peak limiter preventing RF transmitter saturation.

---

## 3. CLI Entry Points & Deployment Tools

Located in `inference/scripts/`:

### 3.1 Live Microphone Enhancement with Hybrid ANC (`live_mic_anc.py`)
Stream live full-duplex audio from local microphone hardware to headphones with active noise cancellation:
```bash
python -m inference.scripts.live_mic_anc \
    --backend onnx \
    --onnx-dir data/onnx_models \
    --onnx-provider CUDAExecutionProvider \
    --onnx-provider CPUExecutionProvider
```

### 3.2 Offline Batch Audio Enhancement (`enhance_audio.py`)
Enhance noisy military audio files using the trained escalation router:
```bash
python -m inference.scripts.enhance_audio \
    --input noisy_cockpit.wav \
    --output clean_cockpit.wav \
    --model router \
    --backend onnx \
    --onnx-dir data/onnx_models
```

### 3.3 Model ONNX & TensorRT Export (`export_onnx.py`)
```bash
# Export DeepFilterNet3 Base for Platform A (CPU/Edge INT8 quantization)
python -m inference.scripts.export_onnx \
    --model se_primary \
    --checkpoint training/checkpoints/aegis-se-primary/best_checkpoint.pt \
    --output data/onnx_models/se_primary.onnx \
    --platform platform_a \
    --quantize

# Export DeepFilterNet3 Base for Platform B (GPU/Laptop TensorRT FP16)
python -m inference.scripts.export_onnx \
    --model se_primary \
    --checkpoint training/checkpoints/aegis-se-primary/best_checkpoint.pt \
    --output data/onnx_models/se_primary_trt.onnx \
    --platform platform_b \
    --fp16

# Export Gating Classifier
python -m inference.scripts.export_onnx \
    --model classifier \
    --checkpoint training/checkpoints/aegis-clf-gate/best_checkpoint.pt \
    --output data/onnx_models/clf_gate.onnx
```
