# Project AEGIS — Real-Time Edge Inference Suite & Hybrid ANC Engine

Project AEGIS (`inference/`) is a low-latency, mission-critical audio inference engine designed for deployment on embedded defense hardware, tactical headsets, and edge AI systems (e.g. **NVIDIA Jetson AGX Orin 64GB Developer Kit**, embedded ARM/x86 platforms, and DSPs).

---

## 1. Architecture & SOTA Edge Features

```
               [Incoming Tactical Audio Stream: 48,000 Hz]
                                  │
                                  ▼
                    [AudioRingBuffer (Zero-Allocation)]
                                  │
                                  ▼
           [AcousticEscalationRouter (Model 4 Classifier)]
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │ (SNR > 25 dB, Clean)    │ (Default Real-Time)     │ (Impulsive / SNR < 0 dB)
        ▼                         ▼                         ▼
   [Bypass Mode]        [Model 1: aegis-se-primary]  [Model 2: aegis-se-escalation]
 (Battery Saving)            (0ms Lookahead)                (40ms Lookahead)
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  │  (Optional Reference Mic)
                                  ▼        │
             [Stage 2: Normalized LMS Filter (NLMS)]
                                  │
                                  ▼
               [StreamingAudioProcessor (50% Overlap-Add)]
                                  │
                                  ▼
                 [Reconstructed Clean Audio: 48kHz]
```

### 1.1 Zero-Allocation Circular Ring Buffer (`inference/runtime/audio_stream.py`)
- **Memory Integrity**: Eliminates dynamic array allocations inside the inner audio callback loop to prevent garbage collection pauses and frame drops.
- **Variable Chunk Handling**: Ingests variable soundcard chunks (e.g., 256, 480, 512, 1024 samples) and buffers them into fixed analysis windows.

### 1.2 Overlap-Add Streaming Processor (`StreamingAudioProcessor`)
- **Acoustic Continuity**: Employs a 50% Overlap-Add (OLA) reconstruction with Hanning synthesis windowing.
- **Discontinuity Prevention**: Ensures frame-by-frame deep filtering processes continuous audio without boundary clicks, pops, or phase jumps.

### 1.3 Hybrid AI + Adaptive Filter (ANC) (`inference/runtime/hybrid_anc.py`)
- **Stage 1 (Deep AI)**: Neural speech enhancement suppresses non-linear, dynamic, and impulsive defence noise (gunfire, artillery, rotor blades).
- **Stage 2 (Adaptive NLMS)**: Normalized Least Mean Squares filter cancels residual stationary leakage and microphone feedthrough:
  $$e(n) = \hat{s}(n) - \mathbf{w}^T(n) \mathbf{x}(n), \quad \mathbf{w}(n+1) = \mathbf{w}(n) + \frac{\mu}{\|\mathbf{x}(n)\|^2 + \epsilon} e(n) \mathbf{x}(n)$$

### 1.4 Dynamic Acoustic Escalation Router (`inference/runtime/escalation_router.py`)
- **Acoustic Environment Monitoring**: Runs Model 4 (`aegis-clf-gate`) to classify audio into `harmonic`, `impulsive`, or `speech_dominant`.
- **Intelligent Switching**:
  - **Clean Speech Bypass** ($\text{SNR} > 25\text{ dB}$): Bypasses neural processing to conserve edge battery.
  - **Primary Streaming (Model 1)**: Runs 0ms lookahead DeepFilterNet3 for real-time edge streaming.
  - **Severe Escalation (Model 2)**: Automatically escalates to 40ms lookahead DeepFilterNet3 when estimated $\text{SNR} < 0\text{ dB}$ or impulsive blasts are detected.
- **Crossfade Blending**: Uses a 5ms crossfade window when switching branches to prevent acoustic clicks.

### 1.5 ONNX Runtime & Dynamic INT8 Quantization (`inference/engines/`)
- **`OnnxRuntimeSession`**: High-performance ONNX Runtime session manager with multi-threading and TensorRT / CUDA execution provider fallback.
- **`quantize_model_dynamic`**: Post-training dynamic quantization (`torch.qint8`) reducing memory footprint by ~75% and accelerating edge CPU inference.

---

## 2. CLI Execution & Operational Runbook

### 2.1 Audio File Enhancer (`enhance_audio.py`)

Enhances noisy WAV/FLAC audio files with optional Hybrid ANC and streaming reconstruction:

```bash
# Enhance file with Model 1 (streaming mode)
python -m inference.scripts.enhance_audio -i noisy.wav -o clean.wav --model aegis-se-primary

# Enhance file with Hybrid AI + NLMS Adaptive Filter
python -m inference.scripts.enhance_audio -i noisy.wav -o clean.wav --model aegis-se-primary --use-hybrid-anc

# Enhance using Dynamic Acoustic Escalation Router
python -m inference.scripts.enhance_audio -i noisy.wav -o clean.wav --model router

# Or using the automation script:
bash scripts/16_enhance_audio.sh -i noisy.wav -o clean.wav
# PowerShell: .\scripts\16_enhance_audio.ps1 -i noisy.wav -o clean.wav
```

### 2.2 Live Microphone & Headset ANC Prototype (`live_mic_anc.py`)

Interactive real-time demonstration simulating tactical headset communication with dynamic noise scenarios (rotor, gunfire, tank engine):

```bash
# Run 5-second live streaming ANC demo with 10ms frame pacing
python -m inference.scripts.live_mic_anc --duration 5.0 --chunk-ms 10.0

# Or using the automation script:
bash scripts/17_live_stream_prototype.sh --duration 5.0
# PowerShell: .\scripts\17_live_stream_prototype.ps1 -duration 5.0
```

### 2.3 Edge ONNX Model Export (`export_onnx.py`)

Exports models with dynamic shapes for NVIDIA Jetson AGX Orin & DSP deployment:

```bash
# Export Model 1 with 10ms chunk profiling
python -m inference.scripts.export_onnx --model aegis-se-primary --chunk-ms 10.0

# Or using the automation script:
bash scripts/15_export_edge_onnx.sh --model aegis-se-primary
# PowerShell: .\scripts\15_export_edge_onnx.ps1 -model aegis-se-primary
```

---

## 3. Directory Layout

```
inference/
├── __init__.py                                 # Public inference API exports
├── README.md                                   # This architectural runbook
├── runtime/
│   ├── hybrid_anc.py                          # NormalizedLMSFilter & HybridAncPipeline (AI + NLMS)
│   ├── audio_stream.py                        # AudioRingBuffer & StreamingAudioProcessor (50% OLA)
│   ├── escalation_router.py                   # AcousticEscalationRouter (Dynamic gating & crossfade)
│   └── __init__.py
├── engines/
│   ├── onnx_engine.py                         # PyTorch-to-ONNX export & latency profiler
│   ├── onnx_runtime_engine.py                 # Accelerated ONNXRuntime execution session
│   ├── quantization.py                        # Post-training dynamic INT8 quantization
│   └── __init__.py
├── utils/
│   ├── audio_io.py                            # 48kHz mono audio reader/writer with clipping protection
│   └── __init__.py
└── scripts/
    ├── enhance_audio.py                       # CLI audio file enhancer
    ├── live_mic_anc.py                        # Real-time live microphone ANC prototype demo
    ├── export_onnx.py                         # CLI ONNX model exporter
    └── __init__.py
```
