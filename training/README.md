# Project AEGIS — Machine Learning Training Pipeline & Model Engine

> **Location:** `training/`  
> **Hardware Target:** NVIDIA DGX Spark GB10 (Grace Blackwell, CUDA 13, Ubuntu Linux)  
> **Acoustic Standards:** 48,000 Hz Fullband Audio, ITU-T P.862.2 (Wideband PESQ), Taal et al. (STOI)  
> **Core Architectures:** DeepFilterNet3 (Models 1 & 2), CleanUMamba SSM (Model 3), Conformer Gate (Model 4), Gated AEC (Model 5)

---

## 1. Overview & End-to-End Training Lifecycle

The `training` package is the deep learning engine for Project AEGIS. It trains, evaluates, guards, and checkpoints a heterogeneous ensemble of neural models designed for speech enhancement, acoustic environment classification, and active noise cancellation under extreme defence-specific acoustic stresses.

```
+===================================================================================================+
|                                    TRAINING ENGINE ARCHITECTURE                                   |
+===================================================================================================+
|                                                                                                   |
|  [WebDataset Tar Shards: data/shards/speech_enhancement/se_train_*.tar]                           |
|                                            │                                                      |
|                                            ▼                                                      |
|  +---------------------------------------------------------------------------------------------+  |
|  | 1. DATA INGESTION & WEIGHTED SAMPLING (training/data/)                                      |  |
|  |    WeightedShardSampler: Sync-Tier down-weighting (0.25x for 16k upsampled audio)           |  |
|  |    Class Oversampling: 6x tank, 8x howitzer, 4x jet, 5x explosion, 4x gunshot                |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                            │                                                      |
|                                            ▼                                                      |
|  +---------------------------------------------------------------------------------------------+  |
|  | 2. MODEL ARCHITECTURES & FACTORY (training/models/)                                         |  |
|  |    Model 1: DeepFilterNet3 Base (Causal, 0ms lookahead, 1.8M params)                        |  |
|  |    Model 2: DeepFilterNet3 Escalation (20ms lookahead, 2.3M params)                         |  |
|  |    Model 3: CleanUMamba (Selective State Space Model, long-horizon transient tracking)      |  |
|  |    Model 4: Gating Classifier (3-way Conformer/Conv1d: harmonic, transient, speech)        |  |
|  |    Model 5: Gated AEC (Dual-branch complex mask post-filter, ERLE > 30 dB)                  |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                            │                                                      |
|                                            ▼                                                      |
|  +---------------------------------------------------------------------------------------------+  |
|  | 3. SOTA PERCEPTUAL LOSS STACK (training/losses/)                                            |  |
|  |    L_total = L_multires + L_local_snr + L_sdr + L_IS3 + L_perc                               |  |
|  |    Multi-Res Spectral (4 FFTs) + SDR (Le Roux) + IS^3 Impulse (3x onset boost) + A-weighted |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                            │                                                      |
|                                            ▼                                                      |
|  +---------------------------------------------------------------------------------------------+  |
|  | 4. TRAINER EXECUTION & GOVERNANCE (training/trainers/ & training/callbacks/)                |  |
|  |    Gradient Accumulation: Effective batch size 128 on DGX Spark GB10                        |  |
|  |    Optimizer: AdamW + Linear Warmup + Cosine Decay Learning Rate                            |  |
|  |    EMA Tracking: Shadow weights (decay=0.999) maintained for validation and export          |  |
|  |    Worst-Class Pareto Guard: Rejects checkpoints if fragile military classes regress        |  |
|  |    Gradual Unfreezer: Progressive unfreezing (df_decoder -> erb_decoder -> encoders)         |  |
|  |    SNR Curriculum: Dynamically shifts training difficulty from +15 dB down to -5 dB         |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                            │                                                      |
|                                            ▼                                                      |
|  [Standardized Checkpoints: training/checkpoints/<model_key>/best_checkpoint.pt]                  |
+===================================================================================================+
```

---

## 2. Deep-Dive: Subsystem Specifications

### 2.1 Model Architectures & Factory (`training/models/`)
The model factory `build_model_for_key(model_key, config)` in [model_loader.py](file:///d:/Nisvana/training/models/model_loader.py) dynamically instantiates the appropriate model architecture:
1. **Model 1 (`aegis-se-primary`) — DeepFilterNet3 Base**:
   - Primary real-time speech enhancement model operating with **strict 0ms lookahead**.
   - Features 32 perceptual ERB bands + deep complex filtering for lower frequencies ($\le 5\text{ kHz}$).
   - Parameter count: $\sim 1.8\text{M}$ parameters ($<10\text{ ms}$ processing time).
2. **Model 2 (`aegis-se-escalation`) — DeepFilterNet3 Escalation**:
   - Configured with `df_lookahead=2` and `conv_lookahead=2` ($20\text{ ms}$ buffered lookahead context).
   - Absorbs extreme acoustic transients and hostile negative-SNR noise ($\text{SNR} < 0\text{ dB}$).
3. **Model 3 (`aegis-se-crosscheck`) — CleanUMamba**:
   - Selective State Space Model (SSM) based on Mamba blocks.
   - Unidirectional causal scanning provides linear-time inference with persistent long-horizon acoustic memory.
   - Leverages `mamba-ssm` and `causal-conv1d` CUDA kernels on DGX Spark GB10.
4. **Model 4 (`aegis-clf-gate`) — Acoustic Gating Classifier**:
   - 4-stage 1D convolutional network with adaptive pooling processing 200ms audio windows.
   - Outputs 3 logits: `stationary_harmonic` (0), `non_stationary_transient` (1), `speech_dominant` (2).
5. **Model 5 (`aegis-aec-gate`) — Gated AEC**:
   - Neural echo post-filter suppressing residual acoustic echo while passing near-end speech uncorrupted.

### 2.2 SOTA Loss Objectives & Co-Design (`training/losses/`)
Implemented in [multires_loss.py](file:///d:/Nisvana/training/losses/multires_loss.py), maximizing generalization from 100% real acoustic recordings without synthetic data:
1. **Multi-Resolution Spectral Loss (`MultiResSpectralLoss`)**:
   $$\mathcal{L}_{\text{mag}} = \frac{1}{M} \sum_{m=1}^M \left\| |\hat{S}_m|^\gamma - |S_m|^\gamma \right\|_1, \quad \gamma = 0.3$$
   Computed across 4 STFT window resolutions ($N_{\text{FFT}} \in \{256, 512, 1024, 2048\}$).
2. **Local SNR Loss (`LocalSnrLoss`)**: Penalizes frames where residual noise exceeds speech energy across 10ms hops.
3. **Speech-Presence-Gated SDR Loss (`SDRLoss`)**:
   $$\mathcal{L}_{\text{SDR}} = - 10 \log_{10} \left( \frac{\|s\|^2}{\|\hat{s} - s\|^2 + \epsilon} \right)$$
   - Penalizes gain mismatches, preserving absolute transmission calibration.
   - **Speech Formant Boosting**: Gated by a 1024-point Hann-windowed energy filter in the critical 300–4000 Hz formant band (`speech_presence_sdr_boost=2.5`, `speech_presence_rms_threshold=0.02`). Applies a clean $2.5\times$ gradient boost during active speech syllables without rectangular spectral sidelobe leakage, guaranteeing zero speech clipping.
4. **Knowledge Distillation Loss (`DistillationLoss`)**:
   $$\mathcal{L}_{\text{distill}} = \frac{1}{T} \sum_{t} \left\| |\text{STFT}(\hat{s}_{\text{student}})| - |\text{STFT}(\hat{s}_{\text{teacher}})| \right\|_2^2$$
   Distills temporal context from a frozen CleanUMamba (SSM) teacher into DeepFilterNet3 students with weight $\lambda_{\text{distill}} = 0.3$.
5. **Impulse-Weighted $\text{IS}^3$ Loss (`ImpulseWeightedLoss`)**:
   - Monitors frame energy ratio $r[k] = E[k] / (E[k-1] + \epsilon)$.
   - Applies a **$3.0\times$ onset error boost** whenever $r[k] > 2.0$, forcing the network to preserve shockwave rise times.
6. **Perceptual Frequency-Weighted Loss (`PerceptualFreqWeightedLoss`)**:
   - Applies IEC 61672-1 A-weighting, emphasizing the 1–6 kHz speech intelligibility band.

### 2.3 Concrete Trainers & Lifecycle (`training/trainers/`)
Inheriting from `BaseTrainer` in [base_trainer.py](file:///d:/Nisvana/training/trainers/base_trainer.py):
- **Native Grace Blackwell GB10 bfloat16 AMP**: Configurable via `precision="bf16"`, running native `torch.autocast('cuda', dtype=torch.bfloat16)` without loss scaling artifacts across the 128GB unified RAM pool.
- **Platform-Adaptive Quantization-Aware Training (QAT)**:
  - Invoked from epoch 1 via `prepare_qat(model)` and `convert_qat(model)`.
  - Attaches fake-quantization observers exclusively to `Conv1d`, `Conv2d`, and `Linear` layers, leaving recurrent state containers (`nn.GRU`) unquantized to prevent tuple-return graph corruption.
- **`SePrimaryTrainer`**: Gradient accumulation (accum steps=4), bfloat16 AMP, CleanUMamba distillation, and deployment model export (`export_deployment_model()`).
- **`SeEscalationTrainer`**: Real lookahead context buffering via `_output_delay_buffer` (1-chunk / 480-sample lookahead delay).
- **`SeCrosscheckTrainer`**: Stateful sequence optimization for CleanUMamba SSM.
- **`ClassifierTrainer`**: Cross-Entropy and focal loss for class-imbalanced gating.
- **`AecGateTrainer`**: Echo Return Loss Enhancement (ERLE) optimization.

### 2.4 Governance Callbacks (`training/callbacks/`)
- **Worst-Class Pareto Checkpoint Guard** (`worst_class_checkpoint_selector.py`):
  Rejects checkpoints if any individual fragile class (`tank_tracked`, `artillery_howitzer`, `jet_cockpit`, `naval_destroyer`, `explosion_blast`, `gunshot_firearm`) regresses by more than $\delta_{\text{margin}} = 0.05$ PESQ or $0.5\text{ dB}$ SNR, even if the global aggregate score increases. Banks individual class high-water marks.
- **Gradual Layer Unfreezer** (`gradual_unfreeze.py`):
  Progressively unfreezes layer groups: `df_decoder` (Epoch 0) $\rightarrow$ `erb_decoder` (Epoch 3) $\rightarrow$ `df_encoder` (Epoch 6) $\rightarrow$ `erb_encoder` (Epoch 9).
- **EMA Shadow Tracker** (`ema.py`): Maintains polyak shadow weights ($\beta = 0.999$) for validation and export.

### 2.5 Dynamic Curriculum Schedulers (`training/schedulers/`)
- **SNR Curriculum** (`snr_curriculum.py`):
  Gradually shifts target SNR bounds from an initial gentle distribution ($+10\text{ dB}$ to $+20\text{ dB}$) down to the hostile operational envelope ($-5.0\text{ dB}$ to $+20.0\text{ dB}$) over training epochs, stabilizing gradient descent.

### 2.6 Configuration Schemas (`training/configs/`)
Validated Python dataclasses in [base_config.py](file:///d:/Nisvana/training/configs/base_config.py) and model-specific configs:
- `SYNC_TIER_SAMPLE_WEIGHT`: `{Tier 1: 1.0, Tier 2: 1.0, Tier 3: 0.25}`.
- `CLASS_OVERSAMPLE_FACTORS`: Tuned to real combat corpus density ($6\times$ tank, $8\times$ howitzer, $4\times$ jet, $5\times$ explosion, $2.5\times$ gunshot/gunfire).
- `QAT & Distillation`: `qat_enabled=True`, `precision="bf16"`, `distillation_factor=0.3`.

### 2.7 Ingestion & Sampling (`training/data/`)
- **Weighted Shard Sampler** (`weighted_shard_sampler.py`): Reads per-sample metadata sidecars from WebDataset `.tar` shards, applying SyncTier down-weighting and class oversampling.
- **SpecMix Masking** (`spec_augment.py`): Dynamic time-frequency masking on mixture spectrograms (clean target speech remains 100% untouched).
- **Streaming Chunker** (`streaming_chunker.py`): Slices continuous training signals into exact 480-sample (10ms @ 48kHz) streaming chunks to maintain zero evaluation discrepancy with production edge inference.

### 2.8 Evaluation Suite & Metrics (`training/utils/`)
Implemented in [metrics.py](file:///d:/Nisvana/training/utils/metrics.py):
- **PESQ Wideband**: ITU-T P.862.2 (scale: $-0.5$ to $4.5$).
- **STOI**: Taal et al. intelligibility index (scale: $0.0$ to $1.0$).
- **SI-SNR & SNR**: Scale-Invariant and empirical Signal-to-Noise Ratios in dB.
- **Segmental SNR (SSNR)**: Clamped to $[-10\text{ dB}, +35\text{ dB}]$.
- **DNSMOS P.835**: SIG, BAK, and OVRL neural MOS proxies.
- **ERLE (dB)**: Echo Return Loss Enhancement for Model 5.

### 2.9 Generalization and split hygiene

Training consumes only `*-train-*.tar` shards. Validation and deployment
readiness must use the independent `*-val-*.tar` and `*-gentest-*.tar`
shards respectively. The shard loader now fails closed when a requested
split is missing; it never falls back to all tar files in a directory.
Evaluation also requires real data and reports both sample-weighted and
class-balanced metrics so frequent broad classes cannot hide regressions in
thin operational classes. Use `--seed` and `--num-workers` on
`training.scripts.train_pipeline` for reproducible worker seeding..

Data-forge provenance (`source_dataset`, `sync_tier`, native sample rate,
measured SNR, and unified class) is retained in every speech-enhancement
mixture sidecar. Training applies the configured class and sync-tier weights
during streaming shard iteration; validation and `test_generalization` are
never oversampled..

---

## 3. Training Execution & CLI Reference

Launch training for any model using the scripts in `training/scripts/`:

```bash
# Model 1: DeepFilterNet3 Base (0ms lookahead streaming)
python -m training.scripts.train_se_primary \
    --epochs 100 \
    --batch-size 32 \
    --grad-accum 4 \
    --lr 0.0005 \
    --device cuda

# Model 2: DeepFilterNet3 Escalation (20ms lookahead)
python -m training.scripts.train_se_escalation \
    --epochs 80 \
    --batch-size 32 \
    --device cuda

# Model 3: CleanUMamba SSM Crosscheck
python -m training.scripts.train_se_crosscheck \
    --epochs 100 \
    --batch-size 16 \
    --device cuda

# Model 4: Gating Classifier
python -m training.scripts.train_classifier \
    --epochs 50 \
    --batch-size 64 \
    --device cuda

# Model 5: Gated AEC Post-Filter
python -m training.scripts.train_aec \
    --epochs 60 \
    --batch-size 32 \
    --device cuda
```

### Checkpoint Storage & Recovery
Checkpoints are saved automatically to `training/checkpoints/<model_key>/`:
- `best_checkpoint.pt`: Optimal weights verified by the Worst-Class Pareto Guard.
- `latest_checkpoint.pt`: Full state dictionary for resuming interrupted runs via `--resume`.
