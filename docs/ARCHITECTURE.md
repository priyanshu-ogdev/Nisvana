# Project AEGIS — Data-Forge System Architecture

## 1. Executive Summary

Project AEGIS is an autonomous acoustic data-forge engineering pipeline designed to generate fullband (48,000 Hz) training corpora for next-generation active noise cancellation (ANC) and speech enhancement systems deployed in high-stress, high-threat defence environments.

Unlike generic speech enhancement pipelines that rely on synthetic procedural noise or ungrounded generative audio, Project AEGIS enforces **strict scientific invariants**:
1. **Zero Pitch-Shift Invariant**: Clean speech targets and mechanical vehicle platforms (tanks, jet aircraft, howitzers, naval destroyers) and ballistics (blasts, gunshots) are **never pitch shifted**. Pitch-shifting corrupts vocal tract formants (arXiv:2407.05471) and warps physical vehicle engine RPM and bore dimensions into acoustically impossible artifacts.
2. **Real Acoustic Primacy**: Only authentic, peer-reviewed, physical acoustic recordings from authoritative repositories (NATO RSG.10, Harvard Dataverse, Dryad, Edinburgh DataShare, Nature Scientific Data, Microsoft Research) are utilized.
3. **5-Model Branch Isolation**: The corpus is branched into five specialized architectures, ensuring task-specific acoustic relationships (e.g. echo path transfer functions in AEC) remain isolated from external noise.

---

## 2. Five-Tier Storage Hierarchy

To eliminate the filesystem/inode bottleneck of loose files at scale while preserving complete data provenance, Project AEGIS structures storage across five distinct tiers:

```
d:\Nisvana\data\
├── raw\                             # Tier 0: Verifiable Raw Sources
│   ├── noisex92\                    # NATO RSG.10 defence vehicle recordings
│   ├── shared_explosions\           # Harvard Dataverse 326 high-explosive detonation waveforms
│   ├── gunshot_dryad\               # Cooper & Shaw multi-mic ballistic gunshot waveforms
│   ├── drone_audioset\              # UAV ego-noise & multi-rotor audio
│   ├── mad\                         # Nature Sci. Data Military Audio Dataset (8,075 clips)
│   ├── vctk_demand\                 # VoiceBank clean speech + 16-ch DEMAND environmental noise
│   ├── dns_challenge\               # DNS-5 training dataset (AudioSet/Freesound noise + clean speech)
│   ├── aec_challenge\               # ICASSP AEC-Challenge synthetic fullband quadruplets
│   ├── sirens_urban\                # ESC-50 & UrbanSound8K sirens & wind
│   └── openslr_rirs\                # OpenSLR-28 RIR archive & calibrated room simulator pool
│
├── processed\                       # Tier 1: 10-Step Standardized Pool
│   │                                # 48kHz polyphase resampled, -23.0 LUFS, mono, PCM 16-bit
│   ├── clean_speech\                # VAD-trimmed speech ground truth
│   ├── general_noise\               # Ambient environmental backgrounds
│   ├── tank_tracked\                # Leopard 1 tank tracks & diesel engine
│   ├── artillery_howitzer\          # M109 155mm howitzer operational noise
│   ├── jet_cockpit\                 # F-16 & Buccaneer cockpit noise
│   ├── naval_destroyer\             # Destroyer engine room & operations room
│   ├── military_vehicle\            # MAD tracked/wheeled combat platforms
│   ├── drone_uav\                   # Multi-rotor UAV signatures
│   ├── siren_emergency\             # Civil defense & emergency vehicle sirens
│   ├── wind_rotor_gap\              # High-velocity turbulent wind
│   ├── explosion_blast\             # Detonation shockwaves (pre-onset preserved)
│   ├── gunshot_firearm\             # Small arms muzzle blasts & shockwaves
│   ├── room_impulse_response\       # Calibrated acoustic impulse responses
│   └── far_end_echo\                # Acoustic echo cancellation signals
│
├── augmented\                       # Tier 2: Grounded Per-Source Variations
│   │                                # Bounded WSOLA time-stretch [0.90, 1.10], gain jitter [-3, +3] dB,
│   │                                # and blast onset windowing for explosions and gunshots.
│
├── splits\                          # Tier 3: Partition Registries & Leak-Free Split Manifests
│   ├── train_manifest.json          # 80% train partition
│   ├── val_manifest.json            # 10% validation partition
│   ├── test_generalization_manifest.json # 10% generalization-test (MUSAN strictly isolated)
│   └── split_summary.json           # Partition statistics & class distributions
│
├── forge\                           # Tier 4: Multi-Branch Model Training Corpora
│   ├── branch_speech_enhancement\   # Models 1-3: Triplets (noisy/, clean/, rir/) at SNR [-5, +20] dB
│   ├── branch_classifier\           # Model 4: 3-way taxonomy audio + labels.json
│   └── branch_aec\                  # Model 5: Isolated AEC quadruplets (mic/, farend/, nearend/, echo/)
│
└── shards\                          # Tier 5: WebDataset Sequential Shards & Dataset Card
    ├── speech_enhancement\          # se-000000.tar, se-000001.tar ... (2048 samples/shard)
    ├── classifier\                  # clf-000000.tar ...
    ├── aec\                         # aec-000000.tar ...
    └── DATASET_CARD.md              # Auto-generated Croissant / HuggingFace metadata card
```

---

## 3. Five Target Model Branches

| Branch | Target Models | Architecture & Characteristics | Target Loss Function | Corpus Structure |
|---|---|---|---|---|
| **Branch 1: Low-Latency Streaming SE** | **Model 1: DeepFilterNet3 Streaming** | Causal ERB filterbank (32 bands) + Order-$N=5$ complex deep filtering ($\le 8\text{ kHz}$); causal 2D-CNN + 2-layer GRU; algorithmic latency $\le 20\text{ ms}$. | Compressed STFT loss ($c=0.3$) + ERB mask loss + Deep Filter loss | Paired triplets `(noisy, clean, rir)` |
| **Branch 2: High-Fidelity Master SE** | **Model 2: DeepFilterNet3 Master** | Semi-causal/offline; 2-frame lookahead (20 ms); Order-$N=8$ deep filtering ($\le 12\text{ kHz}$); 5-layer CNN + Bidirectional GRU (hidden 384). | Multi-Resolution STFT loss + Compressed Spectral Loss + SI-SDR loss | Paired triplets `(noisy, clean, rir)` |
| **Branch 3: State-Space SE** | **Model 3: CleanUMamba** | U-Net backbone with bidirectional Mamba SSM (Selective State Spaces) blocks (Groot et al., 2024); linear $\mathcal{O}(L)$ time complexity. | Time-domain L1 loss + Multi-Resolution STFT spectral loss | Paired triplets `(noisy, clean, rir)` |
| **Branch 4: Acoustic Classifier** | **Model 4: SNR & 3-Way Harmonic Classifier** | MobileNetV3-Audio / Depthwise Separable CNN over 64-band Log-Mel Spectrogram; multi-task classification + continuous SNR regression. | Cross-Entropy Loss + $0.05 \cdot \text{MSE}(\widehat{\text{SNR}}, \text{SNR}_{\text{true}})$ | Single audio clips + 3-way category labels |
| **Branch 5: Gated AEC** | **Model 5: DeepVQE Gated AEC** | Dual-channel complex STFT input $[Y_{\text{mic}}, X_{\text{farend}}]$; complex ratio mask (cRM) for near-end preservation and echo suppression. | Near-end compressed spectral loss + ERLE penalty | Isolated quadruplets `(mic, farend, nearend, echo)` |

---

## 4. Hardware Sizing & Storage Budget (High-Capacity ML Workstation / 4 TB Storage)

- **Available Formatted Capacity**: ~3,725 GB
- **Raw Corpora Downloaded**: ~206.5 GB
- **10-Step Standardized Pool**: ~102.3 GB
- **Grounded Augmentations**: ~2.5 GB
- **Forge Model Training Corpora**: ~281.2 GB (200,000 SE triplets, 50,000 classifier samples, 10,000 AEC quadruplets)
- **WebDataset Shards**: ~281.5 GB
- **Total Storage Utilized**: **~874.0 GB (23.5% of 4 TB volume)**
- **Remaining Free Space**: **~2,851 GB (>2.8 TB)** for PyTorch checkpoints, optimizer states, evaluation caches, and tensorboard logs.
