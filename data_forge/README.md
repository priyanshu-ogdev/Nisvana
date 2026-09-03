# Project AEGIS — Data-Forge Package Guide & Complete Walkthrough

`data_forge` is the autonomous acoustic data-engineering engine for Project AEGIS. It handles the complete lifecycle of multi-source acoustic data: from verifiable remote fetching and 10-step sequential preprocessing, to physics-grounded augmentation, multi-branch training corpus synthesis, and high-throughput WebDataset `.tar` sharding for deep learning model training.

---

## 1. Package Architecture & Submodules

```
data_forge/
├── __init__.py              # Package exports & version
├── __main__.py              # CLI entrypoint (`python -m data_forge ...`)
├── cli.py                   # Command-line interface definition
├── config.py                # Global paths, sample rates, sync tiers, & profiles
├── bibliography.py          # 20 verified literature datasets & citation ledger
│
├── fetcher/                 # Multi-Source Automated Downloaders
│   ├── base.py              # BaseFetcher with chunked streaming, resume, & MD5 hashing
│   ├── manager.py           # FetchManager orchestrating all 10 sources
│   ├── noisex.py            # NATO RSG.10 NOISEX-92 vehicle/cockpit fetcher
│   ├── shared.py            # Harvard Dataverse high-explosive airblast fetcher
│   ├── gunshot.py           # Data Dryad multi-mic ballistic gunshot fetcher
│   ├── drone.py             # HuggingFace DroneAudioSet Parquet unpacker
│   ├── mad.py               # Kaggle API Military Audio Dataset (8,075 clips) fetcher
│   ├── vctk_demand.py       # Edinburgh DataShare VoiceBank-DEMAND fetcher
│   ├── dns.py               # Microsoft DNS-5 training & eval Azure Blob fetcher
│   ├── aec.py               # ICASSP AEC-Challenge synthetic quadruplet fetcher
│   ├── sirens.py            # ESC-50 / UrbanSound8K sirens & wind fetcher
│   └── rir.py               # OpenSLR-28 RIRs & calibrated image-source simulator
│
├── preprocessor/            # 10-Step Sequential Standardization
│   ├── pipeline.py          # PreprocessingPipeline orchestrator with multiprocessing
│   ├── step1_format.py      # Standardizes audio containers to PCM float arrays
│   ├── step2_resample.py    # Kaiser polyphase resampling to 48,000 Hz + Sync Tier tagging
│   ├── step3_loudness.py    # ITU-R BS.1770-4 (-23 LUFS) + -1.0 dBFS true-peak limiter
│   ├── step4_vad.py         # Selective speech VAD (preserves blast/gunshot onsets)
│   ├── step5_channel.py     # Deterministic energy-preserving mono downmixing
│   ├── step6_integrity.py   # Rejects NaN, Inf, clipping (>99.5%), zero-signal flatlines
│   ├── step7_metadata.py    # ClipMetadata records with taxonomy & origin crosswalk
│   ├── step8_dedup.py       # 4-band spectral energy deduplication hashing
│   ├── step9_license.py     # Filters CC-BY-NC files for commercial compliance
│   └── step10_split.py      # Leak-free origin-aware 80/10/10 partitioner
│
├── augmentor/               # Grounded Per-Source Augmentations
│   ├── engine.py            # AugmentationEngine applying class-specific policies
│   ├── policy.py            # Strict enforcement of the Zero Pitch-Shift Invariant
│   ├── time_stretch.py      # WSOLA time-stretch bounded in [0.90, 1.10]
│   ├── gain_jitter.py       # Bounded gain jitter [-3.0 dB, +3.0 dB]
│   └── blast_window.py      # Transient windowing preserving shockwave physics
│
├── mixer/                   # Multi-Branch Model Training Synthesis
│   ├── engine.py            # SNR mixer & causal RIR convolution engine
│   ├── speech_enhancement.py# Models 1-3: Triplet synthesis (noisy, clean, rir)
│   ├── classifier.py        # Model 4: 3-way taxonomy mapping & harmonicity index
│   └── aec.py               # Model 5: Isolated AEC quadruplet organizer
│
├── exporter/                # High-Throughput Storage & PyTorch Integration
│   ├── shard_writer.py      # WebDataset .tar shard writer (~2048 samples/shard)
│   ├── dataset_card.py      # Auto-generates HuggingFace/Croissant DATASET_CARD.md
│   └── torch_dataset.py     # PyTorch IterableDataset wrappers for DataLoader
│
└── verifier/                # Pipeline Quality Assurance
    ├── auditor.py           # PipelineAuditor checking sample rate, leakage, & pitch
    └── reporter.py          # Generates Markdown and JSON pipeline audit reports
```

---

## 2. Complete Step-by-Step Data Flow

```
1. Fetcher (data/raw/)
   └── Retrieves authentic archives from 10 verified sources.
       Supports sample mode, full multi-part Azure Blobs, and offline drop-ins.
          │
          ▼
2. 10-Step Preprocessor (data/processed/ & data/splits/)
   └── Standardizes all audio to 48,000 Hz, -23.0 LUFS, mono, PCM 16-bit.
       Enforces selective speech VAD (leaves blast shockwaves intact).
       Outputs leak-free 80/10/10 split manifests (Train / Val / Test-Generalization).
          │
          ▼
3. Grounded Augmentor (data/augmented/)
   └── Applies bounded WSOLA time-stretch and gain jitter.
       CRITICAL INVARIANT: Zero pitch-shifting on clean speech and military platforms.
          │
          ▼
4. Multi-Branch Mixer (data/forge/)
   └── Generates target corpora across 5 distinct model branches:
       - Branch 1 (Models 1-3 SE): (noisy, clean, rir) triplets at SNR [-5 dB to +20 dB].
       - Branch 2 (Model 4 CLF): 3-way taxonomy (harmonic, transient, speech dominant).
       - Branch 3 (Model 5 AEC): Isolated quadruplets (mic, farend, nearend, echo).
          │
          ▼
5. WebDataset Exporter (data/shards/)
   └── Packs branch audio and JSON metadata into sequential .tar shards.
       Auto-generates Croissant / HuggingFace-compliant DATASET_CARD.md.
          │
          ▼
6. Pipeline Verifier (data/manifests/)
   └── Audits 100% sample rate compliance, zero data leakage, and pitch integrity.
```

---

## 3. CLI Command Walkthrough

You can execute the entire pipeline or any individual stage directly via `python -m data_forge <command>`:

### 1. Endpoint Verification (`--dry-run`)
Probes all remote endpoints, checks HTTP headers, and verifies Azure Blob containers with **zero data written to disk**:
```bash
python -m data_forge run-all --dry-run
```

### 2. Sample Mode (Fast Integration Verification)
Runs a lightweight end-to-end verification pass synthesizing a small sample of mixtures:
```bash
python -m data_forge run-all --sample-mode --num-mixtures 20
```

### 3. Full Production Run (High-Capacity Machine with 4TB Storage)
Executes the complete multi-source download, parallel 10-step preprocessing, augmentation, mixing (200,000 mixtures), and WebDataset sharding:
```bash
python -m data_forge run-all --full-mode --max-workers 16 --num-mixtures 200000
```

### 4. Granular Stage-by-Stage Execution
Each stage can be executed independently:

```bash
# Step 1: Fetch raw datasets (all or a specific source)
python -m data_forge fetch --source all --full-mode

# Step 2: Preprocess with 16 parallel worker threads
python -m data_forge preprocess --max-workers 16

# Step 3: Apply grounded augmentations
python -m data_forge augment

# Step 4: Synthesize multi-branch model corpora
python -m data_forge mix --num-mixtures 10000 --min-snr -5.0 --max-snr 20.0

# Step 5: Pack into WebDataset shards and generate DATASET_CARD.md
python -m data_forge export

# Step 6: Verify compliance and generate audit report
python -m data_forge verify
```

---

## 4. Python API Programmatic Walkthrough

You can also import and use `data_forge` components directly in Python scripts or Jupyter notebooks:

### Consuming WebDataset Shards in PyTorch
```python
from torch.utils.data import DataLoader
from data_forge.exporter.torch_dataset import (
    AegisSpeechEnhancementIterableDataset,
    AegisClassifierIterableDataset,
    AegisAecIterableDataset,
)

# Load speech enhancement triplets for Models 1-3
dataset = AegisSpeechEnhancementIterableDataset("data/shards/speech_enhancement/se-*.tar")
loader = DataLoader(dataset, batch_size=32, num_workers=4)

for batch in loader:
    noisy_audio = batch["noisy"]  # Tensor: [32, 192000] (4.0s @ 48kHz)
    clean_audio = batch["clean"]  # Tensor: [32, 192000]
    rir_audio   = batch["rir"]    # Tensor: [32, L]
    meta        = batch["json"]   # Dict: target SNR, class, sync tier
    break
```

### Programmatic Preprocessing Single Clips
```python
import numpy as np
from data_forge.preprocessor.step2_resample import PolyphaseResampler
from data_forge.preprocessor.step3_loudness import LoudnessNormalizer

resampler = PolyphaseResampler(target_sample_rate=48000)
normalizer = LoudnessNormalizer(target_lufs=-23.0)

# Resample 16kHz audio to 48kHz native standard
audio_16k = np.random.randn(16000).astype(np.float32)
audio_48k, tier = resampler.resample(audio_16k, orig_sr=16000)

# Normalize to -23 LUFS with -1.0 dBFS true-peak limiter
audio_norm, measured_lufs, peak_dbfs = normalizer.normalize(audio_48k, sr=48000)
```
