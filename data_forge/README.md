# Project AEGIS — Data-Forge Architecture & Comprehensive Package Runbook

> **Location:** `data_forge/`  
> **Target Audio Standard:** 48,000 Hz, Mono, PCM 16-bit, $-23.0 \pm 0.5$ LUFS, $-1.0$ dBFS True Peak  
> **Policy:** 100% Real Acoustic Recordings Only (Zero Synthetic Audio Generation / Zero Augmentations)  
> **Storage Target:** Multi-Terabyte NVMe Storage (`data/`) on NVIDIA DGX Spark GB10

---

## 1. Executive Summary & Subsystem Architecture

`data_forge` is the central acoustic data-engineering engine for Project AEGIS. It executes the entire lifecycle of high-fidelity acoustic corpus engineering: resilient remote acquisition across 10 authoritative physical datasets, strict 10-step DSP standardization under ITU-R BS.1770-4, multi-branch model training synthesis, and high-throughput WebDataset `.tar` sharding.

```
+===================================================================================================+
|                                    DATA-FORGE CORE ARCHITECTURE                                   |
+===================================================================================================+
|                                                                                                   |
|  [10 AUTHORITATIVE REAL-WORLD DATASETS]                                                           |
|  NOISEX-92, SHAReD, MAD, DEMAND, DroneAudioSet, Sirens, Dryad Gunshots, OpenSLR RIR, DNS, AEC    |
|                                            │                                                      |
|                                            ▼                                                      |
|  +---------------------------------------------------------------------------------------------+  |
|  | 1. FETCHER SUBMODULE (data_forge/fetcher/)                                                  |  |
|  |    BaseFetcher: TCP session pooling, HTTP Range resume, exponential backoff, disk checks    |  |
|  |    Sources: 10 dedicated fetchers downloading authentic acoustic recordings to data/raw/     |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                            │                                                      |
|                                            ▼                                                      |
|  +---------------------------------------------------------------------------------------------+  |
|  | 2. PREPROCESSOR SUBMODULE (data_forge/preprocessor/)                                        |  |
|  |    10-Step Sequential DSP Pipeline (multiprocessing pool):                                  |  |
|  |    Format -> Polyphase 48kHz (Sync Tiers) -> BS.1770-4 -23 LUFS -> VAD -> Downmix ->       |  |
|  |    Integrity -> Canonical Tagging -> Dedup -> License Filter -> Split Isolation (80/10/10)   |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                            │                                                      |
|                                            ▼                                                      |
|  +---------------------------------------------------------------------------------------------+  |
|  | 3. AUGMENTOR SUBMODULE (data_forge/augmentor/)                                              |  |
|  |    Real-Recordings Policy (DATA_FORGE_REAL_ONLY=true): Bypasses all synthetic augmentations |  |
|  |    Ablation Mode: Zero pitch-shift invariant, WSOLA time stretch [0.90, 1.10], gain jitter   |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                            │                                                      |
|                                            ▼                                                      |
|  +---------------------------------------------------------------------------------------------+  |
|  | 4. MIXER SUBMODULE (data_forge/mixer/)                                                      |  |
|  |    Branch 1: Speech Enhancement (x = s*h + n, -5 to +20 dB SNR, early reverberant target)   |  |
|  |    Branch 2: Gating Classifier (200ms slices, autocorrelation harmonicity index, 3 classes) |  |
|  |    Branch 3: Acoustic Echo Cancellation (matched quadruplets: mic, farend, nearend, echo)   |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                            │                                                      |
|                                            ▼                                                      |
|  +---------------------------------------------------------------------------------------------+  |
|  | 5. EXPORTER & VERIFIER (data_forge/exporter/ & data_forge/verifier/)                        |  |
|  |    Exporter: Packs into WebDataset .tar shards (2,048 samples/tar) + DATASET_CARD.md        |  |
|  |    Verifier: 7-point automated audit checking sample rate, LUFS, clipping, & split leakage  |  |
|  +---------------------------------------------------------------------------------------------+  |
+===================================================================================================+
```

---

## 2. Global Policy: 100% Real Recordings Only

In strict compliance with mission specifications:
- **Zero Synthetic Audio**: Generative mathematical tone generators, synthetic pink/white noise injectors, and algorithmic acoustic simulators are completely disabled.
- **Physical Sensor Integrity**: Downstream models learn to generalize directly from authentic physical sensor measurements: real gunshots (shockwaves and muzzle blasts), combat vehicle engine compartments, supersonic fighter jet cockpits, artillery detonations, and calibrated room impulse responses.
- **Environment Flags** (set in `.env`):
  ```ini
  DATA_FORGE_REAL_ONLY=true
  DATA_FORGE_NO_AUGMENT=true
  ```
- **Generalization Mechanism**: Instead of artificial data augmentation, acoustic robustness is achieved through the upgraded ML training layer: multi-resolution spectral losses, Signal-to-Distortion Ratio (SDR) loss, impulse-weighted $\text{IS}^3$ loss for transient events, perceptual A-weighted loss, tier-weighted shard sampling, and noise-type curriculum learning.

---

## 3. Deep-Dive: Submodule Architecture & Mechanics

### 3.1 Fetcher Submodule (`data_forge/fetcher/`)
Responsible for acquiring all external research corpora with enterprise-grade download resilience, multi-mirror redundancy, and credential injection:
- **`base.py` (`BaseFetcher`)**:
  - **Fallback Mirror Architecture**: Supports `fallback_urls: Optional[List[str]]` in `download_file()`. If primary host returns 4xx/5xx or drops connection, it automatically fails over to mirror endpoints sequentially.
  - **Dynamic Auth Token Injection**: Injects domain-specific authorization headers on the fly via `get_headers_for_url()`:
    - GitHub (`DATA_FORGE_GITHUB_TOKEN` / `GITHUB_TOKEN`): Elevates API / raw release rate limit from 60 req/hr to 5,000 req/hr for NOISEX-92, Sirens, AEC, and MAD annotations.
    - Hugging Face (`DATA_FORGE_HF_TOKEN` / `HF_TOKEN`): Injects `Authorization: Bearer <token>` for DroneAudioSet and Hugging Face mirrors.
    - Data Dryad (`DATA_FORGE_DRYAD_API_TOKEN` / `DRYAD_API_TOKEN`): Injects `Authorization: Bearer <token>` for gunshots data acquisition.
    - Harvard Dataverse (`DATA_FORGE_DATAVERSE_API_TOKEN`): Injects `X-Dataverse-key: <token>` for SHAReD blast shockwave archive access.
  - **Firewall & Proxy Routing**: Routes requests via `DATA_FORGE_HTTP_PROXY` and `DATA_FORGE_HTTPS_PROXY`, spoofing legitimate browser user-agent via `DATA_FORGE_USER_AGENT` to bypass anti-scraping WAFs.
  - **Connection Pooling**: Uses `requests.Session` with `HTTPAdapter(pool_connections=10, pool_maxsize=10)` and HTTP/2 keep-alive.
  - **HTTP Range Resumption**: Incomplete downloads automatically issue `Range: bytes=<offset>-` headers, resuming from byte offsets instead of restarting multi-GB files.
  - **Exponential Backoff**: Up to 10 retries with dynamic backoff ($\text{delay} = 5 \times 2^{\text{attempt}}$ seconds) to survive transient server drops.
  - **Disk Space Safety**: Proactively verifies that free disk space exceeds `DATA_FORGE_DISK_SAFETY_MARGIN_GB` (50 GB) before writing.
  - **Checksum Verification**: Calculates streaming MD5 digests to ensure byte-exact correspondence with upstream releases.
- **Data Source Catalog & Mirror Fallbacks**:
  1. `noisex.py`: NATO RSG.10 NOISEX-92 vehicle and cockpit recordings (Leopard 1 tank, M109 howitzer, F-16 cockpit, Destroyer ops/engine room). Fallbacks: `panandicoding/Build-SE-Dataset`, `haoxiangsnr/UNetGAN-Demo`.
  2. `shared.py`: Harvard Dataverse high-explosive airblast shockwaves (Takazawa et al., Sensors 2024, C-4/TNT detonations). Fallback: Harvard Dataverse direct API token routing.
  3. `drone.py`: DroneAudioSet (Al-Emadi et al.) multi-rotor UAV ego-noise and flybys. Fallback: official `https://hf-mirror.com` endpoint.
  4. `mad.py`: Military Audio Dataset (8,075 clips) acquired via Kaggle API with automated archive checksum verification.
  5. `gunshot.py`: Data Dryad multi-microphone ballistic gunshot acoustic measurements with token header injection.
  6. `sirens.py`: ESC-50 & UrbanSound8K authentic emergency sirens and wind recordings. Primary: Zenodo release; Mirror fallbacks: Zenodo master archive (`records/2452749`) and GitHub release archive (`karolpiczak/ESC-50`).
  7. `rir.py`: OpenSLR-28 authentic measured acoustic impulse responses (`real_rirs` from RWCP, AIR, REVERB). Primary: `openslr.org`; Fallbacks: ELDA European mirror (`openslr.elda.org`) and Magic Data mirror (`openslr.magicdatatech.com`).
  8. `vctk_demand.py`: Edinburgh DataShare clean speech targets and stationary noise. Fallback: Edinburgh DataShare handle routing.
  9. `dns.py`: INTERSPEECH DNS Challenge fullband clean speech and non-stationary noise with parallel chunk extraction.
  10. `aec.py`: ICASSP AEC Challenge acoustic echo cancellation quadruplets.
- **`manager.py` (`FetchManager`)**: Master coordinator with **parallel multi-worker fetching** (`fetch_all(concurrency=N)` via `concurrent.futures.ThreadPoolExecutor`), dry-run endpoint probes, sample-mode subsets, and multi-terabyte production pulls. Available via CLI `--max-workers N`.

### 3.2 Preprocessor Submodule (`data_forge/preprocessor/`)
Executes a 10-step audio standardization pipeline across parallel worker processes (`pipeline.py`):
1. **Step 1: Format Standardization** (`step1_format.py`): Decodes diverse containers into 16-bit uncompressed PCM WAV.
2. **Step 2: Polyphase 48kHz Resampling & Sync Tiers** (`step2_resample.py`):
   - Uses `scipy.signal.resample_poly` with Kaiser windowing for alias-free polyphase rate conversion to 48,000 Hz.
   - Tags **Sync Tiers**:
     - *Tier 1 (Native 48k)*: SHAReD, MAD, DEMAND, VCTK (training weight: `1.0`).
     - *Tier 2 (Resampled 44.1k)*: ESC-50, DroneAudioSet (training weight: `1.0`).
     - *Tier 3 (Upsampled 16k)*: Historic NOISEX-92 audio (training weight: `0.25` to penalize bandwidth-limited audio).
3. **Step 3: ITU-R BS.1770-4 Loudness Normalization** (`step3_loudness.py`):
   - Computes K-weighted integrated loudness via `pyloudnorm`.
   - Normalizes speech to **$-23.0 \pm 0.5$ LUFS** with short-transient RMS fallback for impulsive events $<400$ms.
   - Enforces a true-peak ceiling of **$-1.0$ dBFS** to guarantee zero digital clipping.
4. **Step 4: VAD & Silence Elimination** (`step4_vad.py`): Applies a $-50$ dBFS energy threshold across 20ms frames, trimming dead margins while preserving explosive onsets.
5. **Step 5: Equal-Power Mono Downmix** (`step5_channel.py`): Downmixes stereo/multichannel to mono: $\text{Mono} = \frac{L + R}{\sqrt{2}}$.
6. **Step 6: Numerical Integrity** (`step6_integrity.py`): Nulls DC offsets, eliminates `NaN`/`Inf`, and rejects severely clipped takes.
7. **Step 7: Canonical Metadata Tagging** (`step7_metadata.py`): Generates sidecar JSON mapping dataset classes into the 12-class unified taxonomy.
8. **Step 8: Perceptual Deduplication** (`step8_dedup.py`): 4-band spectral hash deduplication eliminating identical takes.
9. **Step 9: License Compliance Gating** (`step9_license.py`): Filters CC-BY-NC files when `--commercial-strict` is asserted.
10. **Step 10: Split Isolation** (`step10_split.py`): Partitions audio into 80% Train, 10% Val, and 10% Generalization Test, enforcing absolute speaker and environment isolation.

### 3.3 Augmentor Submodule (`data_forge/augmentor/`)
- Under the standard policy (`DATA_FORGE_REAL_ONLY=true`), `run_augmentation()` logs a notice and immediately returns without generating artificial audio.
- For ablation research where `--force-augment` is explicitly passed:
  - **Zero Pitch-Shift Invariant** (`policy.py`): Raises `ForbiddenPitchShiftError` if pitch shifting is attempted on speech, machinery (tanks, howitzers, jets), or ballistic transients.
  - **WSOLA Time Stretching** (`time_stretch.py`): Pitch-preserving waveform overlap-add strictly bounded to $[0.90, 1.10]$.
  - **Gain Jitter** (`gain_jitter.py`): Bounded gain shifts [$-3.0\text{ dB}, +3.0\text{ dB}$].
  - **Blast Onset Windowing** (`blast_window.py`): Preserves shockwave rise times while varying decay tails.

### 3.4 Mixer Submodule (`data_forge/mixer/`)
Synthesizes specialized training corpora for each model family:
- **Branch 1: Speech Enhancement (Models 1–3)** (`se.py`):
  - Acoustic equation: $x[t] = (s * h)[t] + \alpha \cdot n[t]$.
  - $\text{SNR} \sim \mathcal{U}(-5.0\text{ dB}, +20.0\text{ dB})$.
  - Target: Early reverberant speech ($s * h$).
  - Outputs triplet: `noisy.wav`, `clean.wav`, `rir.wav`, plus comprehensive sidecar `json`.
- **Branch 2: Classifier Gating (Model 4)** (`classifier.py`):
  - Slices audio into **200ms windows** (`CLASSIFIER_WINDOW_SEC = 0.2`).
  - Computes normalized autocorrelation harmonicity index:
    - `stationary_harmonic` (0): Drones, tanks, jet engines, sirens (harmonicity $>0.65$).
    - `non_stationary_transient` (1): Explosions, gunshots, impacts, or $\text{SNR} < 0\text{ dB}$.
    - `speech_dominant` (2): Clean speech or $\text{SNR} > 12\text{ dB}$.
- **Branch 3: Acoustic Echo Cancellation (Model 5)** (`aec.py`):
  - Matched quadruplets: `mic.wav`, `farend.wav`, `nearend.wav`, `echo.wav` calibrated for ERLE $>30\text{ dB}$.

### 3.5 Exporter Submodule (`data_forge/exporter/`)
- **WebDataset Sharder** (`shard_writer.py`): Packs flat-file mixtures into sequential POSIX `.tar` archives at 2,048 samples per shard (`se_train_000000.tar`, etc.).
- **DataLoader Iterators** (`torch_dataset.py`): Native PyTorch `IterableDataset` classes streaming samples directly into GPU memory without untarring to disk.
- **Dataset Card Generator** (`dataset_card.py`): Automatically documents sample counts, class distributions, SNR histograms, and licensing in `data/shards/DATASET_CARD.md`.

### 3.6 Verifier Submodule (`data_forge/verifier/`)
- **Pipeline Auditor** (`auditor.py`): Scans raw, processed, splits, forge, and shards directories to verify 48kHz compliance, LUFS bounds, zero clipping, and split segregation.
- **Compliance Reporter** (`reporter.py`): Generates human-readable Markdown audit summaries.

---

## 4. Unified CLI Specification & Command Reference

The entire pipeline is executable via `python -m data_forge <command>`:

```bash
# 1. FETCHING
python -m data_forge fetch --source all --dry-run      # Verify all 10 URLs without downloading
python -m data_forge fetch --source all --sample-mode  # Download verified subset for dev testing
python -m data_forge fetch --source all --full-mode    # Full multi-terabyte production download

# 2. PREPROCESSING
python -m data_forge preprocess --max-workers 64       # 10-step DSP standardization on Grace CPUs
python -m data_forge preprocess --commercial-strict    # Exclude CC-BY-NC datasets

# 3. MIXING
python -m data_forge mix --num-mixtures 200000 --min-snr -5.0 --max-snr 20.0

# 4. EXPORTING
python -m data_forge export                            # Pack into WebDataset shards + DATASET_CARD.md
python -m data_forge export --card-only                # Regenerate dataset card only

# 5. VERIFYING
python -m data_forge verify                            # Run full audit and compliance reporter

# 6. END-TO-END PIPELINE RUNNER
python -m data_forge run-all --sample-mode             # Complete pipeline smoke test
python -m data_forge run-all --full-mode --max-workers 64 # Complete production pipeline execution
```

---

## 5. Python API Usage Examples

### Programmatic Streaming from WebDataset Shards
```python
from torch.utils.data import DataLoader
from data_forge.exporter.torch_dataset import SpeechEnhancementIterableDataset

# Direct streaming into PyTorch training loop
dataset = SpeechEnhancementIterableDataset(
    shard_pattern="data/shards/speech_enhancement/se_train_*.tar",
    shuffle_buffer_size=1000,
)
loader = DataLoader(dataset, batch_size=32, num_workers=8, pin_memory=True)

for batch in loader:
    noisy = batch["noisy.wav"]  # Shape: [32, 192000]
    clean = batch["clean.wav"]  # Shape: [32, 192000]
    meta  = batch["json"]       # Metadata dictionary: SNR, class, tier
    break
```

### Programmatic Preprocessing of Custom Audio
```python
import numpy as np
from data_forge.preprocessor.step2_resample import PolyphaseResampler
from data_forge.preprocessor.step3_loudness import LoudnessNormalizer

resampler = PolyphaseResampler(target_sample_rate=48000)
normalizer = LoudnessNormalizer(target_lufs=-23.0)

# Resample raw 16kHz audio to 48kHz native standard
audio_16k = np.random.randn(16000).astype(np.float32)
audio_48k, tier = resampler.resample(audio_16k, orig_sr=16000)

# Normalize to -23.0 LUFS with true peak limiting
audio_norm, lufs, peak = normalizer.normalize(audio_48k, sr=48000)
```
