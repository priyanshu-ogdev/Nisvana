# Project AEGIS — 10-Step Sequential Preprocessing Pipeline

## 1. Overview

The Project AEGIS Preprocessing Pipeline implements a deterministic, 10-step sequential standardization workflow. Every audio file entering the training system passes through this exact sequence before being exposed to augmentation or multi-branch mixing:

```
Raw Audio File
   │
   ▼
[Step 1: Format Standardization] ─────────► Standard WAV container (PCM 16/24-bit)
   │
   ▼
[Step 2: Polyphase Resampling]  ──────────► Rational polyphase filtering to 48,000 Hz + Sync Tier
   │
   ▼
[Step 5: Channel Standardization] ────────► Energy-preserving mono downmixing
   │
   ▼
[Step 4: Silence & VAD Validation] ───────► Selective VAD speech trimming (transients preserved)
   │
   ▼
[Step 6: Integrity & Anomaly Check] ──────► Rejects NaN, Inf, clipping (>99.5%), zero signal
   │
   ▼
[Step 3: BS.1770-4 Loudness Normalization]► -23.0 LUFS normalization + -1.0 dBFS true-peak limiter
   │
   ▼
[Step 8: Acoustic Fingerprinting] ────────► 4-band spectral energy deduplication hash
   │
   ▼
[Step 7: Metadata Tagging] ───────────────► ClipMetadata record with taxonomy & sync attribution
   │
   ▼
[Step 9: License Compliance Filter] ──────► Retains/excludes CC-BY-NC based on commercial mode
   │
   ▼
[Step 10: Origin-Aware Split Partition] ──► Leak-free 80/10/10 split (train / val / test_gen)
```

---

## 2. Detailed Step Specifications

### Step 1: Format Standardization ([`step1_format.py`](file:///d:/Nisvana/data_forge/preprocessor/step1_format.py))
- **Objective**: Standardizes diverse incoming containers (`.flac`, `.ogg`, `.mp3`, `.wav`, `.aiff`, `.parquet`) into clean uncompressed floating-point audio arrays.
- **Implementation**: Uses `soundfile` with a fallback to `scipy.io.wavfile` and PyArrow byte extraction.
- **Output Range**: Normalized float array $x \in [-1.0, +1.0]$.

### Step 2: Sample-Rate Standardization & Sync-Tier Tagging ([`step2_resample.py`](file:///d:/Nisvana/data_forge/preprocessor/step2_resample.py))
- **Objective**: Converts all audio to the unified Project AEGIS fullband standard ($48,000\text{ Hz}$) using Kaiser-windowed polyphase resampling (`scipy.signal.resample_poly`), preventing the phase distortion of naive FFT resampling.
- **Sync Tier Attribution**:
  - **Tier 1 (Native 48 kHz)**: Clean speech, SHAReD explosions, DroneAudioSet, MAD military combat, VCTK 48k. Pass through untouched with zero resampling penalty.
  - **Tier 2 (Resampled 44.1 kHz)**: Gunshot Dryad, UrbanSound8K/ESC-50 sirens. Rational resampling factors: $up = 160, down = 147$.
  - **Tier 3 (Upsampled 16 kHz)**: NOISEX-92 vehicle recordings, OpenSLR-28 RIRs, MUSAN. Rational resampling factors: $up = 3, down = 1$. Tagged in metadata to guide mixer down-weighting.

### Step 3: ITU-R BS.1770-4 Loudness Normalization ([`step3_loudness.py`](file:///d:/Nisvana/data_forge/preprocessor/step3_loudness.py))
- **Objective**: Eliminates arbitrary recording level variations across disparate datasets to prevent loudness bias during training.
- **Reference Standard**: ITU-R BS.1770-4 integrated loudness via `pyloudnorm`.
  $$\text{Target Integrated Loudness} = -23.0\text{ LUFS}$$
- **Safety True-Peak Limiter**: Clamps signal ceiling to $-1.0\text{ dBFS}$ ($10^{-1.0/20} \approx 0.891$) to prevent inter-sample clipping when convolving with RIRs.
- **Transient Peak/RMS Fallback**: For impulsive signals $< 0.4\text{ s}$ (e.g. sharp gunshots, detonation clicks) where BS.1770 gating cannot integrate, falls back to equivalent sine-wave RMS normalization ($\approx -20\text{ dB RMS}$).

### Step 4: Silence & VAD Validation ([`step4_vad.py`](file:///d:/Nisvana/data_forge/preprocessor/step4_vad.py))
- **Selective Speech VAD**: Per Part 3 Step 4, energy-based voice activity detection (30 ms frames, $-50\text{ dB}$ silence threshold) is applied **strictly to clean speech sources** to trim dead recording tails and drop silent clips ($< 0.4\text{ s}$ active speech).
- **Physical Transient Preservation**: Transient waveforms (`EXPLOSION_BLAST`, `GUNSHOT_FIREARM`), vehicle engine hums, sirens, RIRs, and AEC signals bypass speech VAD trimming, ensuring the shockwave pre-onset lead-in and decay physics remain completely uncorrupted.

### Step 5: Channel Standardization ([`step5_channel.py`](file:///d:/Nisvana/data_forge/preprocessor/step5_channel.py))
- **Mono Downmix**: Multichannel audio is converted to single-channel mono via deterministic energy-preserving averaging:
  $$x_{\text{mono}}[n] = \frac{1}{C} \sum_{c=1}^{C} x_c[n]$$
- **Multichannel Array Pass-Through**: Retains channel metadata for spatial microphone array simulation if needed.

### Step 6: Waveform Integrity & Anomaly Check ([`step6_integrity.py`](file:///d:/Nisvana/data_forge/preprocessor/step6_integrity.py))
- Rejects any clip exhibiting corruptions that destabilize neural network gradient updates:
  - NaN or $\pm\infty$ values
  - Flatline zero recordings ($\max|x| < 10^{-6}$)
  - Severe hard-clipping anomalies ($> 0.5\%$ of samples exceeding $|x| \ge 0.999$)

### Step 7: Metadata Tagging & Taxonomy Crosswalk ([`step7_metadata.py`](file:///d:/Nisvana/data_forge/preprocessor/step7_metadata.py))
- Generates structured [`ClipMetadata`](file:///d:/Nisvana/data_forge/preprocessor/step7_metadata.py#L18-L38) records attaching:
  - Unified Acoustic Class (e.g. `tank_tracked`, `jet_cockpit`, `explosion_blast`)
  - Sync Tier (Tier 1, 2, or 3)
  - Measured LUFS, True-Peak dBFS, duration, and sampling rate
  - Licensing category (`commercial_permissive`, `non_commercial`, `research_only`)

### Step 8: Acoustic Fingerprinting & Deduplication ([`step8_dedup.py`](file:///d:/Nisvana/data_forge/preprocessor/step8_dedup.py))
- Generates a 4-band spectral energy distribution fingerprint across 10 temporal sub-windows.
- Identifies and flags duplicate clips (e.g. Freesound clips repackaged across FSD50K, AudioSet, and ESC-50) to prevent data contamination.

### Step 9: License Compliance Filtering ([`step9_license.py`](file:///d:/Nisvana/data_forge/preprocessor/step9_license.py))
- **`RESEARCH_PROTOTYPE` Mode**: Retains all verified datasets (including CC-BY-NC academic resources like ESC-50).
- **`COMMERCIAL_STRICT` Mode**: Excludes all non-commercial (`CC-BY-NC`) data, retaining strictly commercial-permissive assets (CC0, CC BY, MIT, Apache 2.0).

### Step 10: Leak-Free Origin-Aware Split Partitioning ([`step10_split.py`](file:///d:/Nisvana/data_forge/preprocessor/step10_split.py))
- Partitions the standardized corpus into **Train (80%)**, **Validation (10%)**, and **Generalization-Test (10%)**.
- **Deterministic Fingerprint Hashing**: All clips sharing the same origin or acoustic fingerprint map to the identical partition.
- **Strict Generalization Isolation**: Out-of-distribution evaluation datasets (MUSAN, LibriSpeech test-clean, WHAM) are **strictly isolated to `test_generalization`** and never appear in train or validation sets.
