# Project AEGIS Data-Forge — CHANGELOG (Verification Pass 2, 2026-09-03)

## Authoritative Verifications & Hardened Fixes

### 1. Edinburgh DataShare (VCTK-DEMAND Handle 10283/2791) Fully Resolved & Verified
- **Issue**: Historical handle `10283/1942` is marked "SUPERSEDED: THIS DATASET HAS BEEN REPLACED" on Edinburgh DataShare.
- **Resolution**:
  - Live query against Edinburgh DataShare DSpace REST API (`https://datashare.ed.ac.uk/server/api/core/bitstreams/`) confirmed that all three bitstream UUIDs in `data_forge/fetcher/vctk_demand.py` originate directly from the active, non-superseded record (handle `10283/2791`):
    - `clean_testset_wav.zip` (`dec213d3-bf57-4777-9663-c24bdce92d5e`, 154.3 MB): MD5 `34eb1c0ba7ef667e9b966866c542fc16`
    - `noisy_testset_wav.zip` (`13c1bfbf-14a6-41db-9b41-8f7310f01ad5`, 170.6 MB): MD5 `fb1b86caa31e8ba5b506c0c64da9aab5`
    - `clean_trainset_28spk_wav.zip` (`245452b6-6235-44b6-a6f9-e7eb19797769`, 2.48 GB): MD5 `d2d5a45ec32f8fcbf201bde0447e20ba`
  - All three MD5 checksums are now strictly wired into `VctkDemandFetcher.ARCHIVES`. Corrupted or incomplete downloads will fail hash validation loudly.

### 2. Microsoft DNS-5 Training Corpora & Azure Blob Storage Verified
- **Issue**: Attempting to enumerate Azure Blobs dynamically via `?restype=container&comp=list` returned `HTTP 404: The specified resource does not exist` because Microsoft configured `dnschallengepublic` with Blob-level ACL (container enumeration disabled, direct blob access public).
- **Resolution**:
  - Probed official scripts from `microsoft/DNS-Challenge` repository (`download-dns-challenge-5-headset-training.sh` and `download-dns-challenge-5-noise-ir.sh`).
  - Base URL confirmed: `https://dnschallengepublic.blob.core.windows.net/dns5archive/V5_training_dataset/`
  - All blob URLs probed live and confirmed **200 OK**:
    - `Track1_Headset/emotional_speech.tgz` (200 OK, 38.9 MB)
    - `Track1_Headset/VocalSet_48kHz_mono.tgz` (200 OK, 14.9 MB)
    - `Track1_Headset/vctk_wav48_silence_trimmed.tgz.partaa` (200 OK, 5.24 GB)
    - `Track1_Headset/read_speech.tgz.partaa` (200 OK, 5.24 GB)
    - `noise_fullband/datasets_fullband.noise_fullband.audioset_000.tar.bz2` (200 OK, 5.36 GB)
    - `noise_fullband/datasets_fullband.noise_fullband.freesound_000.tar.bz2` (200 OK, 3.47 GB)
    - `datasets_fullband.impulse_responses_000.tar.bz2` (200 OK, 264.9 MB)
    - `V5_dev_testset.zip` (200 OK, 3.08 GB)
  - `DnsChallengeFetcher` retains dynamic discovery as an opportunistic first attempt, falling back gracefully to the verified official list when container listing is disabled.
  - Added sample audio extraction from `V5_dev_testset.zip` into `data/raw/dns_challenge/dns5_dev/` in `sample_mode` so integration runs immediately have audio for 10-step preprocessing.

### 3. Modular 1-to-1 Test Suite Consolidation
- Integrated all regression tests into [`tests/test_fetchers.py`](file:///d:/Nisvana/tests/test_fetchers.py) under `TestVctkDemandFetcher` and `TestDnsChallengeFetcher`.
- 55/55 unit and integration tests passing in ~62 seconds (100% pass rate).
- Full end-to-end dry run (`python -m data_forge run-all --dry-run`) verified clean with 0 issues.
