# Project AEGIS — Pipeline Operations Runbook (High-Capacity ML Machine / 4 TB Storage)

## 1. Machine Profile & Environment

This runbook covers executing the complete Project AEGIS Data-Forge pipeline on the designated high-capacity machine (equipped with 4 TB of local storage for datasets and ML training):

- **Operating System**: Linux x86_64 (Ubuntu 22.04 LTS / Debian 12 / RHEL 9 recommended)
- **Processor**: 16+ Cores (32 vCPUs recommended for parallel polyphase Kaiser resampling)
- **Memory**: 64 GB+ System RAM (minimum 32 GB)
- **Local Storage**: 4 TB NVMe or high-throughput SSD mounted at `/data` or repo root
- **Python**: Python 3.10, 3.11, 3.12, or 3.13

---

## 2. Environment Preparation

### Step 1: Install System Audio Libraries
```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    sox \
    curl \
    git \
    tmux
```

### Step 2: Clone Repository & Make Scripts Executable
```bash
git clone https://github.com/priyanshu-ogdev/Nisvana.git
cd Nisvana
chmod +x scripts/*.sh
```

### Step 3: Run Environment Initialization
```bash
bash scripts/00_setup_environment.sh
```
This installs Python dependencies, creates the complete `data/` directory hierarchy, and checks API token availability.

---

## 3. External API Credentials (Optional but Recommended)

### 1. Kaggle API (For Military Audio Dataset — MAD)
The real audio archive (~1.1 GB, 8,075 clips) is hosted on Kaggle (`junewookim/mad-dataset-military-audio-dataset`).
```bash
export KAGGLE_USERNAME="your_kaggle_username"
export KAGGLE_KEY="your_kaggle_api_key"
# OR place credentials at ~/.kaggle/kaggle.json
mkdir -p ~/.kaggle
echo '{"username":"your_username","key":"your_key"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```
*(If credentials are not supplied, the pipeline falls back to reading local files placed in `data/raw/mad/`)*.

### 2. Dryad API (For Ballistic Gunshots)
```bash
export DRYAD_API_TOKEN="your_dryad_bearer_token"
```
*(If token is not supplied, drop raw multi-mic gunshot WAVs into `data/raw/gunshot_dryad/`)*.

---

## 4. Pipeline Execution Workflows

### Option A: Full Production Run (Recommended)
Run inside a persistent `tmux` session to ensure uninterrupted execution:
```bash
tmux new -s aegis_pipeline

# Run full pipeline with 16 workers and 200,000 speech enhancement mixtures
bash scripts/03_data_pipeline_full_run.sh --workers 16 --mixtures 200000

# Detach session: Ctrl+B then D
# Re-attach session: tmux attach -t aegis_pipeline
```

### Option B: Dry-Run Reachability Check (Zero Disk Writes)
To verify all remote URLs, Azure Blobs, and format verifiers in seconds without downloading multi-gigabyte files:
```bash
bash scripts/01_data_pipeline_dry_run.sh
```

### Option C: Sample Verification with Auto-Clean Slate
Runs a quick end-to-end test on sample clips, verifies the audit report, and **automatically wipes sample data afterwards** to leave a 100% blank slate:
```bash
bash scripts/02_data_pipeline_sample_test.sh
```

---

## 5. Storage Sizing & Monitoring

During execution, monitor storage utilization on the high-capacity volume using:
```bash
watch -n 10 "df -h . && du -sh data/*"
```

### Expected Storage Breakdown:
| Directory | Contents | Expected Disk Size |
|---|---|---|
| `data/raw/` | Downloaded archives & raw corpora | ~206.5 GB |
| `data/processed/` | 48kHz -23 LUFS standardized mono audio | ~102.3 GB |
| `data/augmented/` | Grounded augmentations (WSOLA, jitter) | ~2.5 GB |
| `data/forge/` | Model-specific paired samples (SE, CLF, AEC) | ~281.2 GB |
| `data/shards/` | High-throughput WebDataset `.tar` shards | ~281.5 GB |
| **Total Pipeline** | **End-to-end dataset footprint** | **~874.0 GB** |
| **Free Headroom** | **Checkpoints, optimizer states, logs** | **> 2,800 GB (>2.8 TB)** |

---

## 6. PyTorch WebDataset Consumption

Training scripts consume the sharded data directly with zero uncompressed disk I/O bottleneck:

```python
from torch.utils.data import DataLoader
from data_forge.exporter.torch_dataset import (
    AegisSpeechEnhancementIterableDataset,
    AegisClassifierIterableDataset,
    AegisAecIterableDataset,
)

# Model 1-3 Speech Enhancement DataLoader (WebDataset)
se_dataset = AegisSpeechEnhancementIterableDataset("data/shards/speech_enhancement/se-*.tar")
se_loader = DataLoader(se_dataset, batch_size=32, num_workers=4)

for batch in se_loader:
    noisy_wav = batch["noisy"]  # Tensor: [B, T] (48kHz)
    clean_wav = batch["clean"]  # Tensor: [B, T] (48kHz)
    rir_wav   = batch["rir"]    # Tensor: [B, T]
    metadata  = batch["json"]   # Dict: SNR, class, sync_tier
    # Forward pass through DeepFilterNet3 / CleanUMamba
```
