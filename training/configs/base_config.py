"""
Project AEGIS — training/configs/base_config.py

Shared config schema every model's training config inherits from. Exists so
the data-forge sync (shard paths, sync-tier weights, unified-class taxonomy)
is defined ONCE and referenced everywhere, instead of five configs each
re-guessing the same paths — the exact class of bug the data-forge review
kept catching in the fetcher layer, now designed out of the training layer
from the start.

Naming convention (applies across training/ and inference/):
  - Model registry keys: "aegis-<branch>-<role>" (e.g. "aegis-se-primary")
  - Checkpoint files: "<model_key>-v<config_version>-step<N>.pt"
  - Config classes: "<ModelKey>Config" in PascalCase matching the key
  - This lets a checkpoint filename alone tell you which config produced it.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Single source of truth: import SyncTier and UnifiedClass directly from data_forge.config
from data_forge.config import (
    DEFAULT_MAX_SNR,
    DEFAULT_MIN_SNR,
    SyncTier,
    TARGET_LUFS,
    TARGET_SAMPLE_RATE,
    UnifiedClass,
)

# SOTA upgrade pass config imports
from training.callbacks.ema import EmaConfig
from training.callbacks.worst_class_checkpoint_selector import WorstClassCheckpointConfig
from training.callbacks.gradual_unfreezing import GradualUnfreezeConfig
from training.data.spec_augment import SpecMixConfig
from training.schedulers.snr_curriculum import SnrCurriculumConfig

# Down-weighting factors applied to WeightedRandomSampler per sync tier.
# Rationale: Tier 3 sources (LibriSpeech, MUSAN, NOISEX-92, OpenSLR RIRs)
# have no real energy above 8kHz -- letting them compete equally with Tier 1
# in a 48kHz training mix teaches the model that "silence above 8kHz" is a
# valid class signature for whichever noise types happen to be Tier 3.
SYNC_TIER_SAMPLE_WEIGHT: Dict[Any, float] = {
    SyncTier.TIER_1_NATIVE_48K: 1.00,
    SyncTier.TIER_2_RESAMPLED_44K: 1.00,
    SyncTier.TIER_3_UPSAMPLED_16K: 0.25,
    1: 1.00,
    2: 1.00,
    3: 0.25,
    "tier_1_native_48k": 1.00,
    "tier_2_resampled_44k": 1.00,
    "tier_3_upsampled_16k": 0.25,
}

# Complete taxonomy containing both granular data_forge classes and broad training aliases
UNIFIED_CLASSES: List[str] = [
    # Granular data_forge UnifiedClass keys
    "clean_speech", "tank_tracked", "artillery_howitzer", "jet_cockpit",
    "naval_destroyer", "military_vehicle", "explosion_blast", "gunshot_firearm",
    "drone_uav", "siren_emergency", "wind_rotor_gap", "general_noise",
    # Broad training bucket aliases
    "gunfire", "explosion_artillery", "rotor_vehicle_drone",
    "armored_vehicle_naval_jet", "siren", "wind", "vehicle_engine_general",
    "babble_crowd", "broad_industrial", "speech_clean",
]

# Canonical crosswalk between granular data_forge class and broad training bucket
TAXONOMY_MAP: Dict[str, str] = {
    "tank_tracked": "armored_vehicle_naval_jet",
    "artillery_howitzer": "armored_vehicle_naval_jet",
    "jet_cockpit": "armored_vehicle_naval_jet",
    "naval_destroyer": "armored_vehicle_naval_jet",
    "military_vehicle": "armored_vehicle_naval_jet",
    "explosion_blast": "explosion_artillery",
    "gunshot_firearm": "gunfire",
    "drone_uav": "rotor_vehicle_drone",
    "clean_speech": "speech_clean",
    "siren_emergency": "siren",
    "wind_rotor_gap": "wind",
    "general_noise": "broad_industrial",
}


@dataclass
class DataForgeSyncConfig:
    """
    Single source of truth for where this training run reads data from.
    Every field here corresponds directly to a path or convention already
    established and verified in data_forge/.
    """
    # Keep the default repository-relative so configs remain portable across
    # worktrees and match the path used by the data-forge CLI.
    shards_root: Path = field(default_factory=lambda: Path("data/shards"))
    speech_enhancement_shards: Path = field(init=False)
    classifier_shards: Path = field(init=False)
    aec_shards: Path = field(init=False)
    dataset_card_path: Path = field(init=False)

    target_sample_rate: int = TARGET_SAMPLE_RATE
    target_lufs: float = TARGET_LUFS
    samples_per_shard: int = 2048            # data_forge.config.SAMPLES_PER_SHARD

    # Splits, matching the dedup/generalization discipline established in
    # the dataset review: WHAM!, MUSAN, LibriSpeech are NEVER in train.
    train_shard_glob: str = "*-train-*.tar"
    val_shard_glob: str = "*-val-*.tar"
    generalization_shard_glob: str = "*-test_generalization-*.tar"
    required_splits: List[str] = field(
        default_factory=lambda: ["train", "val", "test_generalization"]
    )
    require_real_shards: bool = True

    def __post_init__(self):
        self.speech_enhancement_shards = self.shards_root / "speech_enhancement"
        self.classifier_shards = self.shards_root / "classifier"
        self.aec_shards = self.shards_root / "aec"
        self.dataset_card_path = self.shards_root / "DATASET_CARD.md"


@dataclass
class BaseModelConfig:
    """Fields every one of the 5 model configs shares."""
    model_key: str                            # e.g. "aegis-se-primary"
    config_version: int = 1
    seed: int = 1337
    deterministic_training: bool = True

    data: DataForgeSyncConfig = field(default_factory=DataForgeSyncConfig)

    # Checkpointing
    checkpoint_dir: Path = Path("training/checkpoints")
    checkpoint_every_n_steps: int = 5000
    keep_last_n_checkpoints: int = 5
    resume_from: Optional[Path] = None

    # Compute
    mixed_precision: bool = True
    gradient_clip_norm: float = 5.0
    num_workers: int = 8
    device: str = "cuda"

    # Logging
    log_dir: Path = Path("training/runs")
    log_every_n_steps: int = 100
    eval_every_n_steps: int = 2500

    # Optimization & Training Lifecycle
    lr: float = 1e-3
    lr_warmup_steps: int = 5000
    total_finetune_steps: int = 100_000
    early_stopping_patience: int = 10

    # --- SOTA upgrade pass additions, all opt-in/configurable per-model, ---
    # --- not silently forced -- see each module's docstring for grounding. ---
    ema: EmaConfig = field(default_factory=EmaConfig)
    spec_mix: SpecMixConfig = field(default_factory=SpecMixConfig)
    snr_curriculum: SnrCurriculumConfig = field(default_factory=SnrCurriculumConfig)
    worst_class_checkpoint: WorstClassCheckpointConfig = field(default_factory=WorstClassCheckpointConfig)
    gradual_unfreeze: GradualUnfreezeConfig = field(default_factory=GradualUnfreezeConfig)

    # --- Rev 3 P0.4: QAT + bf16 ---
    # QAT from first epoch: fake-quantization observers on Conv1d + GRU
    # layers from the start of fine-tuning, so the checkpoint is already
    # INT8-robust when training completes. No separate PTQ pass needed.
    qat_enabled: bool = True
    qat_backend: str = "qnnpack"  # Platform A (ARM). Platform B (GPU) uses TensorRT path.

    # bf16 precision: Blackwell GB10 has native bf16 Tensor Core support.
    # Avoids fp16's dynamic-range issues with multires_spec_factor=500.0
    # (spectral magnitudes can exceed fp16 max). bf16 doesn't need loss
    # scaling (GradScaler becomes a no-op).
    precision: str = "bf16"  # "bf16", "fp16", or "fp32"

    # --- Rev 3 P1.1: Distillation ---
    # Distillation factor for CleanUMamba teacher → Model 1/2 student.
    # REASONED ENGINEERING CHOICE: zero inference cost, raises accuracy
    # ceiling on naval/armored-vehicle/gunfire classes by providing a
    # second-opinion training signal from a different architecture.
    distillation_factor: float = 0.0  # Set >0 in per-model configs to enable

    min_training_snr_db: float = DEFAULT_MIN_SNR
    max_training_snr_db: float = DEFAULT_MAX_SNR

    def checkpoint_name(self, step: int) -> str:
        return f"{self.model_key}-v{self.config_version}-step{step:08d}.pt"
