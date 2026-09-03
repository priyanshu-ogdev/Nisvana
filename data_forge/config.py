"""
Project AEGIS — Data-Forge Central Configuration and Policy Specifications
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Set, Optional


# ==============================================================================
# Path Definitions
# ==============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
AUGMENTED_DIR = DATA_DIR / "augmented"
SPLITS_DIR = DATA_DIR / "splits"
FORGE_DIR = DATA_DIR / "forge"
MANIFESTS_DIR = DATA_DIR / "manifests"
SHARDS_DIR = DATA_DIR / "shards"  # WebDataset-format sharded export, industry-standard training input

# Shard sizing: ~2048 samples/shard keeps individual .tar files in the low-GB
# range (sequential-read friendly for network/cloud storage) without creating
# so many shard files that shard-listing itself becomes the bottleneck.
SAMPLES_PER_SHARD = 2048

# Data Forge Model Branches
BRANCH_SE = FORGE_DIR / "branch_speech_enhancement"  # Models 1-3: DeepFilterNet3, CleanUMamba
BRANCH_CLASSIFIER = FORGE_DIR / "branch_classifier"  # Model 4: SNR / Harmonic Classifier
BRANCH_AEC = FORGE_DIR / "branch_aec"                # Model 5: Gated AEC


# ==============================================================================
# Audio Processing Standards (Part 3)
# ==============================================================================
TARGET_SAMPLE_RATE = 48000           # 48 kHz standard fullband
TARGET_LUFS = -23.0                  # ITU-R BS.1770-4 loudness target
TARGET_TRUE_PEAK_DBFS = -1.0         # Limiter ceiling to prevent inter-sample clipping
DEFAULT_AUDIO_FORMAT = "WAV"
DEFAULT_SUBTYPE = "PCM_16"           # 16-bit PCM standard (or PCM_24)
MIN_ACTIVE_DURATION_SEC = 0.4        # Minimum non-silent speech/audio length
SILENCE_ENERGY_THRESHOLD_DB = -50.0  # VAD threshold


# ==============================================================================
# Sync Tier Metadata (Part 3, Step 2)
# ==============================================================================
class SyncTier(int, Enum):
    """
    Audio synchronization and native frequency tiering:
    Tier 1: Native 48kHz (passes through without resampling)
    Tier 2: 44.1kHz (clean polyphase resample to 48kHz)
    Tier 3: 16kHz-native (upsampled to 48kHz, tagged and down-weighted in mixing)
    """
    TIER_1_NATIVE_48K = 1
    TIER_2_RESAMPLED_44K = 2
    TIER_3_UPSAMPLED_16K = 3


# ==============================================================================
# Unified Taxonomy Crosswalk (Part 3, Step 7 & Part 5)
# ==============================================================================
class UnifiedClass(str, Enum):
    # Speech Ground Truth
    CLEAN_SPEECH = "clean_speech"
    
    # Defence-Specific Vehicle Noise (NOISEX-92 & MAD)
    TANK_TRACKED = "tank_tracked"              # e.g., Leopard
    ARTILLERY_HOWITZER = "artillery_howitzer"  # e.g., M109
    JET_COCKPIT = "jet_cockpit"                # e.g., F-16, Buccaneer
    NAVAL_DESTROYER = "naval_destroyer"        # e.g., Destroyer Engine/Ops
    MILITARY_VEHICLE = "military_vehicle"      # General military vehicle
    
    # Weapons & Ballistics
    EXPLOSION_BLAST = "explosion_blast"        # High-explosive blast (SHAReD)
    GUNSHOT_FIREARM = "gunshot_firearm"        # Small arms / rifle fire
    
    # Aerial / Drones
    DRONE_UAV = "drone_uav"                    # UAV ego-noise / multi-rotor
    
    # Urban / Environmental
    SIREN_EMERGENCY = "siren_emergency"        # Ambulance, police, fire sirens
    WIND_ROTOR_GAP = "wind_rotor_gap"          # Documented gap classes
    GENERAL_NOISE = "general_noise"            # Ambient, babble, factory, etc.
    
    # Acoustic Coupling
    FAR_END_ECHO = "far_end_echo"              # AEC far-end echo coupling
    ROOM_IMPULSE_RESPONSE = "rir"              # Acoustic room response


# 3-Way Classifier Taxonomy (Model 4)
class ClassifierCategory(str, Enum):
    STATIONARY_HARMONIC = "stationary_harmonic"          # Drones, tanks, jet engines
    NON_STATIONARY_TRANSIENT = "non_stationary_transient" # Gunshots, explosions, impacts
    SPEECH_DOMINANT = "speech_dominant"                  # Clean or high-SNR speech


# ==============================================================================
# Augmentation Policies (Part 1 & Part 2)
# ==============================================================================
class AugmentationPolicy(str, Enum):
    """
    Grounded in 2026 speech enhancement literature (arXiv:2602.14671):
    - NO pitch shifting on speech targets (prevents formant corruption).
    - NO pitch shifting on defence-specific machinery or blast physics (prevents false RPMs).
    - Bounded time-stretch (+/- 5-10%) and level jitter for thin datasets only.
    """
    NONE = "none"                                 # Pure as-is (Clean speech, AEC Challenge)
    MINIMAL_SNR_ONLY = "minimal_snr_only"         # Mixing variation only (DNS noise, FSD50K)
    MINIMAL_LEVEL_CROP = "minimal_level_crop"     # Level jitter & random crop (MAD, DEMAND noise)
    MODERATE_NO_PITCH = "moderate_no_pitch"       # Time-stretch +/-5-10%, gain jitter, strictly NO pitch
    MODERATE_BLAST_WINDOW = "moderate_blast_window" # Onset windowing, gain jitter, strictly NO pitch
    MODERATE_FULL = "moderate_full"               # Time-stretch, gain jitter, layering (sirens, wind)


# Classes where pitch-shifting is strictly forbidden by physics and perceptual invariance
FORBIDDEN_PITCH_CLASSES: Set[UnifiedClass] = {
    UnifiedClass.CLEAN_SPEECH,
    UnifiedClass.TANK_TRACKED,
    UnifiedClass.ARTILLERY_HOWITZER,
    UnifiedClass.JET_COCKPIT,
    UnifiedClass.NAVAL_DESTROYER,
    UnifiedClass.MILITARY_VEHICLE,
    UnifiedClass.EXPLOSION_BLAST,
    UnifiedClass.GUNSHOT_FIREARM,
    UnifiedClass.FAR_END_ECHO,
    UnifiedClass.ROOM_IMPULSE_RESPONSE,
}


# ==============================================================================
# Dataset Profiles & Crosswalk Mapping
# ==============================================================================
@dataclass
class DatasetProfile:
    name: str
    source_citation: str
    license: str
    native_sample_rate: int
    default_sync_tier: SyncTier
    augmentation_policy: AugmentationPolicy
    unified_classes: List[UnifiedClass]
    is_generalization_only: bool = False  # Reserved exclusively for generalization test (e.g. LibriSpeech, MUSAN, WHAM)
    is_non_commercial: bool = False       # True for CC-BY-NC datasets (ESC-50, UrbanSound8K)


DATASET_PROFILES: Dict[str, DatasetProfile] = {
    "noisex92": DatasetProfile(
        name="NOISEX-92",
        source_citation="Varga & Steeneken, NATO RSG.10 / Rice University SPIB",
        license="Public Research / NATO RSG.10",
        native_sample_rate=19980,  # Or standard 16kHz conversion
        default_sync_tier=SyncTier.TIER_3_UPSAMPLED_16K,
        augmentation_policy=AugmentationPolicy.MODERATE_NO_PITCH,
        unified_classes=[
            UnifiedClass.TANK_TRACKED,
            UnifiedClass.ARTILLERY_HOWITZER,
            UnifiedClass.JET_COCKPIT,
            UnifiedClass.NAVAL_DESTROYER,
            UnifiedClass.GUNSHOT_FIREARM,
            UnifiedClass.GENERAL_NOISE,
        ],
    ),
    "shared": DatasetProfile(
        name="SHAReD",
        source_citation="Takazawa et al., Sensors 2024 / Harvard Dataverse (doi:10.7910/DVN/ROWODP)",
        license="CC BY 4.0",
        native_sample_rate=48000,
        default_sync_tier=SyncTier.TIER_1_NATIVE_48K,
        augmentation_policy=AugmentationPolicy.MODERATE_BLAST_WINDOW,
        unified_classes=[UnifiedClass.EXPLOSION_BLAST],
    ),
    "gunshot_dryad": DatasetProfile(
        name="Gunshot Triangulation Dataset",
        source_citation="Cooper & Shaw, Dryad 2020 (doi:10.5061/dryad.wm37pvmkc)",
        license="CC0 1.0 Universal",
        native_sample_rate=44100,
        default_sync_tier=SyncTier.TIER_2_RESAMPLED_44K,
        augmentation_policy=AugmentationPolicy.MODERATE_NO_PITCH,
        unified_classes=[UnifiedClass.GUNSHOT_FIREARM],
    ),
    "drone_audioset": DatasetProfile(
        name="DroneAudioSet",
        source_citation="ahlab-drone-project / Augmented Human Lab (arXiv:2308.10659)",
        license="MIT",
        native_sample_rate=48000,
        default_sync_tier=SyncTier.TIER_1_NATIVE_48K,
        augmentation_policy=AugmentationPolicy.MINIMAL_LEVEL_CROP,
        unified_classes=[UnifiedClass.DRONE_UAV],
    ),
    "mad": DatasetProfile(
        name="Military Audio Dataset (MAD)",
        source_citation="Kim, Yoon, & Jung, Nature Sci Data 11:668 (2024)",
        license="CC BY-SA 4.0",
        # VERIFIED 2026-09-03 (cross-checked twice against the dataset's own Kaggle
        # metadata, incl. an independent listing quoting "192" = 192kbps encoding):
        # MAD is native 48kHz mono, NOT 16kHz. Previously mis-tiered as Tier 3
        # (band-limited, down-weighted in mixing) when it should be Tier 1 (full-band,
        # no penalty). This affects sync-tier weighting in the mixer.
        native_sample_rate=48000,
        default_sync_tier=SyncTier.TIER_1_NATIVE_48K,
        augmentation_policy=AugmentationPolicy.MINIMAL_LEVEL_CROP,
        unified_classes=[
            UnifiedClass.MILITARY_VEHICLE,
            UnifiedClass.GUNSHOT_FIREARM,
            UnifiedClass.EXPLOSION_BLAST,
        ],
    ),
    "vctk_demand": DatasetProfile(
        name="VoiceBank-DEMAND / CSTR VCTK",
        source_citation="Veaux et al. (VCTK) & Thiemann et al. (DEMAND), Univ of Edinburgh",
        license="CC BY 4.0",
        native_sample_rate=48000,
        default_sync_tier=SyncTier.TIER_1_NATIVE_48K,
        augmentation_policy=AugmentationPolicy.NONE,  # Clean speech: NONE; noise: light jitter
        unified_classes=[UnifiedClass.CLEAN_SPEECH, UnifiedClass.GENERAL_NOISE],
    ),
    "dns_challenge": DatasetProfile(
        name="DNS Challenge (Interspeech / ICASSP)",
        source_citation="Reddy et al., Microsoft DNS Challenge",
        license="CC BY 4.0 / Microsoft Research",
        native_sample_rate=48000,
        default_sync_tier=SyncTier.TIER_1_NATIVE_48K,
        augmentation_policy=AugmentationPolicy.MINIMAL_SNR_ONLY,
        unified_classes=[UnifiedClass.CLEAN_SPEECH, UnifiedClass.GENERAL_NOISE],
    ),
    "aec_challenge": DatasetProfile(
        name="AEC Challenge (ICASSP)",
        source_citation="Microsoft ICASSP AEC Challenge (2021-2023)",
        license="CC BY 4.0 / Microsoft Research",
        native_sample_rate=48000,
        default_sync_tier=SyncTier.TIER_1_NATIVE_48K,
        augmentation_policy=AugmentationPolicy.NONE,
        unified_classes=[UnifiedClass.FAR_END_ECHO, UnifiedClass.CLEAN_SPEECH],
    ),
    "sirens_urban": DatasetProfile(
        name="UrbanSound8K / ESC-50 Sirens",
        source_citation="Salamon et al. (UrbanSound8K) & Piczak (ESC-50)",
        license="CC BY-NC 3.0",
        native_sample_rate=44100,
        default_sync_tier=SyncTier.TIER_2_RESAMPLED_44K,
        augmentation_policy=AugmentationPolicy.MODERATE_FULL,
        unified_classes=[UnifiedClass.SIREN_EMERGENCY, UnifiedClass.WIND_ROTOR_GAP],
        is_non_commercial=True,
    ),
    "openslr_rirs": DatasetProfile(
        name="OpenSLR-28 Room Impulse Responses",
        source_citation="Ko et al., ICASSP 2017 (OpenSLR 28)",
        license="Apache 2.0",
        native_sample_rate=16000,
        default_sync_tier=SyncTier.TIER_3_UPSAMPLED_16K,
        augmentation_policy=AugmentationPolicy.NONE,
        unified_classes=[UnifiedClass.ROOM_IMPULSE_RESPONSE],
    ),
    "musan_generalization": DatasetProfile(
        name="MUSAN",
        source_citation="Snyder, Chen, & Povey (OpenSLR 17)",
        license="Creative Commons Attribution 4.0",
        native_sample_rate=16000,
        default_sync_tier=SyncTier.TIER_3_UPSAMPLED_16K,
        augmentation_policy=AugmentationPolicy.NONE,
        unified_classes=[UnifiedClass.GENERAL_NOISE],
        is_generalization_only=True,
    ),
}


# ==============================================================================
# Data-Forge Mixing Configuration (Part 5)
# ==============================================================================
@dataclass
class ForgeMixingConfig:
    # SNR mixing distribution for speech enhancement (Models 1-3)
    min_snr_db: float = -5.0
    max_snr_db: float = 20.0
    
    # RIR convolution probability (how often reverberation is injected)
    rir_probability: float = 0.65
    
    # Target clip duration for training slices
    target_duration_sec: float = 4.0
    
    # Target loudness of mixed audio
    mixture_lufs: float = -23.0
    
    # Random seed for reproducible dataset generation
    seed: int = 42
