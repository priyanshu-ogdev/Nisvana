"""
Project AEGIS — Training Architecture & Data-Forge Synchronization Tests
Verifies that all 5 model configs, WebDataset shard paths, taxonomy classes,
and sample weight calculations sync completely between data_forge/ and training/.
"""

import json
from pathlib import Path
import pytest
import numpy as np

from data_forge.config import (
    BRANCH_SE,
    BRANCH_CLASSIFIER,
    BRANCH_AEC,
    SHARDS_DIR,
    SyncTier,
    UnifiedClass,
    ClassifierCategory,
)
from training.configs.base_config import (
    DataForgeSyncConfig,
    SYNC_TIER_SAMPLE_WEIGHT,
    UNIFIED_CLASSES,
    TAXONOMY_MAP,
)
from training.configs.se_primary_config import SePrimaryConfig, CLASS_OVERSAMPLE_FACTORS
from training.configs.se_escalation_config import SeEscalationConfig
from training.configs.se_crosscheck_config import SeCrosscheckConfig
from training.configs.classifier_config import ClassifierConfig, GATE_CLASSES, UNIFIED_TO_GATE_CLASS
from training.configs.aec_config import AecGateConfig
from training.data.weighted_shard_sampler import compute_sample_weight


class TestModelPathSynchronization:
    """Verifies that all 5 models map to the exact directories created by data_forge."""

    def test_shards_directory_mapping(self):
        sync = DataForgeSyncConfig()
        assert sync.shards_root == Path("data/shards")
        assert sync.speech_enhancement_shards == sync.shards_root / "speech_enhancement"
        assert sync.classifier_shards == sync.shards_root / "classifier"
        assert sync.aec_shards == sync.shards_root / "aec"
        assert sync.dataset_card_path == sync.shards_root / "DATASET_CARD.md"

    def test_all_five_configs_load(self):
        m1 = SePrimaryConfig()
        m2 = SeEscalationConfig()
        m3 = SeCrosscheckConfig()
        m4 = ClassifierConfig()
        m5 = AecGateConfig()

        assert m1.model_key == "aegis-se-primary"
        assert m2.model_key == "aegis-se-escalation"
        assert m3.model_key == "aegis-se-crosscheck"
        assert m4.model_key == "aegis-clf-gate"
        assert m5.model_key == "aegis-aec-gate"

    def test_checkpoint_naming_convention(self):
        m1 = SePrimaryConfig()
        name = m1.checkpoint_name(12000)
        assert name == "aegis-se-primary-v1-step00012000.pt"

        m4 = ClassifierConfig()
        name4 = m4.checkpoint_name(5000)
        assert name4 == "aegis-clf-gate-v1-step00005000.pt"


class TestTaxonomyAndWeightingSynchronization:
    """Verifies taxonomy mapping and combined sample weights (sync_tier * class_oversample)."""

    def test_sync_tier_down_weighting(self):
        assert SYNC_TIER_SAMPLE_WEIGHT[SyncTier.TIER_1_NATIVE_48K] == 1.00
        assert SYNC_TIER_SAMPLE_WEIGHT[SyncTier.TIER_2_RESAMPLED_44K] == 1.00
        assert SYNC_TIER_SAMPLE_WEIGHT[SyncTier.TIER_3_UPSAMPLED_16K] == 0.25
        assert SYNC_TIER_SAMPLE_WEIGHT[1] == 1.00
        assert SYNC_TIER_SAMPLE_WEIGHT[3] == 0.25

    def test_class_oversample_factors_for_all_unified_classes(self):
        # Thin defence vehicle classes get 6.0x oversample factor
        for vehicle_class in ("tank_tracked", "artillery_howitzer", "jet_cockpit", "naval_destroyer", "military_vehicle"):
            assert CLASS_OVERSAMPLE_FACTORS[vehicle_class] == 6.0, f"Missing 6.0x oversample for {vehicle_class}"

        # Explosion blast gets 4.0x
        assert CLASS_OVERSAMPLE_FACTORS["explosion_blast"] == 4.0

        # Siren and wind get 5.0x
        assert CLASS_OVERSAMPLE_FACTORS["siren_emergency"] == 5.0
        assert CLASS_OVERSAMPLE_FACTORS["wind_rotor_gap"] == 5.0

        # Abundant classes get 1.0x
        assert CLASS_OVERSAMPLE_FACTORS["gunshot_firearm"] == 1.0
        assert CLASS_OVERSAMPLE_FACTORS["drone_uav"] == 1.0
        assert CLASS_OVERSAMPLE_FACTORS["clean_speech"] == 1.0

    def test_combined_sample_weight_calculation(self):
        # NOISEX-92 Leopard tank is BOTH Tier 3 (0.25) AND thin class (6.0) -> 0.25 * 6.0 = 1.50
        weight_leopard = compute_sample_weight("tank_tracked", SyncTier.TIER_3_UPSAMPLED_16K, CLASS_OVERSAMPLE_FACTORS)
        assert weight_leopard == pytest.approx(1.50, rel=1e-3)

        # Native 48kHz Drone audio is Tier 1 (1.00) AND abundant class (1.0) -> 1.00 * 1.0 = 1.00
        weight_drone = compute_sample_weight("drone_uav", SyncTier.TIER_1_NATIVE_48K, CLASS_OVERSAMPLE_FACTORS)
        assert weight_drone == pytest.approx(1.00, rel=1e-3)

        # SHAReD explosion blast is Tier 1 (1.00) AND scarce class (4.0) -> 1.00 * 4.0 = 4.00
        weight_blast = compute_sample_weight("explosion_blast", SyncTier.TIER_1_NATIVE_48K, CLASS_OVERSAMPLE_FACTORS)
        assert weight_blast == pytest.approx(4.00, rel=1e-3)


class TestClassifierTaxonomyMapping:
    """Verifies that Model 4 classifier categories map cleanly from UnifiedClass."""

    def test_gate_classes_list(self):
        assert GATE_CLASSES == ["harmonic", "impulsive", "speech_dominant"]

    def test_all_unified_classes_have_gate_mapping(self):
        # Ensure every member in UnifiedClass maps to a valid gate class
        for u in UnifiedClass:
            if u.value in ("rir", "far_end_echo"):
                continue
            assert u.value in UNIFIED_TO_GATE_CLASS, f"Class {u.value} has no gate mapping"
            gate = UNIFIED_TO_GATE_CLASS[u.value]
            assert gate in GATE_CLASSES, f"Mapped class {gate} not in GATE_CLASSES"

        # Transient classes map to impulsive
        assert UNIFIED_TO_GATE_CLASS["explosion_blast"] == "impulsive"
        assert UNIFIED_TO_GATE_CLASS["gunshot_firearm"] == "impulsive"

        # Periodic vehicle classes map to harmonic
        assert UNIFIED_TO_GATE_CLASS["tank_tracked"] == "harmonic"
        assert UNIFIED_TO_GATE_CLASS["drone_uav"] == "harmonic"

        # Speech maps to speech_dominant
        assert UNIFIED_TO_GATE_CLASS["clean_speech"] == "speech_dominant"


class TestPerSampleJsonSidecarRoundTrip:
    """Verifies that mixer branch outputs produce valid JSON sidecars consumed by training."""

    def test_se_json_sidecar_contains_required_keys(self, tmp_path):
        from data_forge.mixer.speech_enhancement import SpeechEnhancementBranch
        import soundfile as sf

        branch = SpeechEnhancementBranch(output_dir=tmp_path)
        # Create dummy clean and noise
        dummy_clean = tmp_path / "clean_test.wav"
        dummy_noise_dir = tmp_path / "tank_tracked"
        dummy_noise_dir.mkdir(parents=True, exist_ok=True)
        dummy_noise = dummy_noise_dir / "noisex92_leopard.wav"

        sr = 48000
        silence = np.zeros(sr * 2, dtype=np.float32)
        sf.write(dummy_clean, silence, sr)
        sf.write(dummy_noise, silence, sr)

        records = branch.generate_mixtures(
            clean_files=[dummy_clean],
            noise_files=[dummy_noise],
            rir_files=[],
            num_mixtures=1,
        )

        assert len(records) == 1
        clip_id = records[0]["clip_id"]

        # Verify JSON sidecar was written
        json_file = tmp_path / f"{clip_id}.json"
        assert json_file.exists(), f"Sidecar {json_file} was not written"

        with open(json_file, "r", encoding="utf-8") as f:
            sidecar = json.load(f)

        assert sidecar["clip_id"] == clip_id
        assert sidecar["unified_class"] == "tank_tracked"
        assert sidecar["sync_tier"] == 3  # NOISEX is Tier 3
        assert "target_snr_db" in sidecar
        assert "measured_snr_db" in sidecar
