"""
tests/test_audit_pass3_fixes.py — regression tests for the 2026-09-03
third audit pass:
  - mixer/speech_enhancement.py's sync_tier now resolves via
    DATASET_PROFILES instead of a crude "noisex" filename heuristic
  - the new gradual_unfreezing.py callback schedules correctly and is
    disabled where it should be (Models 3-4)
"""

import pytest
import torch
import torch.nn as nn
from data_forge.config import DATASET_PROFILES
from data_forge.mixer.speech_enhancement import resolve_source_dataset


def test_source_dataset_resolves_correctly_including_multi_word_keys():
    # These specifically exercise the longest-prefix-match fix -- several
    # DATASET_PROFILES keys contain underscores, which a naive
    # filename.split("_")[0] would get wrong.
    assert resolve_source_dataset("noisex92_leopard.wav") == "noisex92"
    assert resolve_source_dataset("mad_gunshot_001.wav") == "mad"
    assert resolve_source_dataset("gunshot_dryad_398398.wav") == "gunshot_dryad"
    assert resolve_source_dataset("drone_audioset_clip04.wav") == "drone_audioset"
    assert resolve_source_dataset("vctk_demand_p225_001.wav") == "vctk_demand"
    assert resolve_source_dataset("sirens_urban_042.wav") == "sirens_urban"
    assert resolve_source_dataset("openslr_rirs_ir017.wav") == "openslr_rirs"
    assert resolve_source_dataset("totally_unknown_file.wav") == "unknown"


def test_sync_tier_resolution_matches_the_earlier_mad_correction():
    # This is the specific bug the fix addresses: the old heuristic
    # (`"noisex" in filename`) would have silently tagged MAD as Tier 1
    # by accident (right answer, wrong reason) but had no way to ever
    # correctly identify a DIFFERENT band-limited source as Tier 3 --
    # it only special-cased NOISEX-92 by name. This test asserts the fix
    # reads the real, authoritative tier from DATASET_PROFILES instead.
    assert DATASET_PROFILES["mad"].default_sync_tier.value == 1
    assert DATASET_PROFILES["noisex92"].default_sync_tier.value == 3


def test_gradual_unfreeze_schedule_progresses_last_to_first():
    from training.callbacks.gradual_unfreezing import GradualUnfreezeConfig, unfrozen_groups_at_epoch

    config = GradualUnfreezeConfig(epochs_per_unfreeze_step=3)
    assert unfrozen_groups_at_epoch(config, 0) == ["df_decoder"]
    assert unfrozen_groups_at_epoch(config, 3) == ["df_decoder", "erb_decoder"]
    assert unfrozen_groups_at_epoch(config, 6) == ["df_decoder", "erb_decoder", "df_encoder"]
    assert unfrozen_groups_at_epoch(config, 9) == ["df_decoder", "erb_decoder", "df_encoder", "erb_encoder"]
    # Should hold steady, not error, once every group is unfrozen
    assert unfrozen_groups_at_epoch(config, 100) == ["df_decoder", "erb_decoder", "df_encoder", "erb_encoder"]


def test_gradual_unfreeze_disabled_returns_everything_trainable():
    from training.callbacks.gradual_unfreezing import GradualUnfreezeConfig, unfrozen_groups_at_epoch

    config = GradualUnfreezeConfig(enabled=False)
    assert unfrozen_groups_at_epoch(config, 0) == config.layer_groups_last_to_first


def test_gradual_unfreeze_correctly_disabled_per_model():
    from training.configs.se_primary_config import SePrimaryConfig
    from training.configs.se_escalation_config import SeEscalationConfig
    from training.configs.se_crosscheck_config import SeCrosscheckConfig
    from training.configs.classifier_config import ClassifierConfig

    assert SePrimaryConfig().gradual_unfreeze.enabled is True
    assert SeEscalationConfig().gradual_unfreeze.enabled is True
    assert SeCrosscheckConfig().gradual_unfreeze.enabled is False   # possible from-scratch run, see docstring
    assert ClassifierConfig().gradual_unfreeze.enabled is False     # trains from scratch, nothing to protect


def test_apply_freeze_schedule_updates_requires_grad():
    from training.callbacks.gradual_unfreezing import GradualUnfreezeConfig, apply_freeze_schedule

    p1 = nn.Parameter(torch.randn(2, 2))
    p2 = nn.Parameter(torch.randn(2, 2))
    p3 = nn.Parameter(torch.randn(2, 2))
    p4 = nn.Parameter(torch.randn(2, 2))

    layer_map = {
        "df_decoder": [p1],
        "erb_decoder": [p2],
        "df_encoder": [p3],
        "erb_encoder": [p4],
    }

    config = GradualUnfreezeConfig(epochs_per_unfreeze_step=2)

    # Epoch 0: only df_decoder unfrozen
    unfrozen = apply_freeze_schedule(None, config, epoch=0, layer_group_map=layer_map)
    assert unfrozen == ["df_decoder"]
    assert p1.requires_grad is True
    assert p2.requires_grad is False
    assert p3.requires_grad is False
    assert p4.requires_grad is False

    # Epoch 2: df_decoder and erb_decoder unfrozen
    unfrozen = apply_freeze_schedule(None, config, epoch=2, layer_group_map=layer_map)
    assert unfrozen == ["df_decoder", "erb_decoder"]
    assert p1.requires_grad is True
    assert p2.requires_grad is True
    assert p3.requires_grad is False
    assert p4.requires_grad is False
