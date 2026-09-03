"""
training/data/weighted_shard_sampler.py

Wires a model config's sync_tier + class_oversample_factors into an actual
sampling weight per shard sample, on top of the data_forge.exporter
IterableDataset classes already built and verified in the data-forge pass.

This is the ONE place sync-tier down-weighting and class-oversampling
combine -- deliberately not duplicated per-model, so a change to the
down-weighting philosophy (e.g. if a Tier 3 source gets re-verified as
actually full-band, per the MAD correction earlier in this review) only
needs to change in configs/base_config.py, not five times.
"""

import json
from typing import Any, Dict, Optional

from data_forge.exporter import AegisSpeechEnhancementIterableDataset
from training.configs.base_config import SYNC_TIER_SAMPLE_WEIGHT, SyncTier


def compute_sample_weight(
    unified_class: str,
    sync_tier: Any,
    class_oversample_factors: Dict[str, float],
) -> float:
    """
    Combines the two independent weighting axes established across this
    review:
      1. sync_tier weight -- penalizes band-limited (16kHz-native) sources
         so they don't dominate what the model learns "full-band" means.
      2. class_oversample_factor -- compensates for real-hour scarcity on
         thin defence-specific classes (armored vehicle, explosion, siren,
         wind) without inventing data (no synthetic fallback, per the
         explicit removal of synthetic/procedural data from this design).

    These multiply rather than pick-one, because they address different
    problems: a NOISEX-92 clip is BOTH Tier 3 (16kHz-native) AND belongs to
    the thinnest class (armored_vehicle_naval_jet) -- it needs both the
    down-weight (don't let its band-limiting dominate) and the oversample
    (don't let its scarcity starve the class) applied together.
    """
    tier_weight = SYNC_TIER_SAMPLE_WEIGHT.get(sync_tier)
    if tier_weight is None:
        try:
            tier_weight = SYNC_TIER_SAMPLE_WEIGHT.get(SyncTier(sync_tier), 1.0)
        except Exception:
            tier_weight = 1.0

    class_weight = class_oversample_factors.get(unified_class, 1.0)
    return tier_weight * class_weight


def build_weighted_se_dataset(
    shard_dir,
    split: str,
    class_oversample_factors: Dict[str, float],
):
    """
    Returns an AegisSpeechEnhancementIterableDataset with per-sample
    weights attached via its manifest 'json' sidecar. Gracefully returns
    None if webdataset is not installed.
    """
    try:
        dataset = AegisSpeechEnhancementIterableDataset(shard_dir, split=split)
    except ImportError:
        return None

    def _weight_fn(sample: dict) -> float:
        meta = sample.get("json", {})
        if isinstance(meta, (bytes, bytearray)):
            try:
                meta = json.loads(meta.decode("utf-8"))
            except Exception:
                meta = {}
        elif not isinstance(meta, dict):
            meta = {}

        unified_class = meta.get("unified_class", "general_noise")
        sync_tier_raw = meta.get("sync_tier", SyncTier.TIER_1_NATIVE_48K)
        return compute_sample_weight(unified_class, sync_tier_raw, class_oversample_factors)

    dataset.sample_weight_fn = _weight_fn  # consumed by the trainer's sampler construction
    return dataset
