"""
Project AEGIS — Step 7: Metadata Tagging
Attaches unified_class, sync_tier, license, source_dataset, and augmentation_policy to every clip.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional
from data_forge.config import (
    AugmentationPolicy,
    DatasetProfile,
    SyncTier,
    UnifiedClass,
    DATASET_PROFILES,
)


@dataclass
class ClipMetadata:
    clip_id: str
    filename: str
    source_dataset: str
    unified_class: str
    sync_tier: int
    license: str
    augmentation_policy: str
    duration_sec: float
    sample_rate: int
    channels: int
    measured_lufs: float
    true_peak_dbfs: float
    is_non_commercial: bool
    is_generalization_only: bool
    fingerprint: Optional[str] = None
    split: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetadataTagger:
    """Step 7: Metadata enrichment and taxonomy attribution."""

    @staticmethod
    def create_metadata(
        clip_id: str,
        filename: str,
        source_dataset: str,
        unified_class: UnifiedClass,
        sync_tier: SyncTier,
        duration_sec: float,
        sample_rate: int,
        channels: int,
        measured_lufs: float,
        true_peak_dbfs: float,
        fingerprint: Optional[str] = None,
    ) -> ClipMetadata:
        profile = DATASET_PROFILES.get(source_dataset)
        license_str = profile.license if profile else "Unknown"
        policy = profile.augmentation_policy.value if profile else AugmentationPolicy.NONE.value
        is_nc = profile.is_non_commercial if profile else False
        is_gen = profile.is_generalization_only if profile else False

        return ClipMetadata(
            clip_id=clip_id,
            filename=filename,
            source_dataset=source_dataset,
            unified_class=unified_class.value if isinstance(unified_class, UnifiedClass) else str(unified_class),
            sync_tier=int(sync_tier),
            license=license_str,
            augmentation_policy=policy,
            duration_sec=round(duration_sec, 3),
            sample_rate=sample_rate,
            channels=channels,
            measured_lufs=round(measured_lufs, 2),
            true_peak_dbfs=round(true_peak_dbfs, 2),
            is_non_commercial=is_nc,
            is_generalization_only=is_gen,
            fingerprint=fingerprint,
        )
