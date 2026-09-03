"""
Project AEGIS — Step 10: Train / Validation / Generalization-Test Split
Origin-aware and fingerprint-isolated dataset partitioning.
Enforces that generalization-only corpora (MUSAN, LibriSpeech, WHAM) are NEVER in train/val.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List
from data_forge.config import SPLITS_DIR
from .step7_metadata import ClipMetadata


class Splitter:
    """Step 10: Origin-aware and leak-free split generation."""

    def __init__(
        self,
        train_ratio: float = 0.80,
        val_ratio: float = 0.10,
        test_ratio: float = 0.10,
        splits_dir: Path = SPLITS_DIR,
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.splits_dir = Path(splits_dir)
        self.splits_dir.mkdir(parents=True, exist_ok=True)

    def split_clips(self, clips: List[ClipMetadata]) -> Dict[str, List[ClipMetadata]]:
        """
        Partitions clips into train, val, and test_generalization splits.
        """
        train_clips: List[ClipMetadata] = []
        val_clips: List[ClipMetadata] = []
        test_clips: List[ClipMetadata] = []

        for clip in clips:
            # Rule 1: Generalization-only corpora are assigned strictly to generalization test
            if clip.is_generalization_only:
                clip.split = "test_generalization"
                test_clips.append(clip)
                continue

            # Rule 2: Deterministic hash-based assignment using fingerprint or clip_id
            # Ensures duplicate or same-session clips map to the identical split
            hash_key = clip.fingerprint if clip.fingerprint else clip.clip_id
            hash_val = int(hashlib.md5(hash_key.encode()).hexdigest(), 16) % 10000 / 10000.0

            if hash_val < self.train_ratio:
                clip.split = "train"
                train_clips.append(clip)
            elif hash_val < (self.train_ratio + self.val_ratio):
                clip.split = "val"
                val_clips.append(clip)
            else:
                clip.split = "test_generalization"
                test_clips.append(clip)

        splits_dict = {
            "train": train_clips,
            "val": val_clips,
            "test_generalization": test_clips,
        }

        self.save_splits(splits_dict)
        return splits_dict

    def save_splits(self, splits: Dict[str, List[ClipMetadata]]) -> None:
        """Saves partition manifests to disk."""
        summary = {}
        for split_name, clip_list in splits.items():
            manifest_path = self.splits_dir / f"{split_name}_manifest.json"
            data = [c.to_dict() for c in clip_list]
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump({"split": split_name, "count": len(data), "clips": data}, f, indent=2)

            # Record class distribution
            class_counts: Dict[str, int] = {}
            for c in clip_list:
                class_counts[c.unified_class] = class_counts.get(c.unified_class, 0) + 1

            summary[split_name] = {
                "total_clips": len(clip_list),
                "total_duration_hours": round(sum(c.duration_sec for c in clip_list) / 3600.0, 3),
                "class_distribution": class_counts,
            }

        with open(self.splits_dir / "split_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
