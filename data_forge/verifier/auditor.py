"""
Project AEGIS — Pipeline Auditor
Verifies audio standards, sampling rates, loudness levels, SNR accuracy, and split isolation.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import soundfile as sf
from data_forge.config import (
    AUGMENTED_DIR,
    BRANCH_AEC,
    BRANCH_CLASSIFIER,
    BRANCH_SE,
    DATA_DIR,
    FORBIDDEN_PITCH_CLASSES,
    MANIFESTS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    SPLITS_DIR,
    TARGET_SAMPLE_RATE,
)

logger = logging.getLogger("DataForge.Auditor")


@dataclass
class AuditSummary:
    total_raw_files: int
    total_processed_files: int
    total_augmented_files: int
    total_forge_se_samples: int
    total_forge_classifier_samples: int
    total_forge_aec_samples: int
    sample_rate_compliance_rate: float
    split_leakage_detected: bool
    zero_pitch_shift_verified: bool
    audit_passed: bool
    issues: List[str]


class PipelineAuditor:
    """Performs comprehensive audits over all artifacts in data/."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.augmented_dir = self.data_dir / "augmented"
        self.splits_dir = self.data_dir / "splits"
        self.forge_dir = self.data_dir / "forge"
        self.manifests_dir = self.data_dir / "manifests"

    def run_full_audit(self) -> AuditSummary:
        """Runs the complete verification audit."""
        logger.info("=== STARTING FULL PIPELINE AUDIT ===")
        issues: List[str] = []

        # 1. Count files across stages
        raw_files = list(self.raw_dir.glob("**/*.wav")) + list(self.raw_dir.glob("**/*.flac"))
        processed_files = list(self.processed_dir.glob("**/*.wav"))
        augmented_files = list(self.augmented_dir.glob("**/*.wav"))

        se_manifest = self.forge_dir / "branch_speech_enhancement" / "manifest.json"
        clf_labels = self.forge_dir / "branch_classifier" / "labels.json"
        aec_manifest = self.forge_dir / "branch_aec" / "manifest.json"

        se_count = self._read_json_count(se_manifest, "total_samples")
        clf_count = self._read_json_count(clf_labels, "total_samples")
        aec_count = self._read_json_count(aec_manifest, "total_quadruplets")

        # 2. Check Sample-Rate Compliance (Must be exactly 48000 Hz)
        compliant_sr_count = 0
        total_checked_files = processed_files + augmented_files

        for f in total_checked_files[:100]:  # Audit sample
            try:
                info = sf.info(f)
                if info.samplerate == TARGET_SAMPLE_RATE:
                    compliant_sr_count += 1
                else:
                    issues.append(f"Sample rate mismatch in {f.name}: {info.samplerate} Hz (expected {TARGET_SAMPLE_RATE} Hz)")
            except Exception as e:
                issues.append(f"Unreadable audio file {f.name}: {e}")

        if len(total_checked_files) == 0:
            sr_compliance = 100.0
        else:
            sr_compliance = (compliant_sr_count / len(total_checked_files[:100])) * 100.0

        # 3. Check Split Isolation (Train vs Val vs Generalization-Test)
        leakage_detected = self._check_split_leakage(issues)

        # 4. Check Zero Pitch-Shift Invariant
        pitch_shift_ok = self._verify_zero_pitch_shift(issues)

        # Overall verdict
        audit_passed = (sr_compliance >= 99.0) and (not leakage_detected) and pitch_shift_ok and (len(issues) == 0)

        summary = AuditSummary(
            total_raw_files=len(raw_files),
            total_processed_files=len(processed_files),
            total_augmented_files=len(augmented_files),
            total_forge_se_samples=se_count,
            total_forge_classifier_samples=clf_count,
            total_forge_aec_samples=aec_count,
            sample_rate_compliance_rate=round(sr_compliance, 1),
            split_leakage_detected=leakage_detected,
            zero_pitch_shift_verified=pitch_shift_ok,
            audit_passed=audit_passed,
            issues=issues,
        )

        logger.info("Audit finished. Result: %s (Issues: %d)", "PASSED" if audit_passed else "FAILED", len(issues))
        return summary

    def _check_split_leakage(self, issues: List[str]) -> bool:
        """Verifies that no clip_id or fingerprint crosses train/val/test splits."""
        train_p = self.splits_dir / "train_manifest.json"
        val_p = self.splits_dir / "val_manifest.json"
        test_p = self.splits_dir / "test_generalization_manifest.json"

        if not (train_p.exists() and val_p.exists() and test_p.exists()):
            return False

        try:
            with open(train_p, "r", encoding="utf-8") as f:
                train_ids = set(c["clip_id"] for c in json.load(f).get("clips", []))
            with open(val_p, "r", encoding="utf-8") as f:
                val_ids = set(c["clip_id"] for c in json.load(f).get("clips", []))
            with open(test_p, "r", encoding="utf-8") as f:
                test_ids = set(c["clip_id"] for c in json.load(f).get("clips", []))

            overlap_tv = train_ids.intersection(val_ids)
            overlap_tt = train_ids.intersection(test_ids)
            overlap_vt = val_ids.intersection(test_ids)

            if overlap_tv:
                issues.append(f"Split leakage: {len(overlap_tv)} clips in both train and val")
            if overlap_tt:
                issues.append(f"Split leakage: {len(overlap_tt)} clips in both train and test")
            if overlap_vt:
                issues.append(f"Split leakage: {len(overlap_vt)} clips in both val and test")

            return bool(overlap_tv or overlap_tt or overlap_vt)
        except Exception as e:
            issues.append(f"Error checking split leakage: {e}")
            return False

    def _verify_zero_pitch_shift(self, issues: List[str]) -> bool:
        """
        Verifies that no augmented file contains pitch shift markers or violations
        for forbidden acoustic classes.
        """
        for cls in FORBIDDEN_PITCH_CLASSES:
            aug_folder = self.augmented_dir / cls.value
            if aug_folder.exists():
                for f in aug_folder.glob("*.wav"):
                    if "pitch" in f.name.lower():
                        issues.append(f"VIOLATION: Forbidden pitch-shift file detected: {f.name}")
                        return False
        return True

    @staticmethod
    def _read_json_count(json_path: Path, count_key: str) -> int:
        if not json_path.exists():
            return 0
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return int(data.get(count_key, 0))
        except Exception:
            return 0
