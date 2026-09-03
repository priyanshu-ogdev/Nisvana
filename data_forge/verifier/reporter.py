"""
Project AEGIS — Audit Reporter
Generates markdown and JSON audit summaries of pipeline health and data-forge outputs.
"""

import json
from pathlib import Path
from typing import Optional
from data_forge.config import MANIFESTS_DIR
from .auditor import AuditSummary


class AuditReporter:
    """Formats and writes audit reports to disk."""

    @staticmethod
    def generate_markdown_report(summary: AuditSummary, output_path: Optional[Path] = None) -> str:
        output_path = output_path or (MANIFESTS_DIR / "pipeline_audit_report.md")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        md = f"""# Project AEGIS — Data-Forge Pipeline Audit Report

**Status**: {"✅ PASSED" if summary.audit_passed else "❌ FAILED"}

---

## 1. Corpus Inventory Summary

| Pipeline Stage | Total Audio Files | Notes |
|---|---|---|
| **Raw Datasets** (`data/raw/`) | **{summary.total_raw_files}** | Original downloaded corpora across verified bibliography |
| **Standardized Audio** (`data/processed/`) | **{summary.total_processed_files}** | 48 kHz, -23 LUFS, VAD trimmed, mono standardized |
| **Grounded Augmentations** (`data/augmented/`) | **{summary.total_augmented_files}** | Bounded time-stretch (+/- 5-10%), gain jitter; zero pitch-shift |

---

## 2. Multi-Branch Model Training Corpora (`data/forge/`)

| Target Model Branch | Output Samples | Specifications |
|---|---|---|
| **Branch 1: Speech Enhancement** (Models 1-3) | **{summary.total_forge_se_samples}** triplets | `(noisy, clean, rir)` at uniform SNR [-5 dB to +20 dB] |
| **Branch 2: SNR / Harmonic Classifier** (Model 4) | **{summary.total_forge_classifier_samples}** samples | 3-way taxonomy (`stationary_harmonic`, `non_stationary_transient`, `speech_dominant`) |
| **Branch 3: Gated AEC** (Model 5) | **{summary.total_forge_aec_samples}** quadruplets | `(mic, farend, nearend, echo)` isolated from external noise |

---

## 3. Scientific Grounding & Invariant Checks

- **Sample Rate Standardization (48,000 Hz)**: {summary.sample_rate_compliance_rate}% compliant.
- **Zero Pitch-Shift Invariant**: {"✅ VERIFIED (No formant/RPM corruption)" if summary.zero_pitch_shift_verified else "❌ FAILED (Pitch shift detected)"}
- **Split Contamination & Leakage**: {"✅ ZERO LEAKAGE (Partition isolation strictly enforced)" if not summary.split_leakage_detected else "❌ LEAKAGE DETECTED"}

---

## 4. Issues & Warnings

"""
        if summary.issues:
            for issue in summary.issues:
                md += f"- ⚠️ {issue}\n"
        else:
            md += "- ✅ No pipeline issues detected. All audio and manifests conform to specifications.\n"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)

        # Also write JSON summary
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "audit_passed": summary.audit_passed,
                "total_raw_files": summary.total_raw_files,
                "total_processed_files": summary.total_processed_files,
                "total_augmented_files": summary.total_augmented_files,
                "total_forge_se_samples": summary.total_forge_se_samples,
                "total_forge_classifier_samples": summary.total_forge_classifier_samples,
                "total_forge_aec_samples": summary.total_forge_aec_samples,
                "sample_rate_compliance_rate": summary.sample_rate_compliance_rate,
                "split_leakage_detected": summary.split_leakage_detected,
                "zero_pitch_shift_verified": summary.zero_pitch_shift_verified,
                "issues": summary.issues,
            }, f, indent=2)

        return md
