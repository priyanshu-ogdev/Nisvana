"""
Project AEGIS — Gunfire Data Audit

Replaces the stale, uncomputed estimate in
training/configs/se_primary_config.py ("~15-20h combined across
NIJ/Kabealo/Cooper&Shaw/MAD/FSD50K -- abundant") with a real number.

VERIFIED (this review pass): of the five sources named in that comment,
only THREE have any fetcher implementation in data_forge/fetcher/ at all:
  - "gunshot_dryad" (Cooper & Shaw)      -- data_forge/fetcher/gunshot.py
  - "mad" (gunshot-mapped subset only)   -- data_forge/fetcher/mad.py
  - "noisex92" (machinegun.wav ONLY)     -- data_forge/fetcher/noisex.py
NIJ (Cadre Research Labs), Kabealo et al., and FSD50K are cited in
docs/BIBLIOGRAPHY.md and data/shards/DATASET_CARD.md but have zero fetcher
code anywhere in this repo -- they contribute zero real hours today,
regardless of what the stale comment implies. This script measures what
actually landed on disk from the three real sources, nothing else.

Run: python -m data_forge.verifier.gunfire_audit
"""

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List
import json

import soundfile as sf

from data_forge.config import (
    RAW_DIR,
    PROCESSED_DIR,
    UnifiedClass,
    DATASET_PROFILES,
)

logger = logging.getLogger("DataForge.GunfireAudit")

# The only three DATASET_PROFILES keys that actually contribute
# UnifiedClass.GUNSHOT_FIREARM clips via a real, implemented fetcher.
# Kept as an explicit allow-list (not derived from DATASET_PROFILES'
# unified_classes field alone) so this audit fails loudly, not silently,
# if a profile is added/renamed without updating this list.
REAL_GUNFIRE_SOURCES: List[str] = ["gunshot_dryad", "mad", "noisex92"]


@dataclass
class SourceAuditResult:
    source_key: str
    source_name: str
    clip_count: int
    total_duration_sec: float
    unreadable_files: List[str]


@dataclass
class GunfireAuditSummary:
    per_source: List[SourceAuditResult]
    total_clip_count: int
    total_duration_sec: float
    total_hours: float
    cited_but_unfetched_sources: List[str]
    stale_comment_claimed_hours: str
    verdict: str


def _iter_audio_files(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return list(directory.glob("**/*.wav")) + list(directory.glob("**/*.flac"))


def _duration_sec(path: Path) -> float:
    info = sf.info(path)
    return info.frames / float(info.samplerate)


def audit_source(source_key: str) -> SourceAuditResult:
    """
    Audits one real gunfire source. Looks in RAW_DIR/<source_key>/ first
    (fetcher output layout), and cross-checks PROCESSED_DIR/gunshot_firearm/
    for clips whose filename is prefixed with this source_key (preprocessor
    output layout, per mixer/speech_enhancement.py's resolve_source_dataset
    convention: '{source_dataset}_{original_stem}.wav').
    """
    profile = DATASET_PROFILES.get(source_key)
    source_name = profile.name if profile else source_key

    candidates: List[Path] = []
    candidates += _iter_audio_files(RAW_DIR / source_key)

    processed_gunfire_dir = PROCESSED_DIR / UnifiedClass.GUNSHOT_FIREARM.value
    if processed_gunfire_dir.exists():
        candidates += [
            p for p in _iter_audio_files(processed_gunfire_dir)
            if p.name.startswith(source_key + "_")
        ]

    # De-duplicate (a file could theoretically be matched by both globs if
    # RAW_DIR and PROCESSED_DIR overlap in an unusual layout).
    seen = set()
    unique_candidates = []
    for p in candidates:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            unique_candidates.append(p)

    total_dur = 0.0
    unreadable: List[str] = []
    for p in unique_candidates:
        try:
            total_dur += _duration_sec(p)
        except Exception as e:
            unreadable.append(f"{p.name}: {e}")

    return SourceAuditResult(
        source_key=source_key,
        source_name=source_name,
        clip_count=len(unique_candidates),
        total_duration_sec=round(total_dur, 2),
        unreadable_files=unreadable,
    )


def run_gunfire_audit() -> GunfireAuditSummary:
    logger.info("=== GUNFIRE DATA AUDIT ===")
    per_source = [audit_source(k) for k in REAL_GUNFIRE_SOURCES]

    total_clips = sum(r.clip_count for r in per_source)
    total_dur = sum(r.total_duration_sec for r in per_source)
    total_hours = round(total_dur / 3600.0, 3)

    cited_unfetched = ["NIJ (Cadre Research Labs)", "Kabealo et al. 2023", "FSD50K"]

    for r in per_source:
        logger.info(
            "%s (%s): %d clips, %.1f min",
            r.source_name, r.source_key, r.clip_count, r.total_duration_sec / 60.0,
        )
        if r.unreadable_files:
            logger.warning("  %d unreadable files in %s: %s", len(r.unreadable_files), r.source_key, r.unreadable_files)

    if total_clips == 0:
        verdict = (
            "ZERO real gunfire clips found on disk for the three implemented "
            "sources. Either data hasn't been fetched yet (run data_forge's "
            "fetch step for gunshot_dryad/mad/noisex92 first) or RAW_DIR/"
            "PROCESSED_DIR point somewhere unexpected -- check DATA_FORGE_RAW_DIR "
            "env var. Do NOT trust the '~15-20h abundant' comment in "
            "se_primary_config.py until this audit returns a real number."
        )
    elif total_hours < 2.0:
        verdict = (
            f"Real gunfire coverage is THIN ({total_hours}h across {total_clips} "
            f"clips from 3 real sources) -- materially less than the stale "
            f"'~15-20h abundant' estimate, which counted 2 sources (NIJ, "
            f"Kabealo) plus FSD50K that have no fetcher and contribute nothing. "
            f"Recommend: keep gunshot_firearm/gunfire oversample factor boosted "
            f"(not 1.0), keep it on the watched-weak-class regression list, and "
            f"lean harder on augmentation (RIR probability, SNR range, "
            f"multi-shot compositing) since raw hours won't grow without new "
            f"fetcher implementations."
        )
    else:
        verdict = (
            f"Real gunfire coverage: {total_hours}h across {total_clips} clips "
            f"from 3 real sources. Still narrower than the stale 5-source "
            f"estimate implied -- reassess the oversample factor against this "
            f"number specifically, not the old comment."
        )

    return GunfireAuditSummary(
        per_source=per_source,
        total_clip_count=total_clips,
        total_duration_sec=round(total_dur, 2),
        total_hours=total_hours,
        cited_but_unfetched_sources=cited_unfetched,
        stale_comment_claimed_hours="~15-20h combined across NIJ/Kabealo/Cooper&Shaw/MAD/FSD50K",
        verdict=verdict,
    )


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = run_gunfire_audit()
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
