"""
Project AEGIS — Speech Enhancement Branch (Models 1-3)
Target Architectures: DeepFilterNet3 (x2), CleanUMamba
Generates synchronized triplets: (noisy_mixture.wav, clean_target.wav, reverberant_target.wav)
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import soundfile as sf
from data_forge.config import BRANCH_SE, TARGET_SAMPLE_RATE, ForgeMixingConfig, DATASET_PROFILES
from .engine import SnrMixerEngine

logger = logging.getLogger("DataForge.BranchSE")


def resolve_source_dataset(filename: str) -> str:
    """
    Recovers the source_dataset key (matching DATASET_PROFILES) from a
    preprocessed filename of the form '{source_dataset}_{original_stem}.wav'
    -- this naming convention is set by preprocessor/pipeline.py's
    `clip_id = f"{source_dataset}_{input_path.stem}"` and is the ONLY
    reliable way to recover it post-preprocessing, since files are moved
    into unified_class-named directories that don't preserve the raw
    source path substrings (e.g. "noisex") that a naive filename/path
    heuristic would otherwise look for.

    VERIFIED 2026-09-03: several DATASET_PROFILES keys themselves contain
    underscores (e.g. "gunshot_dryad", "drone_audioset", "vctk_demand"),
    so a naive filename.split("_")[0] breaks on those. Longest-prefix
    match against the actual known keys handles this correctly.
    """
    candidates = [k for k in DATASET_PROFILES.keys() if filename.startswith(k + "_")]
    if not candidates:
        return "unknown"
    return max(candidates, key=len)


class SpeechEnhancementBranch:
    """Generates speech enhancement training corpus for Models 1-3."""

    def __init__(self, output_dir: Path = BRANCH_SE, config: Optional[ForgeMixingConfig] = None):
        self.output_dir = Path(output_dir)
        self.config = config or ForgeMixingConfig()
        self.mixer = SnrMixerEngine(sample_rate=TARGET_SAMPLE_RATE)

        # Output subdirectories
        self.noisy_dir = self.output_dir / "noisy"
        self.clean_dir = self.output_dir / "clean"
        self.rir_dir = self.output_dir / "rir"

        for d in (self.noisy_dir, self.clean_dir, self.rir_dir):
            d.mkdir(parents=True, exist_ok=True)

    def generate_mixtures(
        self,
        clean_files: List[Path],
        noise_files: List[Path],
        rir_files: List[Path],
        num_mixtures: int = 100,
        split: str = "train",
    ) -> List[Dict[str, Any]]:
        """
        Generates paired training samples for Models 1-3.
        """
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

        if not clean_files:
            raise ValueError("No clean speech files provided for speech enhancement branch.")
        if not noise_files:
            raise ValueError("No noise files provided for speech enhancement branch.")

        logger.info("Generating %d speech enhancement triplets for split '%s'...", num_mixtures, split)
        records = []

        for idx in range(num_mixtures):
            clean_p = random.choice(clean_files)
            noise_p = random.choice(noise_files)
            rir_p = random.choice(rir_files) if rir_files and (random.random() < self.config.rir_probability) else None

            # Read clean and noise
            clean_audio, _ = sf.read(clean_p, dtype="float32")
            noise_audio, _ = sf.read(noise_p, dtype="float32")
            rir_audio, _ = sf.read(rir_p, dtype="float32") if rir_p else (None, None)

            clean_audio = np.asarray(clean_audio, dtype=np.float32)
            if clean_audio.ndim > 1:
                clean_audio = np.mean(clean_audio, axis=1)

            noise_audio = np.asarray(noise_audio, dtype=np.float32)
            if noise_audio.ndim > 1:
                noise_audio = np.mean(noise_audio, axis=1)

            if rir_audio is not None:
                rir_audio = np.asarray(rir_audio, dtype=np.float32)
                if rir_audio.ndim > 1:
                    rir_audio = np.mean(rir_audio, axis=1)

            # Target slice duration for consistent training tensor dimensions
            if self.config.target_duration_sec > 0:
                target_samples = int(self.config.target_duration_sec * TARGET_SAMPLE_RATE)
                if len(clean_audio) > target_samples:
                    max_start = len(clean_audio) - target_samples
                    start_idx = random.randint(0, max_start)
                    clean_audio = clean_audio[start_idx : start_idx + target_samples]

            # Sample random SNR from uniform range [-5 dB, +20 dB]
            target_snr = round(random.uniform(self.config.min_snr_db, self.config.max_snr_db), 1)

            # Mix
            res = self.mixer.mix_signals(clean_audio, noise_audio, target_snr_db=target_snr, rir=rir_audio)

            clip_id = f"se_{split}_{idx:05d}"
            noisy_file = self.noisy_dir / f"{clip_id}_noisy.wav"
            clean_file = self.clean_dir / f"{clip_id}_clean.wav"
            rir_file = self.rir_dir / f"{clip_id}_rir.wav"

            # Write WAVs
            sf.write(noisy_file, res.noisy_audio, TARGET_SAMPLE_RATE, subtype="PCM_16")
            sf.write(clean_file, res.clean_target, TARGET_SAMPLE_RATE, subtype="PCM_16")
            sf.write(rir_file, res.reverberant_target, TARGET_SAMPLE_RATE, subtype="PCM_16")

            # Determine acoustic class from the directory it was sorted into
            # at preprocessing time (dest_dir = processed_dir/unified_class.value --
            # this part was already correct) and recover the true sync_tier
            # from DATASET_PROFILES via the source-dataset-prefixed filename,
            # NOT the previous "noisex" substring heuristic (confirmed wrong:
            # it silently tagged every non-NOISEX-92 source as Tier 1,
            # including genuinely band-limited ones, and never consulted the
            # authoritative DATASET_PROFILES this project already corrected
            # MAD's tier in).
            noise_class = noise_p.parent.name
            if noise_class in ("processed", "augmented", "audio", "wavs"):
                noise_class = "general_noise"

            source_dataset = resolve_source_dataset(noise_p.name)
            profile = DATASET_PROFILES.get(source_dataset)
            sync_tier = profile.default_sync_tier.value if profile is not None else 1
            if profile is None:
                logger.warning(
                    "Could not resolve source_dataset for noise file '%s' -- "
                    "defaulting sync_tier=1 (native). If this fires frequently, "
                    "check that preprocessing's clip_id naming convention hasn't changed.",
                    noise_p.name,
                )

            record = {
                "clip_id": clip_id,
                "split": split,
                "target_snr_db": target_snr,
                "measured_snr_db": res.measured_snr_db,
                "clean_source": clean_p.name,
                "noise_source": noise_p.name,
                "unified_class": noise_class,
                "sync_tier": sync_tier,
                "rir_applied": res.rir_applied,
                "rir_source": rir_p.name if rir_p else None,
                "duration_sec": round(len(res.noisy_audio) / TARGET_SAMPLE_RATE, 2),
                "noisy_path": str(noisy_file),
                "clean_path": str(clean_file),
                "rir_path": str(rir_file),
            }
            records.append(record)

            # Write per-sample JSON sidecar for WebDataset sharding & weighted sampling
            json_file = self.output_dir / f"{clip_id}.json"
            with open(json_file, "w", encoding="utf-8") as jf:
                json.dump(record, jf, indent=2)

        # Save manifest (accumulative across splits)
        manifest_path = self.output_dir / "manifest.json"
        all_samples = list(records)
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    prev = json.load(f).get("samples", [])
                    existing_ids = {r["clip_id"] for r in records}
                    all_samples = [p for p in prev if p.get("clip_id") not in existing_ids] + records
            except Exception:
                pass

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"branch": "speech_enhancement", "total_samples": len(all_samples), "samples": all_samples}, f, indent=2)

        logger.info("Generated %d speech enhancement samples in %s", len(records), self.output_dir)
        return records
