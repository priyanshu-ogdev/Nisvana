"""
Project AEGIS — 10-Step Sequential Preprocessing Pipeline Orchestrator
Executes: format -> rate -> loudness -> VAD/trim -> channel -> integrity -> metadata -> dedup -> license -> split
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import soundfile as sf
from data_forge.config import (
    DATASET_PROFILES,
    MANIFESTS_DIR,
    PROCESSED_DIR,
    RAW_DIR,
    TARGET_SAMPLE_RATE,
    SyncTier,
    UnifiedClass,
)
from .step1_format import FormatStandardizer
from .step2_resample import Resampler
from .step3_loudness import LoudnessNormalizer
from .step4_vad import VadValidator
from .step5_channel import ChannelStandardizer
from .step6_integrity import IntegrityChecker, IntegrityReport
from .step7_metadata import ClipMetadata, MetadataTagger
from .step8_dedup import Deduplicator
from .step9_license import LicenseComplianceMode, LicenseFilter
from .step10_split import Splitter

logger = logging.getLogger("DataForge.Preprocessor")


class PreprocessingPipeline:
    """Orchestrates the complete 10-step preprocessing sequence."""

    def __init__(
        self,
        raw_dir: Path = RAW_DIR,
        processed_dir: Path = PROCESSED_DIR,
        manifests_dir: Path = MANIFESTS_DIR,
        license_mode: LicenseComplianceMode = LicenseComplianceMode.RESEARCH_PROTOTYPE,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.manifests_dir = Path(manifests_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sequential step processors
        self.format_standardizer = FormatStandardizer()
        self.resampler = Resampler(TARGET_SAMPLE_RATE)
        self.loudness_normalizer = LoudnessNormalizer()
        self.vad_validator = VadValidator()
        self.channel_standardizer = ChannelStandardizer(mode="mono_mean")
        self.integrity_checker = IntegrityChecker()
        self.metadata_tagger = MetadataTagger()
        self.deduplicator = Deduplicator()
        self.license_filter = LicenseFilter(mode=license_mode)
        self.splitter = Splitter()

    def process_single_file(
        self,
        input_path: Path,
        source_dataset: str,
        unified_class: UnifiedClass,
        output_subfolder: Optional[str] = None,
    ) -> Tuple[bool, Optional[ClipMetadata], Optional[str]]:
        """
        Executes steps 1 through 7 on a single audio file.
        Returns: (success, clip_metadata, rejection_reason)
        """
        try:
            # Step 1: Format standardization
            audio_raw, orig_sr = self.format_standardizer.process(input_path)

            # Step 2: Sample-rate standardization (polyphase resample to 48kHz + sync tier)
            audio_48k, sr, sync_tier = self.resampler.process(audio_raw, orig_sr)

            # Step 5: Channel standardization (mono downmix)
            audio_mono, channels = self.channel_standardizer.process(audio_48k)

            # Step 4: Silence and VAD validation (Part 3, Step 4: over clean-speech sources)
            if unified_class == UnifiedClass.CLEAN_SPEECH:
                vad_valid, audio_trimmed, vad_msg = self.vad_validator.process(audio_mono)
                if not vad_valid:
                    return False, None, f"VAD rejected: {vad_msg}"
            else:
                # Environmental noise, transients (gunshots/explosions), RIRs, and AEC retain physical waveforms
                audio_trimmed = audio_mono

            # Step 6: Integrity & corruption check
            integrity = self.integrity_checker.process(audio_trimmed)
            if not integrity.is_clean:
                return False, None, f"Integrity check failed: {integrity.rejection_reason}"

            # Step 3: Loudness normalization (-23 LUFS + peak limiter)
            # For RIRs, retain unit impulse energy without loudness normalization
            if unified_class != UnifiedClass.ROOM_IMPULSE_RESPONSE:
                audio_norm, measured_lufs, peak_dbfs = self.loudness_normalizer.process(audio_trimmed)
            else:
                audio_norm = audio_trimmed
                measured_lufs = -23.0
                peak_dbfs = 0.0

            # Step 8: Acoustic fingerprint calculation
            fingerprint = self.deduplicator.compute_fingerprint(audio_norm, sr)
            clip_id = f"{source_dataset}_{input_path.stem}"

            # Step 7: Metadata tagging
            duration_sec = len(audio_norm) / sr
            meta = self.metadata_tagger.create_metadata(
                clip_id=clip_id,
                filename=f"{clip_id}.wav",
                source_dataset=source_dataset,
                unified_class=unified_class,
                sync_tier=sync_tier,
                duration_sec=duration_sec,
                sample_rate=sr,
                channels=channels,
                measured_lufs=measured_lufs,
                true_peak_dbfs=peak_dbfs,
                fingerprint=fingerprint,
            )

            # Save processed audio to disk
            dest_dir = self.processed_dir / (output_subfolder if output_subfolder else unified_class.value)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / f"{clip_id}.wav"
            sf.write(dest_file, audio_norm, sr, subtype="PCM_16", format="WAV")

            return True, meta, None

        except Exception as e:
            return False, None, f"Exception during processing: {str(e)}"

    def run_pipeline(self, max_workers: int = 4) -> Dict[str, Any]:
        """
        Runs the full 10-step preprocessing pipeline on all raw audio files.
        """
        logger.info("=== PROJECT AEGIS: 10-STEP PREPROCESSING PIPELINE START ===")
        all_metadata: List[ClipMetadata] = []
        rejected_log: List[Dict[str, str]] = []

        # Discover all audio files in RAW_DIR
        raw_files = list(self.raw_dir.glob("**/*.wav")) + list(self.raw_dir.glob("**/*.flac"))
        logger.info("Discovered %d raw audio files across corpora", len(raw_files))

        for file_path in raw_files:
            # Map path to dataset and unified class
            source_dataset, unified_class = self.infer_dataset_and_class(file_path)
            success, meta, reason = self.process_single_file(
                input_path=file_path,
                source_dataset=source_dataset,
                unified_class=unified_class,
            )

            if success and meta:
                # Step 8 check: deduplication registry
                is_dup, orig_id = self.deduplicator.register_and_check(meta.clip_id, meta.fingerprint or "")
                if is_dup:
                    logger.info("Clip %s is acoustic duplicate of %s", meta.clip_id, orig_id)
                all_metadata.append(meta)
            else:
                rejected_log.append({"file": str(file_path), "reason": reason or "Unknown"})

        # Step 9: License filter
        retained_clips, excluded_clips = self.license_filter.filter_manifest(all_metadata)
        logger.info("License filtering: %d retained, %d excluded", len(retained_clips), len(excluded_clips))

        # Step 10: Origin-aware and leak-free split partitioning
        splits = self.splitter.split_clips(retained_clips)
        logger.info(
            "Split complete: Train=%d, Val=%d, Generalization-Test=%d",
            len(splits["train"]),
            len(splits["val"]),
            len(splits["test_generalization"]),
        )

        # Write unified master manifest
        master_manifest_path = self.manifests_dir / "unified_master_manifest.json"
        with open(master_manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_processed": len(retained_clips),
                    "total_rejected": len(rejected_log),
                    "clips": [c.to_dict() for c in retained_clips],
                    "rejections": rejected_log[:100],  # sample rejections
                },
                f,
                indent=2,
            )

        logger.info("Unified master manifest saved to %s", master_manifest_path)
        return {
            "processed_count": len(retained_clips),
            "rejected_count": len(rejected_log),
            "splits": {k: len(v) for k, v in splits.items()},
        }

    @staticmethod
    def infer_dataset_and_class(file_path: Path) -> Tuple[str, UnifiedClass]:
        """Infers source dataset and unified class from file path and name."""
        path_str = str(file_path).lower()
        name = file_path.name.lower()

        if "noisex" in path_str:
            if "leopard" in name:
                return "noisex92", UnifiedClass.TANK_TRACKED
            if "m109" in name:
                return "noisex92", UnifiedClass.ARTILLERY_HOWITZER
            if "f16" in name or "buccaneer" in name:
                return "noisex92", UnifiedClass.JET_COCKPIT
            if "destroyer" in name:
                return "noisex92", UnifiedClass.NAVAL_DESTROYER
            if "machinegun" in name:
                return "noisex92", UnifiedClass.GUNSHOT_FIREARM
            return "noisex92", UnifiedClass.GENERAL_NOISE

        if "shared" in path_str:
            return "shared", UnifiedClass.EXPLOSION_BLAST

        if "gunshot" in path_str:
            return "gunshot_dryad", UnifiedClass.GUNSHOT_FIREARM

        if "drone" in path_str:
            return "drone_audioset", UnifiedClass.DRONE_UAV

        if "siren" in path_str or "esc50" in path_str:
            if "42" in name:
                return "sirens_urban", UnifiedClass.SIREN_EMERGENCY
            if "39" in name:
                return "sirens_urban", UnifiedClass.WIND_ROTOR_GAP
            return "sirens_urban", UnifiedClass.GENERAL_NOISE

        if "aec" in path_str:
            return "aec_challenge", UnifiedClass.FAR_END_ECHO

        if "rir" in path_str:
            return "openslr_rirs", UnifiedClass.ROOM_IMPULSE_RESPONSE

        if "mad" in path_str:
            if "gun" in name or "shot" in name:
                return "mad", UnifiedClass.GUNSHOT_FIREARM
            if "shell" in name or "bomb" in name or "explo" in name:
                return "mad", UnifiedClass.EXPLOSION_BLAST
            return "mad", UnifiedClass.MILITARY_VEHICLE

        if "vctk" in path_str or "demand" in path_str:
            if "noisy" in path_str or "noise" in path_str or "demand" in path_str:
                return "vctk_demand", UnifiedClass.GENERAL_NOISE
            return "vctk_demand", UnifiedClass.CLEAN_SPEECH

        if "dns" in path_str:
            if "noise" in path_str or "noise" in name:
                return "dns_challenge", UnifiedClass.GENERAL_NOISE
            return "dns_challenge", UnifiedClass.CLEAN_SPEECH

        return "unknown", UnifiedClass.GENERAL_NOISE
