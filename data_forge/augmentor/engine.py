"""
Project AEGIS — Batch Augmentation Engine
Executes grounded per-source augmentations strictly in compliance with Part 1 and Part 2.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import soundfile as sf
from data_forge.config import (
    AUGMENTED_DIR,
    AugmentationPolicy,
    PROCESSED_DIR,
    TARGET_SAMPLE_RATE,
    UnifiedClass,
)
from .blast_window import BlastOnsetWindow
from .gain_jitter import GainJitter
from .policy import AugmentationPolicyEngine
from .time_stretch import WsolaTimeStretcher

logger = logging.getLogger("DataForge.Augmentor")


class AugmentationEngine:
    """Orchestrates per-source grounded augmentations."""

    def __init__(
        self,
        processed_dir: Path = PROCESSED_DIR,
        augmented_dir: Path = AUGMENTED_DIR,
        sample_rate: int = TARGET_SAMPLE_RATE,
    ):
        self.processed_dir = Path(processed_dir)
        self.augmented_dir = Path(augmented_dir)
        self.sample_rate = sample_rate
        self.augmented_dir.mkdir(parents=True, exist_ok=True)

        self.time_stretcher = WsolaTimeStretcher(sample_rate=sample_rate)
        self.gain_jitter = GainJitter()
        self.blast_windower = BlastOnsetWindow(sample_rate=sample_rate)

    def augment_clip(
        self,
        audio: np.ndarray,
        clip_id: str,
        unified_class: UnifiedClass,
        policy: AugmentationPolicy,
    ) -> List[Dict[str, Any]]:
        """
        Augments a single audio clip according to its grounded policy.
        Returns list of generated variations: [{"suffix": ..., "audio": ...}]
        """
        # Rule 1: Zero augmentation for clean speech targets and AEC
        if policy == AugmentationPolicy.NONE:
            return []

        # Rule 2: Minimal mixing variety only
        if policy == AugmentationPolicy.MINIMAL_SNR_ONLY:
            return []

        # Pitch-shift invariant assertion
        AugmentationPolicyEngine.assert_pitch_shift_permitted(unified_class)

        variations = []

        # Moderate No-Pitch Policy (Defence vehicles, NOISEX-92 Leopard/M109/F16/Destroyer)
        if policy == AugmentationPolicy.MODERATE_NO_PITCH:
            # 1. Faster time stretch (+6%)
            stretched_fast = self.time_stretcher.process(audio, rate=1.06)
            variations.append({"suffix": "stretch_106", "audio": stretched_fast})

            # 2. Slower time stretch (-6%)
            stretched_slow = self.time_stretcher.process(audio, rate=0.94)
            variations.append({"suffix": "stretch_094", "audio": stretched_slow})

            # 3. Positive level jitter (+1.8 dB)
            gain_up = self.gain_jitter.process(audio, jitter_db=1.8)
            variations.append({"suffix": "gain_p18", "audio": gain_up})

            # 4. Negative level jitter (-1.8 dB)
            gain_down = self.gain_jitter.process(audio, jitter_db=-1.8)
            variations.append({"suffix": "gain_m18", "audio": gain_down})

        # Moderate Blast Window Policy (SHAReD explosions)
        elif policy == AugmentationPolicy.MODERATE_BLAST_WINDOW:
            # Blast onset window variations
            w1 = self.blast_windower.process(audio, target_length_samples=len(audio), onset_offset_samples=int(0.05 * self.sample_rate))
            variations.append({"suffix": "onset_050ms", "audio": w1})

            w2 = self.blast_windower.process(audio, target_length_samples=len(audio), onset_offset_samples=int(0.12 * self.sample_rate))
            variations.append({"suffix": "onset_120ms", "audio": w2})

            # Gain variations
            gain_up = self.gain_jitter.process(audio, jitter_db=2.2)
            variations.append({"suffix": "gain_p22", "audio": gain_up})

            gain_down = self.gain_jitter.process(audio, jitter_db=-2.2)
            variations.append({"suffix": "gain_m22", "audio": gain_down})

        # Moderate Full (Sirens, thin urban classes)
        elif policy == AugmentationPolicy.MODERATE_FULL:
            stretched = self.time_stretcher.process(audio, rate=1.08)
            variations.append({"suffix": "stretch_108", "audio": stretched})

            stretched_slow = self.time_stretcher.process(audio, rate=0.92)
            variations.append({"suffix": "stretch_092", "audio": stretched_slow})

            gain_var = self.gain_jitter.process(audio, jitter_db=2.0)
            variations.append({"suffix": "gain_p20", "audio": gain_var})

        return variations

    def run_augmentation(self, max_clips_per_class: Optional[int] = None) -> Dict[str, int]:
        """
        Runs augmentation across processed audio files.
        """
        logger.info("=== PROJECT AEGIS: GROUNDED AUGMENTATION RUNNER START ===")
        generated_counts: Dict[str, int] = {}

        wav_files = list(self.processed_dir.glob("**/*.wav"))
        for wav_path in wav_files:
            # Determine class and policy from path and metadata
            class_name = wav_path.parent.name
            try:
                unified_class = UnifiedClass(class_name)
            except ValueError:
                unified_class = UnifiedClass.GENERAL_NOISE

            # Determine policy
            if unified_class in (UnifiedClass.TANK_TRACKED, UnifiedClass.ARTILLERY_HOWITZER, UnifiedClass.JET_COCKPIT, UnifiedClass.NAVAL_DESTROYER):
                policy = AugmentationPolicy.MODERATE_NO_PITCH
            elif unified_class in (UnifiedClass.EXPLOSION_BLAST, UnifiedClass.GUNSHOT_FIREARM):
                policy = AugmentationPolicy.MODERATE_BLAST_WINDOW
            elif unified_class == UnifiedClass.SIREN_EMERGENCY:
                policy = AugmentationPolicy.MODERATE_FULL
            else:
                policy = AugmentationPolicy.NONE

            if policy == AugmentationPolicy.NONE:
                continue

            audio, sr = sf.read(wav_path, dtype="float32")
            variations = self.augment_clip(audio, wav_path.stem, unified_class, policy)

            dest_folder = self.augmented_dir / unified_class.value
            dest_folder.mkdir(parents=True, exist_ok=True)

            for var in variations:
                out_name = f"{wav_path.stem}_{var['suffix']}.wav"
                out_path = dest_folder / out_name
                sf.write(out_path, var["audio"], sr, subtype="PCM_16")
                generated_counts[unified_class.value] = generated_counts.get(unified_class.value, 0) + 1

        logger.info("Augmentation complete. Total generated variations: %s", generated_counts)
        return generated_counts
