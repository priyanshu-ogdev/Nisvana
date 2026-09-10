"""
Project AEGIS — SNR / Harmonic Classifier Branch (Model 4)
Relabels mixed acoustic corpus into 3-way category:
1. stationary_harmonic (drones, tanks, jet engines, sirens)
2. non_stationary_transient (gunshots, explosions, blast impacts)
3. speech_dominant (clean speech or speech at high SNR > 12 dB)
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import soundfile as sf
from data_forge.config import (
    BRANCH_CLASSIFIER,
    TARGET_SAMPLE_RATE,
    ClassifierCategory,
    UnifiedClass,
)

logger = logging.getLogger("DataForge.BranchClassifier")


class ClassifierBranch:
    """Generates training dataset and labels for Model 4."""

    # FIX (this pass): previously, build_dataset_from_mixtures read and
    # labeled each WHOLE mixture (4.0s, per ForgeMixingConfig.target_duration_sec)
    # as one classification sample. But Model 4 exists specifically as a
    # low-latency, per-chunk gating signal -- escalation_router.py calls it
    # on 480-sample (10ms) chunks in the live streaming path, a 400x
    # duration mismatch from what it was actually trained on, and more
    # fundamentally a task-granularity mismatch: a 4-second clip can easily
    # contain BOTH harmonic background AND an impulsive transient within
    # it, so a single whole-clip label is a poor training signal for the
    # instantaneous, transient-onset-sensitive decision the model is
    # actually deployed to make. Fixed by slicing each mixture into
    # CLASSIFIER_WINDOW_SEC windows and labeling each independently.
    #
    # 0.2s chosen as a reasoned middle ground -- NOT a literature citation
    # (no published guidance for this exact window size was found for this
    # specific gating task): long enough for compute_harmonicity_index's
    # autocorrelation approach to have enough periods of a low-frequency
    # engine/rotor tone to lock onto, short enough to stay responsive for
    # real-time gating. Validate against AEGIS's own eval curve, same as
    # every other reasoned-default in this design, before treating 0.2s
    # as final.
    CLASSIFIER_WINDOW_SEC: float = 0.2

    def __init__(self, output_dir: Path = BRANCH_CLASSIFIER):
        self.output_dir = Path(output_dir)
        self.audio_dir = self.output_dir / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_harmonicity_index(audio: np.ndarray, sample_rate: int = 48000) -> float:
        """
        Calculates harmonicity index [0.0, 1.0] using normalized autocorrelation.
        Periodic harmonic signals (drones, vehicle engines) yield high values (~0.7-0.95);
        stochastic noise and transients yield low values (<0.3).
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if len(audio) < 1024:
            return 0.0

        # Take a 100ms slice
        slice_len = min(len(audio), int(0.10 * sample_rate))
        sig = audio[:slice_len]
        sig = sig - np.mean(sig)

        # Autocorrelation
        autocorr = np.correlate(sig, sig, mode="full")
        mid = len(autocorr) // 2
        r_0 = autocorr[mid]
        if r_0 < 1e-9:
            return 0.0

        # Search for first harmonic peak between 50 Hz and 1000 Hz
        min_lag = int(sample_rate / 1000.0)  # 1 kHz max pitch
        max_lag = int(sample_rate / 50.0)    # 50 Hz min pitch

        if mid + max_lag >= len(autocorr):
            max_lag = len(autocorr) - mid - 1

        if min_lag >= max_lag:
            return 0.0

        search_window = autocorr[mid + min_lag : mid + max_lag]
        max_peak = np.max(search_window)
        harmonicity = float(np.clip(max_peak / r_0, 0.0, 1.0))
        return round(harmonicity, 3)

    @staticmethod
    def map_to_3way_category(unified_class: str, snr_db: float) -> ClassifierCategory:
        """
        Maps unified acoustic class and SNR to 3-way classifier category.
        """
        # If speech is very dominant (high SNR), categorize as speech_dominant
        if snr_db >= 12.0 or unified_class == UnifiedClass.CLEAN_SPEECH.value:
            return ClassifierCategory.SPEECH_DOMINANT

        # Transient classes
        if unified_class in (UnifiedClass.EXPLOSION_BLAST.value, UnifiedClass.GUNSHOT_FIREARM.value):
            return ClassifierCategory.NON_STATIONARY_TRANSIENT

        # Harmonic / stationary vehicle and machinery classes
        if unified_class in (
            UnifiedClass.DRONE_UAV.value,
            UnifiedClass.TANK_TRACKED.value,
            UnifiedClass.JET_COCKPIT.value,
            UnifiedClass.NAVAL_DESTROYER.value,
            UnifiedClass.ARTILLERY_HOWITZER.value,
            UnifiedClass.SIREN_EMERGENCY.value,
        ):
            return ClassifierCategory.STATIONARY_HARMONIC

        return ClassifierCategory.STATIONARY_HARMONIC

    def build_dataset_from_mixtures(
        self,
        mixture_records: List[Dict[str, Any]],
        noise_class_map: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Converts speech enhancement mixture records into Model 4 classified dataset.
        """
        logger.info("Building Model 4 Classifier dataset from %d mixtures...", len(mixture_records))
        classified_samples = []
        noise_class_map = noise_class_map or {}
        window_samples = int(self.CLASSIFIER_WINDOW_SEC * TARGET_SAMPLE_RATE)

        for rec in mixture_records:
            clip_id = rec["clip_id"]
            noisy_src = Path(rec["noisy_path"])
            if not noisy_src.exists():
                continue

            audio, sr = sf.read(noisy_src, dtype="float32")
            snr = rec["measured_snr_db"]
            noise_src_name = rec["noise_source"]
            unified_class = noise_class_map.get(noise_src_name, "general_noise")

            # Slice into independent windows rather than labeling the whole
            # clip once -- see CLASSIFIER_WINDOW_SEC's docstring above for
            # why. A trailing partial window shorter than half the target
            # length is dropped rather than kept undersized and padded,
            # since a heavily-padded short window would itself be an
            # unrealistic training example relative to real deployment.
            n_windows = max(1, len(audio) // window_samples)
            for w in range(n_windows):
                start = w * window_samples
                end = start + window_samples
                if end > len(audio):
                    if (len(audio) - start) < window_samples // 2:
                        continue
                    end = len(audio)
                window_audio = audio[start:end]

                category = self.map_to_3way_category(unified_class, snr)
                harmonicity = self.compute_harmonicity_index(window_audio, sr)

                window_clip_id = f"{clip_id}_w{w:03d}"
                dest_file = self.audio_dir / f"{window_clip_id}.wav"
                sf.write(dest_file, window_audio, sr, subtype="PCM_16")

                gate_class = (
                    "harmonic" if category == ClassifierCategory.STATIONARY_HARMONIC
                    else "impulsive" if category == ClassifierCategory.NON_STATIONARY_TRANSIENT
                    else "speech_dominant"
                )

                sample_record = {
                    "clip_id": window_clip_id,
                    "split": rec.get("split", "train"),
                    "audio_path": str(dest_file),
                    "gate_class": gate_class,
                    "category_label": category.value,
                    "category_index": (
                        0 if category == ClassifierCategory.STATIONARY_HARMONIC
                        else 1 if category == ClassifierCategory.NON_STATIONARY_TRANSIENT
                        else 2
                    ),
                    "true_snr_db": snr,
                    "harmonicity_index": harmonicity,
                    "noise_class": unified_class,
                    "duration_sec": self.CLASSIFIER_WINDOW_SEC,
                    "source_clip_id": clip_id,   # traceable back to the parent 4.0s mixture
                }
                classified_samples.append(sample_record)

                # Write per-sample JSON sidecar for WebDataset sharding.
                # FIX (this pass): this write was previously OUTSIDE the
                # per-window loop, using the stale outer-loop `clip_id` --
                # meaning only the LAST window of each mixture ever got a
                # JSON sidecar written, and every other window's shard
                # entry would have been missing its metadata entirely.
                # Moved inside the loop, using the correct per-window id.
                json_file = self.output_dir / f"{window_clip_id}.json"
                with open(json_file, "w", encoding="utf-8") as jf:
                    json.dump(sample_record, jf, indent=2)

        # Write labels.json (accumulative across splits)
        labels_path = self.output_dir / "labels.json"
        all_clf = list(classified_samples)
        if labels_path.exists():
            try:
                with open(labels_path, "r", encoding="utf-8") as f:
                    prev = json.load(f).get("samples", [])
                    existing_ids = {r["clip_id"] for r in classified_samples}
                    all_clf = [p for p in prev if p.get("clip_id") not in existing_ids] + classified_samples
            except Exception:
                pass

        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump({
                "branch": "classifier_model_4",
                "taxonomy": [c.value for c in ClassifierCategory],
                "total_samples": len(all_clf),
                "samples": all_clf,
            }, f, indent=2)

        logger.info("Saved Model 4 classifier labels to %s", labels_path)
        return classified_samples
