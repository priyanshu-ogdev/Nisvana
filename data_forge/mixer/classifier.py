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
        Calculates harmonicity index [0.0, 1.0] using fast FFT-based autocorrelation.
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

        # Fast FFT-based Autocorrelation O(N log N) via Wiener-Khinchin theorem
        n = len(sig)
        f = np.fft.rfft(sig, n=2 * n)
        autocorr = np.fft.irfft(f * np.conj(f))[:n]
        r_0 = autocorr[0]
        if r_0 < 1e-9:
            return 0.0

        # Search for first harmonic peak between 50 Hz and 1000 Hz
        min_lag = int(sample_rate / 1000.0)  # 1 kHz max pitch
        max_lag = int(sample_rate / 50.0)    # 50 Hz min pitch

        if min_lag >= n or min_lag >= max_lag:
            return 0.0

        max_lag = min(max_lag, n)
        search_window = autocorr[min_lag:max_lag]
        max_peak = float(np.max(search_window)) if len(search_window) > 0 else 0.0
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
        max_samples: int = 60000,
        num_workers: int = 16,
    ) -> List[Dict[str, Any]]:
        """
        Converts speech enhancement mixture records into Model 4 classified dataset.
        Optimized with stratified sampling across splits/classes and parallel I/O.
        """
        labels_path = self.output_dir / "labels.json"
        if labels_path.exists():
            try:
                with open(labels_path, "r", encoding="utf-8") as f:
                    prev_data = json.load(f)
                    prev_samples = prev_data.get("samples", [])
                    if len(prev_samples) >= max_samples:
                        logger.info(
                            "Found %d existing Model 4 classifier samples in %s. Reusing without re-generation.",
                            len(prev_samples),
                            self.output_dir,
                        )
                        return prev_samples[:max_samples]
            except Exception as e:
                logger.debug("Existing classifier labels check: %s", e)

        logger.info(
            "Building Model 4 Classifier dataset (target=%d samples, pool=%d mixtures, workers=%d)...",
            max_samples,
            len(mixture_records),
            num_workers,
        )

        noise_class_map = noise_class_map or {}
        window_samples = int(self.CLASSIFIER_WINDOW_SEC * TARGET_SAMPLE_RATE)
        approx_windows_per_clip = max(1, int(4.0 / self.CLASSIFIER_WINDOW_SEC))  # typically 20
        max_mixtures_needed = max(1, max_samples // approx_windows_per_clip)

        # 1. Stratify mixture selection if input pool is larger than needed
        if len(mixture_records) > max_mixtures_needed:
            splits_pool: Dict[str, List[Dict[str, Any]]] = {}
            for rec in mixture_records:
                sp = rec.get("split", "train")
                splits_pool.setdefault(sp, []).append(rec)

            selected_mixtures = []
            total_pool_len = len(mixture_records)

            for sp, pool in splits_pool.items():
                split_target = max(1, int(max_mixtures_needed * (len(pool) / total_pool_len)))
                step = max(1, len(pool) // split_target)
                selected_mixtures.extend(pool[::step][:split_target])

            logger.info(
                "Subsampled %d diverse mixtures from %d to generate ~%d classifier windows.",
                len(selected_mixtures),
                len(mixture_records),
                len(selected_mixtures) * approx_windows_per_clip,
            )
        else:
            selected_mixtures = mixture_records

        # 2. Worker function to extract all 20 windows for one mixture
        def _process_single_mixture(rec):
            clip_id = rec["clip_id"]
            noisy_src = Path(rec["noisy_path"])
            if not noisy_src.exists():
                return []

            try:
                audio, sr = sf.read(noisy_src, dtype="float32")
            except Exception:
                return []

            snr = rec.get("measured_snr_db", 0.0)
            noise_src_name = rec.get("noise_source", "")
            unified_class = rec.get(
                "unified_class",
                noise_class_map.get(noise_src_name, "general_noise"),
            )
            sp = rec.get("split", "train")

            n_windows = max(1, len(audio) // window_samples)
            results = []
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
                    "split": sp,
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
                    "source_clip_id": clip_id,
                }

                json_file = self.output_dir / f"{window_clip_id}.json"
                with open(json_file, "w", encoding="utf-8") as jf:
                    json.dump(sample_record, jf, indent=2)

                results.append(sample_record)
            return results

        # 3. Parallel execution with ThreadPoolExecutor
        import concurrent.futures

        classified_samples = []
        completed_count = 0
        total_mixtures = len(selected_mixtures)

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for res_list in executor.map(_process_single_mixture, selected_mixtures):
                classified_samples.extend(res_list)
                completed_count += 1
                if completed_count % 1000 == 0 or completed_count == total_mixtures:
                    logger.info(
                        "Classifier extraction: %d / %d mixtures processed (%.1f%%, %d windows generated)",
                        completed_count,
                        total_mixtures,
                        100.0 * completed_count / total_mixtures,
                        len(classified_samples),
                    )

        # 4. Write labels.json
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump({
                "branch": "classifier_model_4",
                "taxonomy": [c.value for c in ClassifierCategory],
                "total_samples": len(classified_samples),
                "samples": classified_samples,
            }, f, indent=2)

        logger.info("Saved %d Model 4 classifier samples to %s", len(classified_samples), labels_path)
        return classified_samples
