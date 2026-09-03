"""
Project AEGIS — Gated AEC Branch (Model 5)
Strictly dedicated to AEC-Challenge data.
NO mixing with external speech enhancement noise to preserve true acoustic echo path physics.
Quadruplets: (mic_signal.wav, farend_reference.wav, nearend_clean.wav, echo_signal.wav)
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
import soundfile as sf
from data_forge.config import BRANCH_AEC, TARGET_SAMPLE_RATE

logger = logging.getLogger("DataForge.BranchAEC")


class AecBranch:
    """Manages the isolated Gated AEC training branch (Model 5)."""

    def __init__(self, output_dir: Path = BRANCH_AEC):
        self.output_dir = Path(output_dir)
        self.mic_dir = self.output_dir / "mic"
        self.farend_dir = self.output_dir / "farend"
        self.nearend_dir = self.output_dir / "nearend"
        self.echo_dir = self.output_dir / "echo"

        for d in (self.mic_dir, self.farend_dir, self.nearend_dir, self.echo_dir):
            d.mkdir(parents=True, exist_ok=True)

    def organize_aec_pairs(self, aec_raw_or_processed_dir: Path) -> List[Dict[str, Any]]:
        """
        Scans for AEC Challenge file groups (_mic, _farend, _nearend, _echo)
        and structures them cleanly for Model 5 training.
        """
        logger.info("Organizing AEC quadruplet pairs from %s...", aec_raw_or_processed_dir)
        all_wavs = list(aec_raw_or_processed_dir.glob("**/*.wav"))

        # Group by prefix
        groups: Dict[str, Dict[str, Path]] = {}
        for w in all_wavs:
            stem = w.stem
            for suffix in ("_mic", "_farend", "_nearend", "_echo", "_target"):
                if stem.endswith(suffix):
                    prefix = stem[: -len(suffix)]
                    if prefix not in groups:
                        groups[prefix] = {}
                    key = suffix.lstrip("_")
                    groups[prefix][key] = w
                    break

        quadruplets = []
        for prefix, files in groups.items():
            # Check if at least mic and farend exist
            if "mic" in files and "farend" in files:
                mic_src = files["mic"]
                farend_src = files["farend"]
                nearend_src = files.get("nearend", files.get("target", None))
                echo_src = files.get("echo", None)

                dest_mic = self.mic_dir / f"{prefix}_mic.wav"
                dest_farend = self.farend_dir / f"{prefix}_farend.wav"

                shutil.copy2(mic_src, dest_mic)
                shutil.copy2(farend_src, dest_farend)

                dest_nearend_str = None
                if nearend_src and nearend_src.exists():
                    dest_nearend = self.nearend_dir / f"{prefix}_nearend.wav"
                    shutil.copy2(nearend_src, dest_nearend)
                    dest_nearend_str = str(dest_nearend)

                dest_echo_str = None
                if echo_src and echo_src.exists():
                    dest_echo = self.echo_dir / f"{prefix}_echo.wav"
                    shutil.copy2(echo_src, dest_echo)
                    dest_echo_str = str(dest_echo)

                info = sf.info(dest_mic)
                quad = {
                    "prefix": prefix,
                    "mic_path": str(dest_mic),
                    "farend_path": str(dest_farend),
                    "nearend_path": dest_nearend_str,
                    "echo_path": dest_echo_str,
                    "duration_sec": round(info.duration, 2),
                    "sample_rate": info.samplerate,
                }
                quadruplets.append(quad)

        # Write manifest
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "branch": "gated_aec_model_5",
                "isolation_policy": "NO_EXTERNAL_NOISE_MIXING",
                "total_quadruplets": len(quadruplets),
                "samples": quadruplets,
            }, f, indent=2)

        logger.info("Saved %d AEC quadruplets to %s", len(quadruplets), self.output_dir)
        return quadruplets
