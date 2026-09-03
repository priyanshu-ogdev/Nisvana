"""
Project AEGIS — Step 1: Format Standardization
Converts any source audio container to standard WAV PCM (16-bit or 24-bit).
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import soundfile as sf


class FormatStandardizer:
    """Step 1: Standardizes audio format to standard WAV PCM."""

    def __init__(self, target_subtype: str = "PCM_16"):
        self.target_subtype = target_subtype

    def process(self, input_path: Path, output_path: Optional[Path] = None) -> Tuple[np.ndarray, int]:
        """
        Reads input audio file and converts it to standard WAV PCM format.
        Returns: (audio_data_float32, sample_rate)
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Audio file not found: {input_path}")

        # Read using soundfile (supports WAV, FLAC, OGG, etc.)
        data, sr = sf.read(input_path, dtype="float32")

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, data, sr, subtype=self.target_subtype, format="WAV")

        return data, sr
