"""
Project AEGIS — PyTorch DataLoader-Ready Dataset Wrappers

Wraps the WebDataset shards produced by shard_writer.py into PyTorch
IterableDataset classes, so each model's training script can go straight
from data/forge/*_shards/ to a DataLoader with no bespoke glue code:

    from data_forge.exporter import AegisSpeechEnhancementIterableDataset
    ds = AegisSpeechEnhancementIterableDataset(shard_dir, split="train")
    loader = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=4)

Requires the optional `webdataset` and `torch` packages at USE time (not
at shard-creation time — shard_writer.py has zero extra dependencies so
fetch/preprocess/mix/export can run without a torch install on the
storage/build machine). Import errors are raised lazily and with a clear
install hint, not at package import time, so `import data_forge.exporter`
still works in environments that only build shards.
"""

from pathlib import Path
from typing import Iterator, Optional


def _require_webdataset():
    try:
        import webdataset as wds  # type: ignore
        return wds
    except ImportError as e:
        raise ImportError(
            "The `webdataset` package is required to load AEGIS shards as a "
            "PyTorch dataset. Install with: pip install webdataset torch"
        ) from e


class _BaseAegisShardDataset:
    """Common shard-glob + webdataset pipeline construction."""

    def __init__(self, shard_dir: Path, split: str = "train", shard_prefix: str = "shard"):
        self.shard_dir = Path(shard_dir)
        self.split = split

        # Check if split-specific shards exist (e.g. se-train-*.tar, se-val-*.tar, se-gentest-*.tar)
        split_tag = "gentest" if split in ("test", "gentest", "test_generalization") else split
        split_pattern = str(self.shard_dir / f"{shard_prefix}-{split_tag}-{{000000..999999}}.tar")
        legacy_pattern = str(self.shard_dir / f"{shard_prefix}-{{000000..999999}}.tar")

        # Select split pattern if matching shards are found on disk, otherwise legacy fallback
        split_matches = list(self.shard_dir.glob(f"{shard_prefix}-{split_tag}-*.tar"))
        pattern = split_pattern if split_matches else legacy_pattern

        self._wds = _require_webdataset()
        # WebDataset resolves the brace-range against files actually present.
        self.dataset = self._wds.WebDataset(
            pattern,
            shardshuffle=(100 if split == "train" else False),
            nodesplitter=self._wds.split_by_node,
        )

    def __iter__(self) -> Iterator:
        return iter(self.dataset)


class AegisSpeechEnhancementIterableDataset(_BaseAegisShardDataset):
    """
    Yields dicts with keys: 'noisy.wav', 'clean.wav', 'rir.wav' (optional),
    'json' (mixture metadata incl. target/measured SNR, unified_class,
    sync_tier). Feeds Models 1-3 (DeepFilterNet3 x2, CleanUMamba).

    max_sample_len_s: Maximum clip duration in seconds. Clips longer than
        this are cropped; clips shorter than MIN_FFT_LEN (2048 samples) are
        zero-padded. This enforcement was previously documented in config but
        never applied — training received variable-length samples that could
        exceed GPU memory or be too short for the multi-res STFT loss's largest
        FFT size.
    """

    MIN_FFT_LEN = 2048  # Matches multires_loss.py's largest-small FFT size

    def __init__(
        self,
        shard_dir: Path,
        split: str = "train",
        max_sample_len_s: float = 4.0,
        sample_rate: int = 48000,
    ):
        super().__init__(shard_dir, split, shard_prefix="se")
        self.max_sample_len = int(max_sample_len_s * sample_rate)
        self.dataset = (
            self.dataset.decode(wds_decode_audio=True) if hasattr(self.dataset, "decode") else self.dataset
        )

    def _enforce_length(self, sample: dict) -> dict:
        """Crops or pads audio tensors to enforce max_sample_len_s."""
        import numpy as np
        for key in ("noisy.wav", "clean.wav", "rir.wav"):
            audio = sample.get(key)
            if audio is None:
                continue
            if isinstance(audio, (bytes, bytearray)):
                try:
                    import io, soundfile as sf
                    audio, _ = sf.read(io.BytesIO(audio), dtype="float32")
                except Exception:
                    continue
            if hasattr(audio, '__len__'):
                arr = np.asarray(audio, dtype=np.float32).squeeze()
                # Crop
                if len(arr) > self.max_sample_len:
                    arr = arr[:self.max_sample_len]
                # Pad if below minimum FFT length
                elif len(arr) < self.MIN_FFT_LEN:
                    arr = np.pad(arr, (0, self.MIN_FFT_LEN - len(arr)))
                # Lazy normalization to [-1, 1] float32
                peak = np.max(np.abs(arr))
                if peak > 1.0:
                    arr = arr / peak
                sample[key] = arr
        return sample

    def __iter__(self):
        for sample in self.dataset:
            yield self._enforce_length(sample)


class AegisClassifierIterableDataset(_BaseAegisShardDataset):
    """
    Yields dicts with keys: 'wav', 'json' (label: harmonic / impulsive /
    speech_dominant, per the unified 3-way crosswalk). Feeds Model 4.
    """

    def __init__(self, shard_dir: Path, split: str = "train"):
        super().__init__(shard_dir, split, shard_prefix="clf")


class AegisAecIterableDataset(_BaseAegisShardDataset):
    """
    Yields dicts with keys: 'mic.wav', 'farend.wav', 'nearend.wav',
    'echo.wav', 'json'. Feeds Model 5 (gated AEC), only if/when fine-tuning
    the deepvqe-ggml checkpoint is undertaken.
    """

    def __init__(self, shard_dir: Path, split: str = "train"):
        super().__init__(shard_dir, split, shard_prefix="aec")
