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
        pattern = str(self.shard_dir / f"{shard_prefix}-{{000000..999999}}.tar")
        self._wds = _require_webdataset()
        # WebDataset resolves the brace-range against files actually present.
        self.dataset = self._wds.WebDataset(pattern, shardshuffle=(split == "train"), nodesplitter=self._wds.split_by_node)

    def __iter__(self) -> Iterator:
        return iter(self.dataset)


class AegisSpeechEnhancementIterableDataset(_BaseAegisShardDataset):
    """
    Yields dicts with keys: 'noisy.wav', 'clean.wav', 'rir.wav' (optional),
    'json' (mixture metadata incl. target/measured SNR, unified_class,
    sync_tier). Feeds Models 1-3 (DeepFilterNet3 x2, CleanUMamba).
    """

    def __init__(self, shard_dir: Path, split: str = "train"):
        super().__init__(shard_dir, split, shard_prefix="se")
        self.dataset = (
            self.dataset.decode(wds_decode_audio=True) if hasattr(self.dataset, "decode") else self.dataset
        )


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
