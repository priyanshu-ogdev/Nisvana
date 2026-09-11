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
from typing import Callable, Iterator, Optional
import random


def _require_webdataset():
    try:
        import webdataset as wds  # type: ignore
        return wds
    except ImportError as e:
        raise ImportError(
            "The `webdataset` package is required to load AEGIS shards as a "
            "PyTorch dataset. Install with: pip install webdataset torch"
        ) from e


try:
    import torch
    _IterableDatasetBase = torch.utils.data.IterableDataset
except (ImportError, AttributeError):
    _IterableDatasetBase = object


class _BaseAegisShardDataset(_IterableDatasetBase):
    """Common shard-glob + webdataset pipeline construction."""

    def __init__(self, shard_dir: Path, split: str = "train", shard_prefix: str = "shard"):
        self.shard_dir = Path(shard_dir)
        self.split = split
        self.sample_weight_fn: Optional[Callable[[dict], float]] = None
        self._sampling_seed = 1337

        # Generalization exports historically used ``gentest`` while the
        # canonical data-forge split is ``test_generalization``.  Accept both
        # exact tags, but never fall back to unrelated shards.
        split_tags = (
            ("test_generalization", "gentest", "test")
            if split in ("test", "gentest", "test_generalization")
            else (split,)
        )
        branch_subname = "speech_enhancement" if shard_prefix == "se" else ("classifier" if shard_prefix == "clf" else "aec")
        candidates = [
            self.shard_dir,
            self.shard_dir / branch_subname,
            self.shard_dir / shard_prefix,
            self.shard_dir.parent / branch_subname if self.shard_dir.parent.exists() else None,
            self.shard_dir.parent if self.shard_dir.parent.exists() else None,
        ]

        found_shards = []
        for cand in candidates:
            if cand and cand.exists() and cand.is_dir():
                m = sorted(
                    {
                        path
                        for tag in split_tags
                        for path in cand.glob(f"{shard_prefix}-{tag}-*.tar")
                    }
                )
                if m:
                    self.shard_dir = cand
                    found_shards = m
                    break

        if found_shards:
            pattern = [str(p) for p in found_shards]
        else:
            raise FileNotFoundError(
                f"No shards found for split={split!r} under {self.shard_dir}. "
                f"Expected one of {', '.join(f'{shard_prefix}-{tag}-*.tar' for tag in split_tags)}; "
                "refusing to fall back "
                "to another split."
            )

        self._wds = _require_webdataset()
        # WebDataset accepts either a brace-pattern string or a list of concrete tar paths.
        self.dataset = self._wds.WebDataset(
            pattern,
            shardshuffle=(100 if split == "train" else False),
            nodesplitter=self._wds.split_by_node,
        )

    def __iter__(self) -> Iterator:
        if self.sample_weight_fn is None or self.split != "train":
            return iter(self.dataset)

        try:
            import torch
            worker_info = torch.utils.data.get_worker_info()
            worker_offset = worker_info.id if worker_info is not None else 0
        except ImportError:
            worker_offset = 0
        rng = random.Random(self._sampling_seed + worker_offset)

        def weighted_samples():
            for sample in self.dataset:
                weight = max(0.0, float(self.sample_weight_fn(sample)))
                whole_repeats = int(weight)
                fractional_repeat = weight - whole_repeats
                for _ in range(whole_repeats):
                    yield sample
                if fractional_repeat and rng.random() < fractional_repeat:
                    yield sample

        return weighted_samples()


class AegisSpeechEnhancementIterableDataset(_BaseAegisShardDataset):
    """
    Yields dicts with keys: 'noisy.wav', 'clean.wav', 'rir.wav' (optional),
    'json' (mixture metadata incl. target/measured SNR, unified_class,
    sync_tier). Feeds Models 1-3 (DeepFilterNet3 x2, CleanUMamba).

    max_sample_len_s: Maximum clip duration in seconds. Clips longer than
        this are cropped; clips shorter than MIN_FFT_LEN (2048 samples) are
        zero-padded.
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
        if hasattr(self.dataset, "decode"):
            try:
                self.dataset = self.dataset.decode()
            except Exception:
                pass

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
                if arr.ndim > 1:
                    arr = arr.mean(axis=-1)
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
    Yields dicts with keys: 'wav', 'category_index', 'label', 'json'.
    Feeds Model 4 (aegis-clf-gate).
    """

    DEFAULT_SAMPLE_LEN = 9600  # 0.2s @ 48kHz

    def __init__(
        self,
        shard_dir: Path,
        split: str = "train",
        sample_len: int = 9600,
        sample_rate: int = 48000,
    ):
        super().__init__(shard_dir, split, shard_prefix="clf")
        self.sample_len = sample_len
        self.sample_rate = sample_rate

    def _process_sample(self, sample: dict) -> Optional[dict]:
        import numpy as np
        import json

        audio = sample.get("wav")
        if audio is None:
            audio = sample.get("wav.wav")
        if audio is None:
            audio = sample.get("audio")
        if audio is None:
            return None

        if isinstance(audio, (bytes, bytearray)):
            try:
                import io, soundfile as sf
                audio, _ = sf.read(io.BytesIO(audio), dtype="float32")
            except Exception:
                return None

        if hasattr(audio, "__len__"):
            arr = np.asarray(audio, dtype=np.float32).squeeze()
            if arr.ndim > 1:
                arr = arr.mean(axis=-1)
            if len(arr) > self.sample_len:
                arr = arr[:self.sample_len]
            elif len(arr) < self.sample_len:
                arr = np.pad(arr, (0, self.sample_len - len(arr)))
            peak = np.max(np.abs(arr))
            if peak > 1.0:
                arr = arr / peak
        else:
            arr = np.zeros(self.sample_len, dtype=np.float32)

        meta = sample.get("json", {})
        if isinstance(meta, (bytes, bytearray)):
            try:
                meta = json.loads(meta.decode("utf-8"))
            except Exception:
                meta = {}
        elif isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                meta = {}
        elif not isinstance(meta, dict):
            meta = {}

        gate_map = {"harmonic": 0, "impulsive": 1, "speech_dominant": 2}
        cat_idx = meta.get("category_index")
        if cat_idx is None:
            u_class = meta.get("unified_class", "general_noise")
            from training.configs.classifier_config import UNIFIED_TO_GATE_CLASS
            gate_name = UNIFIED_TO_GATE_CLASS.get(u_class, "harmonic")
            cat_idx = gate_map.get(gate_name, 0)
        else:
            cat_idx = int(cat_idx)

        return {
            "wav": arr,
            "category_index": cat_idx,
            "label": cat_idx,
            "json": meta,
        }

    def __iter__(self):
        for sample in self.dataset:
            proc = self._process_sample(sample)
            if proc is not None:
                yield proc


class AegisAecIterableDataset(_BaseAegisShardDataset):
    """
    Yields dicts with keys: 'mic.wav', 'farend.wav', 'nearend.wav',
    'echo.wav', 'json'. Feeds Model 5 (gated AEC).
    """

    MIN_FFT_LEN = 2048

    def __init__(
        self,
        shard_dir: Path,
        split: str = "train",
        max_sample_len_s: float = 4.0,
        sample_rate: int = 48000,
    ):
        super().__init__(shard_dir, split, shard_prefix="aec")
        self.max_sample_len = int(max_sample_len_s * sample_rate)

    def _process_sample(self, sample: dict) -> dict:
        import numpy as np
        for key in ("mic.wav", "farend.wav", "nearend.wav", "echo.wav", "mic", "farend", "nearend", "echo"):
            audio = sample.get(key)
            if audio is None:
                continue
            if isinstance(audio, (bytes, bytearray)):
                try:
                    import io, soundfile as sf
                    audio, _ = sf.read(io.BytesIO(audio), dtype="float32")
                except Exception:
                    continue
            if hasattr(audio, "__len__"):
                arr = np.asarray(audio, dtype=np.float32).squeeze()
                if arr.ndim > 1:
                    arr = arr.mean(axis=-1)
                if len(arr) > self.max_sample_len:
                    arr = arr[:self.max_sample_len]
                elif len(arr) < self.MIN_FFT_LEN:
                    arr = np.pad(arr, (0, self.MIN_FFT_LEN - len(arr)))
                peak = np.max(np.abs(arr))
                if peak > 1.0:
                    arr = arr / peak
                sample[key] = arr
        return sample

    def __iter__(self):
        for sample in self.dataset:
            yield self._process_sample(sample)
