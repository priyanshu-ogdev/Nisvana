"""Dataset and reproducibility guards for production training runs.

The training PRD requires independent train, validation, and generalization
splits.  These helpers make that requirement explicit instead of allowing a
missing split to silently resolve to another shard set.
"""

from pathlib import Path
from typing import Callable, Dict, Iterable, List
import os
import random

import numpy as np


SPLIT_ALIASES = {
    "train": "train",
    "val": "val",
    "gentest": "test_generalization",
    "test": "test_generalization",
    "test_generalization": "test_generalization",
}
SPLIT_TAGS = {
    "train": ("train",),
    "val": ("val",),
    "test_generalization": ("test_generalization", "gentest", "test"),
}
VALID_SPLITS = frozenset(SPLIT_ALIASES)


def find_split_shards(shard_dir: Path, split: str) -> List[Path]:
    """Return only shards belonging to ``split``.

    A split is identified by the conventional ``<branch>-<split>-*.tar``
    filename.  No fallback to unrelated tar files is allowed.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"Unsupported dataset split {split!r}; expected one of {sorted(VALID_SPLITS)}")
    canonical_split = SPLIT_ALIASES[split]
    root = Path(shard_dir)
    if not root.exists():
        return []
    tags = SPLIT_TAGS[canonical_split]
    return sorted(
        path
        for path in root.rglob("*.tar")
        if any(
            f"-{tag}-" in path.stem or path.stem.endswith(f"-{tag}")
            for tag in tags
        )
    )


def validate_split_shards(
    shard_dirs: Iterable[Path],
    required_splits: Iterable[str] = ("train", "val", "test_generalization"),
) -> Dict[str, List[Path]]:
    """Validate that required splits exist and contain disjoint shard files."""
    directories = [Path(directory) for directory in shard_dirs]
    result: Dict[str, List[Path]] = {}
    for split in required_splits:
        paths = sorted({path.resolve() for directory in directories for path in find_split_shards(directory, split)})
        if not paths:
            searched = ", ".join(str(directory) for directory in directories)
            raise FileNotFoundError(f"No {split!r} shards found in: {searched}")
        result[split] = paths

    seen: Dict[Path, str] = {}
    for split, paths in result.items():
        for path in paths:
            previous = seen.setdefault(path, split)
            if previous != split:
                raise ValueError(f"Shard appears in both {previous!r} and {split!r}: {path}")
    return result


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch without requiring CUDA."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker a deterministic, non-overlapping seed."""
    try:
        import torch
        seed = torch.initial_seed() % (2 ** 32)
    except ImportError:
        seed = 1337
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def make_loader_kwargs(config, *, shuffle: bool = False) -> Dict[str, object]:
    """Build consistent DataLoader options for all training entry points."""
    kwargs: Dict[str, object] = {
        "num_workers": max(0, int(getattr(config, "num_workers", 0))),
        "worker_init_fn": worker_init_fn,
    }
    if shuffle:
        kwargs["shuffle"] = True
    if kwargs["num_workers"]:
        kwargs["persistent_workers"] = True
    return kwargs
