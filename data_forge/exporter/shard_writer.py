"""
Project AEGIS — WebDataset-Compatible Shard Writer

WHY THIS EXISTS:
Prior to this upgrade, data/forge/ held one file per sample (e.g. 200,000
individual noisy.wav/clean.wav/rir.wav triplets = 600,000+ loose files).
That is a well-documented anti-pattern for large-scale ML training:
  - Filesystem inode/metadata overhead dominates at this file count
  - Network/cloud storage (the eventual training target) is dramatically
    slower for millions of small random-access reads than sequential reads
  - Every major large-scale audio/speech training pipeline (NVIDIA NeMo,
    OpenAI Whisper's training tooling, LAION/OpenCLIP, HuggingFace's own
    `datasets` "webdataset" loader) converges on the same fix: pack samples
    into sharded .tar archives and stream them sequentially.

This module packs each data/forge/ branch into WebDataset-convention shards:
tar files where each sample is a set of entries sharing a basename
(e.g. 000042.noisy.wav, 000042.clean.wav, 000042.json), read sequentially.
This format is directly consumable by the `webdataset` PyTorch library and
by HuggingFace `datasets` (`load_dataset("webdataset", data_files=...)`)
with no further conversion.
"""

import json
import tarfile
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

DEFAULT_SAMPLES_PER_SHARD = 2048  # ~1-2GB shards at typical 4s/48kHz clip sizes


@dataclass
class ShardManifestEntry:
    shard_file: str
    num_samples: int
    byte_size: int


class ShardWriter:
    """Writes samples into sequential WebDataset-convention .tar shards."""

    def __init__(self, output_dir: Path, shard_prefix: str, samples_per_shard: int = DEFAULT_SAMPLES_PER_SHARD):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_prefix = shard_prefix
        self.samples_per_shard = samples_per_shard

        self._shard_idx = 0
        self._count_in_shard = 0
        self._tar: Optional[tarfile.TarFile] = None
        self._manifest: List[ShardManifestEntry] = []
        self._open_new_shard()

    def _shard_path(self, idx: int) -> Path:
        return self.output_dir / f"{self.shard_prefix}-{idx:06d}.tar"

    def _open_new_shard(self) -> None:
        if self._tar is not None:
            self._close_current_shard()
        path = self._shard_path(self._shard_idx)
        self._tar = tarfile.open(path, mode="w")
        self._count_in_shard = 0

    def _close_current_shard(self) -> None:
        if self._tar is not None:
            path = Path(self._tar.name)
            self._tar.close()
            self._manifest.append(
                ShardManifestEntry(
                    shard_file=path.name,
                    num_samples=self._count_in_shard,
                    byte_size=path.stat().st_size,
                )
            )
            self._shard_idx += 1

    def add_sample(self, key: str, files: Dict[str, bytes]) -> None:
        """
        Adds one sample to the current shard. `files` maps extension
        (e.g. "noisy.wav", "clean.wav", "json") to raw bytes.
        """
        if self._count_in_shard >= self.samples_per_shard:
            self._open_new_shard()

        for ext, data in files.items():
            info = tarfile.TarInfo(name=f"{key}.{ext}")
            info.size = len(data)
            self._tar.addfile(info, io.BytesIO(data))

        self._count_in_shard += 1

    def close(self) -> Dict:
        self._close_current_shard()
        manifest_path = self.output_dir / f"{self.shard_prefix}_shard_index.json"
        summary = {
            "shard_prefix": self.shard_prefix,
            "total_shards": len(self._manifest),
            "total_samples": sum(m.num_samples for m in self._manifest),
            "total_bytes": sum(m.byte_size for m in self._manifest),
            "shards": [m.__dict__ for m in self._manifest],
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary


def pack_branch_to_shards(
    branch_dir: Path,
    output_dir: Path,
    shard_prefix: str,
    file_groups: Callable[[Path], Optional[Dict[str, Path]]],
    samples_per_shard: int = DEFAULT_SAMPLES_PER_SHARD,
    split: Optional[str] = None,
) -> Dict:
    """
    Walks a flat-file branch directory and repacks it into WebDataset shards.
    If `split` is provided (e.g. 'train', 'val', 'gentest'), filters samples
    belonging to that split and prefixes shards as '{shard_prefix}-{split}-*.tar'.
    """
    branch_dir = Path(branch_dir)
    effective_prefix = f"{shard_prefix}-{split}" if split else shard_prefix

    sample_keys: List[str] = []
    manifest_p = branch_dir / "manifest.json"
    labels_p = branch_dir / "labels.json"
    if manifest_p.exists():
        try:
            with open(manifest_p, "r", encoding="utf-8") as f:
                data = json.load(f)
                sample_keys = [
                    s.get("clip_id") or s.get("prefix")
                    for s in data.get("samples", [])
                    if (s.get("clip_id") or s.get("prefix"))
                    and (not split or s.get("split") == split or f"_{split}_" in str(s.get("clip_id")))
                ]
        except Exception:
            pass
    elif labels_p.exists():
        try:
            with open(labels_p, "r", encoding="utf-8") as f:
                data = json.load(f)
                sample_keys = [
                    s.get("clip_id")
                    for s in data.get("samples", [])
                    if s.get("clip_id")
                    and (not split or s.get("split") == split or f"_{split}_" in str(s.get("clip_id")))
                ]
        except Exception:
            pass

    if not sample_keys:
        stems = set()
        for p in branch_dir.rglob("*.wav"):
            base = p.stem
            for sfx in ("_noisy", "_clean", "_rir", "_mic", "_farend", "_nearend", "_echo"):
                if base.endswith(sfx):
                    base = base[:-len(sfx)]
                    break
            if not split or f"_{split}_" in base or base.startswith(f"{split}_"):
                stems.add(base)
        sample_keys = sorted(stems)

    if not sample_keys and split:
        return {
            "shard_prefix": effective_prefix,
            "total_shards": 0,
            "total_samples": 0,
            "total_bytes": 0,
            "shards": [],
            "samples_packed": 0,
        }

    writer = ShardWriter(output_dir, effective_prefix, samples_per_shard)

    packed = 0
    for key in sample_keys:
        group = file_groups(branch_dir / key)
        if not group:
            continue
        files_bytes = {}
        ok = True
        for ext, path in group.items():
            if not path.exists():
                ok = False
                break
            files_bytes[ext] = path.read_bytes()
        if not ok:
            continue
        writer.add_sample(key, files_bytes)
        packed += 1

    summary = writer.close()
    summary["samples_packed"] = packed
    return summary
