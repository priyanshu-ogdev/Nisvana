"""
Project AEGIS — Data-Forge Exporter
Upgrades the flat-file data/forge/ output into industry-standard formats
for large-scale ML training: WebDataset tar-shards (the convention used by
NeMo, OpenCLIP, LAION, and most large audio/speech training pipelines to
avoid the small-files-on-disk problem) plus PyTorch DataLoader-ready wrappers
and an auto-generated dataset card.
"""

from .shard_writer import ShardWriter, pack_branch_to_shards
from .torch_dataset import (
    AegisAecIterableDataset,
    AegisClassifierIterableDataset,
    AegisSpeechEnhancementIterableDataset,
)
from .dataset_card import generate_dataset_card, write_dataset_card

__all__ = [
    "ShardWriter",
    "pack_branch_to_shards",
    "AegisSpeechEnhancementIterableDataset",
    "AegisClassifierIterableDataset",
    "AegisAecIterableDataset",
    "generate_dataset_card",
    "write_dataset_card",
]
