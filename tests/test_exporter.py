"""
Project AEGIS — Tests for Shard Exporter & Dataset Card
Verifies WebDataset shard writing, subdirectory packing, and Croissant/HuggingFace card generation.
"""

import json
from pathlib import Path
import tarfile
import tempfile
import pytest

from data_forge.exporter.shard_writer import ShardWriter, pack_branch_to_shards
from data_forge.exporter.dataset_card import generate_dataset_card, write_dataset_card


class TestShardWriter:
    def test_shard_writer_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            writer = ShardWriter(Path(d), "unit", samples_per_shard=3)
            for i in range(7):
                writer.add_sample(f"{i:06d}", {"wav": b"X" * 10, "json": b"{}"})
            summary = writer.close()

            assert summary["total_samples"] == 7
            assert summary["total_shards"] == 3

            first_shard = Path(d) / summary["shards"][0]["shard_file"]
            with tarfile.open(first_shard) as t:
                names = t.getnames()
                assert "000000.wav" in names
                assert "000000.json" in names

    def test_pack_branch_to_shards_with_subdirectories(self):
        with tempfile.TemporaryDirectory() as d:
            branch = Path(d) / "se"
            (branch / "noisy").mkdir(parents=True)
            (branch / "clean").mkdir(parents=True)
            (branch / "rir").mkdir(parents=True)
            manifest_data = {
                "samples": [
                    {"clip_id": "se_00001"},
                    {"clip_id": "se_00002"},
                ]
            }
            (branch / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")
            (branch / "noisy" / "se_00001_noisy.wav").write_bytes(b"noisy_1")
            (branch / "clean" / "se_00001_clean.wav").write_bytes(b"clean_1")
            (branch / "noisy" / "se_00002_noisy.wav").write_bytes(b"noisy_2")
            (branch / "clean" / "se_00002_clean.wav").write_bytes(b"clean_2")

            def se_groups(sample_root: Path):
                k = sample_root.name
                b = sample_root.parent
                n = b / "noisy" / f"{k}_noisy.wav"
                c = b / "clean" / f"{k}_clean.wav"
                if n.exists() and c.exists():
                    return {"noisy.wav": n, "clean.wav": c}
                return None

            out_shards = Path(d) / "shards"
            summary = pack_branch_to_shards(branch, out_shards, "se", se_groups, samples_per_shard=2)
            assert summary["samples_packed"] == 2
            assert summary["total_shards"] == 1


class TestDatasetCard:
    def test_dataset_card_generation(self, tmp_path):
        card = generate_dataset_card({"speech_enhancement": 100})
        assert "pretty_name: Project AEGIS Defence ANC Training Corpus" in card
        assert "NOISEX-92" in card
        assert "Military Audio Dataset (MAD)" in card

        out_file = tmp_path / "CARD.md"
        write_dataset_card(out_file, {"speech_enhancement": 100})
        assert out_file.exists()
        assert len(out_file.read_text(encoding="utf-8")) > 100
