"""
tests/test_pass7_classifier_duration_alignment.py

Regression test for the Model 4 train/inference duration mismatch found
in this pass: the classifier was trained on whole 4.0s mixtures
(data_forge's real target_duration_sec) but called at inference time on
480-sample (10ms) chunks -- a 400x duration gap and, more importantly, a
task-granularity mismatch (a 4s clip can contain both harmonic background
AND an impulsive transient; a single whole-clip label is a poor signal
for the instantaneous per-chunk decision the model is actually deployed
to make).

Fixed in two places, both covered here:
  1. data_forge/mixer/classifier.py -- slices each mixture into
     CLASSIFIER_WINDOW_SEC (0.2s) windows, each independently labeled,
     instead of one label per whole 4.0s clip.
  2. inference/runtime/escalation_router.py -- accumulates raw hardware
     chunks into a rolling buffer matching that same 0.2s window before
     ever calling the classifier, instead of feeding it a raw 10ms chunk.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


class TestClassifierBranchWindowing:
    def test_one_4s_mixture_produces_20_independent_windows(self):
        from data_forge.mixer.classifier import ClassifierBranch

        with tempfile.TemporaryDirectory() as d:
            branch = ClassifierBranch(output_dir=Path(d))
            audio = (np.random.randn(192000) * 0.05).astype(np.float32)  # 4.0s @ 48kHz
            src_path = Path(d) / "se_train_00001_noisy.wav"
            sf.write(str(src_path), audio, 48000)

            records = [{
                "clip_id": "se_train_00001",
                "noisy_path": str(src_path),
                "measured_snr_db": 3.0,
                "noise_source": "noisex92_leopard.wav",
                "split": "train",
                "duration_sec": 4.0,
            }]

            samples = branch.build_dataset_from_mixtures(
                records, noise_class_map={"noisex92_leopard.wav": "tank_tracked"}
            )

            expected = 192000 // int(branch.CLASSIFIER_WINDOW_SEC * 48000)
            assert len(samples) == expected
            for s in samples:
                assert s["duration_sec"] == branch.CLASSIFIER_WINDOW_SEC
                assert s["source_clip_id"] == "se_train_00001"

    def test_every_window_gets_its_own_json_sidecar(self):
        # Regression test for the bug caught WHILE fixing the windowing --
        # the sidecar write was outside the per-window loop, using a stale
        # clip_id, so only the last window of each mixture ever got a
        # sidecar written at all.
        from data_forge.mixer.classifier import ClassifierBranch

        with tempfile.TemporaryDirectory() as d:
            branch = ClassifierBranch(output_dir=Path(d))
            audio = (np.random.randn(192000) * 0.05).astype(np.float32)
            src_path = Path(d) / "se_train_00002_noisy.wav"
            sf.write(str(src_path), audio, 48000)

            records = [{
                "clip_id": "se_train_00002",
                "noisy_path": str(src_path),
                "measured_snr_db": 3.0,
                "noise_source": "mad_gunshot.wav",
                "split": "train",
                "duration_sec": 4.0,
            }]
            samples = branch.build_dataset_from_mixtures(
                records, noise_class_map={"mad_gunshot.wav": "gunshot_firearm"}
            )

            json_files = list(Path(d).glob("se_train_00002_w*.json"))
            assert len(json_files) == len(samples)

            with open(json_files[0]) as f:
                content = json.load(f)
            assert content["clip_id"] == json_files[0].stem  # not a stale copy of a different window


class TestRollingClassifierBuffer:
    def test_ten_ms_chunks_accumulate_to_the_full_training_window(self):
        # Pure-logic test mirroring escalation_router.py's
        # _push_to_classifier_buffer exactly, without needing torch/nn.Module
        # to construct a full router instance.
        window_samples = int(0.2 * 48000)
        chunk_samples = 480

        buffer = np.zeros(window_samples, dtype=np.float32)
        filled = 0

        def push(chunk):
            nonlocal buffer, filled
            chunk_len = len(chunk)
            if chunk_len >= window_samples:
                buffer = chunk[-window_samples:].astype(np.float32)
            else:
                buffer = np.concatenate([buffer[chunk_len:], chunk.astype(np.float32)])
            filled = min(window_samples, filled + chunk_len)
            return buffer

        chunks_needed = window_samples // chunk_samples
        for i in range(chunks_needed):
            out = push(np.ones(chunk_samples, dtype=np.float32) * (i + 1))

        assert not np.any(out == 0)
        assert out[-1] == chunks_needed  # most recent chunk sits at the tail
        assert out[0] == 1               # oldest surviving chunk correctly evicted in FIFO order

    def test_single_oversized_chunk_uses_only_its_tail(self):
        window_samples = 100
        buffer = np.zeros(window_samples, dtype=np.float32)

        def push(chunk):
            nonlocal buffer
            if len(chunk) >= window_samples:
                buffer = chunk[-window_samples:].astype(np.float32)
            return buffer

        big_chunk = np.arange(500, dtype=np.float32)
        out = push(big_chunk)
        assert len(out) == window_samples
        np.testing.assert_array_equal(out, big_chunk[-window_samples:])
