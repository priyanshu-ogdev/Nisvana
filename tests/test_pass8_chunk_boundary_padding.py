"""
tests/test_pass8_chunk_boundary_padding.py

Regression test for the training/inference discontinuity found this pass:
se_primary_trainer.py feeds each model one continuous 4.0s forward call
during training, but both DeepFilterNet3Wrapper and CleanUMambaWrapper
were zero-padding their causal convolution on EVERY chunked inference
call, fabricating a discontinuity at every ~10ms chunk boundary that
never existed during training. Fixed via real-context carryover between
calls (see model_loader.py's updated docstrings/comments for the full
explanation).

The numpy simulation validating this same algorithm bit-exact-matches a
one-shot pass was already run and confirmed during this fix (see the
review response) -- these tests confirm the ACTUAL shipped PyTorch module
behaves the same way, gated behind a real torch install.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.models.model_loader import DeepFilterNet3Wrapper, CleanUMambaWrapper


class TestChunkBoundaryPaddingFix:
    def test_deepfilternet_chunked_matches_one_shot_processing(self):
        """
        The core claim: processing a signal as one continuous forward call
        (matching training conditions) must produce the same output as
        processing it in sequential chunks with context carried over
        (matching the fixed inference path) -- within floating-point
        tolerance, for the causal (conv_lookahead=0, Model 1) case.
        """
        torch.manual_seed(0)
        model = DeepFilterNet3Wrapper(df_lookahead=0, conv_lookahead=0)
        model.eval()

        full_signal = torch.randn(1, 4800)  # stand-in for a (scaled-down) 4.0s clip

        with torch.no_grad():
            one_shot_output = model(full_signal.clone())

        model.reset_state()
        chunk_size = 480
        chunked_outputs = []
        with torch.no_grad():
            for start in range(0, full_signal.shape[-1], chunk_size):
                chunk = full_signal[:, start:start + chunk_size]
                out = model(chunk)
                chunked_outputs.append(out)
        chunked_output = torch.cat(chunked_outputs, dim=-1)

        assert chunked_output.shape == one_shot_output.shape
        max_diff = torch.max(torch.abs(chunked_output - one_shot_output)).item()
        assert max_diff < 1e-4, (
            f"Chunked (context-carryover) output diverged from one-shot "
            f"(training-equivalent) output by {max_diff} -- the fix is not working correctly."
        )

    def test_deepfilternet_reset_state_clears_context_not_just_hidden_state(self):
        """
        reset_state() must clear the input/hidden context buffers too, not
        just the GRU's hidden_state -- otherwise a "fresh session" would
        silently leak the previous session's final audio into its first
        chunk's padding.
        """
        model = DeepFilterNet3Wrapper(df_lookahead=0, conv_lookahead=0)
        model.eval()
        with torch.no_grad():
            model(torch.randn(1, 480))

        assert model._input_context is not None
        assert model.hidden_state is not None

        model.reset_state()
        assert model._input_context is None
        assert model._hidden_context is None
        assert model.hidden_state is None

    def test_cleanumamba_chunked_matches_one_shot_processing(self):
        torch.manual_seed(0)
        model = CleanUMambaWrapper(target_param_count="1M")
        model.eval()

        full_signal = torch.randn(1, 4800)

        with torch.no_grad():
            one_shot_output = model(full_signal.clone())

        model.reset_state()
        chunk_size = 480
        chunked_outputs = []
        with torch.no_grad():
            for start in range(0, full_signal.shape[-1], chunk_size):
                chunk = full_signal[:, start:start + chunk_size]
                out = model(chunk)
                chunked_outputs.append(out)
        chunked_output = torch.cat(chunked_outputs, dim=-1)

        assert chunked_output.shape == one_shot_output.shape
        max_diff = torch.max(torch.abs(chunked_output - one_shot_output)).item()
        assert max_diff < 1e-4

    def test_deepfilternet_escalation_variant_still_has_zero_lookahead_padding(self):
        """
        Explicit regression test for the KNOWN, NOT-YET-FIXED limitation:
        Model 2 (conv_lookahead=2) should still show a real discrepancy
        against one-shot processing, because its right-side padding is
        still zero, not real future context. This test exists so that if
        someone DOES fix the lookahead-delay-buffering later, they'll see
        this test start failing and know to update it -- not leave a
        stale assumption sitting silently uncovered.
        """
        torch.manual_seed(0)
        model = DeepFilterNet3Wrapper(df_lookahead=2, conv_lookahead=2)
        model.eval()

        full_signal = torch.randn(1, 4800)
        with torch.no_grad():
            one_shot_output = model(full_signal.clone())

        model.reset_state()
        chunk_size = 480
        chunked_outputs = []
        with torch.no_grad():
            for start in range(0, full_signal.shape[-1], chunk_size):
                chunk = full_signal[:, start:start + chunk_size]
                out = model(chunk)
                chunked_outputs.append(out)
        chunked_output = torch.cat(chunked_outputs, dim=-1)

        max_diff = torch.max(torch.abs(chunked_output - one_shot_output)).item()
        # Expected to STILL diverge meaningfully, since only the left-side
        # fix was implemented this pass -- this is a documented limitation,
        # not a passing guarantee.
        assert max_diff > 1e-4, (
            "This test is meant to confirm the KNOWN lookahead limitation "
            "still exists. If it's now passing (no divergence), the "
            "lookahead-delay-buffering fix may have been implemented -- "
            "update this test and model_loader.py's docstring accordingly."
        )
