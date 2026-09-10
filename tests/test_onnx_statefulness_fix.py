"""
tests/test_onnx_statefulness_fix.py

Regression tests for the most severe finding in this project's review
history: export_to_onnx() previously traced every model with a single
input tensor and no declared state input/output, and OnnxRuntimeSession
(the actual edge/Jetson deployment engine) took exactly one input and one
output with no state-threading mechanism at all. Combined, this meant the
REAL deployed ONNX artifact was fully stateless -- every chunk processed
as if it were the first chunk of a session -- silently discarding every
GRU-statefulness and causal-context-carryover fix made elsewhere in this
project. Direct PyTorch inference (where those fixes DO apply) is almost
certainly not how this runs on real Jetson/embedded hardware.

Fixed via an explicit, purely-functional stateful forward() signature
(model_loader.py), a get_initial_state() marker method, an export path
that declares state as real ONNX graph inputs/outputs when that marker
is present, and a runtime session that detects and threads that state
automatically.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.models.model_loader import DeepFilterNet3Wrapper, CleanUMambaWrapper, AudioClassifierNet


class TestExplicitStatefulForward:
    def test_deepfilternet_explicit_mode_does_not_mutate_internal_attributes(self):
        """
        The core correctness requirement for ONNX-traceable statefulness:
        an explicit-state call must be purely functional. If it silently
        touched self.hidden_state/self._input_context/self._hidden_context,
        torch.onnx.export would bake in a side effect the traced graph
        can't represent -- exactly the original bug, reintroduced.
        """
        model = DeepFilterNet3Wrapper(df_lookahead=0, conv_lookahead=0)
        model.eval()
        hidden, in_ctx, hid_ctx = model.get_initial_state(batch_size=1)

        with torch.no_grad():
            model(torch.randn(1, 480), hidden, in_ctx, hid_ctx)

        # Internal attributes must remain untouched by an explicit call.
        assert model.hidden_state is None
        assert model._input_context is None
        assert model._hidden_context is None

    def test_deepfilternet_explicit_mode_returns_four_tensors(self):
        model = DeepFilterNet3Wrapper(df_lookahead=0, conv_lookahead=0)
        model.eval()
        hidden, in_ctx, hid_ctx = model.get_initial_state(batch_size=1)
        with torch.no_grad():
            result = model(torch.randn(1, 480), hidden, in_ctx, hid_ctx)
        assert len(result) == 4
        out, new_hidden, new_in_ctx, new_hid_ctx = result
        assert new_hidden.shape == hidden.shape
        assert new_in_ctx.shape == in_ctx.shape
        assert new_hid_ctx.shape == hid_ctx.shape

    def test_deepfilternet_explicit_state_actually_threads_and_affects_output(self):
        """
        The real proof this matters: feeding the SAME input chunk twice,
        with the state genuinely carried from call 1 into call 2, must
        produce a DIFFERENT output than feeding zero state both times --
        otherwise "statefulness" would be a no-op regardless of wiring.
        """
        torch.manual_seed(0)
        model = DeepFilterNet3Wrapper(df_lookahead=0, conv_lookahead=0)
        model.eval()
        chunk = torch.randn(1, 480)

        hidden, in_ctx, hid_ctx = model.get_initial_state(batch_size=1)
        with torch.no_grad():
            out1, hidden, in_ctx, hid_ctx = model(chunk, hidden, in_ctx, hid_ctx)
            out2_threaded, _, _, _ = model(chunk, hidden, in_ctx, hid_ctx)

        zero_hidden, zero_in_ctx, zero_hid_ctx = model.get_initial_state(batch_size=1)
        with torch.no_grad():
            out2_fresh, _, _, _ = model(chunk, zero_hidden, zero_in_ctx, zero_hid_ctx)

        max_diff = torch.max(torch.abs(out2_threaded - out2_fresh)).item()
        assert max_diff > 1e-6, (
            "Threaded state produced the same output as fresh zero state -- "
            "statefulness is not actually affecting computation."
        )

    def test_deepfilternet_explicit_mode_matches_internal_mode_numerically(self):
        """
        The explicit and internal-state calling conventions must agree --
        otherwise ONNX export would produce a graph that behaves
        differently from the PyTorch reference it was traced from.
        """
        torch.manual_seed(0)
        model_internal = DeepFilterNet3Wrapper(df_lookahead=0, conv_lookahead=0)
        model_explicit = DeepFilterNet3Wrapper(df_lookahead=0, conv_lookahead=0)
        model_explicit.load_state_dict(model_internal.state_dict())
        model_internal.eval()
        model_explicit.eval()

        chunks = [torch.randn(1, 480) for _ in range(3)]

        internal_outputs = []
        with torch.no_grad():
            for c in chunks:
                internal_outputs.append(model_internal(c))

        hidden, in_ctx, hid_ctx = model_explicit.get_initial_state(batch_size=1)
        explicit_outputs = []
        with torch.no_grad():
            for c in chunks:
                out, hidden, in_ctx, hid_ctx = model_explicit(c, hidden, in_ctx, hid_ctx)
                explicit_outputs.append(out)

        for i, (a, b) in enumerate(zip(internal_outputs, explicit_outputs)):
            max_diff = torch.max(torch.abs(a - b)).item()
            assert max_diff < 1e-4, f"Chunk {i}: explicit and internal modes diverged by {max_diff}"

    def test_cleanumamba_has_explicit_state_support_too(self):
        model = CleanUMambaWrapper(target_param_count="1M")
        model.eval()
        hidden, in_ctx = model.get_initial_state(batch_size=1)
        with torch.no_grad():
            result = model(torch.randn(1, 480), hidden, in_ctx)
        assert len(result) == 3

    def test_classifier_has_no_get_initial_state_marker(self):
        """
        The classifier is legitimately stateless (per-window design,
        fixed in an earlier pass) -- it must NOT have get_initial_state,
        so export_to_onnx correctly routes it through the simple,
        unmodified export path rather than expecting state it doesn't have.
        """
        model = AudioClassifierNet()
        assert not hasattr(model, "get_initial_state")


class TestOnnxExportUsesStatefulPath:
    def test_export_to_onnx_declares_state_io_names_for_stateful_models(self, tmp_path):
        from inference.engines.onnx_engine import export_to_onnx

        model = DeepFilterNet3Wrapper(df_lookahead=0, conv_lookahead=0)
        model.eval()
        out_path = tmp_path / "test_stateful.onnx"

        export_to_onnx(model, out_path, sample_rate=48000, chunk_ms=10.0)
        assert out_path.exists()

        # Load back via plain onnx (not onnxruntime) to inspect declared
        # graph inputs/outputs without needing a real inference session.
        onnx = pytest.importorskip("onnx")
        graph = onnx.load(str(out_path)).graph
        input_names = [i.name for i in graph.input]
        output_names = [o.name for o in graph.output]

        assert "state0_in" in input_names
        assert "state1_in" in input_names
        assert "state2_in" in input_names
        assert "state0_out" in output_names
        assert len(input_names) == 4   # input + 3 state tensors
        assert len(output_names) == 4  # output + 3 state tensors
