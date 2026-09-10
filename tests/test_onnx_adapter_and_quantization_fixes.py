"""
tests/test_onnx_adapter_and_quantization_fixes.py

Regression tests for two issues found continuing the inference-layer
review:

1. OnnxRuntimeSession/export_to_onnx's statefulness fix (previous pass)
   had no real caller anywhere in the application -- live_mic_anc.py and
   enhance_audio.py always built raw PyTorch models directly. Fixed via
   OnnxModelAdapter, which makes an OnnxRuntimeSession present the same
   interface AcousticEscalationRouter already calls PyTorch models
   through, so no router/pipeline code needs to change to use it.

2. quantize_model_dynamic's default target_layers ({nn.Linear, nn.GRU})
   silently leaves Conv1d layers at FP32 -- PyTorch dynamic quantization
   does not support Conv layers at all, and this project's actual models
   (DeepFilterNet3Wrapper, CleanUMambaWrapper) are Conv1d-dominant. Fixed
   by adding report_quantization_coverage so this is measured and
   reported, not silently assumed away.
"""

import numpy as np
import pytest


class TestOnnxAdapterParameterNameBug:
    """
    This specific bug was caught by actually RUNNING a mocked-dependency
    test, not by reading the code -- the adapter passed `providers=` to
    OnnxRuntimeSession's constructor, which actually expects
    `execution_providers=`. Locking that in here so it can't silently
    regress if the adapter is edited again later.
    """

    def test_adapter_passes_the_correct_keyword_to_onnx_runtime_session(self):
        import inspect
        from inference.engines import onnx_model_adapter
        import inference.engines.onnx_runtime_engine as ort_engine

        # Confirm OnnxRuntimeSession's real parameter name...
        sig = inspect.signature(ort_engine.OnnxRuntimeSession.__init__)
        assert "execution_providers" in sig.parameters

        # ...and confirm the adapter's source actually uses that name,
        # not the wrong "providers=" this bug shipped with initially.
        source = inspect.getsource(onnx_model_adapter.OnnxModelAdapter.__init__)
        assert "execution_providers=providers" in source
        assert "OnnxRuntimeSession(str(onnx_model_path), providers=providers)" not in source


torch = pytest.importorskip("torch")
onnxruntime = pytest.importorskip("onnxruntime")


class TestOnnxAdapterFullIntegration:
    def test_adapter_presents_pytorch_compatible_interface(self, tmp_path):
        from inference.engines.onnx_engine import export_to_onnx
        from inference.engines.onnx_model_adapter import OnnxModelAdapter
        from training.models.model_loader import DeepFilterNet3Wrapper

        model = DeepFilterNet3Wrapper(df_lookahead=0, conv_lookahead=0)
        model.eval()
        onnx_path = tmp_path / "test_primary.onnx"
        export_to_onnx(model, onnx_path)

        adapter = OnnxModelAdapter(onnx_path)

        # Must support the exact chained call AcousticEscalationRouter.__init__
        # makes unconditionally on every model it's given.
        result = adapter.to(torch.device("cpu")).eval()
        assert result is adapter

        out = adapter(torch.randn(480))
        assert isinstance(out, torch.Tensor)
        assert out.shape == (480,)

    def test_adapter_backed_router_constructs_without_modification(self, tmp_path):
        """
        The real proof this is properly connected: AcousticEscalationRouter
        itself -- unmodified -- accepts ONNX-adapter-wrapped models exactly
        like PyTorch ones.
        """
        from inference.engines.onnx_engine import export_to_onnx
        from inference.engines.onnx_model_adapter import OnnxModelAdapter
        from inference.runtime.escalation_router import AcousticEscalationRouter
        from training.models.model_loader import build_model_for_key

        primary = build_model_for_key("aegis-se-primary")
        escalation = build_model_for_key("aegis-se-escalation")
        classifier = build_model_for_key("aegis-clf-gate")

        primary_path = tmp_path / "primary.onnx"
        escalation_path = tmp_path / "escalation.onnx"
        classifier_path = tmp_path / "classifier.onnx"
        export_to_onnx(primary, primary_path)
        export_to_onnx(escalation, escalation_path)
        export_to_onnx(classifier, classifier_path)

        router = AcousticEscalationRouter(
            model_primary=OnnxModelAdapter(primary_path),
            model_escalation=OnnxModelAdapter(escalation_path),
            classifier=OnnxModelAdapter(classifier_path),
        )
        enhanced, meta = router.route_and_enhance(np.random.randn(480).astype(np.float32) * 0.1)
        assert enhanced.shape == (480,)
        assert "mode" in meta


class TestQuantizationCoverageReporting:
    def test_conv1d_dominant_model_reports_low_quantizable_fraction(self):
        from inference.engines.quantization import report_quantization_coverage
        from training.models.model_loader import DeepFilterNet3Wrapper

        model = DeepFilterNet3Wrapper(df_lookahead=0, conv_lookahead=0)
        coverage = report_quantization_coverage(model)

        # This model is Conv1d+GRU -- only the GRU portion is quantizable
        # by dynamic quantization, so the reported fraction should be
        # meaningfully less than 1.0, proving the report isn't just
        # rubber-stamping "fully quantized" for a Conv-dominant model.
        assert 0.0 < coverage["quantizable_fraction"] < 1.0
        assert coverage["total_params"] > coverage["quantizable_params"]

    def test_unsupported_layer_types_trigger_a_warning_not_silent_failure(self):
        import warnings
        from inference.engines.quantization import quantize_model_dynamic
        from training.models.model_loader import DeepFilterNet3Wrapper
        import torch.nn as nn

        model = DeepFilterNet3Wrapper(df_lookahead=0, conv_lookahead=0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            quantize_model_dynamic(model, target_layers={nn.Linear, nn.GRU, nn.Conv1d})
            assert any("not supported" in str(warning.message) for warning in w)
