"""
tests/test_transmission_prep_wired_in.py

Regression test for the pattern this whole review kept catching: a new
module built, verified correct in isolation, and never actually exported
or called from real application code (the ONNX layer's original
disconnection was the most severe instance of this). Confirms
transmission_prep.py doesn't repeat that mistake.
"""

from pathlib import Path


class TestTransmissionPrepIsActuallyConnected:
    def test_exported_from_utils_package_init(self):
        source = Path("inference/utils/__init__.py").read_text()
        assert "prepare_for_transmission" in source
        assert "TransmissionPrepConfig" in source
        assert "prepare_for_transmission" in source.split("__all__")[1]

    def test_enhance_audio_script_actually_calls_it(self):
        source = Path("inference/scripts/enhance_audio.py").read_text()
        assert "--prepare-for-transmission" in source
        assert "prepare_for_transmission(" in source
        assert "from inference.utils.transmission_prep import" in source

    def test_onnx_model_adapter_exported_from_engines_package_init(self):
        # Same pattern, same fix, different module -- locked in together
        # since both were found the same way (grepping for a definition
        # that existed but was never imported anywhere else).
        source = Path("inference/engines/__init__.py").read_text()
        assert "OnnxModelAdapter" in source
        assert "OnnxModelAdapter" in source.split("__all__")[1]
