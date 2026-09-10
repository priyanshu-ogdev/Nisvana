"""
tests/test_delay_lookup_not_bypassed.py

Regression test for a bug found by following up on my own prior fix:
get_algorithmic_delay_ms() (added to reconcile the fallback-vs-vendored
DeepFilterNet3 delay figures) was correctly used inside onnx_engine.py's
own benchmark_edge_latency, but TWO real application scripts --
enhance_audio.py and export_onnx.py -- still read the static
MODEL_ALGORITHMIC_DELAY_MS dict directly, completely bypassing the fix.

A separate, pre-existing bug was found while fixing this: enhance_audio.py
also had args.model == "router" reporting delay via
MODEL_ALGORITHMIC_DELAY_MS.get("router", 0.0) -- "router" was never a
real key in that dict, so this always silently returned the 0.0 default,
regardless of which internal model the router actually used.
"""

import inspect
from pathlib import Path


class TestNoDirectStaticDictReads:
    """
    Source-inspection tests, not behavioral ones -- the point is to lock
    in that these call sites route through the reconciliation function,
    which is a property of the SOURCE, not something a mocked run would
    necessarily catch if the mock didn't distinguish the two paths.
    """

    def test_enhance_audio_does_not_read_static_dict_directly(self):
        source = Path("inference/scripts/enhance_audio.py").read_text()
        assert "MODEL_ALGORITHMIC_DELAY_MS.get(" not in source
        assert "get_algorithmic_delay_ms(" in source

    def test_export_onnx_does_not_read_static_dict_directly(self):
        source = Path("inference/scripts/export_onnx.py").read_text()
        assert "MODEL_ALGORITHMIC_DELAY_MS.get(" not in source
        assert "get_algorithmic_delay_ms(" in source

    def test_enhance_audio_router_path_uses_a_real_model_object_not_the_string_router(self):
        """
        The pre-existing bug: 'router' was passed as a literal string key
        into the delay dict, which could never match anything. Confirms
        the fixed code instead resolves a real model key + model instance
        for the router path specifically.
        """
        source = Path("inference/scripts/enhance_audio.py").read_text()
        assert 'delay_lookup_key = "aegis-se-primary"' in source
        assert "delay_source_model = router.model_primary" in source
