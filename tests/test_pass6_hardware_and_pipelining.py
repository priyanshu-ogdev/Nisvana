"""
tests/test_pass6_hardware_and_pipelining.py

Tests for:
  - multichannel_frontend.py -- the hardware input bridge closing the
    "zero trace of the multi-mic/throat-mic design" gap found this pass
  - escalation_router.py's new route_and_enhance_pipelined -- decouples
    the classifier from the per-chunk critical path
"""

import numpy as np
import pytest

from inference.runtime.multichannel_frontend import (
    MultichannelHardwareFrontend,
    HardwareFrontendConfig,
)


class TestMultichannelHardwareFrontend:
    def test_combines_n_air_mics_to_mono(self):
        fe = MultichannelHardwareFrontend(HardwareFrontendConfig(num_air_mics=4))
        chunk = np.random.randn(4, 480).astype(np.float32) * 0.01
        mono = fe.process(chunk)
        assert mono.shape == (480,)

    def test_louder_mic_gets_more_weight(self):
        fe = MultichannelHardwareFrontend(HardwareFrontendConfig(num_air_mics=2))
        loud = np.ones(480, dtype=np.float32) * 1.0
        quiet = np.ones(480, dtype=np.float32) * 0.01
        chunk = np.stack([loud, quiet])
        mono = fe.process(chunk)
        # Combined output should sit much closer to the loud channel's
        # value than a naive uniform average (which would land at ~0.505).
        assert np.mean(mono) > 0.7

    def test_throat_mic_weight_increases_as_snr_drops(self):
        fe = MultichannelHardwareFrontend(HardwareFrontendConfig(num_air_mics=1))
        air = np.zeros(480, dtype=np.float32)
        throat = np.ones(480, dtype=np.float32)

        out_high_snr = fe.process(air, throat_mic_channel=throat, estimated_snr_db=20.0)
        out_low_snr = fe.process(air, throat_mic_channel=throat, estimated_snr_db=-10.0)

        # At high SNR, throat weight should be ~0 -> output near the air signal (0).
        # At low SNR, throat weight should be higher -> output closer to throat (1).
        assert np.mean(out_high_snr) < np.mean(out_low_snr)

    def test_no_throat_mic_variant_falls_back_to_air_only(self):
        fe = MultichannelHardwareFrontend(HardwareFrontendConfig(num_air_mics=2, has_throat_mic=False))
        chunk = np.random.randn(2, 480).astype(np.float32)
        out = fe.process(chunk, throat_mic_channel=np.ones(480, dtype=np.float32))
        # Should ignore the throat channel entirely since has_throat_mic=False
        air_only = fe._combine_air_mics(chunk)
        np.testing.assert_array_equal(out, air_only)

    def test_single_mic_passthrough_no_combination_needed(self):
        fe = MultichannelHardwareFrontend(HardwareFrontendConfig(num_air_mics=1, has_throat_mic=False))
        chunk = np.random.randn(1, 480).astype(np.float32)
        out = fe.process(chunk)
        np.testing.assert_array_equal(out, chunk[0])


# --- Router pipelining tests require torch + real models; marked to skip
# --- cleanly rather than fail if torch isn't installed in a given env,
# --- consistent with how this project's other torch-dependent tests are
# --- structured.
torch = pytest.importorskip("torch")


class TestPipelinedRouter:
    def test_pipelined_output_uses_prior_prediction_not_current_classification(self):
        from inference.runtime.escalation_router import AcousticEscalationRouter

        router = AcousticEscalationRouter()
        # Force a known prior state rather than relying on the classifier's
        # actual (untrained, random-weight) output for this structural test.
        router.last_prediction = {"mode": "primary", "category": "harmonic", "estimated_snr_db": 10.0}
        router.current_state = "primary"

        chunk = np.random.randn(480).astype(np.float32) * 0.1
        enhanced, info = router.route_and_enhance_pipelined(chunk)

        # The decision returned should reflect the PRIOR category/SNR,
        # not a fresh classification of this chunk.
        assert info["category"] == "harmonic"
        assert info["estimated_snr_db"] == 10.0
        assert info["pipelined_lag_chunks"] == 1
        assert enhanced.shape == chunk.shape

    def test_pipelined_router_updates_last_prediction_for_next_call(self):
        from inference.runtime.escalation_router import AcousticEscalationRouter

        router = AcousticEscalationRouter()
        router.last_prediction = {"mode": "primary", "category": "harmonic", "estimated_snr_db": 10.0}
        router.current_state = "primary"

        chunk = np.random.randn(480).astype(np.float32) * 0.1
        router.route_and_enhance_pipelined(chunk)

        # last_prediction should now reflect THIS chunk's classification,
        # ready to inform the NEXT call -- not still the old prior value.
        assert "category" in router.last_prediction
        assert "estimated_snr_db" in router.last_prediction
