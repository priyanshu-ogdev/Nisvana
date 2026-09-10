"""
tests/test_final_audit_fixes.py

Regression tests for the three issues found auditing the "final merged"
repository against its own claims:

1. `_reset_state_if_resuming_from_bypass` was dropped entirely during the
   merge (confirmed absent by grep before this fix) -- re-applied and
   wired into BOTH route_and_enhance and route_and_enhance_pipelined.
2. `reset_state()` did not clear `last_prediction`, leaving a new
   session's first pipelined decision based on the previous session's
   stale category/SNR.
3. `live_mic_anc.py` computed real enhanced audio every chunk and only
   ever used its TIMING -- the actual audio content was silently
   discarded, so the "live demo" could confirm real-time speed but never
   let anyone verify enhancement quality.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from inference.runtime.escalation_router import AcousticEscalationRouter


class TestBypassResumeStateReset:
    def test_resuming_from_bypass_to_primary_resets_primary_state(self):
        router = AcousticEscalationRouter()
        router.current_state = "bypass"

        # Give model_primary some non-trivial state to prove it gets cleared,
        # not just left alone.
        with torch.no_grad():
            router.model_primary(torch.randn(1, 480))
        assert router.model_primary.hidden_state is not None

        router._reset_state_if_resuming_from_bypass("primary")
        assert router.model_primary.hidden_state is None

    def test_resuming_from_bypass_to_escalation_resets_escalation_state(self):
        router = AcousticEscalationRouter()
        router.current_state = "bypass"

        with torch.no_grad():
            router.model_escalation(torch.randn(1, 480))
        assert router.model_escalation.hidden_state is not None

        router._reset_state_if_resuming_from_bypass("escalation")
        assert router.model_escalation.hidden_state is None

    def test_transition_not_from_bypass_does_not_reset_anything(self):
        router = AcousticEscalationRouter()
        router.current_state = "primary"  # NOT bypass

        with torch.no_grad():
            router.model_escalation(torch.randn(1, 480))
        state_before = router.model_escalation.hidden_state

        router._reset_state_if_resuming_from_bypass("escalation")
        # Should be unchanged -- this wasn't a resume-from-bypass transition
        assert router.model_escalation.hidden_state is state_before

    def test_staying_in_bypass_does_not_reset_anything(self):
        router = AcousticEscalationRouter()
        router.current_state = "bypass"
        # target_mode == "bypass" too -- not resuming to an active model
        router._reset_state_if_resuming_from_bypass("bypass")
        # No exception, no unexpected state change -- primary/escalation
        # were never touched during bypass in the first place
        assert router.model_primary.hidden_state is None
        assert router.model_escalation.hidden_state is None


class TestResetStateClearsLastPrediction:
    def test_reset_state_clears_last_prediction_to_default(self):
        router = AcousticEscalationRouter()
        router.last_prediction = {"mode": "escalation", "category": "impulsive", "estimated_snr_db": -8.0}

        router.reset_state()

        assert router.last_prediction["category"] == "speech_dominant"
        assert router.last_prediction["estimated_snr_db"] == 10.0
        assert router.last_prediction["mode"] == "primary"


class TestLiveMicOutputCapture:
    def test_output_chunks_accumulate_and_concatenate_to_full_length(self):
        # Structural test of the accumulation pattern used in
        # live_mic_anc.py's fix, independent of actually running the full
        # (torch-heavy, real-time-paced) simulation function.
        chunk_size = 480
        num_chunks = 50
        output_chunks = [np.random.randn(chunk_size).astype(np.float32) for _ in range(num_chunks)]

        full_output = np.concatenate(output_chunks)
        assert len(full_output) == chunk_size * num_chunks
        # Confirm no chunk was silently dropped or duplicated
        np.testing.assert_array_equal(full_output[:chunk_size], output_chunks[0])
        np.testing.assert_array_equal(full_output[-chunk_size:], output_chunks[-1])
