"""
tests/test_audit_pass4_latency_attribution.py

Regression test for the specific failure this pass caught: two entries in
MODEL_ALGORITHMIC_DELAY_MS were confidently-labeled but misattributed
numbers borrowed from unrelated components (RT-SEMamba, a distillation
technique donor never used as any runtime model here; TaylorBeamformer,
a component from an abandoned earlier hardware-ANC design phase, never
Model 5). Both were flagged as suspect BEFORE any code was read, purely
from the numbers not matching what each model actually is -- and both
were confirmed wrong on inspection.

This test can't verify the CORRECTED values are perfectly accurate
either (0.3ms for aegis-se-crosscheck and 32.0ms for aegis-aec-gate are
themselves marked as estimates/unverified-for-this-exact-port in the
source comments) -- but it locks in that the two specific WRONG values
don't silently return, which is the failure mode that actually occurred.
"""

from inference.engines.onnx_engine import MODEL_ALGORITHMIC_DELAY_MS


def test_crosscheck_delay_is_not_the_misattributed_rt_semamba_figure():
    # 25.0ms was RT-SEMamba's window -- a different architecture entirely,
    # never used as Model 3's actual runtime component.
    assert MODEL_ALGORITHMIC_DELAY_MS["aegis-se-crosscheck"] != 25.0


def test_aec_delay_is_not_the_misattributed_taylorbeamformer_figure():
    # 20.0ms was TaylorBeamformer's buffer -- belonged to an abandoned
    # earlier hardware-ANC design phase, never Model 5.
    assert MODEL_ALGORITHMIC_DELAY_MS["aegis-aec-gate"] != 20.0


def test_aec_delay_uses_the_actually_researched_deepvqe_s_figure():
    # The correct, previously-established figure for THIS component
    # (DeepVQE-S: 512 FFT / 256 hop) is ~32ms -- confirm it's actually in use.
    assert MODEL_ALGORITHMIC_DELAY_MS["aegis-aec-gate"] == 32.0


def test_all_five_models_have_an_explicit_delay_entry():
    expected_keys = {
        "aegis-se-primary", "aegis-se-escalation", "aegis-se-crosscheck",
        "aegis-clf-gate", "aegis-aec-gate",
    }
    assert expected_keys.issubset(MODEL_ALGORITHMIC_DELAY_MS.keys())
