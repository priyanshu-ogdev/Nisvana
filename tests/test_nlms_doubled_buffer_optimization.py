"""
tests/test_nlms_doubled_buffer_optimization.py

Regression tests for NormalizedLMSFilter's buffer management, replacing
the original O(filter_length)-per-sample shift with a "doubled buffer"
(O(1) write, plain contiguous slice for the dot product).

WORTH RECORDING HONESTLY: the first attempted fix (a circular buffer with
a split two-segment dot product) was numerically verified correct and
then directly MEASURED to be ~2x SLOWER than the original shift-based
approach, not faster -- at filter_length=64, numpy's per-call overhead
dominates over the O(N) work, and the split-dot-product approach roughly
doubled the number of numpy calls per sample to avoid one small, cheap
array copy. The doubled-buffer approach here keeps the same call-count as
the original (one plain np.dot, no splitting/reversing) while still
getting O(1) writes, and was measured (not just reasoned about) to be
faster before being kept. Complexity-order reasoning without measurement
would have shipped the slower "fix."
"""

import numpy as np


def _extract_normalized_lms_filter_class():
    """
    Extracts and execs ONLY the NormalizedLMSFilter class from the real
    shipped hybrid_anc.py, bypassing HybridAncPipeline (which needs real
    torch.nn.Module/torch.device at class-definition time) -- this class
    itself has zero torch dependency, so this lets it be tested without
    a full torch install.
    """
    src = open("inference/runtime/hybrid_anc.py").read()
    start = src.index("class NormalizedLMSFilter")
    end = src.index("class HybridAncPipeline")
    class_src = src[start:end]

    from typing import Tuple, Optional
    ns = {"np": np, "Tuple": Tuple, "Optional": Optional}
    exec(compile(class_src, "<hybrid_anc.NormalizedLMSFilter>", "exec"), ns)
    return ns["NormalizedLMSFilter"]


class _ReferenceOldNLMS:
    """The original shift-based implementation, kept here as the ground
    truth every optimization attempt must match exactly."""

    def __init__(self, filter_length=64, step_size=0.05, leakage=0.9999, eps=1e-6):
        self.filter_length = filter_length
        self.step_size = step_size
        self.leakage = leakage
        self.eps = eps
        self.weights = np.zeros(filter_length, dtype=np.float32)
        self.buffer = np.zeros(filter_length, dtype=np.float32)

    def step(self, reference, desired):
        self.buffer[1:] = self.buffer[:-1]
        self.buffer[0] = reference
        est_noise = float(np.dot(self.weights, self.buffer))
        error = desired - est_noise
        norm = np.dot(self.buffer, self.buffer) + self.eps
        norm_step = (self.step_size / norm) * error
        self.weights = self.leakage * self.weights + norm_step * self.buffer
        return est_noise, error


class TestNlmsDoubledBufferCorrectness:
    def test_matches_reference_over_many_samples_with_wraparound(self):
        NormalizedLMSFilter = _extract_normalized_lms_filter_class()
        real = NormalizedLMSFilter(filter_length=64)
        ref = _ReferenceOldNLMS(filter_length=64)

        rng = np.random.default_rng(7)
        max_diff = 0.0
        # 3000 samples >> filter_length=64, so many full buffer
        # wraparounds are exercised, not just the first cycle.
        for _ in range(3000):
            r = float(rng.standard_normal())
            d = float(rng.standard_normal())
            e1, err1 = real.step(r, d)
            e2, err2 = ref.step(r, d)
            max_diff = max(max_diff, abs(e1 - e2), abs(err1 - err2))

        assert max_diff < 1e-4  # float32 tolerance

    def test_filter_batch_matches_step_by_step_reference(self):
        NormalizedLMSFilter = _extract_normalized_lms_filter_class()
        real = NormalizedLMSFilter(filter_length=64)
        ref = _ReferenceOldNLMS(filter_length=64)

        rng = np.random.default_rng(99)
        refs_sig = rng.standard_normal(480).astype(np.float32)
        dess_sig = rng.standard_normal(480).astype(np.float32)

        est_batch, err_batch = real.filter_batch(refs_sig, dess_sig)
        ref_est = np.array([ref.step(r, d)[0] for r, d in zip(refs_sig, dess_sig)])

        assert np.max(np.abs(est_batch - ref_est)) < 1e-3

    def test_reset_clears_all_state(self):
        NormalizedLMSFilter = _extract_normalized_lms_filter_class()
        real = NormalizedLMSFilter(filter_length=64)

        rng = np.random.default_rng(1)
        for _ in range(200):
            real.step(float(rng.standard_normal()), float(rng.standard_normal()))
        assert not np.all(real.weights == 0.0)  # confirm it actually adapted first

        real.reset()
        assert np.all(real.weights == 0.0)
        assert np.all(real.dbuf == 0.0)
        assert real._idx == 0

    def test_doubled_buffer_write_is_duplicated_correctly(self):
        """
        Structural sanity check on the doubled-buffer mechanism itself:
        every write must land at both i and i+N, or the contiguous-window
        trick silently breaks the moment the window wraps past N.
        """
        NormalizedLMSFilter = _extract_normalized_lms_filter_class()
        real = NormalizedLMSFilter(filter_length=8)
        real.step(3.5, 0.0)
        assert real.dbuf[0] == 3.5
        assert real.dbuf[8] == 3.5  # i=0, N=8 -> duplicated at index 0 and 8
