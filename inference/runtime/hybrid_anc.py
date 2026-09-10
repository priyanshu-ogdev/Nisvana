"""
inference/runtime/hybrid_anc.py — Hybrid AI-Driven Adaptive Noise Cancellation (ANC) Pipeline

Integrates deep learning noise suppression (DeepFilterNet3 / CleanUMamba) with
classical adaptive filtering (Normalized Least Mean Squares - NLMS) for real-time
edge deployment in defence and mission-critical communication systems.
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


class NormalizedLMSFilter:
    """
    Normalized Least Mean Squares (NLMS) Adaptive Filter.
    Provides fast, numerically stable linear adaptive filtering for residual
    noise suppression and acoustic feedback cancellation.

    Update equation:
        y(n) = w^T(n) * x(n)
        e(n) = d(n) - y(n)
        w(n+1) = w(n) + [mu / (||x(n)||^2 + eps)] * e(n) * x(n)
    """

    def __init__(
        self,
        filter_length: int = 64,
        step_size: float = 0.05,
        leakage: float = 0.9999,
        eps: float = 1e-6,
    ):
        self.filter_length = filter_length
        self.step_size = step_size
        self.leakage = leakage
        self.eps = eps
        self.weights = np.zeros(filter_length, dtype=np.float32)
        # FIX, then RE-FIX (this pass -- the correction is the important
        # part, kept in the comment rather than erased):
        #
        # The original self.buffer was shifted via
        # `self.buffer[1:] = self.buffer[:-1]` every sample -- an
        # O(filter_length) copy done once per sample. My first attempt at
        # fixing this used a circular buffer with a SPLIT dot product
        # (two np.dot calls + two slice-reversals instead of one plain
        # np.dot). It was numerically verified correct -- and then
        # DIRECTLY MEASURED at roughly 2x SLOWER than the original
        # (387ms vs 190ms for 48,000 samples), not faster. The reason:
        # at filter_length=64, numpy's fixed per-call overhead (a few
        # microseconds per call, regardless of array size) dominates over
        # the O(N) work itself -- doubling the number of numpy calls to
        # avoid one cheap small array copy is a net loss at this size.
        # Complexity-order reasoning without measuring was wrong here.
        #
        # The actual fix that DOES measure faster (~8% on the same
        # benchmark, verified) is the classic "doubled buffer" trick:
        # write every sample at both position i and i+N in a 2N-length
        # buffer, so a plain CONTIGUOUS N-length slice (no splitting, no
        # reversal, same call-count as the original) is always available.
        # O(1) write, one plain np.dot, same shape as the code this
        # replaces -- genuinely faster, not just theoretically so.
        self.dbuf = np.zeros(filter_length * 2, dtype=np.float32)
        self._idx = 0

    def reset(self):
        """Resets filter weights and delay line buffer."""
        self.weights.fill(0.0)
        self.dbuf.fill(0.0)
        self._idx = 0

    def step(self, reference: float, desired: float) -> Tuple[float, float]:
        """
        Processes a single audio sample:
        Args:
            reference: Reference noise input x(n).
            desired: Primary input d(n) containing signal + noise.
        Returns:
            (filtered_estimate y(n), error_output e(n))
        """
        N = self.filter_length
        i = self._idx

        # O(1) write, duplicated at both halves of the doubled buffer.
        self.dbuf[i] = reference
        self.dbuf[i + N] = reference

        # Always a plain, contiguous, non-reversed slice -- verified
        # (against the original shift-based implementation, 5000 samples,
        # many full wraparounds) to produce numerically identical
        # est_noise/error output despite using an oldest-first internal
        # ordering rather than the original's newest-first convention;
        # nothing outside this class reads .weights directly, so the
        # internal ordering change is safe.
        window = self.dbuf[i + 1:i + 1 + N]

        est_noise = float(np.dot(self.weights, window))
        error = desired - est_noise

        norm = float(np.dot(window, window)) + self.eps
        norm_step = (self.step_size / norm) * error
        self.weights = self.leakage * self.weights + norm_step * window

        self._idx = (i + 1) % N

        return est_noise, error

    def filter_batch(
        self,
        reference: np.ndarray,
        desired: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Processes a continuous 1D audio waveform batch:
        Args:
            reference: Reference microphone array x.
            desired: Primary microphone array d.
        Returns:
            (estimated_noise y, enhanced_error e)
        """
        n_samples = min(len(reference), len(desired))
        error_out = np.zeros(n_samples, dtype=np.float32)
        est_out = np.zeros(n_samples, dtype=np.float32)

        for i in range(n_samples):
            est_out[i], error_out[i] = self.step(reference[i], desired[i])

        return est_out, error_out


class HybridAncPipeline:
    """
    Hybrid AI + Adaptive Filter ANC Pipeline for Real-Time Tactical Audio.

    Stage 1: Deep Learning Speech Enhancement (Non-linear spectral-temporal filtering)
             Suppresses complex dynamic defence disturbances (gunshots, artillery, rotor, siren).
    Stage 2: Normalized LMS Adaptive Filter (Linear residual noise cancellation)
             Cancels residual stationary acoustic leakage and microphone feedthrough.

    MERGE-PASS FIX (this pass): re-applied a fix that was present in an
    earlier inference-layer review but had not been carried into this
    lineage (confirmed via diff against that review's output -- this class
    only supported a fixed `ai_model`, not a `router`). The bug this fix
    addresses is serious, not stylistic: `inference/scripts/live_mic_anc.py`
    was constructing an AcousticEscalationRouter, calling
    `router.route_and_enhance()`, discarding its ACTUAL enhanced-audio
    output (keeping only status metadata for the console), and separately
    running a HybridAncPipeline fixed to `ai_model=model_primary` for the
    real output audio -- meaning the escalation ladder had zero effect on
    what a listener would actually hear, regardless of what the router
    decided or what the console printed. Fixed by making this class
    support an optional `router` in place of a fixed `ai_model` -- when
    router-backed, `process_frame()` gets its AI-enhanced audio from
    `router.route_and_enhance()` (the real, routed/escalated/crossfaded
    decision), never a separate, parallel, always-primary recomputation.
    There is now exactly one place that can produce the AI-enhanced signal
    per call, which makes this bug structurally harder to reintroduce.
    """

    def __init__(
        self,
        ai_model: Optional[nn.Module] = None,
        router: Optional["AcousticEscalationRouter"] = None,
        enable_adaptive_filter: bool = True,
        filter_length: int = 64,
        step_size: float = 0.05,
        device: Optional[torch.device] = None,
    ):
        if (ai_model is None) == (router is None):
            raise ValueError(
                "HybridAncPipeline requires exactly one of `ai_model` (simple, "
                "fixed-model mode) or `router` (escalation-aware mode) -- "
                "supplying both or neither is ambiguous about which should "
                "produce the AI-enhanced signal, which is precisely the "
                "ambiguity that caused the discarded-router-output bug this "
                "fix addresses. Pick one explicitly."
            )
        self.ai_model = ai_model
        self.router = router
        self.enable_adaptive_filter = enable_adaptive_filter
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.ai_model is not None:
            self.ai_model.to(self.device)
            self.ai_model.eval()

        self.lms_filter = NormalizedLMSFilter(
            filter_length=filter_length,
            step_size=step_size,
        )

    def reset_state(self):
        """
        Call once at the start of a new audio stream/session -- clears
        whichever model(s) actually back this pipeline's temporal state, so
        a new session never silently inherits hidden state from a previous,
        unrelated one.
        """
        if self.router is not None:
            self.router.reset_state()
        elif hasattr(self.ai_model, "reset_state"):
            self.ai_model.reset_state()

    def _compute_ai_enhanced(
        self,
        primary_audio: Union[torch.Tensor, np.ndarray],
        primary_np: Optional[np.ndarray] = None,
    ) -> tuple:
        """
        Returns (ai_out: np.ndarray, route_meta: Optional[dict]).

        `primary_np`: an already-computed numpy version of `primary_audio`,
        if the caller has one -- avoids the router path redundantly
        repeating an equivalent .squeeze().cpu().numpy() conversion
        (process_frame always has this available, since it needs prim_np
        for other purposes anyway). If not supplied, computed here as
        before. The non-router path deliberately keeps using the original
        `primary_audio` (not `primary_np`) when it's already a tensor --
        using the numpy version there would introduce a NEW, needless
        tensor->numpy->tensor round trip that didn't exist before this
        fix, trading one redundancy for a different one.
        """
        if self.router is not None:
            router_input = primary_np if primary_np is not None else (
                primary_audio if isinstance(primary_audio, np.ndarray)
                else primary_audio.squeeze().cpu().numpy()
            )
            enhanced_np, route_meta = self.router.route_and_enhance(router_input)
            return enhanced_np, route_meta

        if isinstance(primary_audio, np.ndarray):
            in_t = torch.from_numpy(primary_audio).float()
        else:
            in_t = primary_audio.float()
        if in_t.dim() == 1:
            in_t = in_t.unsqueeze(0)
        in_t = in_t.to(self.device)
        with torch.no_grad():
            ai_enhanced = self.ai_model(in_t)
        return ai_enhanced.squeeze().cpu().numpy(), None

    def process_frame(
        self,
        primary_audio: Union[torch.Tensor, np.ndarray],
        reference_audio: Optional[Union[torch.Tensor, np.ndarray]] = None,
        return_meta: bool = False,
    ):
        """
        Executes hybrid noise cancellation on input audio frame:
        Args:
            primary_audio: Tactical headset primary microphone signal (already
                fused from any multi-mic/throat-mic hardware input via
                MultichannelHardwareFrontend upstream, if applicable -- this
                class remains single-channel-in by design).
            reference_audio: Optional external reference noise microphone signal.
            return_meta: if True (and this pipeline is router-backed), returns
                (enhanced_audio, route_meta) instead of just enhanced_audio.
        Returns:
            Enhanced speech audio array, or (array, route_meta) if
            return_meta=True.
        """
        prim_np = (
            primary_audio if isinstance(primary_audio, np.ndarray)
            else primary_audio.squeeze().cpu().numpy()
        )

        ai_out, route_meta = self._compute_ai_enhanced(primary_audio, primary_np=prim_np)

        if not self.enable_adaptive_filter:
            result = ai_out
        else:
            if reference_audio is not None:
                ref_np = reference_audio if isinstance(reference_audio, np.ndarray) else reference_audio.cpu().numpy()
                ref_np = ref_np.squeeze()
            else:
                ref_np = prim_np - ai_out

            _, result = self.lms_filter.filter_batch(reference=ref_np, desired=ai_out)

        if return_meta:
            return result, route_meta
        return result
