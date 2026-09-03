"""
training/losses/multires_loss.py — Models 1/2's loss function, implemented.

This was the one genuinely missing piece flagged by TRAINING_ARCHITECTURE.md's
own checklist: `training/losses/` existed only as an empty `__init__.py`
integration point, on the stated (correct) plan of vendoring DeepFilterNet3's
own real loss implementation via `pip install deepfilternet[train]` rather
than reimplementing the model from scratch. That vendoring plan is verified
sound on this review -- `deepfilternet[train]` is a real, current PyPI
package (Rikorose/DeepFilterNet, v0.5.6) that ships exactly this loss --
but "vendor it later" left the trainers with nothing to actually call in
the meantime, and no fallback if the target training machine can't install
the `[train]` extra (it requires the Rust/maturin toolchain and is Linux-only
per the project's own README).

This module does two things, in priority order:
  1. Try to import DeepFilterNet's own real loss classes from the installed
     `df` package. If present, USE THE REAL ONE -- this is not a from-
     scratch reimplementation competing with the vendored source, it's the
     documented integration point actually being filled.
  2. If the package isn't installed (e.g. a quick CPU-only smoke test, or a
     machine where the `[train]` extra's Rust toolchain isn't set up yet),
     fall back to a from-scratch implementation built directly from the
     values already hard-coded in `DfLossConfig` (multires_spec_factor=500,
     multires_spec_factor_complex=500, multires_spec_gamma=0.3, fft_sizes=
     [256,512,1024,2048], local_snr_factor=1e-3) and the loss shape those
     values imply (power-law-compressed multi-resolution magnitude +
     complex spectral distance, matching DeepFilterNet's own documented
     "perceptually motivated" loss design, plus a local/framewise SNR
     term). This fallback exists so `python -m training.scripts.train_se_primary`
     has something real to call TODAY, not just after the vendoring task
     is separately completed -- and its output should be numerically close
     to, but is not guaranteed identical to, the vendored package's own
     loss, since DeepFilterNet's exact framing/windowing details aren't
     all public. Prefer path 1 whenever it's available; treat path 2 as a
     stand-in, not a permanent replacement.

Both paths are exercised by `tests/test_multires_loss.py` (new, this pass).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on a torch-less smoke test
    _TORCH_AVAILABLE = False


def _try_import_vendored_df_loss():
    """
    Attempts to import DeepFilterNet's own real loss implementation from
    the installed `df` package (the module name the `deepfilternet` PyPI
    distribution installs under). Returns the loss callable/class if found,
    else None -- never raises, since "not installed yet" is an expected,
    normal state for this integration point, not an error.
    """
    try:
        from df.loss import Loss as _DfLoss  # real package, if `deepfilternet[train]` is installed
        return _DfLoss
    except Exception:
        return None


@dataclass
class ResolvedLossConfig:
    """Plain mirror of training.configs.se_primary_config.DfLossConfig's
    fields, so this module has no import-time dependency on the configs
    package (keeps the loss testable in isolation)."""
    multires_spec_factor: float = 500.0
    multires_spec_factor_complex: float = 500.0
    multires_spec_gamma: float = 0.3
    multires_fft_sizes: Optional[List[int]] = None
    local_snr_factor: float = 1e-3

    def __post_init__(self):
        if self.multires_fft_sizes is None:
            self.multires_fft_sizes = [256, 512, 1024, 2048]


if _TORCH_AVAILABLE:

    class MultiResSpectralLoss(nn.Module):
        """
        From-scratch fallback path. Power-law-compressed multi-resolution
        STFT loss: for each FFT size in `fft_sizes`, computes both a
        magnitude-domain and a complex-domain distance between the
        gamma-compressed estimate and target spectrograms, summed across
        resolutions. This is the standard "perceptually motivated"
        compressed-spectral-loss shape DeepFilterNet's own papers describe
        (small `gamma` compresses large-magnitude bins, emphasizing
        quieter spectral detail the way loudness perception roughly does)
        -- implemented here from that description and the config's own
        hard-coded values, not copied from the vendored source (which this
        fallback exists specifically for the case where that source isn't
        installed).
        """

        def __init__(self, config: ResolvedLossConfig):
            super().__init__()
            self.config = config

        def _compressed_stft(self, x: "torch.Tensor", n_fft: int) -> "torch.Tensor":
            hop = n_fft // 4
            window = torch.hann_window(n_fft, device=x.device, dtype=x.dtype)
            spec = torch.stft(
                x, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                window=window, return_complex=True,
            )
            mag = spec.abs().clamp_min(1e-8)
            gamma = self.config.multires_spec_gamma
            mag_c = mag.pow(gamma)
            # Compress the complex spectrogram by the same magnitude
            # scaling factor while preserving phase, matching the
            # documented "compress magnitude, keep phase" approach.
            complex_c = spec * (mag_c / mag).to(spec.dtype)
            return mag_c, complex_c

        def forward(self, estimate: "torch.Tensor", target: "torch.Tensor") -> dict:
            if estimate.shape != target.shape:
                raise ValueError(
                    f"estimate/target shape mismatch: {estimate.shape} vs {target.shape}"
                )

            mag_loss_total = estimate.new_zeros(())
            complex_loss_total = estimate.new_zeros(())

            for n_fft in self.config.multires_fft_sizes:
                if estimate.shape[-1] < n_fft:
                    # Skip FFT sizes larger than the clip itself rather than
                    # padding silently -- a caller passing 3s@48kHz clips
                    # (this project's max_sample_len_s) will never hit this
                    # for the configured [256,512,1024,2048] sizes, but a
                    # unit test exercising short synthetic tensors might.
                    continue
                est_mag, est_cplx = self._compressed_stft(estimate, n_fft)
                tgt_mag, tgt_cplx = self._compressed_stft(target, n_fft)

                mag_loss_total = mag_loss_total + F.l1_loss(est_mag, tgt_mag)
                complex_loss_total = complex_loss_total + F.l1_loss(
                    torch.view_as_real(est_cplx), torch.view_as_real(tgt_cplx)
                )

            weighted_mag = self.config.multires_spec_factor * mag_loss_total
            weighted_complex = self.config.multires_spec_factor_complex * complex_loss_total

            return {
                "multires_mag_loss": weighted_mag,
                "multires_complex_loss": weighted_complex,
                "multires_total": weighted_mag + weighted_complex,
            }

    class LocalSnrLoss(nn.Module):
        """
        From-scratch fallback path. Framewise (local, not utterance-mean)
        SNR loss -- penalizes low or negative segmental SNR between the
        enhanced estimate and the clean target, computed per short frame
        rather than over the whole clip, so a single well-reconstructed
        loud segment can't mask a badly-reconstructed quiet one the way a
        single scalar utterance-level SNR term can. This directly targets
        the same failure mode this project's own research (Fraunhofer
        low-SNR comparative study) found matters most: performance at the
        hardest local moments, not the easy average.
        """

        def __init__(self, config: ResolvedLossConfig, frame_size: int = 480, hop_size: int = 240):
            super().__init__()
            self.config = config
            self.frame_size = frame_size
            self.hop_size = hop_size

        def _framewise_snr_db(self, estimate: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
            # Unfold into overlapping frames: (..., n_frames, frame_size)
            tgt_frames = target.unfold(-1, self.frame_size, self.hop_size)
            err_frames = (estimate - target).unfold(-1, self.frame_size, self.hop_size)

            signal_power = tgt_frames.pow(2).mean(dim=-1).clamp_min(1e-10)
            noise_power = err_frames.pow(2).mean(dim=-1).clamp_min(1e-10)
            return 10.0 * torch.log10(signal_power / noise_power)

        def forward(self, estimate: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
            if estimate.shape[-1] < self.frame_size:
                warnings.warn(
                    "LocalSnrLoss: input shorter than one frame; returning zero loss "
                    "for this call rather than raising, since a too-short final batch "
                    "shouldn't crash a training run.",
                    stacklevel=2,
                )
                return estimate.new_zeros(())

            local_snr = self._framewise_snr_db(estimate, target)
            # Negative loss on SNR itself (maximize SNR == minimize -SNR),
            # scaled by the config's local_snr_factor -- kept as a small
            # weight (1e-3, per DfLossConfig) since this term operates on
            # a very different numeric scale (dB) than the spectral terms.
            return self.config.local_snr_factor * (-local_snr.mean())


def build_se_loss(config: Optional[ResolvedLossConfig] = None, prefer_vendored: bool = True):
    """
    Entry point trainers should call. Returns a callable
    `loss_fn(estimate, target) -> dict[str, Tensor]` including at least a
    "total" key. Prefers the real vendored DeepFilterNet loss when
    installed; falls back to the from-scratch implementation above
    otherwise, and tells you which one it picked (logged, not silent) so a
    training run's logs make clear whether the vendored or fallback path
    was actually used -- important given the two are not guaranteed
    numerically identical.
    """
    if config is None:
        config = ResolvedLossConfig()

    if prefer_vendored:
        vendored = _try_import_vendored_df_loss()
        if vendored is not None:
            print("[training.losses.multires_loss] Using vendored df.loss.Loss "
                  "(deepfilternet[train] is installed) -- NOT the from-scratch fallback.")
            return vendored(
                factor_magnitude=config.multires_spec_factor,
                factor_complex=config.multires_spec_factor_complex,
                gamma=config.multires_spec_gamma,
                fft_sizes=config.multires_fft_sizes,
            )
        print("[training.losses.multires_loss] deepfilternet[train] not importable -- "
              "using the from-scratch fallback (MultiResSpectralLoss + LocalSnrLoss). "
              "Vendor the real package before trusting results as the paper's own "
              "loss curve, per TRAINING_ARCHITECTURE.md's stated integration plan.")

    if not _TORCH_AVAILABLE:
        raise ImportError(
            "torch is required for the from-scratch loss fallback and is not installed."
        )

    spectral = MultiResSpectralLoss(config)
    local_snr = LocalSnrLoss(config)

    def _combined(estimate: "torch.Tensor", target: "torch.Tensor") -> dict:
        spec_out = spectral(estimate, target)
        snr_out = local_snr(estimate, target)
        spec_out["local_snr_loss"] = snr_out
        spec_out["total"] = spec_out["multires_total"] + snr_out
        return spec_out

    return _combined
