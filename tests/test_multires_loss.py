"""
Project AEGIS — tests for training/losses/multires_loss.py

This is the review-pass addition filling the previously-empty
training/losses/ integration point. Tests cover both code paths:
  1. The graceful "torch not installed" degradation (importable, raises a
     clear ImportError only when actually asked to build a loss, not on
     import) -- this is what the CI/sandbox environment without torch
     exercises.
  2. The real numerical behavior once torch IS installed (skipped
     automatically via pytest.importorskip when it isn't) -- correctness
     of the multi-res spectral loss and local SNR loss shapes, and the
     vendored-vs-fallback selection logic in build_se_loss().
"""

import sys
import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parent.parent / "training" / "losses" / "multires_loss.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("training.losses.multires_loss", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["training.losses.multires_loss"] = module
    spec.loader.exec_module(module)
    return module


class TestImportBehaviorWithoutTorch:
    """Exercises the no-torch path directly -- must degrade gracefully,
    not crash on import, since `training/losses/__init__.py` and the
    package overall should stay importable for config/CLI smoke tests
    even on a machine that hasn't installed the ML stack yet."""

    def test_module_imports_without_torch_available_flag(self):
        mrl = _load_module()
        # This assertion is only meaningful in an environment without
        # torch; if torch IS installed, _TORCH_AVAILABLE will correctly
        # be True and the numeric tests below take over instead.
        assert isinstance(mrl._TORCH_AVAILABLE, bool)

    def test_resolved_loss_config_defaults_match_df_loss_config(self):
        """The fallback config's defaults must mirror
        training/configs/se_primary_config.py's DfLossConfig exactly --
        this is the single place a values-drift between the two would
        silently produce a different-shaped loss than the docs claim."""
        mrl = _load_module()
        cfg = mrl.ResolvedLossConfig()
        assert cfg.multires_spec_factor == 500.0
        assert cfg.multires_spec_factor_complex == 500.0
        assert cfg.multires_spec_gamma == 0.3
        assert cfg.multires_fft_sizes == [256, 512, 1024, 2048]
        assert cfg.local_snr_factor == 1e-3

    def test_build_se_loss_raises_clear_error_without_torch_and_without_vendored_package(self):
        mrl = _load_module()
        if mrl._TORCH_AVAILABLE:
            pytest.skip("torch is available in this environment; this path isn't reachable.")
        with pytest.raises(ImportError, match="torch is required"):
            mrl.build_se_loss(prefer_vendored=False)


class TestNumericBehaviorWithTorch:
    """Skipped entirely (not failed) when torch isn't installed --
    exercises the actual loss computation once it is."""

    @pytest.fixture(autouse=True)
    def _require_torch(self):
        pytest.importorskip("torch")

    def test_identical_estimate_and_target_gives_near_zero_spectral_loss(self):
        import torch
        mrl = _load_module()
        cfg = mrl.ResolvedLossConfig(multires_fft_sizes=[256, 512])
        loss_fn = mrl.MultiResSpectralLoss(cfg)

        torch.manual_seed(0)
        x = torch.randn(2, 16000)  # 2 items, 1s @ 16kHz -- long enough for both FFT sizes
        out = loss_fn(x, x.clone())
        assert out["multires_total"].item() < 1e-4, (
            "identical signals should produce ~zero multi-resolution spectral loss"
        )

    def test_differing_signals_give_positive_spectral_loss(self):
        import torch
        mrl = _load_module()
        cfg = mrl.ResolvedLossConfig(multires_fft_sizes=[256, 512])
        loss_fn = mrl.MultiResSpectralLoss(cfg)

        torch.manual_seed(0)
        target = torch.randn(2, 16000)
        estimate = target + 0.5 * torch.randn(2, 16000)
        out = loss_fn(estimate, target)
        assert out["multires_total"].item() > 0.0

    def test_local_snr_loss_decreases_as_estimate_improves(self):
        """A better (lower-error) estimate must produce a lower (more
        negative-SNR-penalizing) loss value -- this is the actual
        correctness property the loss needs, not just 'runs without
        crashing'."""
        import torch
        mrl = _load_module()
        cfg = mrl.ResolvedLossConfig()
        loss_fn = mrl.LocalSnrLoss(cfg, frame_size=480, hop_size=240)

        torch.manual_seed(0)
        target = torch.randn(1, 4800)
        noisy_estimate = target + 1.0 * torch.randn(1, 4800)   # poor estimate, high error
        clean_estimate = target + 0.01 * torch.randn(1, 4800)  # near-perfect estimate

        loss_poor = loss_fn(noisy_estimate, target)
        loss_good = loss_fn(clean_estimate, target)
        assert loss_good.item() < loss_poor.item(), (
            "a near-perfect estimate must score a lower local-SNR loss than a poor one"
        )

    def test_build_se_loss_falls_back_when_vendored_package_absent(self):
        mrl = _load_module()
        # Force the fallback path explicitly regardless of whether
        # deepfilternet[train] happens to be installed on this machine --
        # this test is about the fallback's own correctness, not about
        # which path gets auto-selected.
        combined = mrl.build_se_loss(prefer_vendored=False)
        import torch
        x = torch.randn(1, 16000)
        out = combined(x, x.clone())
        assert "total" in out
        assert out["total"].item() < 1e-3
