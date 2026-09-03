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
        self.buffer = np.zeros(filter_length, dtype=np.float32)

    def reset(self):
        """Resets filter weights and delay line buffer."""
        self.weights.fill(0.0)
        self.buffer.fill(0.0)

    def step(self, reference: float, desired: float) -> Tuple[float, float]:
        """
        Processes a single audio sample:
        Args:
            reference: Reference noise input x(n).
            desired: Primary input d(n) containing signal + noise.
        Returns:
            (filtered_estimate y(n), error_output e(n))
        """
        # Shift delay line buffer
        self.buffer[1:] = self.buffer[:-1]
        self.buffer[0] = reference

        # Compute estimated noise
        est_noise = float(np.dot(self.weights, self.buffer))
        error = desired - est_noise

        # Normalized weight update
        norm = np.dot(self.buffer, self.buffer) + self.eps
        norm_step = (self.step_size / norm) * error
        self.weights = self.leakage * self.weights + norm_step * self.buffer

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
    """

    def __init__(
        self,
        ai_model: nn.Module,
        enable_adaptive_filter: bool = True,
        filter_length: int = 64,
        step_size: float = 0.05,
        device: Optional[torch.device] = None,
    ):
        self.ai_model = ai_model
        self.enable_adaptive_filter = enable_adaptive_filter
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ai_model.to(self.device)
        self.ai_model.eval()

        self.lms_filter = NormalizedLMSFilter(
            filter_length=filter_length,
            step_size=step_size,
        )

    def process_frame(
        self,
        primary_audio: Union[torch.Tensor, np.ndarray],
        reference_audio: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Executes hybrid noise cancellation on input audio frame:
        Args:
            primary_audio: Tactical headset primary microphone signal.
            reference_audio: Optional external reference noise microphone signal.
        Returns:
            Enhanced speech audio array.
        """
        # Convert to torch tensor for AI inference
        if isinstance(primary_audio, np.ndarray):
            in_t = torch.from_numpy(primary_audio).float()
        else:
            in_t = primary_audio.float()

        if in_t.dim() == 1:
            in_t = in_t.unsqueeze(0)  # (1, T)

        in_t = in_t.to(self.device)

        with torch.no_grad():
            ai_enhanced = self.ai_model(in_t)

        ai_out = ai_enhanced.squeeze().cpu().numpy()
        prim_np = in_t.squeeze().cpu().numpy()

        if not self.enable_adaptive_filter:
            return ai_out

        # If external reference microphone is provided, use it directly.
        # Otherwise, estimate reference as the removed noise: x(n) = primary(n) - ai_enhanced(n)
        if reference_audio is not None:
            ref_np = reference_audio if isinstance(reference_audio, np.ndarray) else reference_audio.cpu().numpy()
            ref_np = ref_np.squeeze()
        else:
            ref_np = prim_np - ai_out

        # Stage 2: Classical adaptive filtering for residual cancellation
        _, residual_clean = self.lms_filter.filter_batch(reference=ref_np, desired=ai_out)

        return residual_clean
