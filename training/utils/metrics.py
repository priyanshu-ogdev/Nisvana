"""
training/utils/metrics.py — Audio Evaluation Metrics & Worst-Class Aggregator

Computes:
- Empirical SNR (dB)
- Scale-Invariant SNR (SI-SNR dB)
- Echo Return Loss Enhancement (ERLE dB)
- PESQ / MOS proxy
- Per-class breakdown dict for WorstClassCheckpointSelector
"""

from typing import Dict, List, Optional
import torch


def compute_snr_db(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> float:
    """Computes empirical signal-to-noise ratio in decibels."""
    noise = estimate - target
    s_pwr = torch.mean(target ** 2).clamp_min(eps)
    n_pwr = torch.mean(noise ** 2).clamp_min(eps)
    return float(10.0 * torch.log10(s_pwr / n_pwr).item())


def compute_si_snr_db(estimate: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Scale-Invariant Signal-to-Noise Ratio (SI-SNR) in dB."""
    est_zm = estimate - torch.mean(estimate, dim=-1, keepdim=True)
    tgt_zm = target - torch.mean(target, dim=-1, keepdim=True)

    dot = torch.sum(est_zm * tgt_zm, dim=-1, keepdim=True)
    tgt_energy = torch.sum(tgt_zm ** 2, dim=-1, keepdim=True) + eps

    s_target = (dot / tgt_energy) * tgt_zm
    e_noise = est_zm - s_target

    si_snr = 10.0 * torch.log10(
        (torch.sum(s_target ** 2, dim=-1) + eps) /
        (torch.sum(e_noise ** 2, dim=-1) + eps)
    )
    return float(si_snr.mean().item())


def compute_erle_db(mic: torch.Tensor, error: torch.Tensor, eps: float = 1e-10) -> float:
    """Echo Return Loss Enhancement in dB for Acoustic Echo Cancellation."""
    mic_pwr = torch.mean(mic ** 2).clamp_min(eps)
    err_pwr = torch.mean(error ** 2).clamp_min(eps)
    return float(10.0 * torch.log10(mic_pwr / err_pwr).item())


def compute_pesq_proxy(estimate: torch.Tensor, target: torch.Tensor) -> float:
    """
    Lightweight, fast PESQ proxy based on power-law compressed spectral L1 distance.
    Bounded in [1.0, 4.5] MOS range.
    """
    spec_dist = torch.mean(torch.abs(estimate - target)).item()
    return float(max(1.0, min(4.5, 4.5 - spec_dist * 5.0)))


def build_eval_metrics_dict(
    estimates: torch.Tensor,
    targets: torch.Tensor,
    classes: List[str],
    aggregate_metric_name: str = "pesq_aggregate",
) -> Dict[str, float]:
    """
    Constructs the formatted metric dictionary required by WorstClassCheckpointSelector.
    Populates aggregate PESQ, plus f"pesq_{cls}" and f"snr_{cls}" for each class present.
    """
    metrics: Dict[str, float] = {}
    n_samples = estimates.shape[0] if estimates.dim() >= 2 else 1
    if isinstance(classes, str):
        classes = [classes] * n_samples

    all_pesq = []
    class_pesq: Dict[str, List[float]] = {}
    class_snr: Dict[str, List[float]] = {}

    for i in range(min(n_samples, len(classes))):
        est = estimates[i] if estimates.dim() >= 2 else estimates
        tgt = targets[i] if targets.dim() >= 2 else targets
        cls = classes[i]

        p = compute_pesq_proxy(est, tgt)
        s = compute_snr_db(est, tgt)

        all_pesq.append(p)
        class_pesq.setdefault(cls, []).append(p)
        class_snr.setdefault(cls, []).append(s)

    metrics[aggregate_metric_name] = float(sum(all_pesq) / max(len(all_pesq), 1))
    for cls in class_pesq:
        metrics[f"pesq_{cls}"] = float(sum(class_pesq[cls]) / len(class_pesq[cls]))
        metrics[f"snr_{cls}"] = float(sum(class_snr[cls]) / len(class_snr[cls]))

    return metrics
