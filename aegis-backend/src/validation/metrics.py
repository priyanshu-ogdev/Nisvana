"""validation/metrics.py — G3: Offline Validation Metrics.

Calculates PESQ, STOI, absolute SNR, SNR improvement, segmental SNR, 
and clipped-frame recoverability fraction.
"""
from __future__ import annotations
import numpy as np

def compute_pesq(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    try:
        from pesq import pesq
        # pesq accepts 16k or 8k, so typically you'd resample.
        # Here we mock resampling or assume 16k is passed for actual testing.
        # Using 'wb' for wideband
        return pesq(sr, ref, deg, 'wb')
    except Exception:
        return 0.0

def compute_stoi(ref: np.ndarray, deg: np.ndarray, sr: int) -> float:
    try:
        from pystoi import stoi
        return stoi(ref, deg, sr, extended=False)
    except Exception:
        return 0.0

def compute_snr_absolute(signal: np.ndarray) -> float:
    """Absolute SNR: 10 * log10( sum(signal^2) / len ) - not true SNR, but raw energy scale."""
    rms = np.sqrt(np.mean(signal**2) + 1e-10)
    return 20.0 * np.log10(rms)

def compute_snr_improvement(raw: np.ndarray, enhanced: np.ndarray) -> float:
    rms_raw = np.sqrt(np.mean(raw**2) + 1e-10)
    rms_enh = np.sqrt(np.mean(enhanced**2) + 1e-10)
    return 20.0 * np.log10(rms_raw / rms_enh)

def compute_segmental_metrics(ref: np.ndarray, deg: np.ndarray, sr: int, frame_ms: int = 20) -> tuple[float, float]:
    """Returns (segmental_snr, recoverability_fraction)."""
    frame_len = int((frame_ms / 1000.0) * sr)
    if frame_len == 0 or len(ref) < frame_len:
        return 0.0, 0.0
        
    num_frames = len(ref) // frame_len
    snr_segs = []
    clipped_frames = 0
    recovered_frames = 0
    
    for i in range(num_frames):
        start = i * frame_len
        end = start + frame_len
        ref_f = ref[start:end]
        deg_f = deg[start:end]
        
        # Segmental SNR
        sig_energy = np.sum(ref_f**2) + 1e-10
        noise_energy = np.sum((ref_f - deg_f)**2) + 1e-10
        snr_segs.append(10.0 * np.log10(sig_energy / noise_energy))
        
        # Recoverability: count if raw was clipped (abs >= 0.99)
        if np.max(np.abs(ref_f)) >= 0.99:
            clipped_frames += 1
            if np.max(np.abs(deg_f)) < 0.99:
                recovered_frames += 1

    seg_snr = float(np.mean(snr_segs))
    recoverability = float(recovered_frames / clipped_frames) if clipped_frames > 0 else 1.0
    return seg_snr, recoverability
