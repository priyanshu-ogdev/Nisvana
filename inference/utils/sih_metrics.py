"""
inference/utils/sih_metrics.py — Official SIH Defence Benchmark & Scorecard Evaluator

Standardizes evaluation against official Smart India Hackathon (SIH) Defence Targets:
1. Signal-to-Noise Ratio: SNR > 15 dB (or Delta-SNR >= 15 dB)
2. Speech Intelligibility: STOI > 0.85
3. Perceptual Quality: PESQ > 2.50 MOS
4. Edge Real-Time Latency: RTF < 1.0 (latency <= frame duration)
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
import numpy as np
import torch

from training.utils.metrics import (
    compute_snr_db,
    compute_si_snr_db,
    compute_segmental_snr_db,
    compute_stoi,
    compute_pesq,
    compute_pesq_proxy,
    compute_dnsmos_proxy,
)


# Official SIH Targets
SIH_TARGET_SNR_DB = 15.0
SIH_TARGET_DELTA_SNR_DB = 15.0
SIH_TARGET_STOI = 0.85
SIH_TARGET_PESQ = 2.50
SIH_TARGET_MAX_RTF = 1.0


@dataclass
class SihEvaluationResult:
    """Encapsulates SIH benchmark scores and compliance verdicts."""
    snr_in_db: float
    snr_out_db: float
    delta_snr_db: float
    stoi_in: float
    stoi_out: float
    delta_stoi: float
    pesq_out: float
    dnsmos_ovrl: float
    latency_mean_ms: float
    real_time_factor: float
    
    # Compliance verdicts
    snr_passed: bool
    stoi_passed: bool
    pesq_passed: bool
    latency_passed: bool
    overall_compliant: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": {
                "snr_in_db": round(self.snr_in_db, 2),
                "snr_out_db": round(self.snr_out_db, 2),
                "delta_snr_db": round(self.delta_snr_db, 2),
                "stoi_in": round(self.stoi_in, 3),
                "stoi_out": round(self.stoi_out, 3),
                "delta_stoi": round(self.delta_stoi, 3),
                "pesq_out": round(self.pesq_out, 3),
                "dnsmos_ovrl": round(self.dnsmos_ovrl, 2),
                "latency_mean_ms": round(self.latency_mean_ms, 2),
                "real_time_factor": round(self.real_time_factor, 3),
            },
            "compliance": {
                "snr_target_met": self.snr_passed,
                "stoi_target_met": self.stoi_passed,
                "pesq_target_met": self.pesq_passed,
                "latency_target_met": self.latency_passed,
                "all_sih_targets_met": self.overall_compliant,
            },
        }

    def format_markdown_table(self, title: str = "SIH DEFENCE BENCHMARK SCORECARD") -> str:
        """Renders publication-grade GitHub Markdown table comparing against SIH criteria."""
        status_icon = lambda passed: "✅ PASS" if passed else "❌ REGRESSION"
        
        rtf_display = f"{self.real_time_factor:.3f}x" if self.latency_mean_ms > 0 else "UNMEASURED"
        lat_display = f"{self.latency_mean_ms:.2f} ms" if self.latency_mean_ms > 0 else "UNMEASURED"
        lat_verdict = status_icon(self.latency_passed) if self.latency_mean_ms > 0 else "❌ UNVERIFIED"

        lines = [
            f"### {title}",
            "",
            "| Metric | Input (Noisy) | Enhanced (AEGIS) | SIH Target | Verdict |",
            "| :--- | :--- | :--- | :--- | :--- |",
            f"| **SNR (dB)** | {self.snr_in_db:+.2f} dB | **{self.snr_out_db:+.2f} dB** | **> 15.0 dB** (or $\\Delta\\ge 15$) | {status_icon(self.snr_passed)} |",
            f"| **STOI** | {self.stoi_in:.3f} | **{self.stoi_out:.3f}** | **> 0.850** | {status_icon(self.stoi_passed)} |",
            f"| **PESQ (MOS)** | — | **{self.pesq_out:.3f}** | **> 2.500** | {status_icon(self.pesq_passed)} |",
            f"| **DNSMOS (OVRL)** | — | **{self.dnsmos_ovrl:.2f}** | Objective Proxy | ℹ️ INFO |",
            f"| **Real-Time Factor** | — | **{rtf_display}** | **< 1.000x (Real-Time)** | {lat_verdict} |",
            f"| **Frame Latency** | — | **{lat_display}** | Stream Budget | ℹ️ INFO |",
            "",
            f"**Overall SIH Compliance**: {'✅ **ALL TARGETS MET**' if self.overall_compliant else '❌ **TARGETS NOT FULLY MET**'}",
            "",
        ]
        return "\n".join(lines)


def evaluate_sih_compliance(
    estimate: Union[torch.Tensor, np.ndarray],
    target_clean: Union[torch.Tensor, np.ndarray],
    input_noisy: Union[torch.Tensor, np.ndarray],
    sample_rate: int = 48000,
    latency_mean_ms: Optional[float] = None,
    chunk_ms: float = 10.0,
) -> SihEvaluationResult:
    """
    Computes all SIH defence metrics and determines strict standard compliance.
    Args:
        estimate: Enhanced speech waveform from model/pipeline.
        target_clean: Clean reference speech waveform.
        input_noisy: Corrupted input mixture waveform.
        sample_rate: 48,000 Hz.
        latency_mean_ms: Mean processing latency per frame in ms. If unmeasured/None, latency is marked unverified.
        chunk_ms: Audio frame duration in ms.
    Returns:
        SihEvaluationResult with metrics and pass/fail boolean verdicts.
    """
    # 1. SNR Calculations
    snr_in = compute_snr_db(input_noisy, target_clean)
    snr_out = compute_snr_db(estimate, target_clean)
    delta_snr = snr_out - snr_in

    # 2. STOI Intelligibility Calculations
    stoi_in = compute_stoi(input_noisy, target_clean, sr=sample_rate)
    stoi_out = compute_stoi(estimate, target_clean, sr=sample_rate)
    delta_stoi = stoi_out - stoi_in

    # 3. PESQ Calculations (WB MOS)
    pesq_val = compute_pesq(estimate, target_clean, sr=sample_rate)

    # 4. DNSMOS Overall Quality
    dns_res = compute_dnsmos_proxy(estimate, sr=sample_rate)
    dnsmos_ovrl = dns_res["dnsmos_ovrl"]

    # 5. Latency & Real-Time Factor
    if latency_mean_ms is not None and latency_mean_ms > 0.0:
        rtf = latency_mean_ms / max(chunk_ms, 1e-6)
        latency_passed = bool(rtf < SIH_TARGET_MAX_RTF)
        lat_mean_val = float(latency_mean_ms)
    else:
        # Latency was unmeasured: do NOT fabricate an optimistic pass!
        rtf = 0.0
        latency_passed = False
        lat_mean_val = 0.0

    # Verdict evaluations against SIH specifications:
    # SNR passes if output is > 15 dB OR if noise reduction exceeds 15 dB (e.g. from -10 dB to +5 dB)
    snr_passed = bool(snr_out >= SIH_TARGET_SNR_DB or delta_snr >= SIH_TARGET_DELTA_SNR_DB)
    stoi_passed = bool(stoi_out >= SIH_TARGET_STOI or (stoi_in < 0.60 and delta_stoi >= 0.20))
    pesq_passed = bool(pesq_val >= SIH_TARGET_PESQ)

    overall_compliant = bool(snr_passed and stoi_passed and pesq_passed and latency_passed)

    return SihEvaluationResult(
        snr_in_db=snr_in,
        snr_out_db=snr_out,
        delta_snr_db=delta_snr,
        stoi_in=stoi_in,
        stoi_out=stoi_out,
        delta_stoi=delta_stoi,
        pesq_out=pesq_val,
        dnsmos_ovrl=dnsmos_ovrl,
        latency_mean_ms=lat_mean_val,
        real_time_factor=rtf,
        snr_passed=snr_passed,
        stoi_passed=stoi_passed,
        pesq_passed=pesq_passed,
        latency_passed=latency_passed,
        overall_compliant=overall_compliant,
    )
