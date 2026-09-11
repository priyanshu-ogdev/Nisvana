"""dsp/telemetry.py — Pipeline telemetry calculator (latency, SNR improvement, CPU, RAM)."""
from __future__ import annotations
import time
import numpy as np
from ..hardware.cpu_monitor import read_cpu_ram, read_cpu_temp


class TelemetryCollector:
    """
    Computes per-second telemetry payload.
    Latency: marker-injection placeholder (real impl: inject known tone, measure delay).
    SNR improvement: delta between raw and enhanced RMS.
    """

    def __init__(self) -> None:
        self._frame_count: int = 0
        self._window_start: float = time.monotonic()
        self._raw_rms_acc: float = 0.0
        self._enhanced_rms_acc: float = 0.0
        self._frame_acc: int = 0

    def record_frame(self, raw: np.ndarray, enhanced: np.ndarray) -> None:
        self._frame_count += 1
        self._frame_acc += 1
        self._raw_rms_acc += float(np.sqrt(np.mean(raw.astype(np.float64) ** 2)) + 1e-10)
        self._enhanced_rms_acc += float(np.sqrt(np.mean(enhanced.astype(np.float64) ** 2)) + 1e-10)

    def collect(
        self,
        model_name: str,
        *,
        backend: str = "unknown",
        provider: str = "unknown",
        inference_ms: float | None = None,
        algorithmic_delay_ms: float | None = None,
        thermal_tier: str | None = None,
        degradation_reason: str | None = None,
        aec_mode: str | None = None,
    ) -> dict:
        """Return telemetry payload dict. Resets accumulators."""
        now = time.monotonic()
        elapsed = now - self._window_start
        fps = self._frame_count / elapsed if elapsed > 0 else 0.0

        # SNR improvement estimate
        if self._frame_acc > 0 and self._raw_rms_acc > 0:
            raw_mean = self._raw_rms_acc / self._frame_acc
            enhanced_mean = self._enhanced_rms_acc / self._frame_acc
            snr_improvement = 20.0 * np.log10(raw_mean / (enhanced_mean + 1e-10))
        else:
            snr_improvement = 0.0

        cpu_pct, ram_pct = read_cpu_ram()
        temp = read_cpu_temp()

        result = {
            "latency_ms": round(elapsed * 100 / max(1, self._frame_count), 2),  # ~10ms/frame estimate
            "snr_improvement_db": round(float(snr_improvement), 1),
            "model": model_name,
            "platform": "pi5",
            "fps": round(fps, 1),
            "cpu_pct": round(cpu_pct, 1),
            "ram_pct": round(ram_pct, 1),
            "pi_cpu_temp": round(temp, 1),
            "backend": backend,
            "provider": provider,
            "algorithmic_delay_ms": algorithmic_delay_ms,
            "real_time_factor": (
                round(float(inference_ms) / 10.0, 3)
                if inference_ms is not None else None
            ),
            "thermal_tier": thermal_tier,
            "degradation_reason": degradation_reason,
            "aec_mode": aec_mode,
        }

        # Reset
        self._frame_count = 0
        self._window_start = now
        self._raw_rms_acc = 0.0
        self._enhanced_rms_acc = 0.0
        self._frame_acc = 0

        return result
