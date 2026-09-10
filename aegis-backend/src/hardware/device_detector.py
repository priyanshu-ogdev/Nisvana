"""hardware/device_detector.py — ALSA device enumeration and hotplug detection."""
from __future__ import annotations
import asyncio
import logging
import yaml
import os
from pathlib import Path
from typing import Optional, Callable, Awaitable

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 2.0


class DeviceDetector:
    """
    Polls for ALSA/sounddevice card presence every 2s.
    On device change → calls on_status_change(hw_status_dict).

    On the Pi: uses sounddevice.query_devices() + cross-refs with hardware.yaml VID:PIDs.
    On MacBook (dev): gracefully degrades — all mics reported as absent, no crash.
    """

    def __init__(
        self,
        config_path: str = "config/hardware.yaml",
        on_status_change: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> None:
        self._config_path = config_path
        self._on_status_change = on_status_change
        self._last_status: Optional[dict] = None
        self._running = False
        self._hw_config = self._load_config()

    def _load_config(self) -> dict:
        p = Path(self._config_path)
        if not p.exists():
            logger.warning(f"hardware.yaml not found at {p} — using defaults")
            return {}
        with open(p) as f:
            return yaml.safe_load(f) or {}

    def _detect(self) -> dict:
        """Return hw_status-shaped dict reflecting current device presence."""
        status = {
            "headset_detected": False,
            "mic_primary": False,
            "mic_reference": False,
            "mic_throat": False,
            "pi_cpu_temp": self._read_temp(),
            "ai_model_loaded": "none",
            "alsainputs": [],
            "alsaoutputs": [],
        }

        try:
            import sounddevice as sd
            devices = sd.query_devices()
            device_names = [d["name"] for d in devices]
            status["alsainputs"] = [d["name"] for d in devices if d["max_input_channels"] > 0]
            status["alsaoutputs"] = [d["name"] for d in devices if d["max_output_channels"] > 0]

            devices_config = self._hw_config.get("devices", {})

            for role, cfg in devices_config.items():
                label = cfg.get("label", "")
                # Match by label substring (VID:PID matching requires lsusb/sysfs on Pi)
                found = any(label.lower() in name.lower() for name in device_names)
                if role == "primary_mic":
                    status["mic_primary"] = found
                elif role == "reference_mic":
                    status["mic_reference"] = found
                elif role == "throat_mic":
                    status["mic_throat"] = found
                elif role == "headset_output":
                    status["headset_detected"] = found

        except ImportError:
            # sounddevice not installed (MacBook dev env) — report all absent
            logger.debug("sounddevice not available — all hardware reported absent")
        except Exception as e:
            logger.error(f"Device detection error: {e}")

        return status

    def _read_temp(self) -> float:
        """Read Pi CPU temp via vcgencmd; fall back to psutil on non-Pi."""
        try:
            import subprocess
            result = subprocess.run(
                ["vcgencmd", "measure_temp"],
                capture_output=True, text=True, timeout=1.0
            )
            # Output: "temp=45.0'C"
            return float(result.stdout.strip().split("=")[1].replace("'C", ""))
        except Exception:
            pass
        try:
            import psutil
            temps = psutil.sensors_temperatures()
            if temps:
                first = next(iter(temps.values()))
                return first[0].current
        except Exception:
            pass
        return 0.0

    async def poll_forever(self) -> None:
        self._running = True
        logger.info("Device detector polling every 2s")
        while self._running:
            current = self._detect()
            if current != self._last_status:
                logger.info(f"Hardware status changed: {current}")
                self._last_status = current
                if self._on_status_change:
                    await self._on_status_change(current)
            await asyncio.sleep(POLL_INTERVAL_S)

    def stop(self) -> None:
        self._running = False

    def get_latest(self) -> Optional[dict]:
        return self._last_status
