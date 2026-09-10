"""hardware/cpu_monitor.py — CPU, RAM, and temperature readings."""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def read_cpu_temp() -> float:
    """Read Pi CPU temperature. Returns 0.0 on non-Pi systems."""
    try:
        import subprocess
        r = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=1.0)
        return float(r.stdout.strip().split("=")[1].replace("'C", ""))
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


def read_cpu_ram() -> tuple[float, float]:
    """Returns (cpu_pct, ram_pct) using psutil."""
    try:
        import psutil
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        return cpu, ram
    except ImportError:
        return 0.0, 0.0
