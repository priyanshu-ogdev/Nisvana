"""hardware/thermal_guard.py — Thermal model downgrade tiers."""
from __future__ import annotations
import asyncio
import logging
from typing import Callable, Awaitable
from .cpu_monitor import read_cpu_temp

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 5.0

TIER_FULL = 72.0      # below → DeepFilterNet3 + AEC + harmonic
TIER_1 = 78.0         # 72–78 → disable harmonic preproc
TIER_2 = 82.0         # 78–82 → swap to CleanUMamba
                       # above → RNNoise only, bypass AI


class ThermalGuard:
    def __init__(
        self,
        on_tier_change: Callable[[str, str], Awaitable[None]],
    ) -> None:
        """on_tier_change(model_name: str, tier: str)"""
        self._on_tier_change = on_tier_change
        self._current_tier: str = "full"
        self._running = False

    def _temp_to_tier(self, temp: float) -> tuple[str, str]:
        """Returns (tier_label, model_name)."""
        if temp < TIER_FULL:
            return "full", "DeepFilterNet3"
        elif temp < TIER_1:
            return "tier1", "DeepFilterNet3"  # same model, no harmonic preproc
        elif temp < TIER_2:
            return "tier2", "CleanUMamba"
        else:
            return "tier3", "noisereduce-cpu"

    async def monitor_forever(self) -> None:
        self._running = True
        logger.info("Thermal guard monitoring every 5s")
        while self._running:
            temp = read_cpu_temp()
            tier, model = self._temp_to_tier(temp)
            if tier != self._current_tier:
                logger.warning(f"Thermal tier change: {self._current_tier} → {tier} ({temp:.1f}°C) → using {model}")
                self._current_tier = tier
                await self._on_tier_change(model, tier)
            await asyncio.sleep(POLL_INTERVAL_S)

    def stop(self) -> None:
        self._running = False

    @property
    def current_tier(self) -> str:
        return self._current_tier

    @property
    def harmonic_enabled(self) -> bool:
        return self._current_tier == "full"

    @property
    def current_model(self) -> str:
        _, model = self._temp_to_tier(read_cpu_temp())
        return model
