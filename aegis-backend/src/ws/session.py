"""
ws/session.py — Per-client link state machine
==============================================
States: DORMANT → HANDSHAKING → SECURE → DROPPED
Only SECURE clients receive fft_stream / anc_state.
"""

from __future__ import annotations
import asyncio
import time
import logging
from enum import Enum
from typing import Optional, Callable, Awaitable

logger = logging.getLogger(__name__)


class LinkState(str, Enum):
    DORMANT = "dormant"
    HANDSHAKING = "handshaking"
    SECURE = "secure"
    DROPPED = "dropped"


class ClientSession:
    """
    Tracks the state of a single connected client.
    Thread-safe via asyncio — all mutations happen on the event loop.
    """

    def __init__(
        self,
        client_id: str,
        websocket,
        on_state_change: Optional[Callable[["ClientSession"], Awaitable[None]]] = None,
    ) -> None:
        self.client_id = client_id
        self.websocket = websocket
        self.state = LinkState.DORMANT
        self._on_state_change = on_state_change

        # Per-client mute state per target
        self.muted: dict[str, bool] = {
            "primary_mic": False,
            "reference_mic": False,
            "throat_mic": False,
            "headset_output": False,
        }
        self.anc_active: bool = False

        # Heartbeat tracking
        self.last_ping_seq: int = 0
        self.last_pong_ts: float = time.monotonic()
        self.connect_time: float = time.monotonic()

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    async def _transition(self, new_state: LinkState) -> None:
        if self.state == new_state:
            return
        old = self.state
        self.state = new_state
        logger.info(f"[{self.client_id}] {old.value} → {new_state.value}")
        if self._on_state_change:
            await self._on_state_change(self)

    async def begin_handshake(self) -> None:
        """Triggered on receipt of handshake_init."""
        if self.state != LinkState.DORMANT:
            logger.warning(f"[{self.client_id}] Unexpected handshake_init in state {self.state}")
            return
        await self._transition(LinkState.HANDSHAKING)

    async def confirm_secure(self) -> None:
        """Triggered after handshake_ack(ok) is sent."""
        await self._transition(LinkState.SECURE)

    async def drop(self, reason: str = "unknown") -> None:
        """Drop the session (heartbeat timeout, WS close, device unplug)."""
        logger.warning(f"[{self.client_id}] Dropping session: {reason}")
        await self._transition(LinkState.DROPPED)

    async def reset(self) -> None:
        """Reset to DORMANT on reconnect (never carry stale state forward)."""
        await self._transition(LinkState.DORMANT)
        self.muted = {k: False for k in self.muted}
        self.anc_active = False

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def is_secure(self) -> bool:
        return self.state == LinkState.SECURE

    def should_receive_fft(self) -> bool:
        """Only SECURE clients get fft_stream / anc_state."""
        return self.state == LinkState.SECURE

    def update_pong(self, seq: int) -> None:
        self.last_ping_seq = seq
        self.last_pong_ts = time.monotonic()

    def is_pong_overdue(self, timeout_s: float = 3.0) -> bool:
        return (time.monotonic() - self.last_pong_ts) > timeout_s
