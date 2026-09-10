"""
ws/heartbeat.py — Ping/Pong heartbeat manager
==============================================
Sends ping every 1000ms to all sessions.
3 missed pongs → session.drop() + link_status:dropped broadcast.
"""

from __future__ import annotations
import asyncio
import time
import logging
from typing import Callable, Awaitable

from .protocol import Ping, Pong

logger = logging.getLogger(__name__)

PING_INTERVAL_S = 1.0
PONG_TIMEOUT_S = 3.0


class HeartbeatManager:
    def __init__(
        self,
        get_sessions: Callable,
        broadcast_fn: Callable[[dict, str], Awaitable[None]],
    ) -> None:
        """
        get_sessions: callable returning iterable of active ClientSession objects
        broadcast_fn: async fn(message_dict, client_id) to send to a specific ws
        """
        self._get_sessions = get_sessions
        self._broadcast = broadcast_fn
        self._seq = 0
        self._running = False

    async def start(self) -> None:
        self._running = True
        logger.info("Heartbeat manager started (interval=1s, timeout=3s)")
        while self._running:
            await asyncio.sleep(PING_INTERVAL_S)
            await self._tick()

    def stop(self) -> None:
        self._running = False

    async def _tick(self) -> None:
        self._seq += 1
        now_ms = int(time.time() * 1000)

        for session in list(self._get_sessions()):
            # Check for overdue pong
            if session.state.value not in ("dormant", "dropped"):
                if session.is_pong_overdue(PONG_TIMEOUT_S):
                    logger.warning(f"[{session.client_id}] Pong timeout — dropping")
                    await session.drop("heartbeat_timeout")
                    await self._broadcast(
                        {"type": "link_status", "state": "dropped", "clientId": session.client_id},
                        session.client_id,
                    )
                    continue

            # Send ping
            ping = Ping(type="ping", seq=self._seq, timestamp=now_ms)
            try:
                await session.websocket.send(ping.model_dump_json())
            except Exception as e:
                logger.debug(f"[{session.client_id}] Ping send failed: {e}")

    def handle_pong(self, session, pong_data: dict) -> Pong:
        """
        Called when a pong message arrives from the frontend.
        Calculates RTT and updates session liveness.
        """
        now_ms = int(time.time() * 1000)
        original_ts = pong_data.get("timestamp", now_ms)
        rtt_ms = now_ms - original_ts

        session.update_pong(pong_data.get("seq", 0))

        return Pong(
            seq=pong_data.get("seq", 0),
            ts=now_ms,
            rtt_ms=max(0.0, float(rtt_ms)),
        )
