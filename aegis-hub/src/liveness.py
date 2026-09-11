"""src/liveness.py — Node Liveness Monitor.

Catches half-open Wi-Fi sockets that appear connected but are silently dead.

State machine per node:
  online → degraded → offline

Transitions:
  online:   last_pong received within 3s of last expected ping
  degraded: no pong for >3s (Wi-Fi dropout, sleeping device, etc.)
             → broadcast `node_degraded` to all dashboards
  offline:  no pong for >6s OR socket close event
             → remove from registry
             → broadcast `node_offline` to all dashboards

The monitor runs as a single background asyncio.Task shared across all nodes.
It does not send pings itself — the Hub's handle_node() sends pings and updates
NodeConn.last_pong on every node_pong received.

This is separate from the WS protocol-level ping/pong (which operates at the
transport layer). This is application-level heartbeat monitoring.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Callable, Awaitable

logger = logging.getLogger("aegis.hub.liveness")

DEGRADED_THRESHOLD_S = 3.0    # no pong for this long → degraded
OFFLINE_THRESHOLD_S  = 6.0    # no pong for this long → offline + purge
TICK_INTERVAL_S      = 1.0    # how often the monitor checks

NodeState = str   # "online" | "degraded" | "offline"

BroadcastFn = Callable[[str, str | None], Awaitable[None]]


class LivenessMonitor:
    """
    Background task that checks node heartbeat timestamps every second
    and transitions nodes through online → degraded → offline states.

    Args:
        get_nodes:       Callable returning the current {node_id: NodeConn} dict.
        remove_node:     Callable to purge a node from the registry.
        broadcast_json:  Callable(json_str, node_id) to fan-out control messages.
    """

    def __init__(
        self,
        get_nodes,
        remove_node,
        broadcast_json: BroadcastFn,
    ) -> None:
        self._get_nodes   = get_nodes
        self._remove_node = remove_node
        self._broadcast   = broadcast_json
        self._running     = False

    async def monitor_forever(self) -> None:
        """Main monitoring loop — run as a background asyncio.Task."""
        self._running = True
        logger.info("LivenessMonitor: started (tick=1s, degraded>3s, offline>6s)")

        while self._running:
            await asyncio.sleep(TICK_INTERVAL_S)
            await self._tick()

    async def _tick(self) -> None:
        """One monitoring cycle — check all registered nodes."""
        now = time.monotonic()
        nodes = self._get_nodes()

        to_remove = []

        for node_id, conn in list(nodes.items()):
            age = now - conn.last_pong_at

            if age > OFFLINE_THRESHOLD_S:
                if conn.state != "offline":
                    conn.state = "offline"
                    logger.warning(
                        f"LivenessMonitor: node '{node_id}' OFFLINE "
                        f"(no pong for {age:.1f}s)"
                    )
                    to_remove.append(node_id)
                    await self._broadcast(
                        json.dumps({
                            "type":    "node_offline",
                            "node_id": node_id,
                            "reason":  "heartbeat_timeout",
                            "ts":      int(time.time() * 1000),
                        }),
                        None,  # broadcast to all dashboards
                    )

            elif age > DEGRADED_THRESHOLD_S:
                if conn.state == "online":
                    conn.state = "degraded"
                    logger.warning(
                        f"LivenessMonitor: node '{node_id}' DEGRADED "
                        f"(no pong for {age:.1f}s)"
                    )
                    await self._broadcast(
                        json.dumps({
                            "type":    "node_degraded",
                            "node_id": node_id,
                            "age_s":   round(age, 1),
                            "ts":      int(time.time() * 1000),
                        }),
                        None,
                    )
            else:
                if conn.state != "online":
                    prev = conn.state
                    conn.state = "online"
                    logger.info(
                        f"LivenessMonitor: node '{node_id}' recovered "
                        f"({prev} → online)"
                    )
                    await self._broadcast(
                        json.dumps({
                            "type":    "node_online",
                            "node_id": node_id,
                            "ts":      int(time.time() * 1000),
                        }),
                        None,
                    )

        # Purge offline nodes after the iteration (avoids mutating dict mid-loop)
        for node_id in to_remove:
            self._remove_node(node_id)

    def stop(self) -> None:
        self._running = False
