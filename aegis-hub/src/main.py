"""
AEGIS Hub Server (Rev 3 SOTA Architecture)
===========================================
Maintains a dynamic registry of AEGIS edge nodes.
Serves dashboards with subscription-based routing, per-dashboard drop-oldest queues,
non-blocking fan-out (zero backpressure head-of-line stalls), real-time RTT heartbeat
tracking with degraded link state detection, and strict authentication.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import struct
import sys
import time
from typing import Any, Dict, Optional, Set

import websockets
from websockets.asyncio.server import ServerConnection, serve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] HUB: %(message)s")
logger = logging.getLogger("aegis.hub")

# ==============================================================================
# Configuration & Strict Authentication
# ==============================================================================

AEGIS_AUTH_TOKEN = os.getenv("AEGIS_AUTH_TOKEN")
AEGIS_DEV_MODE = os.getenv("AEGIS_DEV_MODE", "0").strip().lower() in ("1", "true", "yes")


def validate_startup_config(token: Optional[str], dev_mode: bool) -> None:
    """
    Ensures no default or hardcoded secrets are shipped.
    AEGIS_AUTH_TOKEN is required to start unless AEGIS_DEV_MODE=1 is explicitly set.
    """
    if not dev_mode and not token:
        msg = (
            "FATAL CONFIGURATION ERROR: AEGIS_AUTH_TOKEN environment variable is not set.\n"
            "The Hub refuses to start with an insecure or shared default secret.\n"
            "To fix:\n"
            "  1. In production: Set AEGIS_AUTH_TOKEN=<secret> in your environment or systemd unit.\n"
            "  2. In local development: Set AEGIS_DEV_MODE=1 to explicitly bypass auth."
        )
        logger.critical(msg)
        raise SystemExit(msg)

    if dev_mode:
        logger.warning("=" * 78)
        logger.warning("  WARNING: AEGIS_DEV_MODE=1 ACTIVATED — AUTHENTICATION IS COMPLETELY BYPASSED!")
        logger.warning("  ALL NODE CONNECTIONS WILL BE ADMITTED WITHOUT CREDENTIAL VERIFICATION.")
        logger.warning("  DO NOT USE THIS MODE IN PRODUCTION.")
        logger.warning("=" * 78)


def check_auth_token(token: Optional[str], expected_token: Optional[str], dev_mode: bool) -> bool:
    """Constant-time token validation."""
    if dev_mode:
        return True
    if not expected_token or not token:
        return False
    import hmac
    return hmac.compare_digest(token.encode("utf-8"), expected_token.encode("utf-8"))


# ==============================================================================
# Low-Latency Drop-Oldest Queue (P1/P2)
# ==============================================================================

class DropOldestQueue:
    """
    Bounded queue that drops the oldest element upon saturation.
    In real-time audio and FFT visualizers, fresh data is strictly superior
    to stale data. Drop-oldest maintains temporal synchronization during bursts.
    """
    def __init__(self, maxsize: int = 128):
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.maxsize = maxsize
        self.dropped_count: int = 0

    def put_nowait(self, item: Any) -> None:
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
                self.dropped_count += 1
            except (asyncio.QueueEmpty, ValueError):
                pass
            try:
                self._queue.put_nowait(item)
            except asyncio.QueueFull:
                pass

    async def get(self) -> Any:
        return await self._queue.get()

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()

    def full(self) -> bool:
        return self._queue.full()

    def get_nowait(self) -> Any:
        return self._queue.get_nowait()

    def clear(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except (asyncio.QueueEmpty, ValueError):
                break


# ==============================================================================
# Dashboard Session Management (Independent Drain Queues)
# ==============================================================================

class DashboardSession:
    """
    Encapsulates a connected dashboard instance with an independent bounded queue
    and isolated drain coroutine. Prevents one slow Wi-Fi tab from blocking fan-out
    to all other dashboards.
    """
    def __init__(self, ws: ServerConnection, session_id: str):
        self.ws = ws
        self.session_id = session_id
        self.queue = DropOldestQueue(maxsize=128)
        self.subscriptions: Optional[Set[str]] = None  # None = all nodes ('*')
        self.drain_task: Optional[asyncio.Task] = None

    def start(self, drain_coro):
        self.drain_task = asyncio.create_task(drain_coro(self))

    def stop(self):
        if self.drain_task and not self.drain_task.done():
            self.drain_task.cancel()


# ==============================================================================
# Dynamic Node Registry with Degraded Link State Detection
# ==============================================================================

class NodeRegistry:
    def __init__(self):
        # Maps node_id -> dict of properties
        self.nodes: Dict[str, dict] = {}
        # Maps node_id -> ServerConnection
        self.node_sockets: Dict[str, ServerConnection] = {}

    def add_node(self, node_id: str, ws: ServerConnection, hw_info: dict):
        self.nodes[node_id] = {
            "node_id": node_id,
            "hw": hw_info,
            "connected_at": time.time(),
            "last_seen_ts": time.time(),
            "status": "online",
            "secure": False,
        }
        self.node_sockets[node_id] = ws
        logger.info(f"Node registered: {node_id}")

    def touch(self, node_id: str) -> None:
        if node_id in self.nodes:
            self.nodes[node_id]["last_seen_ts"] = time.time()

    def set_status(self, node_id: str, status: str) -> bool:
        """Sets node status. Returns True if status changed."""
        if node_id in self.nodes and self.nodes[node_id]["status"] != status:
            self.nodes[node_id]["status"] = status
            return True
        return False

    def set_secure(self, node_id: str, secure: bool = True):
        if node_id in self.nodes:
            self.nodes[node_id]["secure"] = secure

    def remove_node(self, node_id: str, ws: Optional[ServerConnection] = None) -> bool:
        if ws is not None and self.node_sockets.get(node_id) != ws:
            logger.info(f"Ignoring stale disconnect for replaced node: {node_id}")
            return False
        self.nodes.pop(node_id, None)
        self.node_sockets.pop(node_id, None)
        logger.info(f"Node removed: {node_id}")
        return True


# ==============================================================================
# AEGIS Hub Core
# ==============================================================================

class AegisHub:
    def __init__(self, host="0.0.0.0", port=8001, token: Optional[str] = None, dev_mode: bool = False):
        self.host = host
        self.port = port
        self.token = token if token is not None else AEGIS_AUTH_TOKEN
        self.dev_mode = dev_mode or AEGIS_DEV_MODE
        self.registry = NodeRegistry()
        self.dashboards: Dict[ServerConnection, DashboardSession] = {}
        self._session_counter: int = 0
        self._monitor_task: Optional[asyncio.Task] = None

    def _set_tcp_nodelay(self, ws: ServerConnection) -> None:
        """Disables Nagle's algorithm to eliminate small-packet buffering delays."""
        try:
            sock = ws.transport.get_extra_info("socket")
            if sock:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception as e:
            logger.debug(f"Could not set TCP_NODELAY: {e}")

    def broadcast_to_dashboards(self, msg: str | bytes, node_id: Optional[str] = None):
        """
        Non-blocking fan-out to all subscribed dashboards via per-dashboard bounded queues.
        Zero backpressure stalls: if a dashboard socket buffers, only its queue evicts old frames.
        """
        if not self.dashboards:
            return

        # Extract node_id from binary frame if not explicitly passed
        if node_id is None and isinstance(msg, (bytes, bytearray)) and len(msg) >= 2:
            node_id_len = msg[1]
            if len(msg) >= 2 + node_id_len:
                node_id = msg[2:2 + node_id_len].decode("utf-8", errors="ignore")

        for session in list(self.dashboards.values()):
            if session.subscriptions is None or node_id is None or node_id in session.subscriptions:
                session.queue.put_nowait(msg)

    async def _drain_to_dashboard(self, session: DashboardSession):
        """Dedicated drain task per dashboard."""
        try:
            while True:
                msg = await session.queue.get()
                await session.ws.send(msg)
        except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
            pass
        except Exception as e:
            logger.debug(f"Dashboard {session.session_id} drain error: {e}")
        finally:
            self.dashboards.pop(session.ws, None)

    async def handle_dashboard(self, ws: ServerConnection):
        """Dashboard connection handler."""
        self._set_tcp_nodelay(ws)
        self._session_counter += 1
        session_id = f"dash-{self._session_counter}"
        session = DashboardSession(ws, session_id)
        self.dashboards[ws] = session
        session.start(self._drain_to_dashboard)

        logger.info(f"Dashboard connected: {session_id} (active: {len(self.dashboards)})")
        try:
            # On connect, push current state of all online nodes
            for node_id, info in self.registry.nodes.items():
                msg = {
                    "type": "node_online",
                    "node_id": node_id,
                    "hw": info["hw"],
                    "status": info.get("status", "online"),
                    "secure": info.get("secure", False),
                    "ts": int(time.time() * 1000),
                }
                session.queue.put_nowait(json.dumps(msg))

            async for message in ws:
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type")

                        if msg_type == "subscribe":
                            nodes = data.get("nodes", ["*"])
                            if "*" in nodes:
                                session.subscriptions = None
                                logger.info(f"Dashboard {session_id} subscribed to ALL nodes")
                            else:
                                session.subscriptions = set(nodes)
                                logger.info(f"Dashboard {session_id} subscribed to {session.subscriptions}")

                        elif msg_type == "ping":
                            pong = {
                                "type": "pong",
                                "seq": data.get("seq"),
                                "timestamp": data.get("timestamp"),
                                "rtt_ms": 0,
                                "ts": int(time.time() * 1000),
                            }
                            session.queue.put_nowait(json.dumps(pong))

                        else:
                            # Route control messages (anc_set, hardware_mute) to the target node
                            target_id = data.get("node_id") or data.get("clientId")
                            if target_id and target_id in self.registry.node_sockets:
                                try:
                                    await self.registry.node_sockets[target_id].send(message)
                                except Exception as e:
                                    logger.warning(f"Failed to forward message to node {target_id}: {e}")

                    except json.JSONDecodeError:
                        pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            session.stop()
            self.dashboards.pop(ws, None)
            logger.info(f"Dashboard disconnected: {session_id} (remaining: {len(self.dashboards)})")

    async def handle_node(self, ws: ServerConnection):
        """Pi edge node connection handler."""
        self._set_tcp_nodelay(ws)
        node_id = None
        logger.info("New edge node socket connected, awaiting authentication...")
        try:
            async for message in ws:
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type")

                        if msg_type == "node_hello":
                            candidate_id = data.get("node_id")
                            token = data.get("token")

                            # Token validation
                            if not check_auth_token(token, self.token, self.dev_mode):
                                logger.warning(
                                    f"Node '{candidate_id}' REJECTED: invalid or missing auth token."
                                )
                                await ws.send(json.dumps({
                                    "type": "node_ack",
                                    "status": "unauthorized",
                                    "reason": "Invalid or missing AEGIS_AUTH_TOKEN"
                                }))
                                await ws.close(1008, "Policy Violation: Unauthorized")
                                return

                            node_id = candidate_id
                            if not node_id:
                                logger.warning("node_hello missing node_id, dropping")
                                break

                            self.registry.add_node(node_id, ws, data.get("hw", {}))

                            # Broadcast node_online to all dashboards
                            self.broadcast_to_dashboards(json.dumps({
                                "type": "node_online",
                                "node_id": node_id,
                                "hw": data.get("hw", {}),
                                "status": "online",
                                "secure": False,
                                "ts": int(time.time() * 1000),
                            }))

                            # Acknowledge node registration
                            await ws.send(json.dumps({"type": "node_ack", "status": "ok"}))

                        elif msg_type == "node_ping":
                            # 1Hz Node Heartbeat for RTT measurement
                            if node_id:
                                self.registry.touch(node_id)
                                if self.registry.set_status(node_id, "online"):
                                    self.broadcast_to_dashboards(json.dumps({
                                        "type": "node_status",
                                        "node_id": node_id,
                                        "status": "online",
                                        "ts": int(time.time() * 1000),
                                    }))
                            pong = {
                                "type": "node_pong",
                                "node_id": node_id,
                                "seq": data.get("seq", 0),
                                "timestamp": data.get("timestamp"),
                                "ts": int(time.time() * 1000),
                            }
                            await ws.send(json.dumps(pong))

                        elif msg_type == "ping":
                            # Standard ping echo
                            if node_id:
                                self.registry.touch(node_id)
                            pong = {
                                "type": "pong",
                                "seq": data.get("seq"),
                                "timestamp": data.get("timestamp"),
                                "rtt_ms": 0,
                                "ts": int(time.time() * 1000),
                            }
                            await ws.send(json.dumps(pong))

                        elif node_id:
                            self.registry.touch(node_id)
                            if msg_type == "handshake_ack":
                                self.registry.set_secure(node_id, True)
                            # Fan out telemetry, anc_state, and status messages to dashboards
                            self.broadcast_to_dashboards(message, node_id=node_id)

                    except json.JSONDecodeError:
                        pass
                else:
                    # Binary frame (Type 1 FFT, Type 2 Audio, Type 3 Health)
                    if node_id:
                        self.registry.touch(node_id)
                        self.broadcast_to_dashboards(message, node_id=node_id)

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if node_id:
                removed = self.registry.remove_node(node_id, ws)
                if removed:
                    self.broadcast_to_dashboards(json.dumps({
                        "type": "node_offline",
                        "node_id": node_id,
                        "ts": int(time.time() * 1000),
                    }))

    async def _liveness_monitor_loop(self):
        """
        Background liveness monitor.
        If node_pong/activity is not seen for >3.0s (3x ping interval), mark degraded.
        If silent for >15.0s, prune dead connection.
        """
        while True:
            await asyncio.sleep(1.0)
            now = time.time()
            for node_id, info in list(self.registry.nodes.items()):
                elapsed = now - info["last_seen_ts"]
                if elapsed > 15.0:
                    logger.warning(f"Node '{node_id}' silent for {elapsed:.1f}s — pruning dead link")
                    self.registry.remove_node(node_id)
                    self.broadcast_to_dashboards(json.dumps({
                        "type": "node_offline",
                        "node_id": node_id,
                        "ts": int(now * 1000),
                    }))
                elif elapsed > 3.0:
                    if self.registry.set_status(node_id, "degraded"):
                        logger.warning(f"Node '{node_id}' silent for {elapsed:.1f}s — transitioned to DEGRADED")
                        self.broadcast_to_dashboards(json.dumps({
                            "type": "node_status",
                            "node_id": node_id,
                            "status": "degraded",
                            "elapsed_s": round(elapsed, 1),
                            "ts": int(now * 1000),
                        }))

    async def handler(self, ws: ServerConnection):
        clean_path = ws.request.path.split("?")[0].rstrip("/")
        if clean_path == "/dashboard":
            await self.handle_dashboard(ws)
        elif clean_path == "/node":
            await self.handle_node(ws)
        else:
            await ws.close(1002, "Invalid path. Use /dashboard or /node")

    async def serve_forever(self, ssl_context=None):
        validate_startup_config(self.token, self.dev_mode)
        logger.info(f"Starting AEGIS Hub on {self.host}:{self.port} (TLS={'ENABLED' if ssl_context else 'DISABLED'})")
        self._monitor_task = asyncio.create_task(self._liveness_monitor_loop())
        try:
            # compression=None explicitly disables permessage-deflate (pure CPU/latency waste on binary frames)
            async with serve(self.handler, self.host, self.port, ssl=ssl_context, compression=None):
                await asyncio.Future()
        finally:
            if self._monitor_task and not self._monitor_task.done():
                self._monitor_task.cancel()


if __name__ == "__main__":
    validate_startup_config(AEGIS_AUTH_TOKEN, AEGIS_DEV_MODE)
    hub = AegisHub()
    asyncio.run(hub.serve_forever())
