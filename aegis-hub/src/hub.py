"""src/hub.py — AegisHub: Mesh Router + Registry + Control Plane.

Five subsystems, each with one job:
  1. Registry   — nodes: Dict[node_id, NodeConn], dashboards: Dict[ws, DashConn]
  2. Router     — MeshRouter: header fast-parse → echo suppression → fan-out
  3. ConnQueue  — per-connection bounded queue + independent drain task
  4. Liveness   — LivenessMonitor: 1s tick, degraded/offline FSM
  5. Auth       — verify_token, dev-mode per-connection warning

Hub does NOT:
  - Mix audio (nodes mix locally)
  - Transcode (frames pass through byte-identical)
  - Use asyncio.gather over sends (a slow consumer never delays a fast one)
"""
from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set

import websockets.exceptions
from websockets.asyncio.server import ServerConnection

from .auth import verify_token, warn_dev_mode_connection
from .connqueue import ConnQueue
from .frames import parse_source_id
from .liveness import LivenessMonitor
from .routing import MeshRouter, SubscriptionMixin

logger = logging.getLogger("aegis.hub")

HUB_STATS_INTERVAL_S = 10.0   # broadcast hub_stats to dashboards every 10s


# ---------------------------------------------------------------------------
# Connection data objects
# ---------------------------------------------------------------------------

@dataclass
class NodeConn(SubscriptionMixin):
    """State for one connected mesh node."""
    node_id:      str
    ws:           ServerConnection
    q:            ConnQueue
    capabilities: list = field(default_factory=list)
    hw:           dict = field(default_factory=dict)
    joined_at:    float = field(default_factory=time.monotonic)
    last_pong_at: float = field(default_factory=time.monotonic)
    state:        str = "online"        # "online" | "degraded" | "offline"
    secure:       bool = False

    def __post_init__(self):
        self.__init_subscription__()


@dataclass
class DashConn(SubscriptionMixin):
    """State for one connected dashboard client."""
    ws:        ServerConnection
    q:         ConnQueue
    joined_at: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        self.__init_subscription__()


# ---------------------------------------------------------------------------
# AegisHub
# ---------------------------------------------------------------------------

class AegisHub:
    """
    AEGIS Mesh Hub — routes audio between nodes, serves dashboards.

    Instantiate once; call serve_forever() or start() to run.
    The handler() method is passed directly to websockets.serve().
    """

    def __init__(
        self,
        host:                str = "0.0.0.0",
        port:                int = 8001,
        auth_token:          Optional[str] = None,
        dev_mode:            bool = False,
        ssl_cert:            Optional[str] = None,
        ssl_key:             Optional[str] = None,
        node_queue_size:     int = 256,
        dashboard_queue_size: int = 256,
    ) -> None:
        import os
        self.host        = host
        self.port        = port
        self.dev_mode    = dev_mode or (os.getenv("AEGIS_DEV_MODE", "0").lower() in ("1", "true", "yes"))
        self.auth_token  = auth_token or os.getenv("AEGIS_AUTH_TOKEN")
        self.ssl_cert    = ssl_cert or os.getenv("AEGIS_HUB_SSL_CERT")
        self.ssl_key     = ssl_key  or os.getenv("AEGIS_HUB_SSL_KEY")
        self.node_queue_size      = node_queue_size
        self.dashboard_queue_size = dashboard_queue_size

        # Fail-fast auth check (raises RuntimeError in dev-friendly way for test compat)
        if not self.dev_mode and not self.auth_token:
            raise RuntimeError(
                "FATAL: AEGIS_AUTH_TOKEN is required to start AegisHub. "
                "Set AEGIS_AUTH_TOKEN in your environment or launch with "
                "AEGIS_DEV_MODE=1 for local development without credentials."
            )

        if self.dev_mode:
            logger.warning("=" * 60)
            logger.warning("AEGIS_DEV_MODE=1 ACTIVE: AUTHENTICATION CHECKS ARE BYPASSED")
            logger.warning("DO NOT USE IN PRODUCTION OR OUTSIDE ISOLATED LOCAL DEV")
            logger.warning("=" * 60)

        # Registries
        self._nodes:      Dict[str, NodeConn] = {}
        self._dashboards: Dict[object, DashConn] = {}   # ws → DashConn

        # Backward-compat public dicts (test_hub_optimization.py writes directly to these)
        # These are kept in sync with _dashboards in handle_dashboard().
        self.dashboard_queues: Dict[object, asyncio.Queue] = {}
        self.dashboard_subscriptions: Dict[object, Optional[Set[str]]] = {}

        # Subsystems
        self._router   = MeshRouter()
        self._liveness = LivenessMonitor(
            get_nodes       = lambda: self._nodes,
            remove_node     = self._registry_remove_node,
            broadcast_json  = self._broadcast_json_to_dashboards,
        )

        self._hub_stats_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------

    def _registry_remove_node(self, node_id: str) -> bool:
        """Remove node from registry (called by liveness or disconnect)."""
        conn = self._nodes.pop(node_id, None)
        if conn is not None:
            logger.info(f"Registry: removed node '{node_id}' (active: {list(self._nodes.keys())})")
            return True
        return False

    # ------------------------------------------------------------------
    # Broadcast helpers (used by liveness + hub_stats)
    # ------------------------------------------------------------------

    async def _broadcast_json_to_dashboards(
        self, message: str, node_id: Optional[str] = None
    ) -> None:
        """Fan-out a JSON string to all subscribed dashboards."""
        for dash in self._dashboards.values():
            if node_id is None or dash.subscribes(node_id):
                dash.q.enqueue(message)

    # Kept for backward compatibility with existing test suite
    async def broadcast_to_dashboards(
        self, msg: str | bytes, node_id: Optional[str] = None
    ) -> None:
        if isinstance(msg, bytes):
            for dash in self._dashboards.values():
                if node_id is None or dash.subscribes(node_id):
                    dash.q.enqueue(msg)
        else:
            await self._broadcast_json_to_dashboards(msg, node_id)

    # Kept for backward compatibility with existing test suite
    def _enqueue_dashboard(self, ws, msg: str | bytes) -> None:
        """Enqueue to a dashboard using the compat dashboard_queues dict.
        Works for both real DashConn connections and test_04's mock queues.
        """
        # First try the real ConnQueue path
        dash = self._dashboards.get(ws)
        if dash:
            dash.q.enqueue(msg)
            return
        # Fallback: use the raw compat queue dict (test_04 injects a bare asyncio.Queue)
        queue = self.dashboard_queues.get(ws)
        if queue is not None:
            if queue.full():
                try:
                    queue.get_nowait()
                except (asyncio.QueueEmpty, ValueError):
                    pass
            try:
                queue.put_nowait(msg)
            except (asyncio.QueueFull, RuntimeError):
                pass

    # ------------------------------------------------------------------
    # Dashboard handler
    # ------------------------------------------------------------------

    async def handle_dashboard(self, ws: ServerConnection) -> None:
        """Dashboard WebSocket connection lifecycle."""
        _enable_tcp_nodelay(ws)

        cq   = ConnQueue(ws, maxsize=self.dashboard_queue_size, label=f"dash:{id(ws)}")
        dash = DashConn(ws=ws, q=cq)
        self._dashboards[ws] = dash

        drain_task = asyncio.create_task(cq.drain())
        # Keep compat dicts in sync
        self.dashboard_queues[ws] = cq._q
        self.dashboard_subscriptions[ws] = None
        logger.info(f"Dashboard connected (total: {len(self._dashboards)})")

        try:
            # Immediately push node_online for all currently-registered nodes
            for node_id, conn in self._nodes.items():
                online_msg = json.dumps({
                    "type":    "node_online",
                    "node_id": node_id,
                    "hw":      conn.hw,
                    "capabilities": conn.capabilities,
                    "secure":  conn.secure,
                    "state":   conn.state,
                    "ts":      int(time.time() * 1000),
                })
                cq.enqueue(online_msg)

            async for message in ws:
                if isinstance(message, str):
                    try:
                        data     = json.loads(message)
                        msg_type = data.get("type")

                        if msg_type == "subscribe":
                            node_ids = data.get("nodes", []) or data.get("node_ids", [])
                            dash.set_subscriptions(node_ids)
                            cq.enqueue(json.dumps({
                                "type":  "subscription_ack",
                                "nodes": dash.subscription_list,
                            }))

                        elif msg_type == "ping":
                            cq.enqueue(json.dumps({
                                "type":      "pong",
                                "seq":       data.get("seq"),
                                "timestamp": data.get("timestamp"),
                                "ts":        int(time.time() * 1000),
                            }))

                        else:
                            # Forward control messages to target node
                            target_id = data.get("node_id") or data.get("clientId")
                            if target_id and target_id in self._nodes:
                                self._nodes[target_id].q.enqueue(message)

                    except json.JSONDecodeError:
                        pass

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._dashboards.pop(ws, None)
            self.dashboard_queues.pop(ws, None)
            self.dashboard_subscriptions.pop(ws, None)
            drain_task.cancel()
            logger.info(f"Dashboard disconnected (remaining: {len(self._dashboards)})")

    # ------------------------------------------------------------------
    # Node handler
    # ------------------------------------------------------------------

    async def handle_node(self, ws: ServerConnection) -> None:
        """Mesh node WebSocket connection lifecycle."""
        _enable_tcp_nodelay(ws)
        node_id: Optional[str] = None
        cq: Optional[ConnQueue] = None
        drain_task: Optional[asyncio.Task] = None
        logger.info("Node connection attempt...")

        try:
            async for message in ws:
                if isinstance(message, str):
                    try:
                        data     = json.loads(message)
                        msg_type = data.get("type")

                        # ── node_hello: registration ──────────────────────────
                        if msg_type == "node_hello":
                            claimed_id = data.get("node_id")
                            if not claimed_id:
                                logger.warning("node_hello missing node_id — closing")
                                break

                            # Auth gate
                            token = data.get("token")
                            if not self.dev_mode:
                                if not verify_token(token, self.auth_token):
                                    logger.warning(
                                        f"Auth FAILED for '{claimed_id}' — closing 1008"
                                    )
                                    await ws.send(json.dumps({
                                        "type":   "node_ack",
                                        "status": "unauthorized",
                                    }))
                                    await ws.close(1008, "Unauthorized: invalid token")
                                    return
                            else:
                                warn_dev_mode_connection(claimed_id)

                            # Duplicate ID gate
                            if claimed_id in self._nodes:
                                logger.warning(
                                    f"Duplicate node_id '{claimed_id}' rejected — "
                                    "a node with this ID is already registered"
                                )
                                await ws.send(json.dumps({
                                    "type":   "node_ack",
                                    "status": "duplicate_id",
                                    "node_id": claimed_id,
                                }))
                                await ws.close(1008, f"Duplicate node_id: {claimed_id}")
                                return

                            # Register node
                            node_id = claimed_id
                            cq      = ConnQueue(
                                ws, maxsize=self.node_queue_size,
                                label=f"node:{node_id}"
                            )
                            drain_task = asyncio.create_task(cq.drain())

                            conn = NodeConn(
                                node_id      = node_id,
                                ws           = ws,
                                q            = cq,
                                capabilities = data.get("capabilities", []),
                                hw           = data.get("hw", {}),
                            )
                            self._nodes[node_id] = conn
                            logger.info(
                                f"Node '{node_id}' registered "
                                f"(caps={conn.capabilities}, "
                                f"active: {list(self._nodes.keys())})"
                            )

                            # Acknowledge registration
                            # Status is "registered" (legacy compat) — Phase 1 node_client
                            # accepts both "ok" and "registered".
                            await ws.send(json.dumps({
                                "type":    "node_ack",
                                "status":  "registered",
                                "node_id": node_id,
                            }))

                            # Broadcast node_online to all dashboards
                            await self._broadcast_json_to_dashboards(
                                json.dumps({
                                    "type":         "node_online",
                                    "node_id":      node_id,
                                    "hw":           conn.hw,
                                    "capabilities": conn.capabilities,
                                    "secure":       False,
                                    "state":        "online",
                                    "ts":           int(time.time() * 1000),
                                })
                            )

                        # ── node_ping: heartbeat ──────────────────────────────
                        elif msg_type == "node_ping":
                            responding_id = node_id or data.get("node_id")
                            pong = json.dumps({
                                "type":      "node_pong",
                                "node_id":   responding_id,
                                "seq":       data.get("seq", 0),
                                "timestamp": data.get("timestamp"),
                                "ts":        int(time.time() * 1000),
                            })
                            # Send pong directly (bypass drain queue for minimal RTT)
                            await ws.send(pong)

                            # Update liveness timestamp
                            if node_id and node_id in self._nodes:
                                self._nodes[node_id].last_pong_at = time.monotonic()
                                if self._nodes[node_id].state == "degraded":
                                    self._nodes[node_id].state = "online"

                        # ── ping (legacy) ─────────────────────────────────────
                        elif msg_type == "ping":
                            await ws.send(json.dumps({
                                "type":      "pong",
                                "seq":       data.get("seq"),
                                "timestamp": data.get("timestamp"),
                                "ts":        int(time.time() * 1000),
                            }))

                        # ── handshake_ack ─────────────────────────────────────
                        elif msg_type == "handshake_ack" and node_id:
                            if node_id in self._nodes:
                                self._nodes[node_id].secure = True
                            self._router.route_json(message, node_id, self._dashboards)

                        # ── subscribe (node subscribing to other nodes) ───────
                        elif msg_type == "subscribe" and node_id and node_id in self._nodes:
                            node_ids = data.get("node_ids", data.get("nodes", []))
                            self._nodes[node_id].set_subscriptions(node_ids)
                            logger.info(
                                f"Node '{node_id}' subscription: "
                                f"{self._nodes[node_id].subscription_list}"
                            )

                        # ── all other JSON (telemetry, anc_state, etc.) ───────
                        elif node_id:
                            self._router.route_json(message, node_id, self._dashboards)

                    except json.JSONDecodeError:
                        pass

                elif isinstance(message, bytes):
                    # Binary audio/FFT/health frame — router fast-path
                    if node_id:
                        src_id = parse_source_id(message) or node_id
                        self._router.route_binary(
                            message, src_id,
                            self._nodes, self._dashboards,
                        )

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if node_id:
                removed = self._registry_remove_node(node_id)
                if removed:
                    await self._broadcast_json_to_dashboards(
                        json.dumps({
                            "type":    "node_offline",
                            "node_id": node_id,
                            "reason":  "disconnected",
                            "ts":      int(time.time() * 1000),
                        })
                    )
            if drain_task:
                drain_task.cancel()
                try:
                    await drain_task
                except asyncio.CancelledError:
                    pass

    # ------------------------------------------------------------------
    # Hub stats broadcaster
    # ------------------------------------------------------------------

    async def _hub_stats_loop(self) -> None:
        """Broadcast hub_stats to all dashboards every 10s."""
        while True:
            await asyncio.sleep(HUB_STATS_INTERVAL_S)
            dropped: Dict[str, int] = {}
            for node_id, conn in self._nodes.items():
                if conn.q.dropped:
                    dropped[node_id] = conn.q.dropped
            for dash in self._dashboards.values():
                if dash.q.dropped:
                    dropped[f"dash:{id(dash.ws)}"] = dash.q.dropped

            stats_msg = json.dumps({
                "type":          "hub_stats",
                "nodes_online":  len(self._nodes),
                "dashboards":    len(self._dashboards),
                "dropped_frames": dropped,
                "ts":            int(time.time() * 1000),
            })
            await self._broadcast_json_to_dashboards(stats_msg)

    # ------------------------------------------------------------------
    # Request dispatcher
    # ------------------------------------------------------------------

    async def handler(self, ws: ServerConnection) -> None:
        """WebSocket connection dispatcher — routes by path."""
        clean_path = ws.request.path.split("?")[0].rstrip("/")
        if clean_path == "/dashboard":
            await self.handle_dashboard(ws)
        elif clean_path == "/node":
            await self.handle_node(ws)
        else:
            await ws.close(1002, "Invalid path. Use /dashboard or /node")

    # ------------------------------------------------------------------
    # Serve
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the hub (compatibility alias for serve_forever)."""
        await self.serve_forever()

    async def serve_forever(self, sock=None) -> None:
        """
        Start the hub server.

        Args:
            sock: Optional pre-bound socket (from main.py) with TCP_NODELAY
                  already set at the listener level. If None, websockets creates
                  and binds the socket internally.
        """
        import ssl as _ssl
        from websockets.asyncio.server import serve as _serve

        ssl_ctx = None
        if self.ssl_cert and self.ssl_key:
            import os
            if os.path.exists(self.ssl_cert) and os.path.exists(self.ssl_key):
                ssl_ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_SERVER)
                ssl_ctx.load_cert_chain(certfile=self.ssl_cert, keyfile=self.ssl_key)
                logger.info("TLS enabled on hub")
            else:
                logger.warning("SSL cert/key not found — falling back to plain ws://")

        scheme = "wss" if ssl_ctx else "ws"
        logger.info(f"Starting Hub on {scheme}://{self.host}:{self.port} (compression=None)")

        # Start background tasks
        self._hub_stats_task = asyncio.create_task(self._hub_stats_loop())
        liveness_task = asyncio.create_task(self._liveness.monitor_forever())

        serve_kwargs = dict(compression=None, ssl=ssl_ctx)
        if sock is not None:
            async with _serve(self.handler, sock=sock, **serve_kwargs):
                await asyncio.Future()
        else:
            async with _serve(self.handler, self.host, self.port, **serve_kwargs):
                await asyncio.Future()


# ---------------------------------------------------------------------------
# TCP_NODELAY helper
# ---------------------------------------------------------------------------

def _enable_tcp_nodelay(ws: ServerConnection) -> None:
    """Set TCP_NODELAY on this connection's underlying socket."""
    try:
        sock = ws.transport.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception as e:
        logger.debug(f"TCP_NODELAY (per-connection): {e}")
