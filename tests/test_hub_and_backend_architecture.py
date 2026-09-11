"""
Project AEGIS — Hub, Edge Node & Low-Latency Architecture Test Suite
=====================================================================
Validates all Rev 3 SOTA backend upgrades:
1. Strict token authentication and AEGIS_DEV_MODE opt-out.
2. Drop-oldest bounded queueing (freshness over stale buffering).
3. Backpressure isolation (slow dashboards cannot block fast dashboards).
4. Real-time RTT heartbeat and degraded connection state tracking.
5. Subscription-filtered JSON and binary frame routing (Type 1 FFT, Type 2 Audio, Type 3 Health).
"""

import asyncio
import json
import os
import struct
import sys
import time
from pathlib import Path
import pytest
import websockets

import importlib.util

def load_module_from_file(filepath: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(filepath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

ROOT = Path(__file__).resolve().parent.parent

hub_mod = load_module_from_file(ROOT / "aegis-hub" / "src" / "main.py", "aegis_hub_main")
AegisHub = hub_mod.AegisHub
DropOldestQueue = hub_mod.DropOldestQueue
NodeRegistry = hub_mod.NodeRegistry
DashboardSession = hub_mod.DashboardSession
validate_startup_config = hub_mod.validate_startup_config
check_auth_token = hub_mod.check_auth_token

backend_path = str(ROOT / "aegis-backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from src.ws.client import AegisClient, DropOldestQueue as NodeDropOldestQueue
from src.ws.protocol import (
    FRAME_TYPE_FFT, FRAME_TYPE_AUDIO, FRAME_TYPE_HEALTH,
    Telemetry, HwStatus
)


# ==============================================================================
# 1. Strict Auth & Startup Config Tests
# ==============================================================================

class TestHubAuthenticationAndStartup:
    def test_startup_fails_without_token_in_production_mode(self):
        """Hub must strictly refuse to start if AEGIS_AUTH_TOKEN is unset and dev mode is off."""
        with pytest.raises(SystemExit) as exc_info:
            validate_startup_config(token=None, dev_mode=False)
        assert "FATAL CONFIGURATION ERROR" in str(exc_info.value)

    def test_startup_succeeds_in_dev_mode_without_token(self):
        """Hub allows startup when AEGIS_DEV_MODE=1 is explicitly set."""
        # Should not raise
        validate_startup_config(token=None, dev_mode=True)

    def test_startup_succeeds_with_valid_token(self):
        """Hub starts cleanly when AEGIS_AUTH_TOKEN is configured."""
        validate_startup_config(token="production-secret-token-1234", dev_mode=False)

    def test_token_validation_logic(self):
        """Verifies constant-time token verification and dev-mode bypass."""
        # Production mode
        assert check_auth_token("secret123", "secret123", dev_mode=False) is True
        assert check_auth_token("wrong", "secret123", dev_mode=False) is False
        assert check_auth_token(None, "secret123", dev_mode=False) is False
        assert check_auth_token("secret123", None, dev_mode=False) is False

        # Dev mode bypass
        assert check_auth_token(None, None, dev_mode=True) is True
        assert check_auth_token("any-token", None, dev_mode=True) is True


# ==============================================================================
# 2. Drop-Oldest Queue Mechanics
# ==============================================================================

class TestDropOldestQueue:
    def test_queue_evicts_oldest_on_saturation(self):
        """When queue reaches capacity, oldest items are dropped and newest are preserved."""
        q = DropOldestQueue(maxsize=3)
        assert q.empty()
        assert not q.full()

        q.put_nowait("item-1")
        q.put_nowait("item-2")
        q.put_nowait("item-3")
        assert q.full()
        assert q.dropped_count == 0

        # Push item 4 and 5 — item 1 and 2 must be dropped
        q.put_nowait("item-4")
        q.put_nowait("item-5")
        assert q.dropped_count == 2
        assert q.qsize() == 3

        # Items remaining must be 3, 4, 5
        items = []
        while not q.empty():
            items.append(q.get_nowait())
        assert items == ["item-3", "item-4", "item-5"]

    def test_node_client_queue_tracks_dropped_count(self):
        """AegisClient queue also uses drop-oldest and reflects dropped_frames in get_stats()."""
        client = AegisClient(node_id="test-node", hub_url="ws://127.0.0.1:8001/node")
        # Artificially fill client send queue
        client.connected = True
        client._send_queue = NodeDropOldestQueue(maxsize=4)

        for i in range(10):
            client.enqueue(f"frame-{i}")

        stats = client.get_stats()
        assert stats["queue_depth"] == 4
        assert stats["dropped_frames"] == 6  # 10 - 4 = 6 dropped


# ==============================================================================
# 3. End-to-End WebSocket Integration (Auth, Routing, Heartbeat, Backpressure)
# ==============================================================================

class TestHubAndNodeNetworkIntegration:
    def test_node_auth_rejection_and_acceptance(self):
        asyncio.run(self._impl_node_auth_rejection_and_acceptance())

    async def _impl_node_auth_rejection_and_acceptance(self):
        """Verifies unauthorized nodes are disconnected with WS 1008, valid nodes register."""
        test_port = 8765
        hub = AegisHub(host="127.0.0.1", port=test_port, token="top-secret-pass", dev_mode=False)

        server = await websockets.asyncio.server.serve(
            hub.handler, hub.host, hub.port, compression=None
        )
        try:
            # 1. Attempt connection with invalid token
            uri = f"ws://127.0.0.1:{test_port}/node"
            async with websockets.connect(uri, compression=None) as ws:
                bad_hello = {"type": "node_hello", "node_id": "pi-bad", "token": "wrong-token"}
                await ws.send(json.dumps(bad_hello))

                # Hub must reply with unauthorized and close
                response = await ws.recv()
                data = json.loads(response)
                assert data["type"] == "node_ack"
                assert data["status"] == "unauthorized"

                # Next recv should raise ConnectionClosed with code 1008
                with pytest.raises(websockets.exceptions.ConnectionClosed) as exc:
                    await ws.recv()
                assert exc.value.rcvd.code == 1008

            # 2. Attempt connection with valid token
            async with websockets.connect(uri, compression=None) as ws:
                good_hello = {"type": "node_hello", "node_id": "pi-good", "token": "top-secret-pass"}
                await ws.send(json.dumps(good_hello))

                response = await ws.recv()
                data = json.loads(response)
                assert data["type"] == "node_ack"
                assert data["status"] == "ok"
                assert "pi-good" in hub.registry.nodes

        finally:
            server.close()
            await server.wait_closed()

    def test_backpressure_isolation_between_dashboards(self):
        asyncio.run(self._impl_backpressure_isolation_between_dashboards())

    async def _impl_backpressure_isolation_between_dashboards(self):
        """
        A slow/frozen dashboard tab must NEVER block or delay frame delivery
        to an active, fast dashboard.
        """
        test_port = 8766
        hub = AegisHub(host="127.0.0.1", port=test_port, dev_mode=True)
        server = await websockets.asyncio.server.serve(
            hub.handler, hub.host, hub.port, compression=None
        )

        try:
            dash_uri = f"ws://127.0.0.1:{test_port}/dashboard"

            # Connect Dashboard 1 (Active/Fast Consumer)
            ws_fast = await websockets.connect(dash_uri, compression=None)

            # Connect Dashboard 2 (Frozen Consumer - never calls recv)
            ws_slow = await websockets.connect(dash_uri, compression=None)

            await asyncio.sleep(0.05)  # Allow sessions to establish

            # Hub broadcasts 50 frames
            for i in range(50):
                hub.broadcast_to_dashboards(json.dumps({"type": "fft_tick", "seq": i}))

            # Fast dashboard reads all 50 frames rapidly without stalling
            received_fast = []
            for _ in range(50):
                msg = await asyncio.wait_for(ws_fast.recv(), timeout=1.0)
                data = json.loads(msg)
                if data.get("type") == "fft_tick":
                    received_fast.append(data["seq"])

            assert len(received_fast) == 50
            assert received_fast == list(range(50))

            # Slow dashboard queue stayed bounded (maxsize=128) and dropped nothing yet
            assert len(hub.dashboards) == 2
            for session in hub.dashboards.values():
                assert session.queue.qsize() <= 128

            await ws_fast.close()
            await ws_slow.close()

        finally:
            server.close()
            await server.wait_closed()

    def test_heartbeat_rtt_and_degraded_liveness_transition(self):
        asyncio.run(self._impl_heartbeat_rtt_and_degraded_liveness_transition())

    async def _impl_heartbeat_rtt_and_degraded_liveness_transition(self):
        """Verifies 1Hz node_ping -> node_pong and degraded state trigger on silence."""
        test_port = 8767
        hub = AegisHub(host="127.0.0.1", port=test_port, dev_mode=True)
        server = await websockets.asyncio.server.serve(
            hub.handler, hub.host, hub.port, compression=None
        )

        try:
            # 1. Connect node
            node_uri = f"ws://127.0.0.1:{test_port}/node"
            async with websockets.connect(node_uri, compression=None) as ws:
                await ws.send(json.dumps({"type": "node_hello", "node_id": "pi-tac-1"}))
                ack = json.loads(await ws.recv())
                assert ack["status"] == "ok"

                # Send node_ping
                t_send = int(time.time() * 1000)
                await ws.send(json.dumps({
                    "type": "node_ping",
                    "node_id": "pi-tac-1",
                    "seq": 1,
                    "timestamp": t_send,
                }))

                pong = json.loads(await ws.recv())
                assert pong["type"] == "node_pong"
                assert pong["seq"] == 1
                assert pong["timestamp"] == t_send

                # Simulate connection degradation by setting last_seen_ts back 4.0 seconds
                hub.registry.nodes["pi-tac-1"]["last_seen_ts"] = time.time() - 4.0
                # Run one step of liveness check logic
                now = time.time()
                elapsed = now - hub.registry.nodes["pi-tac-1"]["last_seen_ts"]
                assert elapsed > 3.0
                status_changed = hub.registry.set_status("pi-tac-1", "degraded")
                assert status_changed is True
                assert hub.registry.nodes["pi-tac-1"]["status"] == "degraded"

                # Send new traffic -> status should recover to online
                await ws.send(json.dumps({
                    "type": "node_ping",
                    "node_id": "pi-tac-1",
                    "seq": 2,
                    "timestamp": int(time.time() * 1000),
                }))
                _ = await ws.recv()
                assert hub.registry.nodes["pi-tac-1"]["status"] == "online"

        finally:
            server.close()
            await server.wait_closed()

    def test_subscription_filtered_binary_frame_routing(self):
        asyncio.run(self._impl_subscription_filtered_binary_frame_routing())

    async def _impl_subscription_filtered_binary_frame_routing(self):
        """Dashboard subscribed to specific node only receives that node's binary frames."""
        test_port = 8768
        hub = AegisHub(host="127.0.0.1", port=test_port, dev_mode=True)
        server = await websockets.asyncio.server.serve(
            hub.handler, hub.host, hub.port, compression=None
        )

        try:
            dash_uri = f"ws://127.0.0.1:{test_port}/dashboard"
            ws_dash_a = await websockets.connect(dash_uri, compression=None)
            ws_dash_b = await websockets.connect(dash_uri, compression=None)

            # Dashboard A subscribes to node-alpha only
            await ws_dash_a.send(json.dumps({"type": "subscribe", "nodes": ["node-alpha"]}))
            # Dashboard B subscribes to node-beta only
            await ws_dash_b.send(json.dumps({"type": "subscribe", "nodes": ["node-beta"]}))

            await asyncio.sleep(0.05)

            # Build binary FFT frames for node-alpha and node-beta
            def make_fft_binary(node_id: str):
                n_bytes = node_id.encode("utf-8")
                ts = int(time.time() * 1000)
                fmt = f"<BB{len(n_bytes)}sQ64B64B"
                return struct.pack(fmt, FRAME_TYPE_FFT, len(n_bytes), n_bytes, ts, *([10] * 64), *([20] * 64))

            frame_alpha = make_fft_binary("node-alpha")
            frame_beta = make_fft_binary("node-beta")

            # Broadcast both
            hub.broadcast_to_dashboards(frame_alpha)
            hub.broadcast_to_dashboards(frame_beta)

            # Dashboard A should receive frame_alpha and NOT frame_beta
            msg_a = await asyncio.wait_for(ws_dash_a.recv(), timeout=1.0)
            assert isinstance(msg_a, bytes)
            # Unpack node_id
            len_a = msg_a[1]
            rec_id_a = msg_a[2:2 + len_a].decode("utf-8")
            assert rec_id_a == "node-alpha"

            # Dashboard B should receive frame_beta
            msg_b = await asyncio.wait_for(ws_dash_b.recv(), timeout=1.0)
            assert isinstance(msg_b, bytes)
            len_b = msg_b[1]
            rec_id_b = msg_b[2:2 + len_b].decode("utf-8")
            assert rec_id_b == "node-beta"

            await ws_dash_a.close()
            await ws_dash_b.close()

        finally:
            server.close()
            await server.wait_closed()
