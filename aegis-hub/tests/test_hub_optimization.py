import asyncio
import json
import os
import struct
import time
import pytest
import websockets
from src.main import AegisHub


def test_01_hub_failfast_auth_requirement():
    """Verify hub fails to boot without auth token unless dev mode is explicit."""
    async def _run():
        old_token = os.environ.pop("AEGIS_AUTH_TOKEN", None)
        old_dev = os.environ.pop("AEGIS_DEV_MODE", None)
        try:
            with pytest.raises(RuntimeError, match="FATAL: AEGIS_AUTH_TOKEN is required"):
                AegisHub(auth_token=None, dev_mode=False)

            # In dev mode, it must succeed
            hub_dev = AegisHub(auth_token=None, dev_mode=True)
            assert hub_dev.dev_mode is True

            # With token, it must succeed
            hub_token = AegisHub(auth_token="super-secret-token", dev_mode=False)
            assert hub_token.auth_token == "super-secret-token"
        finally:
            if old_token:
                os.environ["AEGIS_AUTH_TOKEN"] = old_token
            if old_dev:
                os.environ["AEGIS_DEV_MODE"] = old_dev

    asyncio.run(_run())


def test_02_node_token_authentication_rejection():
    """Verify unauthorized node connection is rejected with 1008 policy violation."""
    async def _run():
        hub = AegisHub(host="127.0.0.1", port=8901, auth_token="correct-token", dev_mode=False)
        server_task = asyncio.create_task(hub.start())
        await asyncio.sleep(0.1)

        try:
            # Connect with invalid token
            async with websockets.connect("ws://127.0.0.1:8901/node", compression=None) as ws:
                hello = {
                    "type": "node_hello",
                    "node_id": "test-node",
                    "token": "wrong-token",
                    "hw": {}
                }
                await ws.send(json.dumps(hello))
                resp = json.loads(await ws.recv())
                assert resp["type"] == "node_ack"
                assert resp["status"] == "unauthorized"
                
                # The hub must close the socket with 1008
                with pytest.raises(websockets.exceptions.ConnectionClosed) as exc_info:
                    await ws.recv()
                assert exc_info.value.rcvd.code == 1008

            # Connect with valid token
            async with websockets.connect("ws://127.0.0.1:8901/node", compression=None) as ws:
                hello = {
                    "type": "node_hello",
                    "node_id": "test-node",
                    "token": "correct-token",
                    "hw": {}
                }
                await ws.send(json.dumps(hello))
                resp = json.loads(await ws.recv())
                assert resp["type"] == "node_ack"
                assert resp["status"] == "registered"
                assert resp["node_id"] == "test-node"
        finally:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

    asyncio.run(_run())


def test_03_heartbeat_ping_pong_rtt():
    """Verify node_ping -> node_pong and dashboard ping -> pong roundtrips."""
    async def _run():
        hub = AegisHub(host="127.0.0.1", port=8902, dev_mode=True)
        server_task = asyncio.create_task(hub.start())
        await asyncio.sleep(0.1)

        try:
            # Test node heartbeat ping/pong
            async with websockets.connect("ws://127.0.0.1:8902/node", compression=None) as ws:
                await ws.send(json.dumps({"type": "node_hello", "node_id": "node-ping-test", "hw": {}}))
                ack = json.loads(await ws.recv())
                assert ack["status"] == "registered"

                t0 = int(time.time() * 1000)
                await ws.send(json.dumps({
                    "type": "node_ping",
                    "node_id": "node-ping-test",
                    "timestamp": t0,
                    "seq": 42
                }))
                pong = json.loads(await ws.recv())
                assert pong["type"] == "node_pong"
                assert pong["node_id"] == "node-ping-test"
                assert pong["timestamp"] == t0
                assert pong["seq"] == 42

            # Test dashboard ping/pong
            async with websockets.connect("ws://127.0.0.1:8902/dashboard", compression=None) as ws:
                t0 = int(time.time() * 1000)
                await ws.send(json.dumps({
                    "type": "ping",
                    "timestamp": t0,
                    "seq": 99
                }))
                pong = json.loads(await ws.recv())
                assert pong["type"] == "pong"
                assert pong["timestamp"] == t0
                assert pong["seq"] == 99
                assert "ts" in pong
        finally:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

    asyncio.run(_run())


def test_04_drop_oldest_queue_behavior():
    """Verify that drop-oldest evicts the oldest items when a queue saturates."""
    hub = AegisHub(host="127.0.0.1", port=8903, dev_mode=True, dashboard_queue_size=3)
    
    # Mock a dummy websocket key
    dummy_ws = object()
    queue = asyncio.Queue(maxsize=3)
    hub.dashboard_queues[dummy_ws] = queue

    # Push items 1, 2, 3
    hub._enqueue_dashboard(dummy_ws, "msg-1")
    hub._enqueue_dashboard(dummy_ws, "msg-2")
    hub._enqueue_dashboard(dummy_ws, "msg-3")
    assert queue.full()

    # Push item 4: should evict msg-1 and keep 2, 3, 4
    hub._enqueue_dashboard(dummy_ws, "msg-4")
    assert queue.full()

    items = []
    while not queue.empty():
        items.append(queue.get_nowait())

    assert items == ["msg-2", "msg-3", "msg-4"]


def test_05_slow_dashboard_backpressure_isolation():
    """Verify that a slow dashboard does not delay delivery to a fast dashboard."""
    async def _run():
        hub = AegisHub(host="127.0.0.1", port=8904, dev_mode=True, dashboard_queue_size=4)
        server_task = asyncio.create_task(hub.start())
        await asyncio.sleep(0.1)

        try:
            # Fast dashboard reads eagerly
            fast_ws = await websockets.connect("ws://127.0.0.1:8904/dashboard", compression=None)
            # Slow dashboard reads nothing
            slow_ws = await websockets.connect("ws://127.0.0.1:8904/dashboard", compression=None)
            await asyncio.sleep(0.05)

            received = []
            async def fast_reader():
                for _ in range(10):
                    msg = await fast_ws.recv()
                    received.append(json.loads(msg)["num"])

            reader_task = asyncio.create_task(fast_reader())

            # Broadcast 10 messages with small yields to allow drain
            for i in range(10):
                await hub.broadcast_to_dashboards(json.dumps({"type": "test", "num": i}))
                await asyncio.sleep(0.01)

            await asyncio.wait_for(reader_task, timeout=1.0)
            assert received == list(range(10))

            await fast_ws.close()
            await slow_ws.close()
        finally:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

    asyncio.run(_run())


def test_06_targeted_subscriptions():
    """Verify that dashboard subscriptions filter frames by node_id."""
    async def _run():
        hub = AegisHub(host="127.0.0.1", port=8905, dev_mode=True)
        server_task = asyncio.create_task(hub.start())
        await asyncio.sleep(0.1)

        try:
            dash1 = await websockets.connect("ws://127.0.0.1:8905/dashboard", compression=None)
            dash2 = await websockets.connect("ws://127.0.0.1:8905/dashboard", compression=None)

            # dash1 subscribes only to "node-1"
            await dash1.send(json.dumps({"type": "subscribe", "nodes": ["node-1"]}))
            ack1 = json.loads(await dash1.recv())
            assert ack1["type"] == "subscription_ack"
            assert ack1["nodes"] == ["node-1"]

            # dash2 subscribes to "*" (all)
            await dash2.send(json.dumps({"type": "subscribe", "nodes": ["*"]}))
            ack2 = json.loads(await dash2.recv())
            assert ack2["type"] == "subscription_ack"
            assert ack2["nodes"] == ["*"]

            # Broadcast message for node-2
            await hub.broadcast_to_dashboards(
                json.dumps({"type": "telemetry", "node_id": "node-2", "latency_ms": 12.0}),
                node_id="node-2"
            )

            # dash2 must receive it
            dash2_msg = json.loads(await asyncio.wait_for(dash2.recv(), timeout=1.0))
            assert dash2_msg["node_id"] == "node-2"

            # dash1 must NOT receive it (timeout confirms filter worked)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(dash1.recv(), timeout=0.2)

            # Broadcast message for node-1
            await hub.broadcast_to_dashboards(
                json.dumps({"type": "telemetry", "node_id": "node-1", "latency_ms": 9.0}),
                node_id="node-1"
            )

            # Both should receive it
            dash1_msg = json.loads(await asyncio.wait_for(dash1.recv(), timeout=1.0))
            assert dash1_msg["node_id"] == "node-1"
            dash2_msg2 = json.loads(await asyncio.wait_for(dash2.recv(), timeout=1.0))
            assert dash2_msg2["node_id"] == "node-1"

            await dash1.close()
            await dash2.close()
        finally:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

    asyncio.run(_run())


def test_07_multiframe_binary_wire_format():
    """Verify Frame Type 1 (FFT), Type 2 (Audio), Type 3 (Health) routing and unpacking."""
    async def _run():
        hub = AegisHub(host="127.0.0.1", port=8906, dev_mode=True)
        server_task = asyncio.create_task(hub.start())
        await asyncio.sleep(0.1)

        try:
            dash = await websockets.connect("ws://127.0.0.1:8906/dashboard", compression=None)
            # Subscribe specifically to alpha-node
            await dash.send(json.dumps({"type": "subscribe", "nodes": ["alpha-node"]}))
            await dash.recv()  # ack

            node_id_bytes = b"alpha-node"
            ts = 1700000000000

            # 1. FFT Frame (Type 1)
            raw_bins = bytes([10] * 64)
            enh_bins = bytes([20] * 64)
            fft_frame = struct.pack(f"<BB{len(node_id_bytes)}sQ64s64s", 1, len(node_id_bytes), node_id_bytes, ts, raw_bins, enh_bins)
            await hub.broadcast_to_dashboards(fft_frame)
            rec_fft = await asyncio.wait_for(dash.recv(), timeout=1.0)
            assert isinstance(rec_fft, bytes)
            assert rec_fft[0] == 1  # frame_type
            assert rec_fft[1] == len(node_id_bytes)

            # 2. Audio Frame (Type 2)
            pcm_bytes = b"\x00\x01" * 160  # 160 samples of 16-bit PCM
            audio_frame = struct.pack(f"<BB{len(node_id_bytes)}sQ", 2, len(node_id_bytes), node_id_bytes, ts) + pcm_bytes
            await hub.broadcast_to_dashboards(audio_frame)
            rec_audio = await asyncio.wait_for(dash.recv(), timeout=1.0)
            assert isinstance(rec_audio, bytes)
            assert rec_audio[0] == 2
            assert rec_audio[2 + len(node_id_bytes) + 8:] == pcm_bytes

            # 3. Health Frame (Type 3)
            health_bytes = b'{"battery_pct": 98, "temp_c": 42.1}'
            health_frame = struct.pack(f"<BB{len(node_id_bytes)}sQ", 3, len(node_id_bytes), node_id_bytes, ts) + health_bytes
            await hub.broadcast_to_dashboards(health_frame)
            rec_health = await asyncio.wait_for(dash.recv(), timeout=1.0)
            assert isinstance(rec_health, bytes)
            assert rec_health[0] == 3
            assert rec_health[2 + len(node_id_bytes) + 8:] == health_bytes

            await dash.close()
        finally:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

    asyncio.run(_run())
