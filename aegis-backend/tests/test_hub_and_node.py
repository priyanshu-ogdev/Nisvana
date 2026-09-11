import asyncio
import json
import os
import sys
import time
import pytest
import websockets

import importlib.util

hub_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../aegis-hub/src/main.py"))
spec = importlib.util.spec_from_file_location("aegis_hub_module", hub_path)
hub_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hub_module)
AegisHub = hub_module.AegisHub

from src.ws.client import AegisClient
from src.ws.protocol import FRAME_TYPE_FFT, HwStatus


def test_hub_and_node_full_integration():
    """Verify end-to-end integration between AegisClient (edge node) and AegisHub."""
    async def _run():
        auth_token = "production-test-token-42"
        hub = AegisHub(host="127.0.0.1", port=8910, auth_token=auth_token, dev_mode=False)
        server_task = asyncio.create_task(hub.start())
        await asyncio.sleep(0.1)

        try:
            # 1. Test Node with Invalid Auth Token
            bad_client = AegisClient(
                hub_url="ws://127.0.0.1:8910/node",
                node_id="bad-node",
                auth_token="wrong-token",
                reconnect_delay=0.1
            )
            bad_client_task = asyncio.create_task(bad_client.connect())
            await asyncio.sleep(0.2)
            assert not bad_client.connected
            bad_client_task.cancel()

            # 2. Test Node with Valid Auth Token
            client = AegisClient(
                hub_url="ws://127.0.0.1:8910/node",
                node_id="node-alpha",
                auth_token=auth_token,
                reconnect_delay=0.1
            )
            client_task = asyncio.create_task(client.connect())
            await asyncio.sleep(0.2)
            assert client.connected

            # Check that Hub registered node-alpha
            assert "node-alpha" in hub.registry.nodes

            # 3. Connect a Dashboard and verify node_online
            dash = await websockets.connect("ws://127.0.0.1:8910/dashboard", compression=None)
            node_online_msg = json.loads(await asyncio.wait_for(dash.recv(), timeout=1.0))
            assert node_online_msg["type"] == "node_online"
            assert node_online_msg["node_id"] == "node-alpha"

            # 4. Wait for Node Ping Loop to measure real RTT
            # The ping loop runs at 1Hz, give it 1.2s to complete at least one roundtrip
            await asyncio.sleep(1.2)
            stats = client.get_stats()
            assert stats["node_rtt_ms"] is not None
            assert stats["node_rtt_ms"] >= 0.0
            assert client.link_quality == "healthy"

            # 5. Push Binary FFT frame from Node and verify it arrives at Dashboard
            raw_bins = [5] * 64
            enh_bins = [15] * 64
            client.push_fft_binary(raw_bins, enh_bins)

            dash_frame = await asyncio.wait_for(dash.recv(), timeout=1.0)
            assert isinstance(dash_frame, bytes)
            assert dash_frame[0] == FRAME_TYPE_FFT
            id_len = dash_frame[1]
            extracted_id = dash_frame[2 : 2 + id_len].decode("utf-8")
            assert extracted_id == "node-alpha"

            # 6. Test Drop-Oldest Queue on Node Client
            # Fill the queue beyond maxsize (128)
            initial_drops = client.dropped_frames
            for i in range(150):
                client.enqueue(f"filler-{i}")

            stats_after = client.get_stats()
            assert stats_after["dropped_frames"] > initial_drops
            assert stats_after["queue_depth"] <= 128

            # Clean up
            await dash.close()
            client_task.cancel()
        finally:
            server_task.cancel()
            try:
                await server_task
            except asyncio.CancelledError:
                pass

    asyncio.run(_run())
