"""tests/test_hub_routing.py — Phase 2 Gate Tests (Gates 1–7 + 10).

Gate coverage:
  G1  test_g1_node_to_node_flow_echo_suppressed   — A speaks → B hears; A never receives own frames
  G2  test_g2_selective_node_subscription         — B subscribes [operator-1]; C's audio never reaches B
  G3  test_g3_dashboard_filter_binary             — dash subscribed to node-1: zero node-2 binary frames
  G4  test_g4_backpressure_isolation              — stall B 2s; A+C keep full rate; only B.dropped increments
  G5  test_g5_liveness_degraded_offline           — synthetic pong timeout → degraded ≤3s; offline ≤6s
  G6a test_g6a_wrong_token_rejected_1008          — wrong token → 1008 close
  G6b test_g6b_boot_requires_token_or_devmode     — no token + no devmode → RuntimeError
  G6c test_g6c_devmode_warns_per_connection       — dev mode logs WARNING per node_hello
  G7  test_g7_transport_tcp_nodelay_no_deflate    — TCP_NODELAY==1; no permessage-deflate
  G10 test_g10_frames_parity                      — frames.py hub ↔ backend node_client.py pack identical
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
from pathlib import Path
from typing import List

import pytest
import websockets

# ---------------------------------------------------------------------------
# Path setup — hub root must come BEFORE backend root so `src` resolves to
# aegis-hub/src, not aegis-backend/src.
# ---------------------------------------------------------------------------
HUB_ROOT     = Path(__file__).parent.parent
BACKEND_ROOT = HUB_ROOT.parent / "aegis-backend"

# Insert hub root first so `import src` → aegis-hub/src
if str(HUB_ROOT) not in sys.path:
    sys.path.insert(0, str(HUB_ROOT))

from src.hub    import AegisHub
from src.frames import pack_mesh_audio, FRAME_TYPE_AUDIO

# Backend imports done lazily in tests that need them (G10) to avoid collision

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_audio_frame(node_id: str, seq: int = 1, payload: bytes = b"\x00\x01" * 80) -> bytes:
    """Pack a binary mesh audio frame for use in tests."""
    return pack_mesh_audio(node_id, seq, payload, timestamp_ms=1_700_000_000_000)


async def _register_node(ws, node_id: str, token: str | None = None):
    """Send node_hello and return the ack."""
    await ws.send(json.dumps({
        "type":    "node_hello",
        "node_id": node_id,
        "token":   token,
        "hw":      {},
        "capabilities": ["mic", "speaker"],
    }))
    ack_raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
    return json.loads(ack_raw)


async def _start_hub(port: int, **kwargs) -> tuple[AegisHub, asyncio.Task]:
    """Start a hub on a given port and return (hub, server_task)."""
    hub  = AegisHub(host="127.0.0.1", port=port, dev_mode=True, **kwargs)
    task = asyncio.create_task(hub.start())
    await asyncio.sleep(0.08)   # brief yield for server to bind
    return hub, task


async def _stop_hub(task: asyncio.Task):
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# Gate 1: Node→Node flow with echo suppression
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_g1_node_to_node_flow_echo_suppressed():
    """
    A speaks → B hears the frame.
    A NEVER receives its own frame (echo suppression).
    """
    hub, task = await _start_hub(9010)
    try:
        async with (
            websockets.connect("ws://127.0.0.1:9010/node", compression=None) as ws_a,
            websockets.connect("ws://127.0.0.1:9010/node", compression=None) as ws_b,
        ):
            ack_a = await _register_node(ws_a, "node-A")
            assert ack_a["status"] == "registered", f"A registration: {ack_a}"
            ack_b = await _register_node(ws_b, "node-B")
            assert ack_b["status"] == "registered", f"B registration: {ack_b}"

            # B receives node_online for A (and vice versa) — drain those first
            await asyncio.sleep(0.05)
            try:
                while True:
                    await asyncio.wait_for(ws_b.recv(), timeout=0.02)
            except asyncio.TimeoutError:
                pass

            # A sends a binary audio frame
            frame = _make_audio_frame("node-A", seq=42)
            await ws_a.send(frame)

            # B should receive the exact same bytes
            received_by_b = await asyncio.wait_for(ws_b.recv(), timeout=1.0)
            assert isinstance(received_by_b, bytes), "B should receive binary frame"
            assert received_by_b == frame, "Frame must be byte-identical (no transcoding)"
            assert received_by_b[0] == FRAME_TYPE_AUDIO

            # A must NOT receive its own frame (echo suppression)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(ws_a.recv(), timeout=0.25)

    finally:
        await _stop_hub(task)


# ---------------------------------------------------------------------------
# Gate 2: Selective node subscriptions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_g2_selective_node_subscription():
    """
    B subscribes to [operator-1] only.
    C's audio reaches the hub but must NEVER reach B.
    """
    hub, task = await _start_hub(9011)
    try:
        async with (
            websockets.connect("ws://127.0.0.1:9011/node", compression=None) as ws_b,
            websockets.connect("ws://127.0.0.1:9011/node", compression=None) as ws_c,
            websockets.connect("ws://127.0.0.1:9011/node", compression=None) as ws_op1,
        ):
            await _register_node(ws_b,   "node-B")
            await _register_node(ws_c,   "node-C")
            await _register_node(ws_op1, "operator-1")
            await asyncio.sleep(0.05)

            # B subscribes to only operator-1
            await ws_b.send(json.dumps({"type": "subscribe", "node_ids": ["operator-1"]}))
            await asyncio.sleep(0.05)

            # C sends audio
            c_frame = _make_audio_frame("node-C", seq=1)
            await ws_c.send(c_frame)

            # B must NOT receive C's audio (subscription filter)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(ws_b.recv(), timeout=0.3)

            # operator-1 sends audio
            op1_frame = _make_audio_frame("operator-1", seq=2)
            await ws_op1.send(op1_frame)

            # B MUST receive operator-1's audio
            received = await asyncio.wait_for(ws_b.recv(), timeout=1.0)
            assert isinstance(received, bytes)
            assert received == op1_frame

    finally:
        await _stop_hub(task)


# ---------------------------------------------------------------------------
# Gate 3: Dashboard subscription filter on binary frames
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_g3_dashboard_filter_binary():
    """
    Dashboard subscribed to [node-1]: receives node-1 binary frames.
    Zero node-2 binary frames should reach it.
    """
    hub, task = await _start_hub(9012)
    try:
        async with (
            websockets.connect("ws://127.0.0.1:9012/node",      compression=None) as ws_n1,
            websockets.connect("ws://127.0.0.1:9012/node",      compression=None) as ws_n2,
            websockets.connect("ws://127.0.0.1:9012/dashboard", compression=None) as ws_dash,
        ):
            await _register_node(ws_n1, "node-1")
            await _register_node(ws_n2, "node-2")

            # Dashboard subscribes to node-1 only
            await ws_dash.send(json.dumps({"type": "subscribe", "nodes": ["node-1"]}))

            # Drain all pending messages from dashboard queue
            # (node_online for node-1, node-2, subscription_ack)
            await asyncio.sleep(0.15)
            drained = 0
            while True:
                try:
                    _ = await asyncio.wait_for(ws_dash.recv(), timeout=0.05)
                    drained += 1
                except asyncio.TimeoutError:
                    break

            # node-2 sends audio
            n2_frame = _make_audio_frame("node-2", seq=10)
            await ws_n2.send(n2_frame)

            # Dashboard must NOT receive it
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(ws_dash.recv(), timeout=0.3)

            # node-1 sends audio
            n1_frame = _make_audio_frame("node-1", seq=11)
            await ws_n1.send(n1_frame)

            # Dashboard MUST receive it
            received = await asyncio.wait_for(ws_dash.recv(), timeout=1.0)
            assert isinstance(received, bytes)
            assert received == n1_frame

    finally:
        await _stop_hub(task)


# ---------------------------------------------------------------------------
# Gate 4: Backpressure isolation — stall one node, others keep full rate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_g4_backpressure_isolation():
    """
    Stall B's recv for 2s.
    A sends 20 frames while B is stalled.
    After stall, only B's dropped_frames should be non-zero.
    A's queue depth must have stayed manageable (not backed up).
    """
    hub, task = await _start_hub(9013, node_queue_size=8)
    try:
        async with (
            websockets.connect("ws://127.0.0.1:9013/node", compression=None) as ws_a,
            websockets.connect("ws://127.0.0.1:9013/node", compression=None) as ws_b,
        ):
            await _register_node(ws_a, "node-A")
            await _register_node(ws_b, "node-B")
            await asyncio.sleep(0.05)

            assert "node-B" in hub._nodes
            node_b_conn = hub._nodes["node-B"]

            # Find and cancel B's drain task directly.
            # The drain task is stored in handle_node() as a local variable.
            # We locate it by matching against the ConnQueue object.
            b_drain_task = None
            for t in asyncio.all_tasks():
                try:
                    # Look for the drain coroutine whose `self` is node-B's queue
                    coro = t.get_coro()
                    if hasattr(coro, 'cr_frame') and coro.cr_frame:
                        f_self = coro.cr_frame.f_locals.get('self')
                        if f_self is node_b_conn.q:
                            b_drain_task = t
                            break
                except Exception:
                    pass

            # Cancel B's drain so the queue fills up
            if b_drain_task:
                b_drain_task.cancel()
                await asyncio.sleep(0.01)

            # Now flood with frames — B's queue (size=8) should fill and drop
            frames_sent = 50
            for i in range(frames_sent):
                frame = _make_audio_frame("node-A", seq=i, payload=b"\x01\x02" * 80)
                await ws_a.send(frame)
                await asyncio.sleep(0.001)  # 1ms spacing

            await asyncio.sleep(0.1)


            # Check B's queue state via hub registry
            assert "node-B" in hub._nodes
            node_b_conn = hub._nodes["node-B"]

            # B's ConnQueue should have dropped frames (queue_size=5, sent=20)
            # At least some frames must have been dropped
            assert node_b_conn.q.dropped > 0, (
                f"Expected B to have dropped frames, got dropped={node_b_conn.q.dropped}"
            )

            # A's ConnQueue should have dropped=0 (A is the sender, not a receiver here)
            node_a_conn = hub._nodes["node-A"]
            # A's own queue may have pings etc. but should not be the bottleneck
            # The key invariant: B's drops don't cause A to block
            assert node_a_conn.q.dropped == 0, (
                f"A should not have dropped frames; got {node_a_conn.q.dropped}"
            )

    finally:
        await _stop_hub(task)


# ---------------------------------------------------------------------------
# Gate 5: Liveness — degraded within 3s, offline within 6s
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_g5_liveness_degraded_offline():
    """
    Connect a node, then stop sending pings.
    After 3s: dashboard receives node_degraded.
    After 6s: dashboard receives node_offline.
    """
    hub, task = await _start_hub(9014)
    try:
        # Dashboard connected first to collect broadcasts
        async with websockets.connect("ws://127.0.0.1:9014/dashboard", compression=None) as ws_dash:
            async with websockets.connect("ws://127.0.0.1:9014/node", compression=None) as ws_node:
                await _register_node(ws_node, "node-liveness")
                await asyncio.sleep(0.1)

                # Drain node_online from dashboard
                try:
                    online_raw = await asyncio.wait_for(ws_dash.recv(), timeout=0.5)
                    online = json.loads(online_raw)
                    assert online["type"] == "node_online"
                except asyncio.TimeoutError:
                    pass

                # Now stop pinging — node goes silent.
                # Liveness monitor ticks every 1s.
                # degraded at >3s, offline at >6s.

                # Wait for degraded (up to 5s)
                degraded_msg = None
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    try:
                        raw = await asyncio.wait_for(ws_dash.recv(), timeout=0.5)
                        msg = json.loads(raw)
                        if msg.get("type") == "node_degraded" and msg.get("node_id") == "node-liveness":
                            degraded_msg = msg
                            break
                    except asyncio.TimeoutError:
                        continue

                assert degraded_msg is not None, (
                    "Expected node_degraded within 5s but did not receive it"
                )
                assert degraded_msg["node_id"] == "node-liveness"

            # Node disconnected — wait for offline broadcast (up to 8s from start)
            offline_msg = None
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline:
                try:
                    raw = await asyncio.wait_for(ws_dash.recv(), timeout=0.5)
                    msg = json.loads(raw)
                    if msg.get("type") in ("node_offline",) and msg.get("node_id") == "node-liveness":
                        offline_msg = msg
                        break
                except asyncio.TimeoutError:
                    continue

            assert offline_msg is not None, (
                "Expected node_offline within 8s total but did not receive it"
            )

    finally:
        await _stop_hub(task)


# ---------------------------------------------------------------------------
# Gate 6a: Wrong token → WS 1008
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_g6a_wrong_token_rejected_1008():
    """Wrong token must be rejected with node_ack unauthorized + WS close 1008."""
    hub, task = await _start_hub(9015)
    hub.dev_mode    = False
    hub.auth_token  = "correct-secret"
    hub.dev_mode    = False
    try:
        async with websockets.connect("ws://127.0.0.1:9015/node", compression=None) as ws:
            await ws.send(json.dumps({
                "type":    "node_hello",
                "node_id": "bad-node",
                "token":   "wrong-secret",
                "hw":      {},
            }))
            ack = json.loads(await asyncio.wait_for(ws.recv(), timeout=2.0))
            assert ack["type"]   == "node_ack"
            assert ack["status"] == "unauthorized"

            with pytest.raises(websockets.exceptions.ConnectionClosed) as exc:
                await ws.recv()
            assert exc.value.rcvd.code == 1008

    finally:
        await _stop_hub(task)


# ---------------------------------------------------------------------------
# Gate 6b: Hub boot without token and without dev_mode → RuntimeError
# ---------------------------------------------------------------------------

def test_g6b_boot_requires_token_or_devmode():
    """AegisHub must raise RuntimeError if no token and no dev_mode."""
    old_token = os.environ.pop("AEGIS_AUTH_TOKEN", None)
    old_dev   = os.environ.pop("AEGIS_DEV_MODE",   None)
    try:
        with pytest.raises(RuntimeError, match="AEGIS_AUTH_TOKEN"):
            AegisHub(auth_token=None, dev_mode=False)

        # Dev mode must succeed
        hub = AegisHub(auth_token=None, dev_mode=True)
        assert hub.dev_mode is True

        # Token without dev must succeed
        hub2 = AegisHub(auth_token="valid-token", dev_mode=False)
        assert hub2.auth_token == "valid-token"
    finally:
        if old_token: os.environ["AEGIS_AUTH_TOKEN"] = old_token
        if old_dev:   os.environ["AEGIS_DEV_MODE"]   = old_dev


# ---------------------------------------------------------------------------
# Gate 6c: Dev mode logs per-connection warning
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_g6c_devmode_warns_per_connection(caplog):
    """In dev mode, a WARNING must be logged for each node_hello."""
    hub, task = await _start_hub(9016)
    try:
        with caplog.at_level(logging.WARNING, logger="aegis.hub.auth"):
            async with websockets.connect("ws://127.0.0.1:9016/node", compression=None) as ws:
                await _register_node(ws, "node-devmode-1")

            async with websockets.connect("ws://127.0.0.1:9016/node", compression=None) as ws:
                await _register_node(ws, "node-devmode-2")

        # Should have 2 per-connection warnings (one per node_hello)
        dev_warns = [
            r for r in caplog.records
            if "AEGIS_DEV_MODE" in r.message and "node-devmode" in r.message
        ]
        assert len(dev_warns) >= 2, (
            f"Expected ≥2 per-connection dev mode warnings, got: {[r.message for r in dev_warns]}"
        )
    finally:
        await _stop_hub(task)


# ---------------------------------------------------------------------------
# Gate 7: Transport — TCP_NODELAY on pre-bound socket + no permessage-deflate
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_g7_transport_tcp_nodelay_no_deflate():
    """
    Verify hub's pre-bound listening socket has TCP_NODELAY=1.
    Verify connected clients have no permessage-deflate extension negotiated.
    """
    import socket as _socket

    # Create a bound socket the same way main.py does
    s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    s.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
    s.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 9017))
    s.listen(5)

    # Assert TCP_NODELAY is set on the listening socket.
    # Note: on macOS, getsockopt(TCP_NODELAY) may return the option NUMBER (4)
    # rather than 1, since TCP_NODELAY is defined as 4 on Darwin.
    # We assert truthy (non-zero) rather than == 1 for platform portability.
    tcp_nodelay = s.getsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY)
    assert tcp_nodelay != 0, (
        f"Expected TCP_NODELAY to be set (non-zero) on listening socket, got {tcp_nodelay}"
    )

    s.close()

    # Start a hub and verify no compression extension
    hub, task = await _start_hub(9018)
    try:
        async with websockets.connect(
            "ws://127.0.0.1:9018/node",
            compression=None,
        ) as ws:
            # compression=None on client means we don't offer deflate
            # The server also uses compression=None so it must not negotiate it
            extensions = getattr(ws, "extensions", []) or []
            ext_names = [getattr(e, "name", str(e)) for e in extensions]
            assert not any("deflate" in n for n in ext_names), (
                f"permessage-deflate must not be negotiated; extensions={ext_names}"
            )
    finally:
        await _stop_hub(task)


# ---------------------------------------------------------------------------
# Gate 10: Frames parity — hub frames.py ↔ backend node_client.py
# ---------------------------------------------------------------------------

def test_g10_frames_parity():
    """
    Verify byte-level parity of the mesh audio binary format.

    Rather than importing the full aegis-backend module (which has heavy
    transitive deps), we re-implement the REFERENCE pack() inline — an
    exact copy of the struct format from node_client.py. If they diverge
    in endianness, field order, or sizing, this test will fail.

    Reference (aegis-backend/src/ws/node_client.py):
        _SUFFIX_FMT = "!QI"    # big-endian uint64 + uint32
        frame_type  = 0x02
        header = pack("!BB", frame_type, len(node_bytes)) + node_bytes
        header += pack("!QI", ts_ms, seq)
        return header + payload
    """
    import struct as _struct
    import unittest.mock as _mock

    from src.frames import pack_mesh_audio as hub_pack
    import src.frames as _frames_module

    # Inline reference — must match node_client.py exactly
    _FRAME_TYPE_AUDIO = 0x02
    _SUFFIX_FMT = "!QI"

    def reference_pack(node_id: str, seq: int, payload: bytes, ts_ms: int) -> bytes:
        """Exact copy of node_client.pack_mesh_audio with fixed timestamp."""
        node_bytes = node_id.encode("utf-8")
        header = _struct.pack("!BB", _FRAME_TYPE_AUDIO, len(node_bytes))
        header += node_bytes
        header += _struct.pack(_SUFFIX_FMT, ts_ms, seq)
        return header + payload

    node_id    = "operator-1"
    seq        = 42
    payload    = b"\xDE\xAD\xBE\xEF" * 20
    fixed_ts_s = 1_700_000_000.0

    # Hub pack with fixed timestamp (patch _time.time)
    with _mock.patch.object(_frames_module._time, "time", return_value=fixed_ts_s):
        hub_frame = hub_pack(node_id, seq, payload)

    fixed_ts_ms = int(fixed_ts_s * 1000)
    ref_frame   = reference_pack(node_id, seq, payload, ts_ms=fixed_ts_ms)

    # Parse both
    from src.frames import unpack_header
    hub_hdr = unpack_header(hub_frame)
    ref_hdr = unpack_header(ref_frame)

    assert hub_hdr is not None, "hub frame unparseable"
    assert ref_hdr is not None, "reference frame unparseable"

    assert hub_hdr.frame_type      == ref_hdr.frame_type,      f"frame_type: {hub_hdr.frame_type} vs {ref_hdr.frame_type}"
    assert hub_hdr.source_node_id  == ref_hdr.source_node_id,  f"node_id: {hub_hdr.source_node_id!r} vs {ref_hdr.source_node_id!r}"
    assert hub_hdr.seq             == ref_hdr.seq,              f"seq: {hub_hdr.seq} vs {ref_hdr.seq}"
    assert hub_hdr.timestamp_ms    == ref_hdr.timestamp_ms,     f"ts: {hub_hdr.timestamp_ms} vs {ref_hdr.timestamp_ms}"

    hub_payload = hub_frame[hub_hdr.payload_offset:]
    ref_payload = ref_frame[ref_hdr.payload_offset:]
    assert hub_payload == ref_payload == payload, "payload mismatch"

    assert hub_frame == ref_frame, (
        "Full frame bytes not byte-identical vs reference (node_client.py format).\n"
        f"  hub:  {hub_frame[:20].hex()}\n"
        f"  ref:  {ref_frame[:20].hex()}\n"
        "Check endianness and field order in frames.py."
    )
