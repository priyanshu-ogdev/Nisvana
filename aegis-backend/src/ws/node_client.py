"""ws/node_client.py — Mesh Node WebSocket Client.

Connects this node to the central AEGIS Hub using a persistent,
reconnecting WebSocket connection with:
  - Binary mesh audio frame packing/unpacking (Phase 1 wire protocol)
  - Inbound binary audio frames routed to an asyncio.Queue for downstream_task
  - JSON control messages (node_hello, mesh_route, node_pong) handled inline
  - Active 1Hz ping/pong with RTT measurement and link degradation detection
  - TCP_NODELAY for minimal Nagle-induced latency
  - Drop-oldest backpressure on the outbound queue

Binary Audio Frame Wire Layout (big-endian, per Phase 1 spec):
  Byte 0:           0x02 (FRAME_TYPE_MESH_AUDIO)
  Byte 1:           L = len(node_id bytes)
  Bytes 2..2+L:     node_id (UTF-8)
  Bytes 2+L..10+L:  timestamp_ms (uint64)
  Bytes 10+L..14+L: seq (uint32)
  Remaining:        audio payload (PCM int16 or Opus)

Usage:
    client = MeshNodeClient(node_id="operator-1", hub_url="ws://hub:8001/node")
    asyncio.gather(
        client.connect_forever(),
        upstream_task(client),
        downstream_task(client),
    )

    # Upstream: send encoded audio
    await client.send_audio_frame(seq=42, payload=encoded_bytes)

    # Downstream: receive from Hub
    frame = await client.recv_audio_frame()  # MeshAudioFrame
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import ssl
import struct
import time
from typing import Optional

import websockets
from websockets.exceptions import ConnectionClosed

from .protocol import (
    HwStatus, Telemetry, AncState,
    FRAME_TYPE_FFT, FRAME_TYPE_AUDIO, FRAME_TYPE_HEALTH, FRAME_TYPE_MESH_AUDIO,
    MeshAudioFrame,
)

logger = logging.getLogger(__name__)

# Binary header format (big-endian) for the fixed-length suffix after node_id
#   Q = uint64 timestamp_ms
#   I = uint32 seq
_SUFFIX_FMT = "!QI"
_SUFFIX_SIZE = struct.calcsize(_SUFFIX_FMT)  # 12 bytes


def _enable_tcp_nodelay(ws) -> None:
    """Set TCP_NODELAY to prevent Nagle packet coalescing."""
    try:
        sock = ws.transport.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception as e:
        logger.debug(f"TCP_NODELAY: {e}")


def pack_mesh_audio(node_id: str, seq: int, payload: bytes) -> bytes:
    """
    Pack a binary mesh audio frame.

    Args:
        node_id:  This node's identifier (UTF-8 string).
        seq:      Monotonically increasing sequence number (uint32).
        payload:  Encoded audio bytes (PCM int16 or Opus).

    Returns:
        Binary frame bytes ready for websocket send.
    """
    node_bytes = node_id.encode("utf-8")
    ts = int(time.time() * 1000)
    header = struct.pack("!BB", FRAME_TYPE_MESH_AUDIO, len(node_bytes))
    header += node_bytes
    header += struct.pack(_SUFFIX_FMT, ts, seq)
    return header + payload


def unpack_mesh_audio(data: bytes) -> Optional[MeshAudioFrame]:
    """
    Unpack a binary mesh audio frame received from the Hub.

    Returns None if the data is malformed or not a mesh audio frame.
    """
    if len(data) < 2:
        return None

    frame_type = data[0]
    if frame_type != FRAME_TYPE_MESH_AUDIO:
        return None

    node_id_len = data[1]
    offset = 2

    if len(data) < offset + node_id_len + _SUFFIX_SIZE:
        logger.debug("MeshNodeClient: binary frame too short to parse")
        return None

    try:
        source_node_id = data[offset: offset + node_id_len].decode("utf-8")
        offset += node_id_len
        ts_ms, seq = struct.unpack_from(_SUFFIX_FMT, data, offset)
        offset += _SUFFIX_SIZE
        payload = data[offset:]
        return MeshAudioFrame(
            source_node_id=source_node_id,
            timestamp_ms=ts_ms,
            seq=seq,
            payload=payload,
        )
    except Exception as e:
        logger.debug(f"MeshNodeClient: unpack error: {e}")
        return None


class MeshNodeClient:
    """
    WebSocket client for a mesh node.

    Maintains a persistent connection to the Hub with automatic reconnection.
    Provides:
      - send_audio_frame(): enqueue outbound binary mesh audio frame
      - recv_audio_frame(): await next inbound binary audio frame (from Hub)
      - get_stats(): RTT, dropped frames, link quality for telemetry
    """

    def __init__(
        self,
        node_id: str = "pi-demo",
        hub_url: str = "ws://127.0.0.1:8001/node",
        auth_token: Optional[str] = None,
        capabilities: Optional[list] = None,
        queue_size: int = 128,
        inbound_queue_size: int = 50,
        reconnect_delay: float = 1.0,
    ) -> None:
        self.node_id = node_id
        self.hub_url = hub_url
        self.auth_token = auth_token or os.getenv("AEGIS_AUTH_TOKEN")
        self.capabilities = capabilities or ["mic", "speaker"]
        self.queue_size = queue_size

        self.ws: Optional[websockets.client.ClientConnection] = None
        self.connected = False

        # Outbound queue (telemetry + encoded audio)
        self._send_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        # Inbound queue (binary audio frames from Hub → downstream_task)
        self._inbound_queue: asyncio.Queue[MeshAudioFrame] = asyncio.Queue(
            maxsize=inbound_queue_size
        )

        self._hw_status: Optional[HwStatus] = None

        # Reconnect backoff
        self.initial_reconnect_delay = reconnect_delay
        self.reconnect_delay = reconnect_delay

        # Heartbeat / link quality
        self.node_rtt_ms: Optional[float] = None
        self.dropped_frames: int = 0
        self.link_quality: str = "healthy"
        self.last_pong_time: float = time.time()

        # Mute state (updated by Hub control messages)
        self.muted: dict = {
            "primary_mic": False,
            "reference_mic": False,
            "headset_output": False,
        }
        self.anc_active: bool = False

        # Sequence counter for outbound frames
        self._seq: int = 0

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect_forever(self) -> None:
        """Main reconnect loop. Runs until the process exits."""
        while True:
            try:
                ssl_ctx = None
                if self.hub_url.startswith("wss://"):
                    ssl_ctx = ssl.create_default_context()
                    if os.getenv("AEGIS_INSECURE_SSL", "0").lower() in ("1", "true", "yes"):
                        ssl_ctx.check_hostname = False
                        ssl_ctx.verify_mode = ssl.CERT_NONE

                logger.info(f"MeshNodeClient: connecting to {self.hub_url}...")
                async with websockets.connect(
                    self.hub_url,
                    compression=None,
                    ssl=ssl_ctx,
                ) as ws:
                    _enable_tcp_nodelay(ws)
                    self.ws = ws
                    self.connected = True
                    self.reconnect_delay = self.initial_reconnect_delay
                    self.last_pong_time = time.time()
                    self.link_quality = "healthy"
                    logger.info(f"MeshNodeClient: connected as '{self.node_id}'")

                    # Send node_hello with capabilities
                    await self._send_node_hello(ws)

                    recv_task = asyncio.create_task(self._recv_loop())
                    send_task = asyncio.create_task(self._drain_send_queue())
                    ping_task = asyncio.create_task(self._ping_loop())

                    done, pending = await asyncio.wait(
                        [recv_task, send_task, ping_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()

            except (ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                logger.warning(
                    f"MeshNodeClient: connection lost ({e}). "
                    f"Retrying in {self.reconnect_delay:.1f}s..."
                )
            except Exception as e:
                logger.error(f"MeshNodeClient: unexpected error: {e}")
            finally:
                self.connected = False
                self.ws = None
                self._clear_send_queue()
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 1.5, 10.0)

    async def _send_node_hello(self, ws) -> None:
        """Send node_hello handshake to Hub."""
        hello = {
            "type": "node_hello",
            "node_id": self.node_id,
            "token": self.auth_token,
            "capabilities": self.capabilities,
            "hw": self._hw_status.model_dump() if self._hw_status else {},
        }
        await ws.send(json.dumps(hello))
        logger.info(f"MeshNodeClient: node_hello sent (caps={self.capabilities})")

    # ------------------------------------------------------------------
    # Receive loop
    # ------------------------------------------------------------------

    async def _recv_loop(self) -> None:
        """Dispatch inbound messages: binary frames → inbound queue; JSON → control handler."""
        if not self.ws:
            return
        async for message in self.ws:
            if isinstance(message, bytes):
                self._handle_binary(message)
            elif isinstance(message, str):
                self._handle_json(message)

    def _handle_binary(self, data: bytes) -> None:
        """Route inbound binary frame. Mesh audio frames go to the inbound queue."""
        frame = unpack_mesh_audio(data)
        if frame is None:
            return  # Unknown binary frame type; ignore
        if frame.source_node_id == self.node_id:
            # Echo from our own audio (expected during mock hub testing)
            pass
        try:
            self._inbound_queue.put_nowait(frame)
        except asyncio.QueueFull:
            # Drop oldest inbound frame to make room
            try:
                self._inbound_queue.get_nowait()
                self.dropped_frames += 1
            except asyncio.QueueEmpty:
                pass
            try:
                self._inbound_queue.put_nowait(frame)
            except asyncio.QueueFull:
                pass

    def _handle_json(self, raw: str) -> None:
        """Handle JSON control messages from Hub."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        msg_type = data.get("type")

        if msg_type == "node_pong":
            ts = data.get("timestamp")
            if ts:
                self.node_rtt_ms = max(0.0, round(time.time() * 1000 - ts, 2))
                self.last_pong_time = time.time()
                self.link_quality = "healthy"

        elif msg_type == "node_ack":
            if data.get("status") == "unauthorized":
                logger.critical(
                    "MeshNodeClient: Hub rejected connection — UNAUTHORIZED. "
                    "Verify AEGIS_AUTH_TOKEN."
                )

        elif msg_type == "mesh_route":
            source = data.get("source", "?")
            action = data.get("action", "subscribe")
            logger.info(f"MeshNodeClient: mesh_route {action} ← {source}")

        elif msg_type == "hardware_mute":
            target = data.get("target")
            state = data.get("state")
            if target in self.muted:
                self.muted[target] = bool(state)
                logger.info(f"MeshNodeClient: mute {target}={state}")

        elif msg_type == "anc_set":
            self.anc_active = bool(data.get("enabled", False))
            logger.info(f"MeshNodeClient: ANC set={self.anc_active}")

        elif msg_type == "ping":
            pong = {
                "type": "pong",
                "seq": data.get("seq", 0),
                "timestamp": data.get("timestamp"),
                "ts": int(time.time() * 1000),
            }
            self.enqueue(json.dumps(pong))

    # ------------------------------------------------------------------
    # Ping / heartbeat
    # ------------------------------------------------------------------

    async def _ping_loop(self) -> None:
        """1Hz active heartbeat with link degradation detection."""
        seq = 0
        while self.connected and self.ws:
            try:
                seq += 1
                now = time.time()
                if now - self.last_pong_time > 3.0:
                    if self.link_quality != "degraded":
                        logger.warning(
                            f"MeshNodeClient: link degraded — "
                            f"no pong for {now - self.last_pong_time:.1f}s"
                        )
                    self.link_quality = "degraded"
                else:
                    self.link_quality = "healthy"

                ping = {
                    "type": "node_ping",
                    "node_id": self.node_id,
                    "timestamp": int(now * 1000),
                    "seq": seq,
                }
                self.enqueue(json.dumps(ping))
            except Exception as e:
                logger.debug(f"MeshNodeClient ping error: {e}")
            await asyncio.sleep(1.0)

    # ------------------------------------------------------------------
    # Send queue
    # ------------------------------------------------------------------

    def enqueue(self, msg: str | bytes) -> None:
        """Enqueue outbound message with drop-oldest backpressure."""
        if not self.connected:
            return
        if self._send_queue.full():
            try:
                self._send_queue.get_nowait()
                self.dropped_frames += 1
            except (asyncio.QueueEmpty, ValueError):
                pass
        try:
            self._send_queue.put_nowait(msg)
        except (asyncio.QueueFull, RuntimeError):
            pass

    async def _drain_send_queue(self) -> None:
        """Drain outbound queue to websocket."""
        while self.connected and self.ws:
            msg = await self._send_queue.get()
            try:
                await self.ws.send(msg)
            except Exception as e:
                logger.debug(f"MeshNodeClient send failed: {e}")
                break

    def _clear_send_queue(self) -> None:
        while not self._send_queue.empty():
            try:
                self._send_queue.get_nowait()
            except (asyncio.QueueEmpty, ValueError):
                break

    # ------------------------------------------------------------------
    # Audio I/O API (called by orchestrator tasks)
    # ------------------------------------------------------------------

    async def send_audio_frame(self, payload: bytes) -> None:
        """
        Pack and enqueue a binary mesh audio frame for transmission to Hub.

        Args:
            payload: Encoded audio bytes (from AudioCodec.encode()).
        """
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        frame = pack_mesh_audio(self.node_id, self._seq, payload)
        self.enqueue(frame)

    async def recv_audio_frame(self) -> MeshAudioFrame:
        """
        Wait for and return the next inbound binary mesh audio frame from Hub.

        This is the async generator consumed by downstream_task.
        Blocks until a frame arrives or the connection is lost.
        """
        while True:
            try:
                frame = await asyncio.wait_for(
                    self._inbound_queue.get(), timeout=0.1
                )
                return frame
            except asyncio.TimeoutError:
                # No frame yet — yield control back to event loop
                await asyncio.sleep(0)

    # ------------------------------------------------------------------
    # Telemetry / state helpers
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        return {
            "node_rtt_ms": self.node_rtt_ms,
            "dropped_frames": self.dropped_frames,
            "queue_depth": self._send_queue.qsize(),
            "inbound_queue_depth": self._inbound_queue.qsize(),
            "link_quality": self.link_quality,
        }

    def push_hw_status(self, status: HwStatus) -> None:
        self._hw_status = status
        if self.connected:
            hello = {
                "type": "node_hello",
                "node_id": self.node_id,
                "token": self.auth_token,
                "capabilities": self.capabilities,
                "hw": status.model_dump(),
            }
            self.enqueue(json.dumps(hello))

    def push_fft_binary(self, raw_bins: list[int], enhanced_bins: list[int]) -> None:
        """Binary FFT frame (same format as legacy AegisClient)."""
        import struct as _struct
        node_bytes = self.node_id.encode("utf-8")
        ts = int(time.time() * 1000)
        fmt = f"<BB{len(node_bytes)}sQ64B64B"
        try:
            binary = _struct.pack(
                fmt,
                FRAME_TYPE_FFT,
                len(node_bytes),
                node_bytes,
                ts,
                *raw_bins,
                *enhanced_bins,
            )
            self.enqueue(binary)
        except Exception as e:
            logger.error(f"MeshNodeClient: FFT pack failed: {e}")

    def push_anc_state(self, anc_msg: AncState) -> None:
        msg = anc_msg.model_dump()
        msg["node_id"] = self.node_id
        self.enqueue(json.dumps(msg))

    def push_telemetry(self, tel_msg: Telemetry) -> None:
        msg = tel_msg.model_dump()
        msg["node_id"] = self.node_id
        self.enqueue(json.dumps(msg))
