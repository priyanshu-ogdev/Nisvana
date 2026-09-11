"""
ws/client.py — WebSocket Client for Pi Node
===========================================
Connects to the AEGIS Hub using low-latency transport options.
Implements:
  - Required token authentication
  - Drop-oldest queue backpressure
  - TCP_NODELAY and compression=None
  - Active 1Hz heartbeat measuring real node_rtt_ms & degraded liveness detection
  - Multi-frame binary streaming (FFT, Audio PCM, Health)
"""

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
    HwStatus, Telemetry, AncState, HardwareMute, AncSet, Ping,
    HandshakeAck, LinkStatus, Pong,
    FRAME_TYPE_FFT, FRAME_TYPE_AUDIO, FRAME_TYPE_HEALTH
)

logger = logging.getLogger(__name__)


def _enable_tcp_nodelay(ws) -> None:
    """Set TCP_NODELAY on client socket to prevent Nagle packet coalescing delay."""
    try:
        sock = ws.transport.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except Exception as e:
        logger.debug(f"Could not set TCP_NODELAY on client socket: {e}")


class AegisClient:
    def __init__(
        self,
        hub_url: str = "ws://127.0.0.1:8001/node",
        node_id: str = "pi-demo",
        auth_token: Optional[str] = None,
        queue_size: int = 128,
        reconnect_delay: float = 1.0
    ):
        self.hub_url = hub_url
        self.node_id = node_id
        self.auth_token = auth_token or os.getenv("AEGIS_AUTH_TOKEN")
        self.queue_size = queue_size
        
        self.ws: Optional[websockets.client.ClientConnection] = None
        # Drop-oldest bounded queue
        self._send_queue: asyncio.Queue = asyncio.Queue(maxsize=self.queue_size)
        self._hw_status: Optional[HwStatus] = None
        self.connected = False
        self.is_secure = False
        self.initial_reconnect_delay = reconnect_delay
        self.reconnect_delay = self.initial_reconnect_delay

        # Heartbeat & link telemetry
        self.node_rtt_ms: Optional[float] = None
        self.dropped_frames: int = 0
        self.link_quality: str = "healthy"
        self.last_pong_time: float = time.time()

        # Mutable state updated by incoming control messages
        self.anc_active = False
        self.muted = {"primary_mic": False, "reference_mic": False, "throat_mic": False, "headset_output": False}

    def _clear_send_queue(self) -> None:
        while not self._send_queue.empty():
            try:
                self._send_queue.get_nowait()
            except (asyncio.QueueEmpty, ValueError):
                break

    def get_stats(self) -> dict:
        """Return socket RTT, backpressure dropped frames, and link quality for telemetry."""
        return {
            "node_rtt_ms": self.node_rtt_ms,
            "dropped_frames": self.dropped_frames,
            "queue_depth": self._send_queue.qsize(),
            "link_quality": self.link_quality,
        }

    async def connect(self) -> None:
        """Connect to Hub."""
        await self.connect_forever()

    async def connect_forever(self) -> None:
        """Main connection loop with exponential backoff and active heartbeat."""
        while True:
            try:
                ssl_ctx = None
                if self.hub_url.startswith("wss://"):
                    ssl_ctx = ssl.create_default_context()
                    if os.getenv("AEGIS_INSECURE_SSL", "0").lower() in ("1", "true", "yes"):
                        ssl_ctx.check_hostname = False
                        ssl_ctx.verify_mode = ssl.CERT_NONE

                logger.info(f"Connecting to Hub at {self.hub_url} (compression=None)...")
                async with websockets.connect(
                    self.hub_url,
                    compression=None,
                    ssl=ssl_ctx
                ) as ws:
                    _enable_tcp_nodelay(ws)
                    self.ws = ws
                    self.connected = True
                    self.reconnect_delay = self.initial_reconnect_delay
                    self.last_pong_time = time.time()
                    self.link_quality = "healthy"
                    logger.info("Connected to Hub.")

                    # Send node_hello with token authentication
                    hello_msg = {
                        "type": "node_hello",
                        "node_id": self.node_id,
                        "token": self.auth_token,
                        "hw": self._hw_status.model_dump() if self._hw_status else {}
                    }
                    await ws.send(json.dumps(hello_msg))

                    # Start communication tasks
                    recv_task = asyncio.create_task(self._recv_loop())
                    send_task = asyncio.create_task(self._drain_send_queue())
                    ping_task = asyncio.create_task(self._ping_loop())

                    done, pending = await asyncio.wait(
                        [recv_task, send_task, ping_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    for task in pending:
                        task.cancel()

            except (ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                logger.warning(f"Connection lost/failed: {e}. Retrying in {self.reconnect_delay}s...")
            except Exception as e:
                logger.error(f"Unexpected WS client error: {e}. Retrying in {self.reconnect_delay}s...")
            finally:
                self.connected = False
                self.ws = None
                self._clear_send_queue()
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 1.5, 10.0)

    async def _ping_loop(self) -> None:
        """Active 1Hz heartbeat ping measuring socket RTT and detecting link degradation."""
        seq = 0
        while self.connected and self.ws:
            try:
                seq += 1
                now = time.time()
                # Check for lost heartbeat (no pong in > 3 seconds)
                if now - self.last_pong_time > 3.0:
                    if self.link_quality != "degraded":
                        logger.warning(f"Link to Hub degraded: no pong received in {now - self.last_pong_time:.1f}s")
                    self.link_quality = "degraded"
                else:
                    self.link_quality = "healthy"

                ping_msg = {
                    "type": "node_ping",
                    "node_id": self.node_id,
                    "timestamp": int(now * 1000),
                    "seq": seq
                }
                self.enqueue(json.dumps(ping_msg))
            except Exception as e:
                logger.debug(f"Ping loop exception: {e}")
            await asyncio.sleep(1.0)

    async def _recv_loop(self):
        """Handle incoming messages from Hub (relayed from dashboard)."""
        if not self.ws:
            return
        async for message in self.ws:
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "node_pong":
                        ts = data.get("timestamp")
                        if ts:
                            self.node_rtt_ms = max(0.0, round(time.time() * 1000 - ts, 2))
                            self.last_pong_time = time.time()
                            self.link_quality = "healthy"

                    elif msg_type == "node_ack":
                        status = data.get("status")
                        if status == "unauthorized":
                            logger.critical(
                                "Hub rejected node connection: UNAUTHORIZED. Verify AEGIS_AUTH_TOKEN."
                            )

                    elif msg_type == "handshake_init":
                        self.is_secure = True
                        logger.info(f"Handshake accepted for node: {self.node_id}")
                        ack = HandshakeAck(clientId=self.node_id, status="ok")
                        self.enqueue(ack.model_dump_json())
                        link = LinkStatus(clientId=self.node_id, state="streaming")
                        self.enqueue(link.model_dump_json())

                    elif msg_type == "hardware_mute":
                        target = data.get("target")
                        state = data.get("state")
                        if target in self.muted:
                            self.muted[target] = state
                            logger.info(f"Hardware mute updated: {target}={state}")

                    elif msg_type == "anc_set":
                        self.anc_active = data.get("enabled", False)
                        logger.info(f"ANC set: {self.anc_active}")

                    elif msg_type == "ping":
                        pong_payload = {
                            "type": "pong",
                            "seq": data.get("seq", 0),
                            "timestamp": data.get("timestamp"),
                            "rtt_ms": 0.0,
                            "ts": int(time.time() * 1000)
                        }
                        self.enqueue(json.dumps(pong_payload))

                except json.JSONDecodeError:
                    pass

    # ------------------------------------------------------------------
    # Send helpers (drop-oldest queue backpressure)
    # ------------------------------------------------------------------

    def enqueue(self, msg: str | bytes) -> None:
        """
        Put a message on the send queue.
        Enforces drop-oldest policy: when queue is full, evicts stale frame so
        freshest audio/FFT real-time data is prioritized.
        """
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
        """Drain enqueued messages onto the websocket."""
        while self.connected and self.ws:
            msg = await self._send_queue.get()
            try:
                await self.ws.send(msg)
            except Exception as e:
                logger.debug(f"Send failed: {e}")
                break

    # ------------------------------------------------------------------
    # Data Pushing Helpers
    # ------------------------------------------------------------------

    def push_hw_status(self, status: HwStatus) -> None:
        self._hw_status = status
        if self.connected:
            msg = {
                "type": "node_hello",
                "node_id": self.node_id,
                "token": self.auth_token,
                "hw": status.model_dump()
            }
            self.enqueue(json.dumps(msg))

    def push_fft_binary(self, raw_bins: list[int], enhanced_bins: list[int]) -> None:
        """
        Binary wire framing for FFT stream (Frame Type 1).
        Header format:
          1 byte: frame_type (1)
          1 byte: node_id length (L)
          L bytes: node_id (utf-8)
          8 bytes: timestamp_ms (uint64)
          64 bytes: raw_bins (uint8)
          64 bytes: enhanced_bins (uint8)
        """
        node_id_bytes = self.node_id.encode("utf-8")
        ts = int(time.time() * 1000)
        fmt = f"<BB{len(node_id_bytes)}sQ64B64B"
        try:
            binary_frame = struct.pack(
                fmt,
                FRAME_TYPE_FFT,
                len(node_id_bytes),
                node_id_bytes,
                ts,
                *raw_bins,
                *enhanced_bins
            )
            self.enqueue(binary_frame)
        except Exception as e:
            logger.error(f"Failed to pack FFT binary frame: {e}")

    def push_audio_binary(self, audio_pcm: bytes) -> None:
        """
        Binary wire framing for raw PCM audio passthrough (Frame Type 2).
        Header format:
          1 byte: frame_type (2)
          1 byte: node_id length (L)
          L bytes: node_id (utf-8)
          8 bytes: timestamp_ms (uint64)
          Remaining: raw 16-bit PCM bytes
        """
        node_id_bytes = self.node_id.encode("utf-8")
        ts = int(time.time() * 1000)
        fmt = f"<BB{len(node_id_bytes)}sQ"
        try:
            header = struct.pack(fmt, FRAME_TYPE_AUDIO, len(node_id_bytes), node_id_bytes, ts)
            self.enqueue(header + audio_pcm)
        except Exception as e:
            logger.error(f"Failed to pack audio binary frame: {e}")

    def push_health_binary(self, health_payload: bytes) -> None:
        """
        Binary wire framing for high-rate HW health telemetry (Frame Type 3).
        Header format:
          1 byte: frame_type (3)
          1 byte: node_id length (L)
          L bytes: node_id (utf-8)
          8 bytes: timestamp_ms (uint64)
          Remaining: health payload bytes
        """
        node_id_bytes = self.node_id.encode("utf-8")
        ts = int(time.time() * 1000)
        fmt = f"<BB{len(node_id_bytes)}sQ"
        try:
            header = struct.pack(fmt, FRAME_TYPE_HEALTH, len(node_id_bytes), node_id_bytes, ts)
            self.enqueue(header + health_payload)
        except Exception as e:
            logger.error(f"Failed to pack health binary frame: {e}")

    def push_anc_state(self, anc_msg: AncState) -> None:
        msg = anc_msg.model_dump()
        msg["node_id"] = self.node_id
        self.enqueue(json.dumps(msg))

    def push_telemetry(self, tel_msg: Telemetry) -> None:
        msg = tel_msg.model_dump()
        msg["node_id"] = self.node_id
        self.enqueue(json.dumps(msg))
