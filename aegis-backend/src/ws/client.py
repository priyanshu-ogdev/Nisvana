"""
ws/client.py — WebSocket Client for Pi Edge Node (Rev 3 SOTA)
============================================================
Connects to the AEGIS Hub using TCP_NODELAY, compression=None,
drop-oldest bounded queues, active 1Hz heartbeat ping for RTT measurement,
and strict token authentication.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import struct
import time
from typing import Any, Optional

import websockets
from websockets.exceptions import ConnectionClosed

from .protocol import (
    HwStatus, Telemetry, AncState, HardwareMute, AncSet, Ping,
    HandshakeAck, LinkStatus, Pong,
    FRAME_TYPE_FFT, FRAME_TYPE_AUDIO, FRAME_TYPE_HEALTH
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Low-Latency Drop-Oldest Queue (P1/P2)
# ==============================================================================

class DropOldestQueue:
    """
    Bounded queue that drops the oldest element upon saturation.
    Preserves freshness for high-fps audio and spectral data under link congestion.
    """
    def __init__(self, maxsize: int = 256):
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
# Edge Node Client
# ==============================================================================

class AegisClient:
    def __init__(
        self,
        hub_url: Optional[str] = None,
        node_id: Optional[str] = None,
        auth_token: Optional[str] = None,
        dev_mode: Optional[bool] = None,
    ):
        self.hub_url = hub_url or os.getenv("AEGIS_HUB_URL", "ws://127.0.0.1:8001/node")
        self.node_id = node_id or os.getenv("AEGIS_NODE_ID", "pi-demo")
        self.auth_token = auth_token if auth_token is not None else os.getenv("AEGIS_AUTH_TOKEN")
        self.dev_mode = dev_mode if dev_mode is not None else (
            os.getenv("AEGIS_DEV_MODE", "0").strip().lower() in ("1", "true", "yes")
        )

        self.ws: Optional[websockets.client.ClientConnection] = None
        self._send_queue = DropOldestQueue(maxsize=256)
        self._hw_status: Optional[HwStatus] = None
        self.connected = False
        self.is_secure = False
        self.reconnect_delay = 1.0

        # Link metrics
        self.node_rtt_ms: Optional[float] = None
        self.link_state: str = "offline"
        self._ping_seq: int = 0

        # Mutable state updated by incoming control messages
        self.anc_active = False
        self.muted = {"primary_mic": False, "reference_mic": False, "throat_mic": False, "headset_output": False}

    def _clear_send_queue(self) -> None:
        self._send_queue.clear()

    def _set_tcp_nodelay(self, ws: websockets.client.ClientConnection) -> None:
        """Disables Nagle's algorithm for low-latency transmission."""
        try:
            sock = ws.transport.get_extra_info("socket")
            if sock:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception as e:
            logger.debug(f"Could not set TCP_NODELAY: {e}")

    def get_stats(self) -> dict:
        """Telemetry statistics query for link health and queue depth."""
        return {
            "node_rtt_ms": self.node_rtt_ms,
            "dropped_frames": self._send_queue.dropped_count,
            "queue_depth": self._send_queue.qsize(),
            "link_state": self.link_state,
        }

    async def connect_forever(self) -> None:
        """Main connection loop with exponential backoff and heartbeat."""
        while True:
            try:
                logger.info(f"Connecting to Hub at {self.hub_url} (node_id={self.node_id})...")
                # compression=None avoids permessage-deflate overhead on small binary frames
                async with websockets.connect(self.hub_url, compression=None) as ws:
                    self.ws = ws
                    self.connected = True
                    self.link_state = "online"
                    self.reconnect_delay = 1.0
                    self._set_tcp_nodelay(ws)
                    logger.info("Connected to Hub.")

                    # Send node_hello with authentication token
                    hello_msg = {
                        "type": "node_hello",
                        "node_id": self.node_id,
                        "token": self.auth_token,
                        "hw": self._hw_status.model_dump() if self._hw_status else {},
                    }
                    await ws.send(json.dumps(hello_msg))

                    # Run receive, drain, and heartbeat tasks concurrently
                    recv_task = asyncio.create_task(self._recv_loop())
                    send_task = asyncio.create_task(self._drain_send_queue())
                    heartbeat_task = asyncio.create_task(self._heartbeat_loop())

                    done, pending = await asyncio.wait(
                        [recv_task, send_task, heartbeat_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    for task in pending:
                        task.cancel()

            except (ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                logger.warning(f"Hub connection dropped: {e}. Reconnecting in {self.reconnect_delay:.1f}s...")
            except Exception as e:
                logger.error(f"Unexpected WS client error: {e}. Reconnecting in {self.reconnect_delay:.1f}s...")
            finally:
                self.connected = False
                self.link_state = "offline"
                self.ws = None
                self._send_queue.clear()
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 1.5, 10.0)

    async def _heartbeat_loop(self) -> None:
        """Active 1Hz ping loop measuring real Node <-> Hub network RTT."""
        while self.connected and self.ws:
            await asyncio.sleep(1.0)
            self._ping_seq += 1
            ping_payload = {
                "type": "node_ping",
                "node_id": self.node_id,
                "seq": self._ping_seq,
                "timestamp": int(time.time() * 1000),
            }
            self.enqueue(json.dumps(ping_payload))

    async def _recv_loop(self) -> None:
        """Handle incoming messages from Hub (control commands, pongs)."""
        if not self.ws:
            return
        async for message in self.ws:
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "node_ack":
                        status = data.get("status")
                        if status == "unauthorized":
                            logger.error(f"Node authentication failed: {data.get('reason')}")
                        else:
                            logger.info("Node registration acknowledged by Hub.")

                    elif msg_type == "node_pong":
                        sent_ts = data.get("timestamp", 0)
                        if sent_ts > 0:
                            self.node_rtt_ms = round(time.time() * 1000 - sent_ts, 2)
                            self.link_state = "online"

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
                            "ts": int(time.time() * 1000),
                        }
                        self.enqueue(json.dumps(pong_payload))

                except json.JSONDecodeError:
                    pass

    # ------------------------------------------------------------------
    # Send helpers (thread-safe, drop-oldest upon queue saturation)
    # ------------------------------------------------------------------

    def enqueue(self, msg: str | bytes) -> None:
        """Thread-safe enqueue with drop-oldest behavior."""
        if not self.connected:
            return
        self._send_queue.put_nowait(msg)

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
    # Data Push Helpers
    # ------------------------------------------------------------------

    def push_hw_status(self, status: HwStatus) -> None:
        self._hw_status = status
        if self.connected:
            msg = {
                "type": "node_hello",
                "node_id": self.node_id,
                "token": self.auth_token,
                "hw": status.model_dump(),
            }
            self.enqueue(json.dumps(msg))

    def push_fft_binary(self, raw_bins: list[int], enhanced_bins: list[int]) -> None:
        """
        Binary framing for FFT stream (Frame Type 1).
        Header format:
          1 byte:  frame_type (1)
          1 byte:  node_id length (L)
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
                *enhanced_bins,
            )
            self.enqueue(binary_frame)
        except Exception as e:
            logger.error(f"Failed to pack FFT binary frame: {e}")

    def push_audio_binary(self, pcm_bytes: bytes) -> None:
        """
        Binary framing for raw/enhanced PCM audio stream (Frame Type 2).
        Header: 1B frame_type (2) + 1B node_id_len (L) + L bytes node_id + 8B ts_ms (uint64) + PCM payload.
        """
        node_id_bytes = self.node_id.encode("utf-8")
        ts = int(time.time() * 1000)
        header_fmt = f"<BB{len(node_id_bytes)}sQ"
        try:
            header = struct.pack(header_fmt, FRAME_TYPE_AUDIO, len(node_id_bytes), node_id_bytes, ts)
            self.enqueue(header + pcm_bytes)
        except Exception as e:
            logger.error(f"Failed to pack audio binary frame: {e}")

    def push_health_binary(self, battery_pct: int, temp_c: float, cpu_pct: float) -> None:
        """
        Binary framing for hardware health stream (Frame Type 3).
        Header + 1B battery_pct + 4B temp_c (float32) + 4B cpu_pct (float32).
        """
        node_id_bytes = self.node_id.encode("utf-8")
        ts = int(time.time() * 1000)
        fmt = f"<BB{len(node_id_bytes)}sQEff"
        try:
            binary_frame = struct.pack(
                fmt,
                FRAME_TYPE_HEALTH,
                len(node_id_bytes),
                node_id_bytes,
                ts,
                battery_pct,
                temp_c,
                cpu_pct,
            )
            self.enqueue(binary_frame)
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
