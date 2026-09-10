"""
ws/client.py — WebSocket Client for Pi Node
===========================================
Replaces the legacy server.py. Connects to the AEGIS Hub.
Handles automatic reconnection, binary frame pushing, and telemetry.
"""

import asyncio
import json
import logging
import struct
import time
from typing import Optional

import websockets
from websockets.exceptions import ConnectionClosed

from .protocol import (
    HwStatus, Telemetry, AncState, HardwareMute, AncSet, Ping
)

logger = logging.getLogger(__name__)

class AegisClient:
    def __init__(self, hub_url: str = "ws://127.0.0.1:8001/node", node_id: str = "pi-demo"):
        self.hub_url = hub_url
        self.node_id = node_id
        
        self.ws: Optional[websockets.client.ClientConnection] = None
        self._send_queue: asyncio.Queue = asyncio.Queue(maxsize=2048)
        self._hw_status: Optional[HwStatus] = None
        self.connected = False
        self.reconnect_delay = 1.0

        # Mutable state updated by incoming control messages
        self.anc_active = False
        self.muted = {"primary_mic": False, "reference_mic": False, "throat_mic": False, "headset_output": False}

    async def connect_forever(self) -> None:
        """Main connection loop with exponential backoff."""
        while True:
            try:
                logger.info(f"Connecting to Hub at {self.hub_url}...")
                async with websockets.connect(self.hub_url) as ws:
                    self.ws = ws
                    self.connected = True
                    self.reconnect_delay = 1.0
                    logger.info("Connected to Hub.")

                    # Send node_hello
                    hello_msg = {
                        "type": "node_hello",
                        "node_id": self.node_id,
                        "hw": self._hw_status.model_dump() if self._hw_status else {}
                    }
                    await ws.send(json.dumps(hello_msg))

                    # Start tasks
                    recv_task = asyncio.create_task(self._recv_loop())
                    send_task = asyncio.create_task(self._drain_send_queue())

                    done, pending = await asyncio.wait(
                        [recv_task, send_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    for task in pending:
                        task.cancel()

            except (ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                self.connected = False
                self.ws = None
                logger.warning(f"Connection lost/failed: {e}. Retrying in {self.reconnect_delay}s...")
                await asyncio.sleep(self.reconnect_delay)
                self.reconnect_delay = min(self.reconnect_delay * 1.5, 10.0)

    async def _recv_loop(self):
        """Handle incoming messages from Hub (sent by dashboard)."""
        if not self.ws:
            return
        async for message in self.ws:
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    if msg_type == "hardware_mute":
                        target = data.get("target")
                        state = data.get("state")
                        if target in self.muted:
                            self.muted[target] = state
                            logger.info(f"Hardware mute updated: {target}={state}")
                    elif msg_type == "anc_set":
                        self.anc_active = data.get("enabled", False)
                        logger.info(f"ANC set: {self.anc_active}")
                except json.JSONDecodeError:
                    pass

    # ------------------------------------------------------------------
    # Send helpers (called by orchestrator — thread-safe via queue)
    # ------------------------------------------------------------------

    def enqueue(self, msg: str | bytes) -> None:
        """Thread-safe: put a message on the send queue."""
        if not self.connected:
            return
        try:
            self._send_queue.put_nowait(msg)
        except asyncio.QueueFull:
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
    # Called by orchestrator to push data
    # ------------------------------------------------------------------

    def push_hw_status(self, status: HwStatus) -> None:
        self._hw_status = status
        # If we are connected, we can send a hot-update (optional, but good)
        if self.connected:
            msg = {
                "type": "node_hello",
                "node_id": self.node_id,
                "hw": status.model_dump()
            }
            self.enqueue(json.dumps(msg))

    def push_fft_binary(self, raw_bins: list[int], enhanced_bins: list[int]) -> None:
        """P1: Binary framing for FFT stream."""
        # Frame format:
        # 1 byte: frame_type (1 = FFT)
        # 1 byte: node_id length (L)
        # L bytes: node_id (utf-8)
        # 8 bytes: timestamp (uint64)
        # 64 bytes: raw_bins (uint8)
        # 64 bytes: enhanced_bins (uint8)
        
        node_id_bytes = self.node_id.encode('utf-8')
        ts = int(time.time() * 1000)
        
        # Packing
        fmt = f"<BB{len(node_id_bytes)}sQ64B64B"
        try:
            binary_frame = struct.pack(
                fmt,
                1, # frame_type = 1
                len(node_id_bytes),
                node_id_bytes,
                ts,
                *raw_bins,
                *enhanced_bins
            )
            self.enqueue(binary_frame)
        except Exception as e:
            logger.error(f"Failed to pack FFT binary frame: {e}")

    def push_anc_state(self, anc_msg: AncState) -> None:
        msg = anc_msg.model_dump()
        msg["node_id"] = self.node_id
        self.enqueue(json.dumps(msg))

    def push_telemetry(self, tel_msg: Telemetry) -> None:
        msg = tel_msg.model_dump()
        msg["node_id"] = self.node_id
        self.enqueue(json.dumps(msg))
