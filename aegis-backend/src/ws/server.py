"""
ws/server.py — WebSocket Server
================================
Accepts up to 2 simultaneous client connections.
Rejects extras with handshake_ack(denied).
Dispatches incoming messages to the session state machine.
"""

from __future__ import annotations
import asyncio
import json
import logging
import time
from typing import Dict, Optional

import websockets
from websockets.asyncio.server import ServerConnection

from .protocol import (
    HandshakeAck, HwStatus, LinkStatus,
    parse_incoming, HandshakeInit, HardwareMute, AncSet, Ping, Pong
)
from .session import ClientSession, LinkState
from .heartbeat import HeartbeatManager

logger = logging.getLogger(__name__)

MAX_SLOTS = 2
SLOT_IDS = ["person-1", "person-2"]


class AegisServer:
    """
    Main WebSocket server. Orchestrator calls self.broadcast_*() to push data.
    Audio threads MUST use send_queue, not await directly (never block audio).
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.host = host
        self.port = port

        # Active sessions keyed by client_id
        self._sessions: Dict[str, ClientSession] = {}

        # Asyncio queue so audio threads can enqueue WS messages without awaiting
        self._send_queue: asyncio.Queue = asyncio.Queue(maxsize=2048)

        self._hw_status: Optional[HwStatus] = None
        self._heartbeat = HeartbeatManager(
            get_sessions=lambda: list(self._sessions.values()),
            broadcast_fn=self._send_to_client,
        )

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    async def serve_forever(self) -> None:
        """Start WS server and heartbeat loop concurrently."""
        logger.info(f"AEGIS WS server listening on ws://{self.host}:{self.port}/ws")
        async with websockets.serve(self._handle_connection, self.host, self.port):
            await asyncio.gather(
                asyncio.get_event_loop().create_future(),  # run forever
                self._heartbeat.start(),
                self._drain_send_queue(),
            )

    # ------------------------------------------------------------------
    # Connection handler
    # ------------------------------------------------------------------

    async def _handle_connection(self, ws: ServerConnection) -> None:
        """Called by websockets library for each new TCP connection."""

        # Reject if all slots are taken
        if len(self._sessions) >= MAX_SLOTS:
            await ws.send(HandshakeAck(
                clientId="unknown",
                status="denied",
                reason="slot_full",
            ).model_dump_json())
            await ws.close()
            logger.warning("Rejected connection: all slots full")
            return

        # Assign a temporary placeholder; real clientId comes from handshake_init
        temp_id = f"_pending_{int(time.time()*1000)}"
        session = ClientSession(
            client_id=temp_id,
            websocket=ws,
            on_state_change=self._on_session_state_change,
        )
        logger.info(f"New connection accepted (temp_id={temp_id})")

        # Send initial hw_status if available
        if self._hw_status:
            try:
                await ws.send(self._hw_status.model_dump_json())
            except Exception:
                pass

        try:
            async for raw in ws:
                await self._dispatch(session, raw)
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"[{session.client_id}] Connection closed: {e}")
        finally:
            await self._remove_session(session)

    async def _dispatch(self, session: ClientSession, raw: str) -> None:
        """Parse and route an incoming message."""
        try:
            msg = parse_incoming(raw)
        except Exception as e:
            logger.warning(f"[{session.client_id}] Bad message: {e} — raw={raw[:120]}")
            return

        if isinstance(msg, HandshakeInit):
            await self._handle_handshake_init(session, msg)
        elif isinstance(msg, HardwareMute):
            await self._handle_hardware_mute(session, msg)
        elif isinstance(msg, AncSet):
            session.anc_active = msg.enabled
            logger.info(f"[{session.client_id}] ANC set to {msg.enabled}")
        elif isinstance(msg, Ping):
            pong = self._heartbeat.handle_pong(session, {"seq": msg.seq, "timestamp": msg.timestamp})
            try:
                await session.websocket.send(pong.model_dump_json())
            except Exception:
                pass

    async def _handle_handshake_init(self, session: ClientSession, msg: HandshakeInit) -> None:
        client_id = msg.clientId

        # If this slot already has an active session, deny
        if client_id in self._sessions and self._sessions[client_id].state != LinkState.DROPPED:
            ack = HandshakeAck(clientId=client_id, status="denied", reason="slot_taken")
            await session.websocket.send(ack.model_dump_json())
            return

        # Re-key the session with the real client_id
        old_id = session.client_id
        if old_id in self._sessions:
            del self._sessions[old_id]
        session.client_id = client_id
        self._sessions[client_id] = session

        # Transition state machine
        await session.begin_handshake()

        # 1s simulated Pi processing delay (real backend does ALSA probe here)
        await asyncio.sleep(1.0)

        ack = HandshakeAck(clientId=client_id, status="ok")
        await session.websocket.send(ack.model_dump_json())
        await session.confirm_secure()

        link = LinkStatus(state="link_secure", clientId=client_id)
        await session.websocket.send(link.model_dump_json())

        # If both slots are secure, broadcast streaming
        if all(s.is_secure for s in self._sessions.values()) and len(self._sessions) == 2:
            for s in self._sessions.values():
                try:
                    stream_msg = LinkStatus(state="streaming", clientId=s.client_id)
                    await s.websocket.send(stream_msg.model_dump_json())
                except Exception:
                    pass

    async def _handle_hardware_mute(self, session: ClientSession, msg: HardwareMute) -> None:
        if msg.clientId != session.client_id:
            logger.warning(f"clientId mismatch: msg={msg.clientId} session={session.client_id}")
        target = msg.target
        session.muted[target] = msg.state
        logger.info(f"[{session.client_id}] hardware_mute {target}={msg.state}")
        # Orchestrator will pick up muted state in its next frame

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    async def _remove_session(self, session: ClientSession) -> None:
        cid = session.client_id
        self._sessions.pop(cid, None)
        if session.state != LinkState.DROPPED:
            await session.drop("connection_closed")

    async def _on_session_state_change(self, session: ClientSession) -> None:
        """Hook called by session on state transition."""
        if session.state == LinkState.DROPPED:
            ls = LinkStatus(state="dropped", clientId=session.client_id)
            try:
                await session.websocket.send(ls.model_dump_json())
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Send helpers (called by orchestrator — thread-safe via queue)
    # ------------------------------------------------------------------

    def enqueue(self, msg_json: str, client_id: Optional[str] = None) -> None:
        """
        Thread-safe: put a message on the send queue.
        Audio threads use this instead of awaiting directly.
        client_id=None → broadcast to all secure sessions.
        """
        try:
            self._send_queue.put_nowait((msg_json, client_id))
        except asyncio.QueueFull:
            logger.warning("WS send queue full — dropping frame")

    async def _drain_send_queue(self) -> None:
        """Drain enqueued messages onto their target websockets."""
        while True:
            msg_json, client_id = await self._send_queue.get()
            if client_id:
                await self._send_to_client(json.loads(msg_json), client_id)
            else:
                for session in list(self._sessions.values()):
                    if session.is_secure:
                        try:
                            await session.websocket.send(msg_json)
                        except Exception:
                            pass

    async def _send_to_client(self, msg_dict: dict, client_id: str) -> None:
        session = self._sessions.get(client_id)
        if session:
            try:
                await session.websocket.send(json.dumps(msg_dict))
            except Exception as e:
                logger.debug(f"[{client_id}] Send failed: {e}")

    # ------------------------------------------------------------------
    # Called by orchestrator to push data
    # ------------------------------------------------------------------

    def push_hw_status(self, status: HwStatus) -> None:
        """Store latest hw_status and broadcast to all connections."""
        self._hw_status = status
        msg_json = status.model_dump_json()
        for session in list(self._sessions.values()):
            self.enqueue(msg_json, session.client_id)

    def push_fft(self, fft_msg_json: str, client_id: str) -> None:
        """Push fft_stream for a specific secure client."""
        session = self._sessions.get(client_id)
        if session and session.should_receive_fft():
            self.enqueue(fft_msg_json, client_id)

    def push_anc_state(self, anc_json: str, client_id: str) -> None:
        session = self._sessions.get(client_id)
        if session and session.is_secure:
            self.enqueue(anc_json, client_id)

    def push_telemetry(self, telemetry_json: str) -> None:
        self.enqueue(telemetry_json)

    def get_session(self, client_id: str) -> Optional[ClientSession]:
        return self._sessions.get(client_id)

    def get_all_sessions(self):
        return list(self._sessions.values())
