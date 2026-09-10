"""
AEGIS Hub Server
Maintains a dynamic registry of AEGIS Pi nodes.
Serves dashboards, fanning out node events and telemetry.
"""

import asyncio
import json
import logging
import struct
import time
from typing import Dict, Set

import websockets
from websockets.asyncio.server import ServerConnection, serve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] HUB: %(message)s")
logger = logging.getLogger("aegis.hub")

class NodeRegistry:
    def __init__(self):
        # Maps node_id -> dict of properties (hw status, telemetry, etc.)
        self.nodes: Dict[str, dict] = {}
        # Maps node_id -> ServerConnection
        self.node_sockets: Dict[str, ServerConnection] = {}

    def add_node(self, node_id: str, ws: ServerConnection, hw_info: dict):
        self.nodes[node_id] = {
            "node_id": node_id,
            "hw": hw_info,
            "connected_at": time.time()
        }
        self.node_sockets[node_id] = ws
        logger.info(f"Node registered: {node_id}")

    def remove_node(self, node_id: str):
        self.nodes.pop(node_id, None)
        self.node_sockets.pop(node_id, None)
        logger.info(f"Node removed: {node_id}")


class AegisHub:
    def __init__(self, host="0.0.0.0", port=8001):
        self.host = host
        self.port = port
        self.registry = NodeRegistry()
        self.dashboards: Set[ServerConnection] = set()

    async def broadcast_to_dashboards(self, msg: str | bytes):
        """Concurrent fan-out to all connected dashboards."""
        if not self.dashboards:
            return
        
        # P1: Concurrent fan-out
        coros = []
        for ws in self.dashboards:
            coros.append(self._send_safe(ws, msg))
        await asyncio.gather(*coros)

    async def _send_safe(self, ws: ServerConnection, msg: str | bytes):
        try:
            await ws.send(msg)
        except Exception:
            pass # Socket likely closed

    async def handle_dashboard(self, ws: ServerConnection):
        """Dashboard client connection."""
        self.dashboards.add(ws)
        logger.info(f"Dashboard connected (total: {len(self.dashboards)})")
        try:
            # On connect, immediately send node_online for all existing nodes
            for node_id, info in self.registry.nodes.items():
                msg = {
                    "type": "node_online",
                    "node_id": node_id,
                    "hw": info["hw"],
                    "ts": int(time.time() * 1000)
                }
                await ws.send(json.dumps(msg))

            async for message in ws:
                # Dashboard sends control messages (anc_set, hardware_mute) targeted at specific nodes
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                        target_id = data.get("node_id") or data.get("clientId") # back-compat for a moment
                        if target_id and target_id in self.registry.node_sockets:
                            # Forward the control message verbatim to the target node
                            await self.registry.node_sockets[target_id].send(message)
                        elif data.get("type") == "ping":
                            # Respond to ping immediately (for RTT)
                            await ws.send(json.dumps({"type": "pong", "seq": data.get("seq"), "rtt_ms": 0, "ts": int(time.time()*1000)}))
                    except json.JSONDecodeError:
                        pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.dashboards.discard(ws)
            logger.info(f"Dashboard disconnected (remaining: {len(self.dashboards)})")

    async def handle_node(self, ws: ServerConnection):
        """Pi Node connection."""
        node_id = None
        logger.info("New node connection attempting registration...")
        try:
            async for message in ws:
                if isinstance(message, str):
                    # Control plane (JSON)
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type")

                        if msg_type == "node_hello":
                            node_id = data.get("node_id")
                            if not node_id:
                                logger.warning("node_hello missing node_id, dropping")
                                break
                            
                            self.registry.add_node(node_id, ws, data.get("hw", {}))
                            
                            # Broadcast to dashboards
                            await self.broadcast_to_dashboards(json.dumps({
                                "type": "node_online",
                                "node_id": node_id,
                                "hw": data.get("hw", {}),
                                "ts": int(time.time() * 1000)
                            }))
                            
                            # Ack the node
                            await ws.send(json.dumps({"type": "node_ack", "status": "ok"}))
                        
                        elif msg_type == "ping":
                            await ws.send(json.dumps({"type": "pong", "seq": data.get("seq"), "rtt_ms": 0, "ts": int(time.time()*1000)}))
                            
                        elif node_id:
                            # Forward all other JSON (telemetry, anc_state) to dashboards
                            # We blindly forward, P1 rule: skip schema validation on hub hot-path
                            await self.broadcast_to_dashboards(message)

                    except json.JSONDecodeError:
                        pass
                else:
                    # Binary plane (Audio/FFT frames)
                    # Relayed blindly to all dashboards
                    if node_id:
                        await self.broadcast_to_dashboards(message)

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if node_id:
                self.registry.remove_node(node_id)
                await self.broadcast_to_dashboards(json.dumps({
                    "type": "node_offline",
                    "node_id": node_id,
                    "ts": int(time.time() * 1000)
                }))

    async def handler(self, ws: ServerConnection):
        path = ws.request.path
        if path == "/dashboard":
            await self.handle_dashboard(ws)
        elif path == "/node":
            await self.handle_node(ws)
        else:
            await ws.close(1002, "Invalid path. Use /dashboard or /node")

    async def serve_forever(self):
        logger.info(f"Starting Hub WS on {self.host}:{self.port}")
        async with serve(self.handler, self.host, self.port):
            await asyncio.Future()  # run forever

if __name__ == "__main__":
    hub = AegisHub()
    asyncio.run(hub.serve_forever())
