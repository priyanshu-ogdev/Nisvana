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
            "connected_at": time.time(),
            "secure": False
        }
        self.node_sockets[node_id] = ws
        logger.info(f"Node registered: {node_id}")

    def set_secure(self, node_id: str, secure: bool = True):
        if node_id in self.nodes:
            self.nodes[node_id]["secure"] = secure

    def remove_node(self, node_id: str, ws: ServerConnection | None = None) -> bool:
        if ws is not None and self.node_sockets.get(node_id) != ws:
            logger.info(f"Ignoring stale disconnect for replaced node: {node_id}")
            return False
        self.nodes.pop(node_id, None)
        self.node_sockets.pop(node_id, None)
        logger.info(f"Node removed: {node_id}")
        return True


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
        
        # P1: Concurrent fan-out (copy set to avoid mutation during iteration)
        coros = [self._send_safe(ws, msg) for ws in list(self.dashboards)]
        await asyncio.gather(*coros)

    async def _send_safe(self, ws: ServerConnection, msg: str | bytes):
        try:
            await ws.send(msg)
        except Exception:
            # Prune closed/errored dashboard socket
            self.dashboards.discard(ws)

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
                    "secure": info.get("secure", False),
                    "ts": int(time.time() * 1000)
                }
                await ws.send(json.dumps(msg))

            async for message in ws:
                # Dashboard sends control messages (anc_set, hardware_mute) targeted at specific nodes
                if isinstance(message, str):
                    try:
                        data = json.loads(message)
                        target_id = data.get("node_id") or data.get("clientId") # back-compat
                        if target_id and target_id in self.registry.node_sockets:
                            # Forward the control message verbatim to the target node
                            try:
                                await self.registry.node_sockets[target_id].send(message)
                            except Exception as e:
                                logger.warning(f"Failed to forward message to node {target_id}: {e}")
                        elif data.get("type") == "ping":
                            # Respond to ping echoing original timestamp for frontend RTT calculation
                            pong = {
                                "type": "pong",
                                "seq": data.get("seq"),
                                "timestamp": data.get("timestamp"),
                                "rtt_ms": 0,
                                "ts": int(time.time() * 1000)
                            }
                            await ws.send(json.dumps(pong))
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
                                "secure": False,
                                "ts": int(time.time() * 1000)
                            }))
                            
                            # Ack the node
                            await ws.send(json.dumps({"type": "node_ack", "status": "ok"}))
                        
                        elif msg_type == "ping":
                            pong = {
                                "type": "pong",
                                "seq": data.get("seq"),
                                "timestamp": data.get("timestamp"),
                                "rtt_ms": 0,
                                "ts": int(time.time() * 1000)
                            }
                            await ws.send(json.dumps(pong))
                            
                        elif node_id:
                            if msg_type == "handshake_ack":
                                self.registry.set_secure(node_id, True)
                            # Forward all other JSON (telemetry, anc_state, handshake_ack) to dashboards
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
                removed = self.registry.remove_node(node_id, ws)
                if removed:
                    await self.broadcast_to_dashboards(json.dumps({
                        "type": "node_offline",
                        "node_id": node_id,
                        "ts": int(time.time() * 1000)
                    }))

    async def handler(self, ws: ServerConnection):
        # Normalize path: strip query parameters and trailing slashes
        clean_path = ws.request.path.split("?")[0].rstrip("/")
        if clean_path == "/dashboard":
            await self.handle_dashboard(ws)
        elif clean_path == "/node":
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
