#!/usr/bin/env python3
"""scripts/mock_hub.py — Mesh Hub Mock for Gate 2 Verification.

Acts as a minimal AEGIS Hub:
  - Accepts node WebSocket connections
  - Reads binary mesh audio frames from any connected node
  - Immediately echoes each frame back to the *same* node (loopback)
    OR broadcasts to all other nodes (multi-node mode)
  - Logs frame rate, byte rate, and node ID to stdout

Gate 2 test:
  Terminal 1: python scripts/mock_hub.py --port 8001
  Terminal 2: NODE_ID=operator-1 HUB_WS_URL=ws://127.0.0.1:8001/node python -m src.main

Gate 5 test (two distinct nodes):
  Terminal 1: python scripts/mock_hub.py --port 8001 --broadcast
  Terminal 2: NODE_ID=node-A HUB_WS_URL=ws://127.0.0.1:8001/node python -m src.main
  Terminal 3: NODE_ID=node-B HUB_WS_URL=ws://127.0.0.1:8001/node python -m src.main
"""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import struct
import sys
import time
from typing import Dict, Optional

import websockets
from websockets.server import WebSocketServerProtocol

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [HUB] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("mock_hub")

FRAME_TYPE_MESH_AUDIO = 0x02


def parse_node_id(data: bytes) -> Optional[str]:
    """Extract source node_id from a binary mesh audio frame header."""
    if len(data) < 2:
        return None
    if data[0] != FRAME_TYPE_MESH_AUDIO:
        return None
    node_id_len = data[1]
    if len(data) < 2 + node_id_len:
        return None
    try:
        return data[2: 2 + node_id_len].decode("utf-8")
    except Exception:
        return None


# Global registry of connected nodes: node_id → websocket
_nodes: Dict[str, WebSocketServerProtocol] = {}
_stats: Dict[str, dict] = {}  # node_id → {frames, bytes, last_log_t}

_broadcast_mode = False


async def handle_node(ws: WebSocketServerProtocol) -> None:
    """Handle one connected node."""
    node_id: Optional[str] = None

    try:
        async for message in ws:
            # JSON control message
            if isinstance(message, str):
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "node_hello":
                        node_id = data.get("node_id", f"unknown-{id(ws)}")
                        caps = data.get("capabilities", [])
                        _nodes[node_id] = ws
                        _stats[node_id] = {"frames": 0, "bytes": 0, "last_log_t": time.time()}
                        logger.info(f"  ✅ Node connected: '{node_id}'  caps={caps}")
                        logger.info(f"     Active nodes: {list(_nodes.keys())}")

                        # Send node_ack
                        ack = json.dumps({"type": "node_ack", "node_id": node_id, "status": "ok"})
                        await ws.send(ack)

                    elif msg_type == "node_ping":
                        # Reply with pong
                        pong = json.dumps({
                            "type": "node_pong",
                            "node_id": node_id,
                            "timestamp": data.get("timestamp"),
                            "seq": data.get("seq"),
                        })
                        await ws.send(pong)

                except json.JSONDecodeError:
                    pass

            # Binary audio frame
            elif isinstance(message, bytes):
                src = parse_node_id(message) or node_id or "unknown"
                frame_len = len(message)

                # Update stats
                if src in _stats:
                    _stats[src]["frames"] += 1
                    _stats[src]["bytes"] += frame_len
                    now = time.time()
                    if now - _stats[src]["last_log_t"] >= 5.0:
                        fps = _stats[src]["frames"] / (now - _stats[src]["last_log_t"] + 1e-6)
                        kbps = (_stats[src]["bytes"] * 8 / 1000) / (now - _stats[src]["last_log_t"] + 1e-6)
                        logger.info(
                            f"  📦 Node '{src}': "
                            f"{_stats[src]['frames']} frames  "
                            f"~{fps:.0f} fps  ~{kbps:.0f} kbps  "
                            f"frame_size={frame_len}B"
                        )
                        _stats[src]["frames"] = 0
                        _stats[src]["bytes"] = 0
                        _stats[src]["last_log_t"] = now

                if _broadcast_mode:
                    # Broadcast to all other connected nodes
                    targets = [
                        nws for nid, nws in _nodes.items()
                        if nid != src and nws.open
                    ]
                    if targets:
                        await asyncio.gather(*(t.send(message) for t in targets))
                else:
                    # Loopback: echo back to sender (Gate 2 — node hears own voice)
                    try:
                        await ws.send(message)
                    except Exception as e:
                        logger.debug(f"Echo send failed: {e}")

    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        logger.error(f"Node handler error: {e}")
    finally:
        if node_id and node_id in _nodes:
            del _nodes[node_id]
            if node_id in _stats:
                del _stats[node_id]
            logger.info(f"  ❌ Node disconnected: '{node_id}'  active: {list(_nodes.keys())}")


async def run_server(host: str, port: int, broadcast: bool) -> None:
    global _broadcast_mode
    _broadcast_mode = broadcast
    mode = "broadcast" if broadcast else "loopback-echo"

    logger.info(f"Mock Hub starting on ws://{host}:{port}  mode={mode}")
    logger.info("Waiting for nodes to connect...")
    logger.info("  Gate 2: connect a single node — it will hear its own audio echoed back")
    logger.info("  Gate 5: connect two nodes with distinct NODE_IDs\n")

    async with websockets.serve(handle_node, host, port, compression=None):
        await asyncio.Future()  # run forever


def main() -> None:
    parser = argparse.ArgumentParser(description="AEGIS Mock Hub — Phase 1 Gate 2/5 Verifier")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8001, help="Bind port (default: 8001)")
    parser.add_argument(
        "--broadcast",
        action="store_true",
        help="Broadcast mode: relay audio from each node to all others (Gate 5). "
             "Default: loopback-echo mode (Gate 2).",
    )
    args = parser.parse_args()
    asyncio.run(run_server(args.host, args.port, args.broadcast))


if __name__ == "__main__":
    main()
