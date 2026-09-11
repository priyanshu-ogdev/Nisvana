"""src/healthcheck.py — WS-based Docker HEALTHCHECK for aegis-node container.

Opens a WebSocket to HUB_WS_URL, sends a node_ping, expects node_pong within
2 seconds. Exit 0 = healthy (Hub reachable + responding). Exit 1 = unhealthy.

This measures ACTUAL mesh reachability, not just "is the process alive."
A node that has lost its Hub connection will correctly report unhealthy.

Usage (in Dockerfile):
    HEALTHCHECK CMD python -m src.healthcheck

Or standalone:
    HUB_WS_URL=ws://192.168.1.100:8001/node python -m src.healthcheck
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time

logging.basicConfig(level=logging.WARNING)


async def _check() -> bool:
    """Return True if Hub responds to node_ping within 2 seconds."""
    try:
        import websockets
    except ImportError:
        print("healthcheck: websockets not installed", file=sys.stderr)
        return False

    hub_url  = os.getenv("HUB_WS_URL", "ws://127.0.0.1:8001/node")
    node_id  = os.getenv("NODE_ID", "healthcheck")
    timeout  = float(os.getenv("HEALTHCHECK_TIMEOUT_S", "2.0"))

    try:
        async with websockets.connect(hub_url, compression=None, open_timeout=timeout) as ws:
            # Send node_hello (dev mode — no token needed for healthcheck)
            hello = json.dumps({
                "type":    "node_hello",
                "node_id": node_id,
                "token":   os.getenv("AEGIS_AUTH_TOKEN"),
                "hw":      {},
            })
            await ws.send(hello)

            # Expect node_ack (ok or dev mode skips it)
            try:
                ack_raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                ack = json.loads(ack_raw) if isinstance(ack_raw, str) else {}
                if ack.get("status") == "unauthorized":
                    print("healthcheck: unauthorized (check AEGIS_AUTH_TOKEN)", file=sys.stderr)
                    return False
            except asyncio.TimeoutError:
                print("healthcheck: no ack from Hub", file=sys.stderr)
                return False

            # Send node_ping
            ts = int(time.time() * 1000)
            await ws.send(json.dumps({
                "type":      "node_ping",
                "node_id":   node_id,
                "timestamp": ts,
                "seq":       1,
            }))

            # Expect node_pong
            try:
                pong_raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                pong = json.loads(pong_raw) if isinstance(pong_raw, str) else {}
                if pong.get("type") == "node_pong":
                    rtt = int(time.time() * 1000) - ts
                    print(f"healthcheck: OK (rtt={rtt}ms)", file=sys.stderr)
                    return True
                else:
                    print(f"healthcheck: unexpected response: {pong}", file=sys.stderr)
                    return False
            except asyncio.TimeoutError:
                print(f"healthcheck: pong timeout after {timeout}s", file=sys.stderr)
                return False

    except (OSError, ConnectionRefusedError) as e:
        print(f"healthcheck: connection failed: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"healthcheck: error: {e}", file=sys.stderr)
        return False


def main() -> None:
    ok = asyncio.run(_check())
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
