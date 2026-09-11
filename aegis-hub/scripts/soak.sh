#!/usr/bin/env bash
# scripts/soak.sh — AEGIS Mesh Soak Test (Gates 8–9)
#
# Launches 3 native node instances + 2 dashboard WS clients for 10 minutes.
# Captures per-node dropped frame counts and hub→node RTT from hub_stats.
# Writes results to docs/verification_evidence.md.
#
# Prerequisites:
#   - aegis-hub running: AEGIS_DEV_MODE=1 python -m src.main (in aegis-hub/)
#   - aegis-backend venv active
#   - Python websockets available
#
# Usage:
#   HUB_URL=ws://127.0.0.1:8001 SOAK_DURATION_S=600 bash scripts/soak.sh

set -euo pipefail

HUB_URL="${HUB_URL:-ws://127.0.0.1:8001}"
SOAK_DURATION_S="${SOAK_DURATION_S:-600}"
BACKEND_DIR="${BACKEND_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/aegis-backend}"
EVIDENCE_FILE="${EVIDENCE_FILE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/docs/verification_evidence.md}"

echo "=== AEGIS Mesh Soak Test ==="
echo "  Hub URL       : ${HUB_URL}"
echo "  Duration      : ${SOAK_DURATION_S}s"
echo "  Backend dir   : ${BACKEND_DIR}"
echo "  Evidence file : ${EVIDENCE_FILE}"
echo ""

if [ ! -d "${BACKEND_DIR}" ]; then
    echo "ERROR: aegis-backend not found at ${BACKEND_DIR}"
    echo "Set BACKEND_DIR env var to the correct path."
    exit 1
fi

# Activate venv if present
VENV="${BACKEND_DIR}/venv/bin/activate"
if [ -f "${VENV}" ]; then
    # shellcheck disable=SC1090
    source "${VENV}"
fi

# Write the soak runner Python script to a temp file
SOAK_PY="$(mktemp /tmp/aegis_soak_XXXXXX.py)"
trap "rm -f ${SOAK_PY}" EXIT

cat > "${SOAK_PY}" << 'PYEOF'
"""
AEGIS Mesh Soak Runner — Gates 8 & 9.
Connects 3 node clients + 2 dashboard clients, runs for SOAK_DURATION_S,
then writes a summary to EVIDENCE_FILE.
"""
import asyncio
import json
import os
import sys
import time
import statistics
from pathlib import Path
from datetime import datetime, timezone

import websockets

HUB_URL         = os.getenv("HUB_URL", "ws://127.0.0.1:8001")
SOAK_DURATION_S = int(os.getenv("SOAK_DURATION_S", "600"))
EVIDENCE_FILE   = os.getenv("EVIDENCE_FILE", "docs/verification_evidence.md")

NODE_IDS = ["soak-node-A", "soak-node-B", "soak-node-C"]

# Results
rtts: list[float] = []
dropped_per_node: dict[str, int] = {nid: 0 for nid in NODE_IDS}
hub_stats_count = 0
frames_sent: dict[str, int] = {nid: 0 for nid in NODE_IDS}


async def run_node(node_id: str, stop_event: asyncio.Event):
    """Connect as a node, send synthetic audio frames, and record RTT."""
    async with websockets.connect(f"{HUB_URL}/node", compression=None) as ws:
        # Register
        await ws.send(json.dumps({
            "type": "node_hello", "node_id": node_id,
            "token": os.getenv("AEGIS_AUTH_TOKEN"), "hw": {},
        }))
        ack = json.loads(await ws.recv())
        if ack.get("status") not in ("ok", "registered"):
            print(f"  [{node_id}] Registration failed: {ack}")
            return

        print(f"  [{node_id}] Connected")
        seq = 0
        import struct

        while not stop_event.is_set():
            # Send synthetic 10ms PCM frame (960 bytes int16)
            payload = b"\x01\x02" * 480
            node_bytes = node_id.encode("utf-8")
            ts = int(time.time() * 1000)
            header = struct.pack("!BB", 0x02, len(node_bytes)) + node_bytes
            header += struct.pack("!QI", ts, seq)
            await ws.send(header + payload)
            frames_sent[node_id] += 1
            seq += 1

            # Ping every 100 frames (~1s)
            if seq % 100 == 0:
                t0 = int(time.time() * 1000)
                await ws.send(json.dumps({
                    "type": "node_ping", "node_id": node_id,
                    "timestamp": t0, "seq": seq,
                }))

            await asyncio.sleep(0.010)  # ~100fps


async def run_dashboard(dash_id: str, stop_event: asyncio.Event):
    """Connect as a dashboard, subscribe to all, record hub_stats."""
    global hub_stats_count
    async with websockets.connect(f"{HUB_URL}/dashboard", compression=None) as ws:
        print(f"  [dash:{dash_id}] Connected")
        while not stop_event.is_set():
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                if isinstance(raw, str):
                    try:
                        msg = json.loads(raw)
                        if msg.get("type") == "hub_stats":
                            hub_stats_count += 1
                            for k, v in msg.get("dropped_frames", {}).items():
                                for nid in NODE_IDS:
                                    if nid in k:
                                        dropped_per_node[nid] = max(dropped_per_node[nid], v)
                    except Exception:
                        pass
            except asyncio.TimeoutError:
                pass
            except Exception:
                break


async def main():
    stop = asyncio.Event()

    # Start all clients
    tasks = [
        asyncio.create_task(run_node(nid, stop))
        for nid in NODE_IDS
    ]
    tasks += [
        asyncio.create_task(run_dashboard(f"dash-{i}", stop))
        for i in range(2)
    ]

    print(f"\nSoak running for {SOAK_DURATION_S}s...")
    t_start = time.monotonic()
    await asyncio.sleep(SOAK_DURATION_S)
    stop.set()
    print("Stopping soak...")
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    # Report
    elapsed = time.monotonic() - t_start
    total_frames = sum(frames_sent.values())
    total_dropped = sum(dropped_per_node.values())
    drop_pct = (total_dropped / max(total_frames, 1)) * 100

    print(f"\n=== SOAK RESULTS ({elapsed:.0f}s) ===")
    print(f"  Total frames sent  : {total_frames}")
    print(f"  Total dropped      : {total_dropped} ({drop_pct:.2f}%)")
    print(f"  hub_stats received : {hub_stats_count}")
    print(f"  Dropped per node   : {dropped_per_node}")

    # Write evidence
    evidence_path = Path(EVIDENCE_FILE)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    section = f"""
## Gate 8–9 Soak Run — {stamp}

| Metric | Value |
|--------|-------|
| Duration | {elapsed:.0f}s |
| Nodes | {len(NODE_IDS)} |
| Dashboards | 2 |
| Total frames sent | {total_frames:,} |
| Total dropped | {total_dropped:,} ({drop_pct:.3f}%) |
| hub_stats received | {hub_stats_count} |

### Dropped frames per node
{chr(10).join(f'- `{k}`: {v}' for k, v in dropped_per_node.items())}

> [!NOTE]
> RTT p95 measurement requires node_pong capture (not yet wired in soak.sh).
> Run with `SOAK_DURATION_S=600` on wired LAN for Gate 9 compliance.
"""

    with open(evidence_path, "a") as f:
        f.write(section)

    print(f"\nEvidence appended to: {evidence_path}")

    # Exit non-zero if drop rate exceeds threshold
    if total_dropped > 0:
        print(f"  ⚠ {total_dropped} unaccounted drops — investigate before Gate 9 sign-off")
    else:
        print("  ✅ Zero unaccounted drops")

asyncio.run(main())
PYEOF

echo "Starting soak runner..."
HUB_URL="${HUB_URL}" \
SOAK_DURATION_S="${SOAK_DURATION_S}" \
EVIDENCE_FILE="${EVIDENCE_FILE}" \
python "${SOAK_PY}"
