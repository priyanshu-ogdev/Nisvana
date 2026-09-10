"""
pi_backend_stub.py — Project AEGIS v5 Backend Stub (upgraded)
==============================================================
Runs on a laptop to simulate the Pi backend for frontend development.

Changes from v4:
  - ping → pong roundtrip with rtt_ms measurement
  - Telemetry includes cpu_pct, ram_pct (F-5)
  - fft_stream includes raw_bins + bins (dual-stream, F-4)
  - hw_status includes alsainputs, alsaoutputs (F-7)
  - link_status includes clientId
  - All outgoing messages match protocol.py exactly

Usage:
  python3 pi_backend_stub.py                         # default (8s person-2 delay)
  python3 pi_backend_stub.py --ack2-delay 3          # faster demo
  python3 pi_backend_stub.py --no-headset            # simulate missing headset
  python3 pi_backend_stub.py --leak-pre-ack          # test frontend gate (F-1 test)
  python3 pi_backend_stub.py --kill-after 30         # auto-exit after 30s
"""
import asyncio
import websockets
import json
import time
import argparse
import random
import math
import psutil

clients = set()

state = {
    "person-1": {"secure": False, "muted_primary": False, "muted_ref": False,
                 "muted_throat": False, "muted_output": False, "anc_active": False},
    "person-2": {"secure": False, "muted_primary": False, "muted_ref": False,
                 "muted_throat": False, "muted_output": False, "anc_active": False},
}

# Track ping timestamps for RTT calculation
ping_times: dict[int, float] = {}


async def handler(websocket, path=None, args=None):
    global clients
    clients.add(websocket)
    print(f"[STUB] Client connected (total={len(clients)})")

    # Reject if > 2 slots
    if len(clients) > 2:
        await websocket.send(json.dumps({
            "type": "handshake_ack",
            "clientId": "unknown",
            "status": "denied",
            "reason": "slot_full"
        }))
        await websocket.close()
        clients.discard(websocket)
        return

    # Send initial hw_status with all new fields
    await websocket.send(json.dumps({
        "type": "hw_status",
        "headset_detected": not args.no_headset,
        "mic_primary": True,
        "mic_reference": True,
        "mic_throat": True,
        "pi_cpu_temp": 45.0,
        "ai_model_loaded": "DeepFilterNet3",
        "alsainputs": ["USB Audio (C-Media) [Capture]", "USB Mic (Throat) [Capture]"],
        "alsaoutputs": ["USB Audio (Headset) [Playback]"],
    }))

    try:
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get("type")

            if msg_type == "ping":
                # Measure RTT from original timestamp
                seq = data.get("seq", 0)
                original_ts = data.get("timestamp", int(time.time() * 1000))
                now_ms = int(time.time() * 1000)
                rtt_ms = max(0.0, float(now_ms - original_ts))
                await websocket.send(json.dumps({
                    "type": "pong",
                    "seq": seq,
                    "ts": now_ms,
                    "rtt_ms": round(rtt_ms, 2),
                }))

            elif msg_type == "handshake_init":
                client_id = data.get("clientId")
                delay = args.ack2_delay if client_id == "person-2" else 1.0
                print(f"[STUB] handshake_init from {client_id} — ack in {delay}s")

                async def send_ack(cid, d):
                    # Broadcast: handshaking
                    for ws in list(clients):
                        try:
                            await ws.send(json.dumps({
                                "type": "link_status",
                                "state": "handshaking",
                                "clientId": cid,
                            }))
                        except Exception:
                            pass

                    await asyncio.sleep(d)
                    state[cid]["secure"] = True
                    state[cid]["anc_active"] = True

                    # Ack to the sender
                    await websocket.send(json.dumps({
                        "type": "handshake_ack",
                        "clientId": cid,
                        "status": "ok"
                    }))

                    # Broadcast: link_secure
                    for ws in list(clients):
                        try:
                            await ws.send(json.dumps({
                                "type": "link_status",
                                "state": "link_secure",
                                "clientId": cid,
                            }))
                        except Exception:
                            pass

                    # Both secure → streaming
                    if state["person-1"]["secure"] and state["person-2"]["secure"]:
                        for ws in list(clients):
                            for c in ["person-1", "person-2"]:
                                try:
                                    await ws.send(json.dumps({
                                        "type": "link_status",
                                        "state": "streaming",
                                        "clientId": c,
                                    }))
                                except Exception:
                                    pass

                asyncio.create_task(send_ack(client_id, delay))

            elif msg_type == "hardware_mute":
                client_id = data.get("clientId")
                target = data.get("target", "primary_mic")
                mute_state = data.get("state", False)
                key = f"muted_{target.replace('_mic', '').replace('headset_output', 'output')}"
                if key in state.get(client_id, {}):
                    state[client_id][key] = mute_state
                print(f"[STUB] hardware_mute {client_id}.{target}={mute_state}")

            elif msg_type == "anc_set":
                client_id = data.get("clientId")
                state[client_id]["anc_active"] = data.get("enabled", False)
                print(f"[STUB] anc_set {client_id}={state[client_id]['anc_active']}")

    except websockets.exceptions.ConnectionClosed:
        print("[STUB] Client disconnected.")
    finally:
        clients.discard(websocket)


async def stream_data(args):
    start_time = time.time()

    while True:
        await asyncio.sleep(1 / 30)  # 30fps loop

        now = time.time()
        if args.kill_after > 0 and (now - start_time) > args.kill_after:
            print("[STUB] Killing per --kill-after")
            import os
            os._exit(0)

        for ws in list(clients):
            try:
                # FFT stream (30fps) — dual stream: raw_bins + bins
                for cid in ["person-1", "person-2"]:
                    if state[cid]["secure"] or args.leak_pre_ack:
                        primary_muted = state[cid]["muted_primary"]

                        # Raw bins: noisy signal
                        if primary_muted:
                            raw_bins = [0] * 64
                            enhanced_bins = [0] * 64
                        else:
                            raw_bins = []
                            enhanced_bins = []
                            for i in range(64):
                                # Raw: high noise floor
                                raw_base = max(0, 200 - i * 2)
                                raw_val = min(255, raw_base + random.randint(-40, 40))
                                raw_bins.append(max(0, raw_val))

                                # Enhanced: noticeably cleaner (lower floor, peaks only where voice is)
                                enhanced_base = max(0, 120 - i * 3)
                                enhanced_val = min(255, enhanced_base + random.randint(-20, 20))
                                enhanced_bins.append(max(0, enhanced_val))

                        await ws.send(json.dumps({
                            "type": "fft_stream",
                            "clientId": cid,
                            "bins": enhanced_bins,
                            "raw_bins": raw_bins,
                            "sampleRate": 48000,
                            "ts": int(now * 1000),
                        }))

                # ANC state (10Hz)
                if int(now * 30) % 3 == 0:
                    for cid in ["person-1", "person-2"]:
                        if state[cid]["secure"]:
                            cycle = (now + (4 if cid == 'person-2' else 0)) % 8
                            vad_speech = cycle < 3
                            anc_active = state[cid]["anc_active"]
                            out_db = -20.0 + math.sin(now) * 2
                            in_db = -45.0 + math.sin(now) * 1.5 if anc_active else out_db

                            await ws.send(json.dumps({
                                "type": "anc_state",
                                "clientId": cid,
                                "anc_active": anc_active,
                                "vad_speech": vad_speech,
                                "ambient_out_db": round(out_db, 1),
                                "ambient_in_ear_db": round(in_db, 1),
                                "sidetone_on": True,
                            }))

                # Telemetry (1Hz) — now with cpu_pct + ram_pct
                if int(now * 30) % 30 == 0:
                    try:
                        cpu_pct = psutil.cpu_percent(interval=None)
                        ram_pct = psutil.virtual_memory().percent
                    except Exception:
                        cpu_pct = 0.0
                        ram_pct = 0.0

                    await ws.send(json.dumps({
                        "type": "telemetry",
                        "latency_ms": round(13.0 + math.sin(now) * 5, 1),
                        "snr_improvement_db": round(22.0 + math.sin(now * 0.5) * 4, 1),
                        "model": "DeepFilterNet3",
                        "platform": "pi5",
                        "fps": round(100.0 + math.sin(now * 2) * 5, 1),
                        "cpu_pct": round(cpu_pct, 1),
                        "ram_pct": round(ram_pct, 1),
                        "pi_cpu_temp": round(45.0 + math.sin(now * 0.1) * 5, 1),
                    }))

            except Exception:
                pass


async def main():
    parser = argparse.ArgumentParser(description="AEGIS v5 Backend Stub")
    parser.add_argument("--ack2-delay", type=float, default=8.0)
    parser.add_argument("--no-headset", action="store_true")
    parser.add_argument("--leak-pre-ack", action="store_true")
    parser.add_argument("--kill-after", type=float, default=0)
    args = parser.parse_args()

    print(f"[STUB] Starting on ws://localhost:8000/ws")
    print(f"[STUB] person-2 ack delay: {args.ack2_delay}s")

    server = await websockets.serve(
        lambda ws, path="": handler(ws, path, args),
        "localhost", 8000
    )
    asyncio.create_task(stream_data(args))
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
