import asyncio
import websockets
import json
import time
import argparse
import random
import math

clients = set()
state = {
    "person-1": {"secure": False, "muted": False, "anc_active": False},
    "person-2": {"secure": False, "muted": False, "anc_active": False}
}

async def handler(websocket, path=None, args=None):
    global clients
    clients.add(websocket)
    print("Client connected!")
    
    # Send initial hw_status
    await websocket.send(json.dumps({
        "type": "hw_status",
        "headset_detected": not args.no_headset,
        "mic_primary": True,
        "mic_reference": True,
        "mic_throat": True,
        "pi_cpu_temp": 45.0,
        "ai_model_loaded": "DeepFilterNet3"
    }))
    
    try:
        async for message in websocket:
            data = json.loads(message)
            print(f"Received: {data}")
            
            if data.get("type") == "handshake_init":
                client_id = data.get("clientId")
                delay = args.ack2_delay if client_id == "person-2" else 0.5
                
                async def send_ack(cid, d):
                    await asyncio.sleep(d)
                    state[cid]["secure"] = True
                    await websocket.send(json.dumps({
                        "type": "handshake_ack",
                        "clientId": cid,
                        "status": "ok"
                    }))
                    
                    # Also broadcast link_status if both secure
                    if state["person-1"]["secure"] and state["person-2"]["secure"]:
                        await websocket.send(json.dumps({
                            "type": "link_status",
                            "state": "streaming"
                        }))
                    
                    # Auto-enable ANC on secure
                    state[cid]["anc_active"] = True

                asyncio.create_task(send_ack(client_id, delay))

            elif data.get("type") == "hardware_mute":
                client_id = data.get("clientId")
                state[client_id]["muted"] = data.get("state")
                print(f"Mute state updated: {client_id} -> {state[client_id]['muted']}")
                
            elif data.get("type") == "anc_set":
                client_id = data.get("clientId")
                state[client_id]["anc_active"] = data.get("enabled")
                print(f"ANC state updated: {client_id} -> {state[client_id]['anc_active']}")

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
    finally:
        clients.remove(websocket)

async def stream_data(args):
    start_time = time.time()
    
    while True:
        await asyncio.sleep(1/30) # 30fps
        
        now = time.time()
        if args.kill_after > 0 and (now - start_time) > args.kill_after:
            print("Killing stub per --kill-after")
            import os
            os._exit(0)
            
        for ws in list(clients):
            try:
                # FFT stream (30fps)
                for cid in ["person-1", "person-2"]:
                    # Gating: only send if secure, OR if leak_pre_ack is True
                    if state[cid]["secure"] or args.leak_pre_ack:
                        if state[cid]["muted"]:
                            bins = [0] * 64
                        else:
                            # Generate some dummy audio signal
                            # We want something that looks like an FFT: low frequencies higher, high frequencies lower
                            bins = []
                            for i in range(64):
                                base = max(0, 255 - i * 4) 
                                noise = random.randint(0, 50)
                                val = min(255, base + noise) if random.random() > 0.1 else 0
                                bins.append(int(val))
                        
                        await ws.send(json.dumps({
                            "type": "fft_stream",
                            "clientId": cid,
                            "bins": bins
                        }))

                # ANC state (10Hz -> every 3rd frame)
                if int(now * 30) % 3 == 0:
                    for cid in ["person-1", "person-2"]:
                        if state[cid]["secure"]:
                            # Burst vad_speech: 3s on, 5s off -> 8s cycle
                            cycle = (now + (4 if cid == 'person-2' else 0)) % 8
                            vad_speech = cycle < 3
                            
                            anc_active = state[cid]["anc_active"]
                            out_db = 92 + math.sin(now)*2
                            in_ear_db = 58 + math.sin(now)*1 if anc_active else out_db
                            
                            await ws.send(json.dumps({
                                "type": "anc_state",
                                "clientId": cid,
                                "anc_active": anc_active,
                                "vad_speech": vad_speech,
                                "ambient_out_db": out_db,
                                "ambient_in_ear_db": in_ear_db,
                                "sidetone_on": True
                            }))
                            
                # Telemetry (1Hz -> every 30th frame)
                if int(now * 30) % 30 == 0:
                    await ws.send(json.dumps({
                        "type": "telemetry",
                        "latency_ms": int(13 + math.sin(now)*5),
                        "snr_improvement_db": int(22 + math.sin(now*0.5)*4),
                        "model": "DeepFilterNet3",
                        "platform": "pi5"
                    }))
                    
            except Exception as e:
                pass


async def main():
    parser = argparse.ArgumentParser(description="AEGIS v4 Backend Stub")
    parser.add_argument("--ack2-delay", type=float, default=8.0, help="Delay for person-2 handshake ack (seconds)")
    parser.add_argument("--no-headset", action="store_true", help="Simulate headset not detected")
    parser.add_argument("--leak-pre-ack", action="store_true", help="Leak fft_stream before ack")
    parser.add_argument("--kill-after", type=float, default=0, help="Kill server after N seconds")
    args = parser.parse_args()

    print(f"Starting backend stub on ws://localhost:8000/ws with args: {args}")
    
    server = await websockets.serve(lambda ws, path: handler(ws, path, args), "localhost", 8000)
    
    # Start data streaming loop
    asyncio.create_task(stream_data(args))
    
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
