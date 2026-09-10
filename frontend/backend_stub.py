import asyncio
import websockets
import json
import time

async def handler(websocket, path=None):
    print("Client connected!")
    try:
        async for message in websocket:
            data = json.loads(message)
            print(f"Received: {data}")
            
            if data.get("type") == "handshake_init":
                # Simulate backend latency
                await asyncio.sleep(0.5)
                # Send ack
                await websocket.send(json.dumps({
                    "type": "handshake_ack",
                    "clientId": data.get("clientId"),
                    "status": "ok"
                }))
                
                # Start sending telemetry stream
                asyncio.create_task(send_telemetry(websocket))

            elif data.get("type") == "mute_state":
                print(f"Mute state updated: {data.get('clientId')} -> {data.get('muted')}")

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")
    except Exception as e:
        print(f"Error: {e}")

async def send_telemetry(websocket):
    latency = 12
    snr = 15
    try:
        while True:
            await asyncio.sleep(0.5)
            # Drift values slightly for realism
            latency += (time.time() % 3) - 1.5
            snr += (time.time() % 2) - 1
            
            await websocket.send(json.dumps({
                "type": "telemetry",
                "latencyMs": int(max(4, latency)),
                "snrImprovementDb": int(max(0, snr)),
                "model": "DeepFilterNet3",
                "platform": "A"
            }))
    except:
        pass

async def main():
    print("Starting backend stub on ws://localhost:8000/ws")
    async with websockets.serve(handler, "localhost", 8000):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
