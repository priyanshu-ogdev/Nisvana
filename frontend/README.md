# Project AEGIS - Tactical ANC Link

This is the frontend demonstration screen for Project AEGIS. It visualizes a real-time, hybrid AI/ML active noise cancellation system link between two operators. 

It is built with a single-screen, no-scroll WebGL-native aesthetic (the "Nirvana" theme), providing a calm, focused, and responsive demonstration of the tactical voice pipeline.

## Features
- **WebGL-native rendering:** Uses React Three Fiber and Three.js.
- **Audio-reactive UI:** Real-time waveform and halo visualizers built with Web Audio API. Fallback synthetic signals are provided.
- **Handshake Sequence:** An automated state machine visualizes the secure link handshake, transitioning from connecting to streaming.
- **Nirvana Dark Aesthetic:** Polished with bloom post-processing, glassmorphic UI via Tailwind CSS v4, and dynamic, smooth particle animations.

## Setup & Running

1. Install dependencies:
   \`\`\`bash
   npm install --legacy-peer-deps
   \`\`\`
   
2. Start the development server:
   \`\`\`bash
   npm run dev
   \`\`\`

## Frontend-only demonstration

The final dashboard can be shown without local model files, Python, or a
running Pi/backend. Start Vite in explicit simulation mode:

```bash
VITE_DEMO_MODE=1 npm run dev -- --host 0.0.0.0
```

On Windows PowerShell:

```powershell
$env:VITE_DEMO_MODE = "1"
npm run dev -- --host 0.0.0.0
```

Open the URL printed by Vite, normally `http://localhost:5173/`. The
dashboard labels the connection **OFFLINE / SIMULATED DEMO**, shows synthetic
waveforms, and leaves live SIH/model measurements unavailable. This is
intentional: no model or backend measurement is represented as real.

## Live backend configuration

To connect the UI to a live backend WebSocket, create a \`.env\` file in the \`frontend/\` directory and add the URL:
\`\`\`env
VITE_BACKEND_WS_URL=ws://localhost:8000/ws
\`\`\`
Leave `VITE_DEMO_MODE` unset for live connection attempts. If no valid
backend is found or the connection fails, the frontend falls back to a
high-fidelity visual simulation after the connection timeout.

## Backend/model integration path

The production path is separate from this frontend-only demo:

```text
trained/exported models
  -> aegis-backend node
  -> authenticated hub WebSocket
  -> frontend protocol schemas and telemetry store
  -> dashboard
```

The backend protocol is authoritative. Export its schemas with:

```bash
cd aegis-backend
python -m src.ws.protocol --export-schemas --output-dir ../frontend/src/ws/schemas
```

Do not start the backend for the frontend-only presentation unless exported
model artifacts and hardware/audio devices are available.

## Demo Controls
There is a hidden "Trigger Handshake" button in the bottom right corner (hover over it to make it fully visible) to manually trigger the connection sequence for live demonstrations.
