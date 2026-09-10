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

## Configuration

To connect the UI to a live backend WebSocket, create a \`.env\` file in the \`frontend/\` directory and add the URL:
\`\`\`env
VITE_BACKEND_WS_URL=ws://localhost:8000/ws
\`\`\`
If no valid backend is found or the connection fails, the frontend will automatically fall back to a high-fidelity visual simulation of the handshake process.

## Demo Controls
There is a hidden "Trigger Handshake" button in the bottom right corner (hover over it to make it fully visible) to manually trigger the connection sequence for live demonstrations.
