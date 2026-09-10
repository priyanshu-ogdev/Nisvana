# Project AEGIS — Demo Runbook

## Prerequisites
- Pi 5 is on the same WiFi as the demo laptop
- Pi IP: `192.168.x.x` (check with `hostname -I`)
- Frontend: `VITE_PI_WS_URL=ws://192.168.x.x:8000/ws` in `.env`
- All 4 USB audio devices plugged into Pi

---

## Step-by-Step Demo Sequence

### 1. Boot → Frontend shows OFFLINE
- Pi boots → systemd starts `aegis.service` → LED blinks amber
- LED turns green when WS server is ready (~5s)
- **Frontend shows:** `OFFLINE / SIMULATED DEMO` (amber) → switches to `PI REACHED — NO LINK` once WS connects

### 2. Person 1 Handshake (1s delay)
- Click **INITIATE PI HANDSHAKE**
- Wait 1 second
- **Frontend shows:** Person 1 card border glows teal, `LINK SECURE`, waveform animates
- **Particle field:** shockwave radiates outward from core
- **Beam:** teal energy line appears from core to Person 1 card with travelling pulses

### 3. Person 2 Handshake
- Same button triggers Person 2 simultaneously
- After 1s delay Person 2 goes SECURE
- **Frontend shows:** `STREAMING` status pill, both beams lit, both waveforms alive
- **Telemetry bar:** latency, SNR, temp, CPU, RAM, RTT, model — all live numbers

### 4. Live Noise Demonstration
- Play 85dB defense noise from a speaker
- **Frontend shows:** waveform bars jump in the RAW view; ENC view is visibly quieter
- Click the waveform strip to toggle between **ENC** (teal) and **RAW** (amber/rose)
- Point at the SNR improvement reading in the telemetry bar — it's live, not canned

### 5. Mute Primary Mic
- Short-click mute button on Person 1 → waveform flatlines within 1 frame
- Long-press (500ms) → dropdown reveals 4 targets: Primary / Reference / Throat / Output
- **Frontend shows:** MUTED overlay, bars go to zero

### 6. Unplug Throat Mic
- Physically unplug the throat mic USB cable
- **Within 2 seconds:** amber ⚠ indicator appears on the headphone icon tooltip (hover to see ALSA device list)
- Plug back in → auto-recovers

### 7. Thermal Stress Demo (optional)
- Run `stress-ng --cpu 4 --timeout 30` on Pi in a separate terminal
- Watch telemetry chip **MODEL** change in real time:
  - `DeepFilterNet3` → (at 72°C harmonic preproc disables)
  - `CleanUMamba` → (at 78°C)
  - `noisereduce-cpu` → (at 82°C)

### 8. WS Kill Demo (robustness)
- `sudo systemctl stop aegis` on Pi
- **Within 3 seconds:** frontend flips to `OFFLINE / SIMULATED DEMO`
- **RECONNECT** button appears in top bar
- `sudo systemctl start aegis` → auto-reconnect in 5s

---

## Judge Q&A Cheat Sheet

| Q | A |
|---|---|
| "Is the SNR number real?" | "Yes — measured live as delta between primary mic RMS and post-model RMS per second. Watch it change when noise level changes." |
| "What model is running?" | "Shown live in MODEL chip. Backend thermal guard can swap it — we just demonstrated that with stress-ng." |
| "Can you fake it?" | "No — the `OFFLINE / SIMULATED DEMO` label always appears if the Pi is not connected. Everything on this screen came from the Pi or is labeled SIM." |
| "What happens if a mic dies?" | "ALSA polling detects it within 2s. That client's waveform flatlines and the HW status tooltip shows the missing device." |
