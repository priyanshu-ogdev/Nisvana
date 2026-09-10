# Project AEGIS — Multi-User Backend & Streaming Subsystem

> **Location:** `backend/`  
> **Target Deployment:** Tactical Edge Servers, Base Station Nodes, Multi-Channel Vehicle Intercoms, Cloud Relays  
> **Throughput Envelope:** 100+ Concurrent Fullband Audio Streams per Node (<50 KB State per Session)  
> **Acoustic Standards:** 48,000 Hz Fullband Audio, 10ms (480-sample) Streaming Frames, PCM-16 Wire Protocol  

---

## 1. Executive Summary & Architecture

The `backend` package provides the high-concurrency, multi-user streaming audio infrastructure for Project AEGIS. It bridges real-time tactical radio/network endpoints to shared neural speech enhancement backbones without replicating model weights per user.

The architecture is partitioned into three decoupled, high-performance layers:
1. **Session Lifecycle & State Management (`session_manager.py`)**:
   - Maintains isolated, ultra-compact state instances (<50 KB RAM per user) containing recurrent hidden states (`h_in`), lookahead context buffers, and speech-floor crossfader registers.
   - Enforces configurable TTL idle pruning (`session_ttl_sec=120.0`), preventing memory leaks in hostile reconnect scenarios.
2. **Batched Neural Inference Engine (`batch_inference.py`)**:
   - Aggregates frame requests across concurrent active sessions into unified PyTorch tensor batches $(B, 1, 480)$.
   - Runs a single forward pass over shared neural weights, eliminating redundant memory footprint and maximizing Tensor Core utilization on NVIDIA Grace Blackwell / edge GPUs.
   - Enforces the speech intelligibility floor ($\ge 0.15$ dry mix during speech dominance) per session.
3. **Asynchronous Audio Transport (`transport.py`)**:
   - Non-blocking async queue handlers (`StreamQueue`) with zero-copy PCM-16 byte serialization (`AudioPacket`).
   - Bounded ring buffers with an atomic drop-oldest backpressure policy, guaranteeing that network jitter or client lag never builds pipeline latency.

```
+===================================================================================================+
|                                  MULTI-USER BACKEND ARCHITECTURE                                  |
+===================================================================================================+
|                                                                                                   |
|  [User 1 Stream]       [User 2 Stream]       [User 3 Stream]       ...      [User N Stream]       |
|  (PCM-16 / 48kHz)      (PCM-16 / 48kHz)      (PCM-16 / 48kHz)               (PCM-16 / 48kHz)      |
|         │                     │                     │                              │              |
|         ▼                     ▼                     ▼                              ▼              |
|  +---------------------------------------------------------------------------------------------+  |
|  | 1. ASYNC AUDIO TRANSPORT LAYER (backend/transport.py)                                       |  |
|  |    * AudioPacket: PCM-16 16-bit signed integer byte packing & unpacking (float32 <-> int16)  |  |
|  |    * StreamQueue: Async bounded queue with 'drop_oldest' backpressure overflow control       |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                                │                                                  |
|                                                ▼                                                  |
|  +---------------------------------------------------------------------------------------------+  |
|  | 2. SESSION MANAGER (backend/session_manager.py)                                             |  |
|  |    * Lightweight UserSession: <50 KB state per user (h_gru, lookahead context, VAD history) |  |
|  |    * Session Registry: Fast thread-safe session lookup, heartbeat tracking, & TTL reap      |  |
|  +---------------------------------------------------------------------------------------------+  |
|                                                │                                                  |
|                                                ▼                                                  |
|  +---------------------------------------------------------------------------------------------+  |
|  | 3. BATCH INFERENCE ENGINE (backend/batch_inference.py)                                      |  |
|  |    * Batch Stacking: Packs N active session frames into tensor shape (B, 1, 480)            |  |
|  |    * Shared Weights: Single neural forward pass across active sessions                      |  |
|  |    * State Demuxing: Unpacks output chunks and writes updated states back to UserSession     |  |
|  |    * Intelligibility Floor: Guarantees dry_mix >= 0.15 on speech-dominant frames           |  |
|  +---------------------------------------------------------------------------------------------+  |
|         │                     │                     │                              │              |
|         ▼                     ▼                     ▼                              ▼              |
|  [Enhanced Stream 1]   [Enhanced Stream 2]   [Enhanced Stream 3]            [Enhanced Stream N]   |
+===================================================================================================+
```

---

## 2. Deep-Dive: Core Subsystems

### 2.1 Session Lifecycle (`session_manager.py`)
- **`UserSession`**:
  - Encapsulates per-user streaming state: `session_id`, `created_at`, `last_active_at`, `model_states` dictionary (containing recurrent hidden states and lookahead delay buffers), and a moving-average speech confidence metric.
  - **Memory Footprint**: Verified at `< 50 KB` per session, enabling 100+ active tactical channels in under 5 MB total RAM.
- **`SessionManager`**:
  - Manages session lifecycle: `get_or_create_session(session_id)`, `remove_session(session_id)`, and `cleanup_idle_sessions(max_idle_seconds)`.
  - Supports dynamic state resets without connection drops via `session.reset_state()`.

### 2.2 Batched Tensor Inference (`batch_inference.py`)
- **`BatchInferenceEngine`**:
  - Supports both single-session execution (`process_session_frame(session, audio_frame)`) and batched multi-session execution (`process_batch(sessions, audio_frames)`).
  - Handles state demultiplexing: updates each user's isolated hidden state after batched computation.
  - Enforces speech intelligibility preservation: automatically applies the dry mix floor ($\ge 0.15$) whenever speech dominance is detected.

### 2.3 Audio Transport & Serialization (`transport.py`)
- **`AudioPacket`**:
  - Encapsulates 48 kHz mono streaming frames (default: 480 samples = 10ms).
  - Converts between normalized `float32` audio tensors $[-1.0, +1.0]$ and standard 16-bit linear PCM byte buffers (`pcm16_bytes = int16.tobytes()`).
- **`StreamQueue`**:
  - High-throughput `asyncio.Queue` wrapper with bounded capacity (`max_queue_frames=10`).
  - Implements drop-oldest semantics on overflow: when network backpressure occurs, the oldest frame is discarded to preserve strict low-latency streaming guarantees.

---

## 3. Python Quickstart Example

```python
import asyncio
import numpy as np
from backend import SessionManager, BatchInferenceEngine, StreamQueue, AudioPacket

async def main():
    # 1. Initialize session manager and batch engine
    session_mgr = SessionManager(session_ttl_sec=60.0)
    batch_engine = BatchInferenceEngine(model=None)  # Model loaded on demand or injected
    
    # 2. Allocate sessions for tactical users
    session_alpha = session_mgr.get_or_create_session("callsign_alpha")
    session_bravo = session_mgr.get_or_create_session("callsign_bravo")
    
    # 3. Simulate 10ms audio chunk (480 samples @ 48kHz)
    input_chunk = np.random.randn(480).astype(np.float32) * 0.1
    
    # 4. Process multi-user batch through shared neural engine
    results = batch_engine.process_batch(
        sessions=[session_alpha, session_bravo],
        audio_frames=[input_chunk, input_chunk]
    )
    
    # 5. Pack enhanced output to PCM-16 wire format
    packet = AudioPacket.from_float32(results[0], sample_rate=48000)
    wire_bytes = packet.to_bytes()
    print(f"Serialized {len(wire_bytes)} bytes for transmission")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 4. Verification & Testing

The backend subsystem is validated by automated test suites in `tests/test_backend_subsystem.py`:
```bash
pytest tests/test_backend_subsystem.py -v
```
Verified metrics:
- State footprint `< 50 KB` per user session.
- Thread-safe session CRUD operations and TTL cleanup.
- Batched multi-session tensor alignment and state isolation.
- PCM-16 serialization roundtrip with zero bit-depth distortion.
- Async queue backpressure and non-blocking drop-oldest flow control.
