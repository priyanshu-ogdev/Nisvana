"""
ws/protocol.py — Project AEGIS WebSocket Protocol
===================================================
THIS FILE IS THE SINGLE SOURCE OF TRUTH FOR THE WS CONTRACT.

Every message type sent or received by the AEGIS backend is defined here
as a Pydantic BaseModel with a Literal discriminator on the `type` field.

Frontend must consume messages matching these schemas exactly.
Run `python -m src.ws.protocol --export-schemas` to generate JSON Schemas
into `frontend_schemas/` for parity testing.

DO NOT change field names, types, or optionality without updating
`frontend/src/state/useConnectionStore.js` and re-running parity tests.
"""

from __future__ import annotations
from typing import Literal, Optional, List, Union
from pydantic import BaseModel, Field
import json
import sys


# ===========================================================================
# INCOMING MESSAGES (frontend → Pi)
# ===========================================================================

class HandshakeInit(BaseModel):
    """Frontend requests a secure link for a given client slot."""
    type: Literal["handshake_init"]
    clientId: str = Field(..., pattern=r"^person-[12]$")
    timestamp: int = Field(..., description="Unix ms from frontend clock")


class HardwareMute(BaseModel):
    """Mute or unmute a specific audio channel on the hardware."""
    type: Literal["hardware_mute"]
    clientId: str = Field(..., pattern=r"^person-[12]$")
    target: Literal["primary_mic", "reference_mic", "throat_mic", "headset_output"]
    state: bool = Field(..., description="True = muted")


class AncSet(BaseModel):
    """Enable or disable ANC for a client."""
    type: Literal["anc_set"]
    clientId: str = Field(..., pattern=r"^person-[12]$")
    enabled: bool


class Ping(BaseModel):
    """Heartbeat from frontend, expects Pong reply."""
    type: Literal["ping"]
    seq: int
    timestamp: int = Field(..., description="Unix ms")


# Discriminated union of all incoming messages
IncomingMessage = Union[HandshakeInit, HardwareMute, AncSet, Ping]


def parse_incoming(raw: str) -> IncomingMessage:
    """Parse raw JSON string into a typed IncomingMessage."""
    from pydantic import TypeAdapter
    adapter = TypeAdapter(IncomingMessage)
    return adapter.validate_json(raw)


# ===========================================================================
# OUTGOING MESSAGES (Pi → frontend)
# ===========================================================================

class HandshakeAck(BaseModel):
    """Response to HandshakeInit — ok or denied."""
    type: Literal["handshake_ack"] = "handshake_ack"
    clientId: str
    status: Literal["ok", "denied"]
    reason: Optional[str] = None


class HwStatus(BaseModel):
    """Broadcast when ALSA state changes or on connection."""
    type: Literal["hw_status"] = "hw_status"
    headset_detected: bool
    mic_primary: bool
    mic_reference: bool
    mic_throat: bool
    pi_cpu_temp: Optional[float] = None
    ai_model_loaded: Optional[str] = None
    alsainputs: list[str] = Field(default_factory=list)
    alsaoutputs: list[str] = Field(default_factory=list)


class FftStream(BaseModel):
    """30fps spectral data."""
    type: Literal["fft_stream"] = "fft_stream"
    clientId: str
    bins: list[int] = Field(..., min_length=64, max_length=64, description="Enhanced output bins [0-255]")
    raw_bins: list[int] = Field(..., min_length=64, max_length=64, description="Raw input bins [0-255]")
    sampleRate: int
    ts: int


class AncState(BaseModel):
    """ANC pipeline state, broadcast at 10Hz per secure client."""
    type: Literal["anc_state"] = "anc_state"
    clientId: str
    anc_active: bool
    vad_speech: bool = Field(..., description="True when voice detected (AEC gated)")
    ambient_out_db: float = Field(..., description="Primary mic RMS→dBFS A-weighted")
    ambient_in_ear_db: float = Field(..., description="Post-limiter in-ear dBFS")
    sidetone_on: bool


class Telemetry(BaseModel):
    """System performance telemetry, broadcast at 1Hz."""
    type: Literal["telemetry"] = "telemetry"
    latency_ms: float = Field(..., description="End-to-end pipeline latency ms")
    snr_improvement_db: float = Field(..., description="Estimated SNR delta dB")
    model: str = Field(..., description="Currently active AI model name")
    platform: Literal["pi5"] = "pi5"
    fps: float = Field(..., description="Audio pipeline frames/sec")
    cpu_pct: float = Field(..., description="Pi CPU usage 0–100")
    ram_pct: float = Field(..., description="Pi RAM usage 0–100")
    pi_cpu_temp: Optional[float] = Field(None, description="°C, duplicated for convenience")
    aec_active: Optional[bool] = None
    snr_state: Optional[str] = None
    blend_weight: Optional[float] = None


class LinkStatus(BaseModel):
    """State machine event for a specific client link."""
    type: Literal["link_status"] = "link_status"
    state: Literal["handshaking", "link_secure", "streaming", "dropped"]
    clientId: str


class Pong(BaseModel):
    """Response to Ping with RTT measurement."""
    type: Literal["pong"] = "pong"
    seq: int
    ts: int = Field(..., description="Unix ms at time of pong send")
    rtt_ms: float = Field(..., description="Measured round-trip time ms")


# Discriminated union of all outgoing messages
OutgoingMessage = Union[
    HandshakeAck, HwStatus, FftStream, AncState, Telemetry, LinkStatus, Pong
]


# ===========================================================================
# SCHEMA EXPORT (run as __main__ to generate frontend_schemas/)
# ===========================================================================

_MESSAGE_REGISTRY = {
    "HandshakeInit": HandshakeInit,
    "HardwareMute": HardwareMute,
    "AncSet": AncSet,
    "Ping": Ping,
    "HandshakeAck": HandshakeAck,
    "HwStatus": HwStatus,
    "FftStream": FftStream,
    "AncState": AncState,
    "Telemetry": Telemetry,
    "LinkStatus": LinkStatus,
    "Pong": Pong,
}


def export_schemas(output_dir: str = "frontend_schemas") -> None:
    """Export all message schemas as JSON Schema files for frontend parity tests."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    for name, model in _MESSAGE_REGISTRY.items():
        schema = model.model_json_schema()
        path = os.path.join(output_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(schema, f, indent=2)
        print(f"  Exported {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AEGIS Protocol Schema Exporter")
    parser.add_argument("--export-schemas", action="store_true")
    parser.add_argument("--output-dir", default="frontend_schemas")
    args = parser.parse_args()

    if args.export_schemas:
        print("Exporting schemas...")
        export_schemas(args.output_dir)
        print("Done.")
    else:
        print("Available messages:", list(_MESSAGE_REGISTRY.keys()))
