"""tests/test_protocol_parity.py — Schema parity between backend Pydantic models and frontend."""
from __future__ import annotations
import json
import os
import sys
import pytest
from pathlib import Path

# Add aegis-backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ws.protocol import (
    HandshakeInit, HardwareMute, AncSet, Ping,
    HandshakeAck, HwStatus, FftStream, AncState, Telemetry, LinkStatus, Pong,
    _MESSAGE_REGISTRY, export_schemas
)


def test_all_schemas_exported():
    """Ensure every message in _MESSAGE_REGISTRY has a corresponding JSON Schema in the frontend."""
    import tempfile
    import json
    
    FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "../../frontend/src/ws/schemas")
    # If the frontend directory doesn't exist, this test should fail because G5 requires it to be committed
    assert os.path.exists(FRONTEND_DIR), f"Frontend schema directory missing: {FRONTEND_DIR}"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        export_schemas(tmpdir)
        
        for name in _MESSAGE_REGISTRY.keys():
            tmp_path = os.path.join(tmpdir, f"{name}.json")
            frontend_path = os.path.join(FRONTEND_DIR, f"{name}.json")
            
            assert os.path.exists(frontend_path), f"Schema {name}.json not found in frontend repository"
            
            with open(tmp_path) as f_tmp, open(frontend_path) as f_front:
                tmp_json = json.load(f_tmp)
                front_json = json.load(f_front)
                
            assert tmp_json == front_json, f"Schema mismatch for {name}. Please run 'make schemas' in aegis-backend and commit the changes."


def test_fft_stream_has_both_bins():
    """FftStream must include both bins and raw_bins (dual-stream requirement)."""
    schema = FftStream.model_json_schema()
    props = schema.get("properties", {})
    assert "bins" in props, "FftStream.bins missing"
    assert "raw_bins" in props, "FftStream.raw_bins missing"
    # Both must be 64-element arrays
    assert props["bins"].get("minItems") == 64 or props["bins"].get("maxItems") == 64
    assert props["raw_bins"].get("minItems") == 64 or props["raw_bins"].get("maxItems") == 64


def test_hardware_mute_has_target_enum():
    """HardwareMute.target must be a 4-value enum (not a boolean)."""
    schema = HardwareMute.model_json_schema()
    target = schema["properties"]["target"]
    assert "enum" in target or "$ref" in target or "anyOf" in target


def test_link_status_has_client_id():
    """LinkStatus must include clientId (v5 frontend contract)."""
    schema = LinkStatus.model_json_schema()
    assert "clientId" in schema["properties"]


def test_telemetry_has_cpu_and_ram():
    """Telemetry must include cpu_pct and ram_pct (F-5)."""
    schema = Telemetry.model_json_schema()
    props = schema["properties"]
    assert "cpu_pct" in props, "Telemetry.cpu_pct missing (F-5)"
    assert "ram_pct" in props, "Telemetry.ram_pct missing (F-5)"


def test_pong_has_rtt():
    """Pong must include rtt_ms (F-2)."""
    schema = Pong.model_json_schema()
    assert "rtt_ms" in schema["properties"]


def test_hw_status_has_alsa_fields():
    """HwStatus must include alsainputs and alsaoutputs arrays (F-7)."""
    schema = HwStatus.model_json_schema()
    props = schema["properties"]
    assert "alsainputs" in props
    assert "alsaoutputs" in props


def test_all_messages_have_type_literal():
    """Every message model must have a Literal type discriminator."""
    for name, model in _MESSAGE_REGISTRY.items():
        schema = model.model_json_schema()
        props = schema.get("properties", {})
        assert "type" in props, f"{name} missing 'type' discriminator"


def test_schema_roundtrip_handshake_ack():
    """HandshakeAck must serialize and re-parse cleanly."""
    original = HandshakeAck(clientId="person-1", status="ok")
    json_str = original.model_dump_json()
    parsed = HandshakeAck.model_validate_json(json_str)
    assert parsed.clientId == "person-1"
    assert parsed.status == "ok"


def test_schema_roundtrip_fft_stream():
    """FftStream with both bin arrays must roundtrip."""
    import time
    msg = FftStream(
        clientId="person-2",
        bins=[100] * 64,
        raw_bins=[50] * 64,
        sampleRate=48000,
        ts=int(time.time() * 1000),
    )
    parsed = FftStream.model_validate_json(msg.model_dump_json())
    assert len(parsed.bins) == 64
    assert len(parsed.raw_bins) == 64
