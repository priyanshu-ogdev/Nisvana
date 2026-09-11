"""src/frames.py — Shared Binary Frame Specification.

THIS IS THE SINGLE SOURCE OF TRUTH for mesh audio binary wire format.
Both aegis-hub and aegis-backend/src/ws/node_client.py MUST produce
byte-identical output for the same inputs. Gate 10 enforces this.

Wire layout (big-endian):
  Byte 0:           frame_type (uint8)
  Byte 1:           L = len(node_id bytes) (uint8)
  Bytes 2..2+L:     node_id (UTF-8)
  Bytes 2+L..10+L:  timestamp_ms (uint64 BE)
  Bytes 10+L..14+L: seq (uint32 BE)
  Remaining:        payload (untouched by hub)

Frame types:
  0x01 = FFT stream
  0x02 = Mesh audio (PCM int16 or Opus)
  0x03 = Health telemetry

DO NOT add hub-side payload inspection — the router reads only the header.
"""
from __future__ import annotations

import struct
import time as _time
import time
from typing import Optional, NamedTuple


# ---------------------------------------------------------------------------
# Frame type constants (shared contract)
# ---------------------------------------------------------------------------
FRAME_TYPE_FFT    = 0x01
FRAME_TYPE_AUDIO  = 0x02
FRAME_TYPE_HEALTH = 0x03

# Header suffix format: timestamp_ms (uint64 BE) + seq (uint32 BE)
_SUFFIX_FMT  = "!QI"
_SUFFIX_SIZE = struct.calcsize(_SUFFIX_FMT)   # 12 bytes


class FrameHeader(NamedTuple):
    """Parsed result of unpack_header()."""
    frame_type:    int
    source_node_id: str
    timestamp_ms:  int
    seq:           int
    payload_offset: int   # byte index where payload begins


# ---------------------------------------------------------------------------
# Pack
# ---------------------------------------------------------------------------

def pack_mesh_audio(node_id: str, seq: int, payload: bytes, *,
                    timestamp_ms: Optional[int] = None) -> bytes:
    """
    Pack a binary mesh audio frame.

    Args:
        node_id:      Source node identifier (UTF-8).
        seq:          Monotonically increasing sequence counter (uint32).
        payload:      Encoded audio bytes (PCM int16 or Opus).
        timestamp_ms: Override timestamp; defaults to current wall-clock ms.

    Returns:
        Packed frame bytes, byte-identical regardless of caller (hub or node).
    """
    node_bytes = node_id.encode("utf-8")
    ts = timestamp_ms if timestamp_ms is not None else int(_time.time() * 1000)
    header = struct.pack("!BB", FRAME_TYPE_AUDIO, len(node_bytes))
    header += node_bytes
    header += struct.pack(_SUFFIX_FMT, ts, seq)
    return header + payload


def pack_fft(node_id: str, seq: int, raw_bins: bytes, enhanced_bins: bytes, *,
             timestamp_ms: Optional[int] = None) -> bytes:
    """Pack a binary FFT frame (frame_type=0x01)."""
    node_bytes = node_id.encode("utf-8")
    ts = timestamp_ms if timestamp_ms is not None else int(_time.time() * 1000)
    header = struct.pack("!BB", FRAME_TYPE_FFT, len(node_bytes))
    header += node_bytes
    header += struct.pack(_SUFFIX_FMT, ts, seq)
    return header + raw_bins + enhanced_bins


def pack_health(node_id: str, seq: int, health_payload: bytes, *,
                timestamp_ms: Optional[int] = None) -> bytes:
    """Pack a binary health telemetry frame (frame_type=0x03)."""
    node_bytes = node_id.encode("utf-8")
    ts = timestamp_ms if timestamp_ms is not None else int(_time.time() * 1000)
    header = struct.pack("!BB", FRAME_TYPE_HEALTH, len(node_bytes))
    header += node_bytes
    header += struct.pack(_SUFFIX_FMT, ts, seq)
    return header + health_payload


# ---------------------------------------------------------------------------
# Unpack
# ---------------------------------------------------------------------------

def unpack_header(data: bytes) -> Optional[FrameHeader]:
    """
    Parse the fixed-layout binary header without touching the payload.

    Returns None if the data is too short or malformed.
    This is the hub router's hot path — O(1), no payload inspection.
    """
    if len(data) < 2:
        return None

    frame_type   = data[0]
    node_id_len  = data[1]
    offset       = 2

    if len(data) < offset + node_id_len + _SUFFIX_SIZE:
        return None

    try:
        source_node_id = data[offset: offset + node_id_len].decode("utf-8")
    except UnicodeDecodeError:
        return None

    offset += node_id_len
    ts_ms, seq = struct.unpack_from(_SUFFIX_FMT, data, offset)
    offset += _SUFFIX_SIZE

    return FrameHeader(
        frame_type=frame_type,
        source_node_id=source_node_id,
        timestamp_ms=ts_ms,
        seq=seq,
        payload_offset=offset,
    )


def parse_source_id(data: bytes) -> Optional[str]:
    """
    Extract only the source node_id from a binary frame.

    This is the absolute minimum work the router must do — avoids
    constructing a full FrameHeader when only routing decisions are needed.
    """
    if len(data) < 2:
        return None
    node_id_len = data[1]
    end = 2 + node_id_len
    if len(data) < end:
        return None
    try:
        return data[2:end].decode("utf-8")
    except UnicodeDecodeError:
        return None
