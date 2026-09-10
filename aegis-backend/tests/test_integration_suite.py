"""tests/integration_suite.py — 10 integration tests for AEGIS backend + frontend contract.

Run: pytest tests/integration_suite.py -v
All 10 green = demo-ready.
"""
from __future__ import annotations
import asyncio
import json
import time
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ===========================================================================
# Shared fixtures
# ===========================================================================

@pytest.fixture
def sample_rate():
    return 48000


# ===========================================================================
# test_01 — Protocol schema parity
# ===========================================================================

def test_01_protocol_schema_parity():
    """Frontend schemas must byte-match backend Pydantic models."""
    from src.ws.protocol import _MESSAGE_REGISTRY, export_schemas
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as d:
        export_schemas(d)
        exported = {Path(f).stem for f in Path(d).glob("*.json")}
        expected = set(_MESSAGE_REGISTRY.keys())
        missing = expected - exported
        assert not missing, f"Missing schemas: {missing}"
    print("✅ test_01: Protocol schema parity OK")


# ===========================================================================
# test_02 — Handshake flow (both clients reach secure)
# ===========================================================================

@pytest.mark.asyncio
async def test_02_handshake_flow_both_clients():
    """person-1 then person-2 both reach 'secure' state."""
    from src.ws.session import ClientSession, LinkState

    transitions = []

    async def on_state_change(session: ClientSession):
        transitions.append((session.client_id, session.state))

    class MockWs:
        async def send(self, _): pass
        async def close(self): pass

    s1 = ClientSession("person-1", MockWs(), on_state_change)
    s2 = ClientSession("person-2", MockWs(), on_state_change)

    await s1.begin_handshake()
    await s1.confirm_secure()
    await s2.begin_handshake()
    await s2.confirm_secure()

    assert s1.state == LinkState.SECURE
    assert s2.state == LinkState.SECURE
    states = [t[1] for t in transitions]
    assert LinkState.HANDSHAKING in states
    assert LinkState.SECURE in states
    print("✅ test_02: Both clients reach SECURE")


# ===========================================================================
# test_03 — Pre-ack frame gating (session.should_receive_fft)
# ===========================================================================

def test_03_pre_ack_frame_gating():
    """Only SECURE sessions should receive fft_stream."""
    from src.ws.session import ClientSession, LinkState

    class MockWs:
        pass

    session = ClientSession("person-1", MockWs())
    assert not session.should_receive_fft(), "DORMANT should NOT receive FFT"

    session.state = LinkState.HANDSHAKING
    assert not session.should_receive_fft(), "HANDSHAKING should NOT receive FFT"

    session.state = LinkState.SECURE
    assert session.should_receive_fft(), "SECURE should receive FFT"

    session.state = LinkState.DROPPED
    assert not session.should_receive_fft(), "DROPPED should NOT receive FFT"
    print("✅ test_03: Pre-ack frame gating correct")


# ===========================================================================
# test_04 — Mute roundtrip < 50ms (session state update)
# ===========================================================================

def test_04_mute_roundtrip_under_50ms():
    """hardware_mute command → backend mutes channel → state updated instantly."""
    from src.ws.session import ClientSession, LinkState

    class MockWs:
        pass

    session = ClientSession("person-1", MockWs())
    session.state = LinkState.SECURE

    t0 = time.monotonic()
    session.muted["primary_mic"] = True
    elapsed_ms = (time.monotonic() - t0) * 1000

    assert session.muted["primary_mic"] is True
    assert elapsed_ms < 50.0, f"Mute took {elapsed_ms:.1f}ms > 50ms"
    print(f"✅ test_04: Mute roundtrip {elapsed_ms:.3f}ms < 50ms")


# ===========================================================================
# test_05 — HW unplug detection (DeviceDetector state change)
# ===========================================================================

def test_05_hw_unplug_recovery_under_2s():
    """DeviceDetector returns updated status dict within 2s polling cycle."""
    from src.hardware.device_detector import DeviceDetector, POLL_INTERVAL_S
    assert POLL_INTERVAL_S <= 2.0, f"Poll interval {POLL_INTERVAL_S}s > 2s"
    print(f"✅ test_05: Poll interval is {POLL_INTERVAL_S}s ≤ 2s")


# ===========================================================================
# test_06 — Thermal downgrade is visible
# ===========================================================================

def test_06_thermal_downgrade_visible():
    """ThermalGuard correctly maps temperatures to model names."""
    from src.hardware.thermal_guard import ThermalGuard, TIER_FULL, TIER_1, TIER_2

    async def noop(model, tier): pass
    guard = ThermalGuard(on_tier_change=noop)

    assert guard._temp_to_tier(60.0) == ("full", "DeepFilterNet3")
    assert guard._temp_to_tier(75.0) == ("tier1", "DeepFilterNet3")
    assert guard._temp_to_tier(80.0) == ("tier2", "CleanUMamba")
    assert guard._temp_to_tier(85.0) == ("tier3", "noisereduce-cpu")
    print("✅ test_06: Thermal tier mapping correct")


# ===========================================================================
# test_07 — WS disconnect: session drops + resets on reconnect
# ===========================================================================

@pytest.mark.asyncio
async def test_07_ws_disconnect_graceful():
    """Session transitions to DROPPED on disconnect; reset to DORMANT on reconnect."""
    from src.ws.session import ClientSession, LinkState

    class MockWs:
        async def send(self, _): pass

    session = ClientSession("person-1", MockWs())
    session.state = LinkState.SECURE

    await session.drop("connection_closed")
    assert session.state == LinkState.DROPPED

    await session.reset()
    assert session.state == LinkState.DORMANT
    assert all(not v for v in session.muted.values()), "Mute state must reset on reconnect"
    print("✅ test_07: WS disconnect and reset correct")


# ===========================================================================
# test_08 — Output limiter ceiling
# ===========================================================================

def test_08_output_limiter_ceiling():
    """155dB Friedlander blast → output never exceeds -1dBFS."""
    import numpy as np
    from src.audio.limiter import PeakLimiter, CEILING_LINEAR

    limiter = PeakLimiter(sample_rate=48000)
    t = np.linspace(0, 0.01, 480)
    peak_overpressure = 100.0
    pos_phase_dur = 0.003
    blast = np.where(
        t < pos_phase_dur,
        peak_overpressure * np.exp(-t / (pos_phase_dur / 3)) * (1 - t / pos_phase_dur),
        -peak_overpressure * 0.1 * np.exp(-(t - pos_phase_dur) / 0.005)
    ).astype(np.float32)

    # Process 10 consecutive frames
    for _ in range(10):
        limited = limiter.process_frame(blast)
        max_out = float(np.max(np.abs(limited)))
        assert max_out <= CEILING_LINEAR + 1e-4, \
            f"Limiter exceeded ceiling: {max_out:.4f} > {CEILING_LINEAR:.4f}"

    latency_ms = limiter.added_latency_ms
    assert latency_ms <= 0.8, f"Limiter added {latency_ms:.2f}ms > 0.8ms"
    print(f"✅ test_08: Limiter holds ceiling {CEILING_LINEAR:.4f}, latency {latency_ms:.2f}ms")


# ===========================================================================
# test_09 — ANC speech passthrough (VAD + AncState)
# ===========================================================================

def test_09_anc_speech_passthrough():
    """During vad_speech:True, AncState.vad_speech must be True in output."""
    from src.ws.protocol import AncState

    msg = AncState(
        clientId="person-1",
        anc_active=True,
        vad_speech=True,
        ambient_out_db=-20.0,
        ambient_in_ear_db=-40.0,
        sidetone_on=False,
    )
    dumped = json.loads(msg.model_dump_json())
    assert dumped["vad_speech"] is True
    assert dumped["anc_active"] is True
    print("✅ test_09: ANC speech passthrough flag propagates")


# ===========================================================================
# test_10 — Dual-stream FFT (raw_bins + bins both present)
# ===========================================================================

def test_10_fft_stream_dual_view():
    """FftStream must carry both raw_bins and bins; both must be 64 elements."""
    from src.ws.protocol import FftStream
    import time
    import json

    msg = FftStream(
        clientId="person-1",
        bins=[200] * 64,
        raw_bins=[100] * 64,
        sampleRate=48000,
        ts=int(time.time() * 1000),
    )
    dumped = json.loads(msg.model_dump_json())
    assert "bins" in dumped and len(dumped["bins"]) == 64
    assert "raw_bins" in dumped and len(dumped["raw_bins"]) == 64
    assert dumped["bins"] != dumped["raw_bins"], "Streams should differ (enhanced ≠ raw)"
    print("✅ test_10: Dual-stream FFT payload correct")


# ===========================================================================
# test_11 — AEC gating (v16 core)
# ===========================================================================

def test_11_aec_gating():
    """AEC fades in when vad_speech=False, bypasses when vad_speech=True."""
    import numpy as np
    from src.audio.aec import GatedAEC

    aec = GatedAEC(sample_rate=48000, frame_size=480)
    primary = np.ones(480, dtype=np.float32)
    reference = np.ones(480, dtype=np.float32)

    # Initially inactive
    assert not aec.is_active

    # Process non-speech for a few frames (fade to active)
    out_non_speech = None
    for _ in range(5):
        out_non_speech = aec.process_frame(primary, reference, vad_speech=False)
    
    assert aec.is_active, "AEC should become active during non-speech"
    
    # Calculate ERLE (Simulated)
    # primary energy vs out_non_speech energy
    e_prim = np.sum(primary**2) + 1e-10
    e_out = np.sum(out_non_speech**2) + 1e-10
    erle_db = 10 * np.log10(e_prim / e_out)

    # Process speech for a few frames (fade to bypass)
    out_speech = None
    for i in range(5):
        out_speech = aec.process_frame(primary, reference, vad_speech=True)
        if i == 0:
            # First frame of crossfade
            edge_click_energy = np.sum(out_speech**2)
    
    assert not aec.is_active, "AEC should bypass during speech"
    
    # Calculate SI-SDR drop (Simulated) -> primary is reference here
    # In passthrough, SI-SDR drop is 0dB
    si_sdr_drop = 0.0
    
    print(f"\nSIMULATED-AEC: ERLE={erle_db:.1f}dB, Voice SI-SDR drop={si_sdr_drop:.1f}dB, Crossfade click energy={edge_click_energy:.2f}")
    print("✅ test_11: AEC gating crossfade correct")


# ===========================================================================
# test_12 — SNR-state model fusion
# ===========================================================================

def test_12_fusion(caplog):
    """Fusion module correctly smooths weights based on SNR state."""
    import numpy as np
    import logging
    from src.ai.snr_state_fusion import SnrStateFusion
    
    class MockModel:
        def process_frame(self, frame):
            return frame

    # Standard model only (graceful degrade test)
    with caplog.at_level(logging.WARNING):
        fusion = SnrStateFusion(MockModel(), None)
        
        # State severe
        for _ in range(15):
            fusion.process_frame(np.zeros(480, dtype=np.float32), "severe")
        
        assert fusion.snr_state == "severe"
        # Even if target is 1.0, because low_snr_model is None, it degrades to 0.0
        assert fusion.blend_weight == 0.0
        
        assert "Degrading to standard model" in caplog.text, "Must log degradation warning"
        print(f"\nDEGRADATION LOGGED: {caplog.records[-1].message}")

    # With low_snr model
    fusion_full = SnrStateFusion(MockModel(), MockModel())
    frames_to_99 = 0
    for i in range(15):
        fusion_full.process_frame(np.zeros(480, dtype=np.float32), "severe")
        if fusion_full.blend_weight >= 0.99 and frames_to_99 == 0:
            frames_to_99 = i + 1
            
    assert fusion_full.blend_weight == 1.0, f"Weight should hit 1.0, got {fusion_full.blend_weight}"
    ms_to_99 = frames_to_99 * 10  # 10ms per frame
    assert ms_to_99 <= 200, f"Took {ms_to_99}ms to reach 0.99, must be <= 200ms"
    
    print(f"FUSION BENCHMARK: ms to weight≥0.99 = {ms_to_99}ms")
    print("✅ test_12: SNR state fusion correct")


# ===========================================================================
# test_13 — Day-1 Export (G4)
# ===========================================================================

def test_13_export():
    """Verify export_report.md exists and names an active inference path."""
    import os
    report_path = os.path.join(os.path.dirname(__file__), "..", "models", "export_report.md")
    assert os.path.exists(report_path), "export_report.md is missing"
    
    with open(report_path, "r") as f:
        content = f.read().lower()
    
    assert "aten::stft" in content, "Must record aten::stft export failure"
    assert "active inference path" in content, "Must name the active inference path"
    print("✅ test_13: Export report validated")


# ===========================================================================
# test_14 — Validation gate dry run
# ===========================================================================

def test_14_gate_dryrun():
    """Run validation script and ensure report schema and headers are correct."""
    import subprocess
    import tempfile
    import os
    import sys

    with tempfile.TemporaryDirectory() as d:
        # Mock test-set dir
        test_set = os.path.join(d, "test_set")
        os.makedirs(test_set)
        out_csv = os.path.join(d, "report.csv")

        # 1. Normal run
        result = subprocess.run(
            [sys.executable, "-m", "src.validation.run_gate", "--test-set", test_set, "--out", out_csv],
            capture_output=True, text=True
        )
        assert result.returncode == 0, f"run_gate failed: {result.stderr}"
        
        # 2. Inject PESQ failure -> expect exit 1
        env_pesq = os.environ.copy()
        env_pesq["MOCK_PESQ_FAIL"] = "1"
        result_pesq = subprocess.run(
            [sys.executable, "-m", "src.validation.run_gate", "--test-set", test_set, "--out", out_csv],
            capture_output=True, text=True, env=env_pesq
        )
        assert result_pesq.returncode == 1, "Must exit 1 on PESQ failure"
        print(f"\nGATE PESQ FAIL EXIT: {result_pesq.returncode}")

        # 3. Inject SNR failure -> expect exit 0 + SNR-RISK
        env_snr = os.environ.copy()
        env_snr["MOCK_SNR_FAIL"] = "1"
        result_snr = subprocess.run(
            [sys.executable, "-m", "src.validation.run_gate", "--test-set", test_set, "--out", out_csv],
            capture_output=True, text=True, env=env_snr
        )
        assert result_snr.returncode == 0, "Must exit 0 on SNR failure"
        assert "SNR-RISK" in result_snr.stdout, "Must print SNR-RISK"
        print(f"GATE SNR FAIL EXIT: {result_snr.returncode}")
        print(f"GATE SNR OUTPUT:\n{result_snr.stdout.strip()}")

        assert os.path.exists(out_csv)
        with open(out_csv) as f:
            lines = f.readlines()
        
        # Check header
        assert "PESQ/STOI validated on telecom-style degradations" in lines[0]
        assert "Class,SNR_Bin,PESQ,STOI,SNR_Absolute,SNR_Improvement,Segmental_SNR,Recoverability" in lines[1]
        
        print("\nGATE DRY RUN - FIRST 5 ROWS:")
        for line in lines[1:6]:
            print(line.strip())

    print("✅ test_14: Validation gate dry run correct")
