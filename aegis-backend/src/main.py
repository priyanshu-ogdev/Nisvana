"""src/main.py — AEGIS Mesh Node Entrypoint.

Boot sequence:
  1. Load config (node_config.yaml + env overrides)
  2. Detect hardware (device_detector)
  3. Load AI model (model_loader)
  4. Instantiate MeshNodeClient, AudioCodec, MeshPlaybackBuffer, MeshRecorder
  5. Start bidirectional Orchestrator
  6. GPIO LED: amber → green when ready

Usage:
  python -m src.main                 # normal mesh node operation
  python -m src.main --self-test     # 60s loopback self-test
  python -m src.main --node-id foo   # override NODE_ID from CLI
"""
from __future__ import annotations
import asyncio
import logging
import os
import sys
import argparse
import time
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("aegis.main")


def _load_node_config(path: str = "config/node_config.yaml") -> dict:
    """Load node_config.yaml, applying env variable overrides."""
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"Node config not found at {path}, using defaults")
        cfg = {}

    # Environment overrides (env takes priority over YAML)
    node_id = os.getenv("NODE_ID") or os.getenv("AEGIS_NODE_ID") or cfg.get("node_id", "pi-demo")
    hub_url = os.getenv("HUB_WS_URL") or cfg.get("hub", {}).get("ws_url", "ws://127.0.0.1:8001/node")
    auth_token = os.getenv("AEGIS_AUTH_TOKEN")

    # Audio device indices
    audio_cfg = cfg.get("audio", {})
    input_idx = int(os.getenv("AUDIO_INPUT_INDEX", audio_cfg.get("input", {}).get("device_index", 0)))
    output_idx = int(os.getenv("AUDIO_OUTPUT_INDEX", audio_cfg.get("output", {}).get("device_index", 1)))

    return {
        "node_id": node_id,
        "hub_url": hub_url,
        "auth_token": auth_token,
        "audio_input_index": input_idx,
        "audio_output_index": output_idx,
        "capabilities": cfg.get("capabilities", ["mic", "speaker"]),
        "log_dir": cfg.get("logging", {}).get("log_dir", "logs/"),
        "record_downstream": cfg.get("logging", {}).get("record_downstream", True),
        "playback_max_queue": cfg.get("playback", {}).get("max_queue_frames", 50),
        "raw": cfg,
    }


def _setup_gpio_led(pin: int, state: str) -> None:
    """Blink LED on GPIO pin. Silently fails on non-Pi."""
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.OUT)
        if state == "amber":
            pwm = GPIO.PWM(pin, 2)
            pwm.start(50)
        elif state == "green":
            GPIO.output(pin, GPIO.HIGH)
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"GPIO LED error (non-fatal): {e}")


async def _self_test() -> bool:
    """60-second self-test mode: validates all subsystems."""
    logger.info("=== AEGIS MESH NODE SELF-TEST MODE ===")
    results = {}

    # 1. Protocol schema export
    try:
        from .ws.protocol import export_schemas
        export_schemas("frontend_schemas")
        results["protocol_schemas"] = "PASS"
    except Exception as e:
        results["protocol_schemas"] = f"FAIL: {e}"

    # 2. ALSA bridge (uses hardware.yaml fallback for legacy compat)
    try:
        hw_cfg_path = "config/hardware.yaml"
        if os.path.exists(hw_cfg_path):
            with open(hw_cfg_path) as f:
                hw_cfg = yaml.safe_load(f)
        else:
            hw_cfg = {"devices": {}}
        from .audio.alsa_bridge import AlsaBridge
        bridge = AlsaBridge(hw_cfg)
        bridge.self_test()
        results["alsa_bridge"] = "PASS"
    except Exception as e:
        results["alsa_bridge"] = f"FAIL: {e}"

    # 3. Model loader
    try:
        from .ai.model_loader import ModelLoader
        ml = ModelLoader()
        model_name = ml.load()
        results["model_loader"] = f"PASS ({model_name})"
    except Exception as e:
        results["model_loader"] = f"FAIL: {e}"

    # 4. Limiter safety test (155dB Friedlander)
    try:
        import numpy as np
        from .audio.limiter import PeakLimiter, CEILING_LINEAR
        limiter = PeakLimiter()
        t = np.linspace(0, 0.01, 480)
        peak_overpressure = 100.0
        pos_phase_dur = 0.003
        blast = np.where(
            t < pos_phase_dur,
            peak_overpressure * np.exp(-t / (pos_phase_dur / 3)) * (1 - t / pos_phase_dur),
            -peak_overpressure * 0.1 * np.exp(-(t - pos_phase_dur) / 0.005),
        ).astype(np.float32)
        limited = limiter.process_frame(blast)
        max_out = float(np.max(np.abs(limited)))
        if max_out <= CEILING_LINEAR + 0.001:
            results["limiter_safety"] = f"PASS (max={max_out:.4f} ≤ {CEILING_LINEAR:.4f})"
        else:
            results["limiter_safety"] = f"FAIL (max={max_out:.4f} > {CEILING_LINEAR:.4f})"
    except Exception as e:
        results["limiter_safety"] = f"FAIL: {e}"

    # 5. Codec smoke test
    try:
        import numpy as np
        from .audio.codec import AudioCodec
        codec = AudioCodec()
        test_frame = np.random.randn(480).astype(np.float32) * 0.5
        encoded = codec.encode(test_frame)
        decoded = codec.decode(encoded)
        assert len(decoded) == 480, f"decoded length mismatch: {len(decoded)}"
        results["codec"] = f"PASS ({codec.mode}, {len(encoded)} bytes/frame)"
    except Exception as e:
        results["codec"] = f"FAIL: {e}"

    # 6. Jitter buffer smoke test
    try:
        import numpy as np
        from .audio.playback import MeshPlaybackBuffer
        buf = MeshPlaybackBuffer()
        pcm = np.zeros(480, dtype=np.float32)

        async def _fill():
            for _ in range(5):
                await buf.push_network_audio(pcm)

        await _fill()
        frame = await buf.pop_for_dac()
        assert len(frame) == 480
        results["playback_buffer"] = "PASS"
    except Exception as e:
        results["playback_buffer"] = f"FAIL: {e}"

    # 7. MeshNodeClient instantiation
    try:
        from .ws.node_client import MeshNodeClient
        client = MeshNodeClient(hub_url="ws://127.0.0.1:8001/node", node_id="test-node")
        results["mesh_node_client"] = "PASS (instantiation)"
    except Exception as e:
        results["mesh_node_client"] = f"FAIL: {e}"

    # Report
    logger.info("\n=== SELF-TEST RESULTS ===")
    all_pass = True
    for subsystem, result in results.items():
        status = "✅" if result.startswith("PASS") else "❌"
        logger.info(f"  {status} {subsystem}: {result}")
        if not result.startswith("PASS"):
            all_pass = False

    return all_pass


async def main() -> None:
    parser = argparse.ArgumentParser(description="Project AEGIS Mesh Node")
    parser.add_argument("--self-test", action="store_true", help="Run self-test and exit")
    parser.add_argument("--node-id", default=None, help="Override NODE_ID env var")
    parser.add_argument("--port", type=int, default=None, help="Legacy: local WS port (unused in mesh mode)")
    args = parser.parse_args()

    led_pin = int(os.getenv("PI_LED_GPIO", 17))

    if args.self_test:
        ok = await _self_test()
        sys.exit(0 if ok else 1)

    # Override NODE_ID if passed via CLI
    if args.node_id:
        os.environ["NODE_ID"] = args.node_id

    logger.info("=== PROJECT AEGIS MESH NODE STARTING ===")
    _setup_gpio_led(led_pin, "amber")

    # 1. Load config
    node_cfg = _load_node_config()
    node_id = node_cfg["node_id"]
    hub_url = node_cfg["hub_url"]
    logger.info(f"Node ID  : {node_id}")
    logger.info(f"Hub URL  : {hub_url}")
    logger.info(f"Caps     : {node_cfg['capabilities']}")

    # 2. Hardware detection
    from .hardware.device_detector import DeviceDetector
    from .ws.node_client import MeshNodeClient
    from .ws.protocol import HwStatus
    from .audio.codec import AudioCodec
    from .audio.playback import MeshPlaybackBuffer
    from .audio.recorder import MeshRecorder
    from .orchestrator import Orchestrator

    # 3. Instantiate MeshNodeClient
    client = MeshNodeClient(
        node_id=node_id,
        hub_url=hub_url,
        auth_token=node_cfg["auth_token"],
        capabilities=node_cfg["capabilities"],
    )

    async def on_hw_change(status_dict: dict) -> None:
        hw = HwStatus(**status_dict)
        client.push_hw_status(hw)

    detector = DeviceDetector(on_status_change=on_hw_change)
    initial_hw = detector._detect()
    client.push_hw_status(HwStatus(**initial_hw))

    # 4. Load AI model
    from .ai.model_loader import ModelLoader
    ml = ModelLoader()
    model_name = ml.load()
    logger.info(f"Active model: {model_name}")

    # 5. Instantiate mesh subsystems
    codec = AudioCodec()
    playback_buffer = MeshPlaybackBuffer(max_queue_frames=node_cfg["playback_max_queue"])

    recorder = None
    if node_cfg["record_downstream"]:
        recorder = MeshRecorder(node_id=node_id, log_dir=node_cfg["log_dir"])
        logger.info(f"Downstream recorder: {recorder.wav_path}")

    # 6. ALSA bridge for DAC output (optional; gracefully absent on dev machines)
    alsa_bridge = None
    try:
        hw_cfg_path = "config/hardware.yaml"
        if os.path.exists(hw_cfg_path):
            with open(hw_cfg_path) as f:
                hw_cfg = yaml.safe_load(f)
            from .audio.alsa_bridge import AlsaBridge
            alsa_bridge = AlsaBridge(hw_cfg)
            alsa_bridge.open_streams()
            logger.info("ALSA bridge: streams opened")
    except Exception as e:
        logger.info(f"ALSA bridge not available (dev mode): {e}")

    # 7. Load pipeline config
    pipeline_cfg_path = "config/audio_pipeline.yaml"
    pipeline_cfg = {}
    if os.path.exists(pipeline_cfg_path):
        with open(pipeline_cfg_path) as f:
            pipeline_cfg = yaml.safe_load(f) or {}

    # 8. Orchestrator
    orchestrator = Orchestrator(
        client=client,
        config=pipeline_cfg,
        codec=codec,
        playback_buffer=playback_buffer,
        recorder=recorder,
        alsa_bridge=alsa_bridge,
    )
    orchestrator.load_model()

    _setup_gpio_led(led_pin, "green")
    logger.info(f"Mesh node '{node_id}' ready — connecting to Hub at {hub_url}")

    # 9. Run: WS client + bidirectional orchestrator + hardware detector
    try:
        await asyncio.gather(
            client.connect_forever(),
            orchestrator.run_forever(),
            detector.poll_forever(),
        )
    finally:
        orchestrator.stop()
        if alsa_bridge:
            alsa_bridge.close_all()


def run() -> None:
    """Entry point for `aegis` CLI command."""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
