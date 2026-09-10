"""src/main.py — AEGIS Backend Entrypoint.

Boot sequence:
  1. Load config
  2. Detect hardware (device_detector)
  3. Load AI model (model_loader)
  4. Bind WS server
  5. Start orchestrator + thermal guard
  6. GPIO LED: amber → green when WS ready

Usage:
  python -m src.main                 # normal operation
  python -m src.main --self-test     # 60s loopback self-test
"""
from __future__ import annotations
import asyncio
import logging
import os
import sys
import argparse
import time

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


def _setup_gpio_led(pin: int, state: str) -> None:
    """Blink LED on GPIO pin. Silently fails on non-Pi."""
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.OUT)
        if state == "amber":
            # PWM blink at 2Hz
            pwm = GPIO.PWM(pin, 2)
            pwm.start(50)
        elif state == "green":
            GPIO.output(pin, GPIO.HIGH)
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"GPIO LED error (non-fatal): {e}")


async def _self_test() -> bool:
    """60-second self-test mode: loopback all subsystems."""
    logger.info("=== AEGIS SELF-TEST MODE ===")
    results = {}

    # 1. Protocol schema export
    try:
        from .ws.protocol import export_schemas
        export_schemas("frontend_schemas")
        results["protocol_schemas"] = "PASS"
    except Exception as e:
        results["protocol_schemas"] = f"FAIL: {e}"

    # 2. ALSA bridge self-test
    try:
        import yaml
        with open("config/hardware.yaml") as f:
            hw_cfg = yaml.safe_load(f)
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
        # Synthesize Friedlander blast (amplitude >> 1.0)
        t = np.linspace(0, 0.01, 480)
        peak_overpressure = 100.0  # far above full scale
        pos_phase_dur = 0.003
        blast = np.where(
            t < pos_phase_dur,
            peak_overpressure * np.exp(-t / (pos_phase_dur / 3)) * (1 - t / pos_phase_dur),
            -peak_overpressure * 0.1 * np.exp(-(t - pos_phase_dur) / 0.005)
        ).astype(np.float32)
        limited = limiter.process_frame(blast)
        max_out = float(np.max(np.abs(limited)))
        if max_out <= CEILING_LINEAR + 0.001:
            results["limiter_safety"] = f"PASS (max={max_out:.4f} ≤ {CEILING_LINEAR:.4f})"
        else:
            results["limiter_safety"] = f"FAIL (max={max_out:.4f} > {CEILING_LINEAR:.4f})"
    except Exception as e:
        results["limiter_safety"] = f"FAIL: {e}"

    # 5. WS client smoke test (brief)
    try:
        from .ws.client import AegisClient
        client = AegisClient(hub_url="ws://127.0.0.1:8001/node", node_id="test-node")
        results["ws_client"] = "PASS (instantiation)"
    except Exception as e:
        results["ws_client"] = f"FAIL: {e}"

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
    parser = argparse.ArgumentParser(description="Project AEGIS Edge Backend")
    parser.add_argument("--self-test", action="store_true", help="Run 60s self-test and exit")
    parser.add_argument("--port", type=int, default=int(os.getenv("PI_WS_PORT", 8000)))
    parser.add_argument("--host", default=os.getenv("PI_WS_HOST", "0.0.0.0"))
    args = parser.parse_args()

    led_pin = int(os.getenv("PI_LED_GPIO", 17))

    if args.self_test:
        ok = await _self_test()
        sys.exit(0 if ok else 1)

    logger.info("=== PROJECT AEGIS BACKEND STARTING ===")
    _setup_gpio_led(led_pin, "amber")

    # 1. Hardware detection
    from .hardware.device_detector import DeviceDetector
    from .ws.client import AegisClient
    from .ws.protocol import HwStatus
    from .orchestrator import Orchestrator

    # Read node identity from env (or generate one)
    node_id = os.getenv("AEGIS_NODE_ID", "pi-demo")
    hub_url = os.getenv("AEGIS_HUB_URL", "ws://127.0.0.1:8001/node")

    client = AegisClient(hub_url=hub_url, node_id=node_id)

    async def on_hw_change(status_dict: dict) -> None:
        hw = HwStatus(**status_dict)
        client.push_hw_status(hw)

    detector = DeviceDetector(on_status_change=on_hw_change)
    initial_hw = detector._detect()
    client.push_hw_status(HwStatus(**initial_hw))

    # 2. Load AI model
    from .ai.model_loader import ModelLoader
    ml = ModelLoader()
    model_name = ml.load()
    logger.info(f"Active model: {model_name}")

    # 3. Orchestrator
    import yaml
    with open("config/audio_pipeline.yaml") as f:
        pipeline_cfg = yaml.safe_load(f)

    orchestrator = Orchestrator(client=client, config=pipeline_cfg)
    orchestrator.load_model()

    # Signal ready with green LED
    _setup_gpio_led(led_pin, "green")
    logger.info(f"Connecting node {node_id} to {hub_url}")

    # 4. Run all tasks
    await asyncio.gather(
        client.connect_forever(),
        orchestrator.run_forever(),
        detector.poll_forever(),
    )


def run() -> None:
    """Entry point for `aegis` CLI command."""
    asyncio.run(main())


if __name__ == "__main__":
    asyncio.run(main())
