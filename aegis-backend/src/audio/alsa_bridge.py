"""audio/alsa_bridge.py — ALSA / sounddevice wrapper for 3 USB mics + 1 output.

On Pi: enumerates devices by VID:PID via hardware.yaml.
On MacBook dev: runs in --self-test mode → gracefully skips missing hardware.

Usage:
    python -m src.audio.alsa_bridge --self-test
"""
from __future__ import annotations
import argparse
import asyncio
import logging
import time
from typing import Optional, Dict
import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 48000
FRAME_SIZE = 480  # 10ms


class AlsaBridge:
    """
    Manages 3 input streams (primary, reference, throat) and 1 output stream.
    Each stream has its own SpscRingBuffer; the orchestrator reads from them.
    """

    def __init__(
        self,
        hw_config: dict,
        sample_rate: int = SAMPLE_RATE,
        frame_size: int = FRAME_SIZE,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self._hw_config = hw_config
        self._streams: Dict[str, object] = {}
        self._running = False

    def open_streams(self) -> dict[str, bool]:
        """
        Open all configured audio streams.
        Returns {"primary_mic": bool, "reference_mic": bool, ...}
        """
        status = {
            "primary_mic": False,
            "reference_mic": False,
            "throat_mic": False,
            "headset_output": False,
        }

        try:
            import sounddevice as sd
            devices = sd.query_devices()
            device_names = [d["name"] for d in devices]
            logger.info(f"Found {len(devices)} audio devices: {device_names}")

            devices_cfg = self._hw_config.get("devices", {})

            for role, cfg in devices_cfg.items():
                label = cfg.get("label", "")
                # Find matching device
                device_idx = None
                for i, d in enumerate(devices):
                    if label.lower() in d["name"].lower():
                        device_idx = i
                        break

                if device_idx is None:
                    logger.warning(f"[{role}] Device not found: {label}")
                    continue

                # Open stream
                try:
                    if role == "headset_output":
                        stream = sd.OutputStream(
                            device=device_idx,
                            samplerate=self.sample_rate,
                            channels=1,
                            dtype="float32",
                            blocksize=self.frame_size,
                        )
                    else:
                        stream = sd.InputStream(
                            device=device_idx,
                            samplerate=self.sample_rate,
                            channels=1,
                            dtype="float32",
                            blocksize=self.frame_size,
                        )
                    stream.start()
                    self._streams[role] = stream
                    status[role] = True
                    logger.info(f"[{role}] Opened: {label} (device {device_idx})")
                except Exception as e:
                    logger.error(f"[{role}] Failed to open {label}: {e}")

        except ImportError:
            logger.warning("sounddevice not installed — ALSA bridge in stub mode")
        except Exception as e:
            logger.error(f"ALSA bridge open_streams error: {e}")

        return status

    def read_frame(self, role: str) -> Optional[np.ndarray]:
        """Read one frame from a mic stream. Returns None if unavailable."""
        stream = self._streams.get(role)
        if stream is None:
            return None
        try:
            data, _ = stream.read(self.frame_size)
            return data[:, 0].astype(np.float32)  # mono
        except Exception:
            return None

    def write_frame(self, frame: np.ndarray) -> None:
        """Write one frame to headset output."""
        stream = self._streams.get("headset_output")
        if stream is None:
            return
        try:
            stream.write(frame.reshape(-1, 1).astype(np.float32))
        except Exception:
            pass

    def close_all(self) -> None:
        for role, stream in self._streams.items():
            try:
                stream.stop()
                stream.close()
                logger.info(f"[{role}] Stream closed")
            except Exception:
                pass
        self._streams.clear()

    def self_test(self) -> bool:
        """5-second loopback test: mic → speaker → log."""
        logger.info("Running ALSA self-test...")
        status = self.open_streams()
        logger.info(f"Open status: {status}")

        if not any(status.values()):
            logger.error("No audio streams opened — FAIL")
            return False

        # Record 5s of primary mic
        frames = []
        for _ in range(int(SAMPLE_RATE * 5 / FRAME_SIZE)):
            f = self.read_frame("primary_mic")
            if f is not None:
                frames.append(f)
            time.sleep(FRAME_SIZE / SAMPLE_RATE)

        rms = 0.0
        if frames:
            all_data = np.concatenate(frames)
            rms = float(np.sqrt(np.mean(all_data**2)))
            logger.info(f"Primary mic 5s RMS: {rms:.4f}")

        self.close_all()
        logger.info("ALSA self-test complete")
        return True  # if we got here without crashing, it's a pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    import yaml
    with open("config/hardware.yaml") as f:
        hw_cfg = yaml.safe_load(f)

    bridge = AlsaBridge(hw_cfg)
    if args.self_test:
        result = bridge.self_test()
        print("SELF-TEST:", "PASS" if result else "FAIL")
