"""ai/model_loader.py — ONNX Runtime model loader with 3-attempt fallback chain.

Chain: DeepFilterNet3 → CleanUMamba → noisereduce (CPU passthrough)
Broadcasts current model name for thermal guard and telemetry.
"""
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Optional
import yaml

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self, models_config_path: str = "config/models.yaml") -> None:
        with open(models_config_path) as f:
            self._config = yaml.safe_load(f)
        self._current_model = None
        self._current_name = "none"
        self._session = None

    def load(self) -> str:
        """
        Load the best available model.
        Returns the model name that was actually loaded.
        """
        chain = ["primary", "cleanumamba", "noisereduce"]
        for key in chain:
            cfg = self._config.get(key, {})
            name = cfg.get("name", key)
            path = cfg.get("path")

            if path is None:
                # noisereduce — pure Python, always available
                logger.info(f"Using noisereduce passthrough (no ONNX needed)")
                self._current_name = name
                self._current_model = "noisereduce"
                return name

            if not Path(path).exists():
                logger.warning(f"Model not found at {path} — trying next fallback")
                continue

            try:
                import onnxruntime as ort
                sess_opts = ort.SessionOptions()
                sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess = ort.InferenceSession(
                    path,
                    sess_options=sess_opts,
                    providers=["CPUExecutionProvider"],
                )
                self._session = sess
                self._current_name = name
                self._current_model = "onnx"
                logger.info(f"Loaded ONNX model: {name} from {path}")
                return name
            except Exception as e:
                logger.error(f"Failed to load {name}: {e} — trying next fallback")

        # Should never reach here (noisereduce always available)
        self._current_name = "none"
        return "none"

    def get_session(self):
        return self._session

    @property
    def model_name(self) -> str:
        return self._current_name

    def swap_model(self, model_key: str) -> str:
        """Swap to a different model (thermal guard trigger)."""
        cfg = self._config.get(model_key, {})
        name = cfg.get("name", model_key)
        path = cfg.get("path")

        if path and Path(path).exists():
            try:
                import onnxruntime as ort
                self._session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                self._current_name = name
                self._current_model = "onnx"
                logger.info(f"Swapped to model: {name}")
                return name
            except Exception as e:
                logger.error(f"Model swap failed for {name}: {e}")

        # Fall back to noisereduce
        self._session = None
        self._current_name = "noisereduce-cpu"
        self._current_model = "noisereduce"
        return "noisereduce-cpu"
