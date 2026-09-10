"""ai/snr_state_fusion.py — G2: SNR-State Model Fusion.

Standard and low-SNR checkpoint blending based on rnnoise_vad snr_state.
Weights: severe=1.0, low=0.7, mid=0.2, high=0.0.
Smooths over 100ms to avoid audible artifacts.
"""
from __future__ import annotations
import numpy as np
import logging
from .deepfilternet3 import DeepFilterNet3

logger = logging.getLogger(__name__)

class SnrStateFusion:
    """
    Blends outputs of two models based on SNR state.
    """
    def __init__(
        self,
        standard_model: DeepFilterNet3,
        low_snr_model: DeepFilterNet3 | None,
        sample_rate: int = 48000,
        frame_size: int = 480
    ) -> None:
        self._standard = standard_model
        self._low_snr = low_snr_model
        
        self._target_weights = {
            "severe": 1.0,
            "low": 0.7,
            "mid": 0.2,
            "high": 0.0
        }
        
        self._current_weight = 0.0
        self._current_state = "high"
        
        # 100ms smoothing = 10 frames at 10ms
        self._fade_step = 1.0 / 10.0

    def process_frame(self, frame: np.ndarray, snr_state: str) -> np.ndarray:
        self._current_state = snr_state
        target = self._target_weights.get(snr_state, 0.0)
        
        # If low_snr_model is missing, we gracefully degrade by forcing weight to 0
        if self._low_snr is None:
            if target > 0.0:
                # Log once per transition? To avoid spam, just silently act as 0
                pass
            target = 0.0
            
        if self._current_weight < target:
            self._current_weight = min(target, self._current_weight + self._fade_step)
        elif self._current_weight > target:
            self._current_weight = max(target, self._current_weight - self._fade_step)
            
        out_std = self._standard.process_frame(frame)
        
        if self._current_weight > 0.0 and self._low_snr is not None:
            out_lsnr = self._low_snr.process_frame(frame)
            return (out_std * (1.0 - self._current_weight) + out_lsnr * self._current_weight).astype(np.float32)
            
        return out_std

    @property
    def blend_weight(self) -> float:
        return self._current_weight
        
    @property
    def snr_state(self) -> str:
        return self._current_state
