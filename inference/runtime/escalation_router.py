"""
inference/runtime/escalation_router.py — Dynamic Acoustic Escalation Router

Orchestrates multi-model tactical execution:
- Model 4 (aegis-clf-gate): Continuously monitors acoustic environment & SNR
- Model 1 (aegis-se-primary): 0ms latency streaming enhancement for standard tactical noise
- Model 2 (aegis-se-escalation): 40ms lookahead enhancement for severe negative SNR / impulsive blasts
- Bypass Mode: Energy-saving mode when speech is clean (SNR > 25 dB)
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from training.models.model_loader import build_model_for_key


class AcousticEscalationRouter:
    """
    Intelligent dynamic routing engine between primary streaming SE,
    lookahead escalation SE, and low-power clean speech bypass.
    """

    def __init__(
        self,
        model_primary: Optional[nn.Module] = None,
        model_escalation: Optional[nn.Module] = None,
        classifier: Optional[nn.Module] = None,
        escalation_snr_threshold_db: float = 0.0,
        bypass_snr_threshold_db: float = 25.0,
        crossfade_samples: int = 240,  # 5 ms crossfade @ 48kHz
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Instantiate models if not supplied
        self.model_primary = model_primary or build_model_for_key("aegis-se-primary")
        self.model_escalation = model_escalation or build_model_for_key("aegis-se-escalation")
        self.classifier = classifier or build_model_for_key("aegis-clf-gate")

        self.model_primary.to(self.device).eval()
        self.model_escalation.to(self.device).eval()
        self.classifier.to(self.device).eval()

        self.escalation_snr_threshold_db = escalation_snr_threshold_db
        self.bypass_snr_threshold_db = bypass_snr_threshold_db
        self.crossfade_samples = crossfade_samples

        self.current_state: Optional[str] = None  # None until first audio frame is processed
        self.last_prediction: Dict[str, Union[str, float]] = {
            "mode": "primary",
            "category": "speech_dominant",
            "estimated_snr_db": 10.0,
        }

    def analyze_audio(self, audio_chunk: np.ndarray) -> Dict[str, Union[str, float]]:
        """
        Runs acoustic environment classification & fast SNR estimation on frame:
        Returns:
            Dictionary with category ('harmonic', 'impulsive', 'speech_dominant') and estimated_snr_db.
        """
        in_t = torch.from_numpy(audio_chunk).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.classifier(in_t)
            pred_idx = int(torch.argmax(logits, dim=-1).item())

        categories = ["harmonic", "impulsive", "speech_dominant"]
        category = categories[min(pred_idx, len(categories) - 1)]

        # Empirical crest factor and energy proxy for instantaneous SNR
        rms = float(np.sqrt(np.mean(audio_chunk ** 2)) + 1e-8)
        peak = float(np.max(np.abs(audio_chunk)))
        crest = peak / rms

        # Speech dominant with high dynamic crest indicates clean speech
        estimated_snr = float(10.0 * np.log10(max(crest, 1.0)) * 2.0 - 5.0)

        return {
            "category": category,
            "estimated_snr_db": estimated_snr,
        }

    def route_and_enhance(
        self,
        audio_chunk: np.ndarray,
        forced_mode: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Dynamically routes audio frame to optimal processing branch:
        Args:
            audio_chunk: 1D audio waveform array.
            forced_mode: Optional override ("bypass", "primary", "escalation").
        Returns:
            (enhanced_audio, routing_metadata)
        """
        meta = self.analyze_audio(audio_chunk)
        category = meta["category"]
        snr_est = meta["estimated_snr_db"]

        # Determine target mode
        if forced_mode is not None:
            target_mode = forced_mode
        elif category == "speech_dominant" and snr_est >= self.bypass_snr_threshold_db:
            target_mode = "bypass"
        elif category == "impulsive" or snr_est < self.escalation_snr_threshold_db:
            # Escalation triggered by extreme impulsive blasts or severe negative SNR
            target_mode = "escalation"
        else:
            target_mode = "primary"

        in_t = torch.from_numpy(audio_chunk).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            if target_mode == "bypass":
                enhanced = audio_chunk.copy()
            elif target_mode == "escalation":
                out_t = self.model_escalation(in_t)
                enhanced = out_t.squeeze().cpu().numpy()
            else:
                out_t = self.model_primary(in_t)
                enhanced = out_t.squeeze().cpu().numpy()

        # Handle smooth crossfade if mode switched from an active prior state
        if self.current_state is not None and target_mode != self.current_state and len(enhanced) >= self.crossfade_samples:
            # Generate previous mode's enhanced output for transition blend
            with torch.no_grad():
                if self.current_state == "bypass":
                    prev_enhanced = audio_chunk
                elif self.current_state == "escalation":
                    prev_out = self.model_escalation(in_t)
                    prev_enhanced = prev_out.squeeze().cpu().numpy()
                else:
                    prev_out = self.model_primary(in_t)
                    prev_enhanced = prev_out.squeeze().cpu().numpy()

            fade_in = np.linspace(0.0, 1.0, self.crossfade_samples, dtype=np.float32)
            fade_out = 1.0 - fade_in
            # Blend smoothly between previous mode's enhanced output and new mode's enhanced output
            enhanced[: self.crossfade_samples] = (
                enhanced[: self.crossfade_samples] * fade_in + prev_enhanced[: self.crossfade_samples] * fade_out
            )

        self.current_state = target_mode
        routing_info = {
            "mode": target_mode,
            "category": category,
            "estimated_snr_db": round(snr_est, 2),
        }
        self.last_prediction = routing_info

        return enhanced, routing_info
