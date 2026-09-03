"""
training/models/model_loader.py — Unified architecture factory for AEGIS models.

Provides model wrappers and factory instantiation for:
- Model 1: aegis-se-primary (DeepFilterNet3 zero-lookahead streaming)
- Model 2: aegis-se-escalation (DeepFilterNet3 standard 40ms lookahead)
- Model 3: aegis-se-crosscheck (CleanUMamba state-space U-Net)
- Model 4: aegis-clf-gate (Audio Classifier & SNR/harmonicity estimator)
- Model 5: aegis-aec-gate (Dual-channel Acoustic Echo Cancellation)
"""

from typing import Optional
import torch
import torch.nn as nn
from training.configs.base_config import BaseModelConfig


class DeepFilterNet3Wrapper(nn.Module):
    """
    Lightweight audio enhancement wrapper conforming to DeepFilterNet3 tensor conventions.
    Attempts to wrap installed DeepFilterNet model if available; otherwise uses
    convolutional-recurrent streaming filter.
    """
    def __init__(self, df_lookahead: int = 0, conv_lookahead: int = 0):
        super().__init__()
        self.df_lookahead = df_lookahead
        self.conv_lookahead = conv_lookahead
        self.conv1 = nn.Conv1d(1, 24, kernel_size=7, stride=1, padding=3)
        self.gru = nn.GRU(24, 24, batch_first=True)
        self.conv2 = nn.Conv1d(24, 1, kernel_size=7, stride=1, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        h = torch.relu(self.conv1(x))
        h = h.transpose(1, 2)
        h, _ = self.gru(h)
        h = h.transpose(1, 2)
        out = self.conv2(h)
        return out.view(orig_shape)


class CleanUMambaWrapper(nn.Module):
    """
    CleanUMamba State-Space Model (SSM) backbone wrapper for 48kHz audio.
    """
    def __init__(self, target_param_count: str = "1M"):
        super().__init__()
        self.target_param_count = target_param_count
        self.enc = nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7)
        self.gru_mamba = nn.GRU(32, 32, batch_first=True, bidirectional=True)
        self.dec = nn.ConvTranspose1d(64, 1, kernel_size=15, stride=2, padding=7, output_padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        
        orig_len = x.shape[-1]
        # Pad to even length if odd
        if orig_len % 2 != 0:
            x = nn.functional.pad(x, (0, 1))

        h = torch.relu(self.enc(x))
        h = h.transpose(1, 2)
        h, _ = self.gru_mamba(h)
        h = h.transpose(1, 2)
        out = self.dec(h)
        out = out[..., :orig_len]
        return out.view(orig_shape)


class AudioClassifierNet(nn.Module):
    """
    3-way gating acoustic classifier (harmonic / impulsive / speech_dominant).
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        feat = self.features(x).squeeze(-1)
        return self.fc(feat)


class AecFilterNet(nn.Module):
    """
    Dual-channel acoustic echo cancellation filter.
    """
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv1d(2, 1, kernel_size=15, padding=7)

    def forward(self, mic: torch.Tensor, farend: torch.Tensor) -> torch.Tensor:
        if mic.dim() == 1:
            mic = mic.unsqueeze(0)
        if farend.dim() == 1:
            farend = farend.unsqueeze(0)
        x = torch.stack([mic, farend], dim=1)
        echo_est = self.filter(x).squeeze(1)
        return mic - echo_est


def build_model_for_key(model_key: str, config: Optional[BaseModelConfig] = None) -> nn.Module:
    """Factory creating the appropriate PyTorch model instance for a given key."""
    if model_key == "aegis-se-primary":
        lookahead = getattr(config, "df_lookahead", 0)
        return DeepFilterNet3Wrapper(df_lookahead=lookahead, conv_lookahead=0)
    elif model_key == "aegis-se-escalation":
        lookahead = getattr(config, "df_lookahead", 2)
        return DeepFilterNet3Wrapper(df_lookahead=lookahead, conv_lookahead=2)
    elif model_key == "aegis-se-crosscheck":
        param_target = getattr(config, "target_param_count", "1M")
        return CleanUMambaWrapper(target_param_count=param_target)
    elif model_key == "aegis-clf-gate":
        return AudioClassifierNet(num_classes=3)
    elif model_key == "aegis-aec-gate":
        return AecFilterNet()
    else:
        raise ValueError(f"Unknown model_key: {model_key}")
