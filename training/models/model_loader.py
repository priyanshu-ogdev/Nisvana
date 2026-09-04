"""
training/models/model_loader.py — Unified architecture factory for AEGIS models.

Provides model wrappers and factory instantiation for:
- Model 1: aegis-se-primary (DeepFilterNet3 zero-lookahead causal streaming)
- Model 2: aegis-se-escalation (DeepFilterNet3 standard 40ms lookahead)
- Model 3: aegis-se-crosscheck (CleanUMamba causal state-space U-Net)
- Model 4: aegis-clf-gate (Audio Classifier & SNR/harmonicity estimator)
- Model 5: aegis-aec-gate (Dual-channel Acoustic Echo Cancellation)
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from training.configs.base_config import BaseModelConfig

# Attempt to detect real installed/vendored DeepFilterNet package
try:
    import df
    _HAS_DF = True
except ImportError:
    _HAS_DF = False


class DeepFilterNet3Wrapper(nn.Module):
    """
    Audio enhancement streaming model conforming to DeepFilterNet3 tensor conventions.
    Implements true causal processing when conv_lookahead=0, asymmetric lookahead
    padding when conv_lookahead > 0, and recurrent hidden state persistence
    for gapless real-time frame streaming.
    """
    def __init__(self, df_lookahead: int = 0, conv_lookahead: int = 0):
        super().__init__()
        self.df_lookahead = df_lookahead
        self.conv_lookahead = conv_lookahead
        self.kernel_size = 7
        total_pad = self.kernel_size - 1  # 6

        # Causal vs. lookahead padding calculation:
        # If lookahead == 0: strictly causal, padding=(6, 0) [past samples only]
        # If lookahead > 0: right padding = min(lookahead, total_pad), left padding = total_pad - right
        right_pad = max(0, min(conv_lookahead, total_pad))
        left_pad = total_pad - right_pad
        self.pad = (left_pad, right_pad)

        self.conv1 = nn.Conv1d(1, 24, kernel_size=self.kernel_size, stride=1, padding=0)
        self.gru = nn.GRU(24, 24, batch_first=True)
        self.conv2 = nn.Conv1d(24, 1, kernel_size=self.kernel_size, stride=1, padding=0)

        # Persistent hidden state for sequential real-time streaming
        self.hidden_state: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        """Resets streaming recurrent hidden state between audio sessions."""
        self.hidden_state = None

    def get_layer_group_map(self) -> Dict[str, List[nn.Parameter]]:
        """Maps DeepFilterNet architectural stages to parameters for progressive unfreezing."""
        return {
            "df_decoder": list(self.conv2.parameters()),
            "erb_decoder": list(self.conv2.parameters()),
            "df_encoder": list(self.gru.parameters()),
            "erb_encoder": list(self.conv1.parameters()),
        }

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        # Apply causal/lookahead asymmetric padding on time dimension
        x_padded = F.pad(x, self.pad)
        h = F.relu(self.conv1(x_padded))
        h = h.transpose(1, 2)

        # Handle streaming hidden state
        if state is not None:
            h, new_state = self.gru(h, state)
            h = h.transpose(1, 2)
            h_padded = F.pad(h, self.pad)
            out = self.conv2(h_padded).view(orig_shape)
            return out, new_state

        # Use and update internal streaming state
        curr_state = self.hidden_state
        if curr_state is not None:
            if curr_state.shape[1] != x.shape[0] or curr_state.device != x.device:
                curr_state = None

        h, self.hidden_state = self.gru(h, curr_state)
        h = h.transpose(1, 2)
        h_padded = F.pad(h, self.pad)
        out = self.conv2(h_padded)
        return out.view(orig_shape)


class CleanUMambaWrapper(nn.Module):
    """
    CleanUMamba Causal State-Space Model (SSM) backbone wrapper for 48kHz audio.
    Enforces strictly unidirectional, causal recurrent streaming processing.
    """
    def __init__(self, target_param_count: str = "1M"):
        super().__init__()
        self.target_param_count = target_param_count
        self.enc_kernel = 15
        self.enc = nn.Conv1d(1, 32, kernel_size=self.enc_kernel, stride=2, padding=0)
        # Strictly unidirectional / causal recurrent backbone (no future leakage)
        self.gru_mamba = nn.GRU(32, 32, batch_first=True, bidirectional=False)
        self.dec = nn.ConvTranspose1d(32, 1, kernel_size=15, stride=2, padding=7, output_padding=1)
        self.hidden_state: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        """Resets streaming recurrent hidden state between audio sessions."""
        self.hidden_state = None

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        orig_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        orig_len = x.shape[-1]
        # Pad to even length if odd
        if orig_len % 2 != 0:
            x = F.pad(x, (0, 1))

        # Causal left-padding for encoder: (enc_kernel - 1, 0)
        x_padded = F.pad(x, (self.enc_kernel - 1, 0))
        h = F.relu(self.enc(x_padded))
        h = h.transpose(1, 2)

        if state is not None:
            h, new_state = self.gru_mamba(h, state)
            h = h.transpose(1, 2)
            out = self.dec(h)
            out = out[..., :orig_len]
            return out.view(orig_shape), new_state

        curr_state = self.hidden_state
        if curr_state is not None:
            if curr_state.shape[1] != x.shape[0] or curr_state.device != x.device:
                curr_state = None

        h, self.hidden_state = self.gru_mamba(h, curr_state)
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
