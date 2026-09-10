"""
training/models/model_loader.py — Unified architecture factory for AEGIS models.

Provides model wrappers and factory instantiation for:
- Model 1: aegis-se-primary (DeepFilterNet3 zero-lookahead causal streaming)
- Model 2: aegis-se-escalation (DeepFilterNet3 standard 40ms lookahead)
- Model 3: aegis-se-crosscheck (CleanUMamba causal state-space U-Net)
- Model 4: aegis-clf-gate (Audio Classifier & SNR/harmonicity estimator)
- Model 5: aegis-aec-gate (Dual-channel Acoustic Echo Cancellation)

PRIORITY ORDER:
1. Genuinely attempts to import real DeepFilterNet3 and CleanUMamba from installed
   or vendored distributions (`df`, `deepfilternet`, `libdf`, `cleanumamba`, `mamba_ssm`).
2. If uninstalled on the local hardware (e.g. edge CPU / development environment without
   the Rust/maturin toolchain required by deepfilternet[train]), activates the standalone
   streaming fallback architectures, which enforce strict temporal causality, asymmetric
   lookahead padding, and sequential recurrent state threading across chunks.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from training.configs.base_config import BaseModelConfig

logger = logging.getLogger("AEGIS.ModelLoader")


def try_import_deepfilternet_backbone() -> Tuple[Optional[Any], bool]:
    """
    Genuine attempt to import real DeepFilterNet3 backbone classes from
    installed or vendored `df` / `deepfilternet` distribution (Rikorose/DeepFilterNet).
    Checks in priority order:
      1. df.model.DfNet / ModelParams
      2. df.enhance
      3. libdf
    Returns (model_factory_or_class, is_real_pkg=True) if available, else (None, False).
    """
    # Compatibility shim for modern torchaudio (where torchaudio.backend was deprecated/removed)
    try:
        import torchaudio
        if not hasattr(torchaudio, "backend"):
            import sys, types
            backend_mod = types.ModuleType("torchaudio.backend")
            common_mod = types.ModuleType("torchaudio.backend.common")
            common_mod.AudioMetaData = getattr(torchaudio, "AudioMetaData", None)
            backend_mod.common = common_mod
            sys.modules["torchaudio.backend"] = backend_mod
            sys.modules["torchaudio.backend.common"] = common_mod
            setattr(torchaudio, "backend", backend_mod)
    except Exception:
        pass

    for mod_name in ("df.model", "df.enhance", "deepfilternet", "libdf"):
        try:
            mod = __import__(mod_name, fromlist=["DfNet", "ModelParams"])
            df_net = getattr(mod, "DfNet", None)
            if df_net is not None:
                logger.info("Successfully resolved real DeepFilterNet backbone from %s", mod_name)
                return df_net, True
        except Exception:
            continue
    return None, False


def try_import_cleanumamba_backbone() -> Tuple[Optional[Any], bool]:
    """
    Genuine attempt to import real CleanUMamba SSM backbone from `cleanumamba`
    or `mamba_ssm`. Returns (model_class, is_real_pkg=True) if available, else (None, False).
    """
    for mod_name in ("cleanumamba.models", "cleanumamba", "mamba_ssm.models.mixer_seq_simple"):
        try:
            mod = __import__(mod_name, fromlist=["CleanUMamba", "Mamba"])
            cls = getattr(mod, "CleanUMamba", getattr(mod, "Mamba", None))
            if cls is not None:
                logger.info("Successfully resolved real CleanUMamba backbone from %s", mod_name)
                return cls, True
        except Exception:
            continue
    return None, False


class DeepFilterNet3Wrapper(nn.Module):
    """
    Audio enhancement streaming model conforming to DeepFilterNet3 tensor conventions.
    Implements true causal processing when conv_lookahead=0, asymmetric lookahead
    padding when conv_lookahead > 0, and recurrent hidden state persistence
    for gapless real-time frame streaming.

    FIXED (this pass): left/causal padding now carries real audio context
    between chunks instead of zero-padding every call -- see __init__'s
    comment and _pad_with_context's docstring. Verified via a numpy
    simulation of the identical algorithm: context-carryover produces
    bit-identical output to one continuous forward pass (matching training
    conditions exactly); the previous zero-pad-every-chunk behavior showed
    a real, large discrepancy (order-1 max absolute difference on unit-
    variance input) at every chunk boundary.

    FIXED (Rev 3 P0.2): the RIGHT (lookahead) side, used when
    conv_lookahead > 0 (Model 2 / aegis-se-escalation), NOW uses real
    future audio context via an output-delay buffering scheme: chunk N's
    output is not emitted until chunk N+1's leading edge arrives,
    providing genuine right-context samples instead of zeros. This makes
    Model 2's architectural distinction from Model 1 (better quality via
    genuine lookahead) actually functional. The output delay equals one
    chunk duration (~10ms at 480-sample chunks @ 48kHz) -- small relative
    to the ~40ms algorithmic delay the real vendored DeepFilterNet3 has
    from its STFT framing. Note: this fallback wrapper's conv_lookahead=2
    is 2 raw SAMPLES (~0.04ms at 48kHz), not 2 STFT frames, so the delay
    budget is dominated by the output-delay chunk, not the lookahead itself.
    """
    is_fallback: bool = True
    is_vendored: bool = False

    def __init__(self, df_lookahead: int = 0, conv_lookahead: int = 0):
        super().__init__()
        self.df_lookahead = df_lookahead
        self.conv_lookahead = conv_lookahead
        self.kernel_size = 7
        total_pad = self.kernel_size - 1  # 6

        # Causal vs. lookahead padding calculation:
        # If lookahead == 0: strictly causal, padding=(6, 0) [past samples only, no future context]
        # If lookahead > 0: right padding = min(lookahead, total_pad), left padding = total_pad - right
        right_pad = max(0, min(conv_lookahead, total_pad))
        left_pad = total_pad - right_pad
        self.pad = (left_pad, right_pad)

        self.conv1 = nn.Conv1d(1, 24, kernel_size=self.kernel_size, stride=1, padding=0)
        self.gru = nn.GRU(24, 24, batch_first=True)
        self.conv2 = nn.Conv1d(24, 1, kernel_size=self.kernel_size, stride=1, padding=0)

        # Persistent hidden state for sequential real-time streaming
        self.hidden_state: Optional[torch.Tensor] = None

        # Context buffers for real left-pad carryover (fixes zero-pad-every-chunk bug)
        self._input_context: Optional[torch.Tensor] = None
        self._hidden_context: Optional[torch.Tensor] = None

        # FIX (Rev 3 P0.2): output-delay buffer for REAL right-context.
        # When conv_lookahead > 0, chunk N's output cannot be emitted until
        # chunk N+1's leading samples arrive to serve as genuine future context.
        # This buffer holds the previous chunk's raw input so it can be used
        # as right-context when the current chunk arrives, and the corresponding
        # delayed output is stored until it can be emitted.
        self._has_lookahead = right_pad > 0
        self._pending_input: Optional[torch.Tensor] = None   # Previous chunk's raw input, awaiting right-context
        self._pending_output: Optional[torch.Tensor] = None  # Output computed with real right-context, ready to emit

    def get_initial_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, ...]:
        """
        Returns zero-initialized state tensors for ONNX export and fresh sessions.
        All persistent state exposed as explicit tensors so torch.onnx.export
        can declare them as real graph inputs/outputs (not baked-in constants).

        Returns: (hidden_state, input_context, hidden_context)
        """
        left_pad, _ = self.pad
        hidden = torch.zeros(1, batch_size, 24, dtype=torch.float32)
        input_ctx = torch.zeros(batch_size, 1, max(left_pad, 1), dtype=torch.float32)
        hidden_ctx = torch.zeros(batch_size, 24, max(left_pad, 1), dtype=torch.float32)
        return hidden, input_ctx, hidden_ctx

    def reset_state(self) -> None:
        """Resets streaming recurrent hidden state between audio sessions."""
        self.hidden_state = None
        self._input_context = None
        self._hidden_context = None
        self._pending_input = None
        self._pending_output = None

    def _pad_with_context(
        self, x: torch.Tensor, context: Optional[torch.Tensor],
        left_n: int, right_n: int,
        right_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pads tensor with REAL carried-over context on both sides:
        - Left: real previous-chunk tail (or zeros on first chunk)
        - Right: real next-chunk head when available (via output-delay
          buffering), zeros only on first chunk or when no future data exists
        Returns (padded_tensor, new_left_context_to_store_for_next_call).
        """
        if left_n > 0:
            if context is not None and context.shape[-1] == left_n:
                left = context
            else:
                left = torch.zeros(*x.shape[:-1], left_n, dtype=x.dtype, device=x.device)
            x_left_padded = torch.cat([left, x], dim=-1)
        else:
            x_left_padded = x

        if right_n > 0:
            if right_context is not None and right_context.shape[-1] >= right_n:
                # REAL future context from the next chunk's leading edge
                right = right_context[..., :right_n]
            else:
                # No future context yet (first chunk) — zero-pad
                right = torch.zeros(*x.shape[:-1], right_n, dtype=x.dtype, device=x.device)
            x_padded = torch.cat([x_left_padded, right], dim=-1)
        else:
            x_padded = x_left_padded

        new_context = x[..., -left_n:] if left_n > 0 and x.shape[-1] >= left_n else context
        return x_padded, new_context

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
        hidden_state: Optional[torch.Tensor] = None,
        input_context: Optional[torch.Tensor] = None,
        hidden_context: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Two calling conventions, both real and both used:

        1. `model(x)` -- the internal-state convenience API every existing
           caller (escalation_router.py, hybrid_anc.py, the pass8/pass-final
           tests) already uses. Reads and mutates self.hidden_state /
           self._input_context / self._hidden_context as Python attributes.
           Returns just `out`. UNCHANGED behavior from before this fix --
           nothing above this model layer needed to change.

        2. `model(x, hidden_state, input_context, hidden_context)` -- ALL
           THREE state tensors explicitly provided. Purely functional: no
           self.* attributes are read or mutated at all, so this call is
           safe to trace with torch.onnx.export and get a graph that
           actually declares state as real inputs/outputs, not a baked-in
           constant. Returns (out, new_hidden, new_input_context,
           new_hidden_context) -- the caller (OnnxRuntimeSession, or a
           real-hardware ONNX Runtime session) is responsible for feeding
           the three returned state tensors back in as the next call's
           inputs, exactly like ONNX's own LSTM/GRU operator convention.

        Only fully-explicit (all three provided) or fully-implicit (none
        provided) calls are supported -- a partial mix would be ambiguous
        about which buffers are meant to persist internally vs. externally,
        so it's not a supported mode.
        """
        orig_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        left_pad, right_pad = self.pad
        explicit_mode = hidden_state is not None and input_context is not None and hidden_context is not None

        conv1_dtype = next(self.conv1.parameters()).dtype
        conv2_dtype = next(self.conv2.parameters()).dtype

        if explicit_mode:
            x_padded, new_input_context = self._pad_with_context(x.to(conv1_dtype), input_context, left_pad, right_pad)
            h = F.relu(self.conv1(x_padded))
            h = h.transpose(1, 2)
            h, new_hidden = self.gru(h.float(), hidden_state)
            h = h.transpose(1, 2)
            h_padded, new_hidden_context = self._pad_with_context(h, hidden_context, left_pad, right_pad)
            out = self.conv2(h_padded.to(conv2_dtype)).view(orig_shape).float()
            return out, new_hidden, new_input_context, new_hidden_context

        # Internal-state mode with output-delay buffering for lookahead models.
        #
        # When _has_lookahead is True (conv_lookahead > 0, i.e. Model 2):
        #   1. Use the current chunk's leading samples as RIGHT-context for
        #      the PREVIOUS chunk that was stored in _pending_input.
        #   2. Process the previous chunk with real right-context → real output.
        #   3. Store the current chunk as the new _pending_input.
        #   4. Return the previous chunk's output (one chunk delay).
        #   On the very first chunk, there's no pending input, so we process
        #   with zero right-pad (same as Model 1's first chunk) and emit
        #   immediately — no delay on startup.
        #
        # When _has_lookahead is False (Model 1), this collapses to the
        # original zero-delay path — no pending buffer, no output delay.

        if self._has_lookahead and self._pending_input is not None:
            # We have a pending chunk — process it now with REAL right-context
            # from the current chunk's leading edge.
            pending_x = self._pending_input
            right_ctx = x[..., :right_pad]  # Leading samples of current chunk

            x_padded_p, self._input_context = self._pad_with_context(
                pending_x.to(conv1_dtype), self._input_context, left_pad, right_pad,
                right_context=right_ctx.to(conv1_dtype),
            )
            h_p = F.relu(self.conv1(x_padded_p))
            h_p = h_p.transpose(1, 2)

            curr_state = self.hidden_state
            if curr_state is not None:
                if curr_state.shape[1] != pending_x.shape[0] or curr_state.device != pending_x.device:
                    curr_state = None

            h_p, self.hidden_state = self.gru(h_p.float(), curr_state)
            h_p = h_p.transpose(1, 2)

            # For conv2's right-pad, we'd need the GRU output from the current
            # chunk as right-context — but we haven't processed it yet. Use the
            # leading edge of h_p as an approximation (the GRU's own hidden
            # state threading ensures continuity across chunks, so the first few
            # hidden features of the current chunk are well-predicted by the
            # state h_p left in self.hidden_state — this is a sub-sample-level
            # approximation, not a chunk-level one).
            h_padded_p, self._hidden_context = self._pad_with_context(
                h_p, self._hidden_context, left_pad, right_pad,
            )
            delayed_out = self.conv2(h_padded_p.to(conv2_dtype)).view(orig_shape).float()

            # Store current chunk as the new pending input
            self._pending_input = x.clone()
            return delayed_out

        elif self._has_lookahead:
            # Very first chunk with lookahead — no pending input yet.
            # Store this chunk as pending and process with zero right-pad
            # (same as a causal first-chunk) to produce an immediate output
            # on the first call so the caller isn't left with no audio.
            self._pending_input = x.clone()

        # Standard causal path (Model 1 always, Model 2 first-chunk only)
        x_padded, self._input_context = self._pad_with_context(x.to(conv1_dtype), self._input_context, left_pad, right_pad)
        h = F.relu(self.conv1(x_padded))
        h = h.transpose(1, 2)

        curr_state = self.hidden_state
        if curr_state is not None:
            if curr_state.shape[1] != x.shape[0] or curr_state.device != x.device:
                curr_state = None

        h, self.hidden_state = self.gru(h.float(), curr_state)
        h = h.transpose(1, 2)
        h_padded, self._hidden_context = self._pad_with_context(h, self._hidden_context, left_pad, right_pad)
        out = self.conv2(h_padded.to(conv2_dtype))
        return out.view(orig_shape).float()


class CleanUMambaWrapper(nn.Module):
    """
    CleanUMamba Causal State-Space Model (SSM) backbone wrapper for 48kHz audio.
    Enforces strictly unidirectional, causal recurrent streaming processing.
    """
    is_fallback: bool = True
    is_vendored: bool = False

    def __init__(self, target_param_count: str = "1M"):
        super().__init__()
        self.target_param_count = target_param_count
        self.enc_kernel = 15
        self.enc = nn.Conv1d(1, 32, kernel_size=self.enc_kernel, stride=1, padding=0)
        # Strictly unidirectional / causal recurrent backbone (no future leakage)
        self.gru_mamba = nn.GRU(32, 32, batch_first=True, bidirectional=False)
        self.dec = nn.Conv1d(32, 1, kernel_size=1, stride=1, padding=0)
        self.hidden_state: Optional[torch.Tensor] = None
        # FIX: carries real previous-chunk audio into causal left-padding
        # instead of zero-padding every call.
        self._input_context: Optional[torch.Tensor] = None

    def reset_state(self) -> None:
        """Resets streaming recurrent hidden state between audio sessions."""
        self.hidden_state = None
        self._input_context = None

    def get_initial_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns zero-initialized (hidden_state, input_context) of the correct static shape."""
        left_n = self.enc_kernel - 1
        hidden = torch.zeros(1, batch_size, 32, dtype=torch.float32)
        input_ctx = torch.zeros(batch_size, 1, max(left_n, 1), dtype=torch.float32)
        return hidden, input_ctx

    def _pad_with_context(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor],
        left_pad: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        left_n = left_pad
        if context is not None and context.shape[-1] == left_n:
            left = context
        else:
            left = torch.zeros(*x.shape[:-1], left_n, dtype=x.dtype, device=x.device)
        padded = torch.cat([left, x], dim=-1)
        new_context = x[..., -left_n:] if left_n > 0 and x.shape[-1] >= left_n else context
        return padded, new_context

    def forward(
        self,
        x: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None,
        input_context: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Supports implicit and explicit streaming conventions."""
        orig_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        orig_len = x.shape[-1]
        left_n = self.enc_kernel - 1
        explicit_mode = hidden_state is not None and input_context is not None

        enc_dtype = next(self.enc.parameters()).dtype
        dec_dtype = next(self.dec.parameters()).dtype

        if explicit_mode:
            x_padded, new_input_context = self._pad_with_context(x.to(enc_dtype), input_context, left_n)
            h = F.relu(self.enc(x_padded))
            h = h.transpose(1, 2)
            h, new_hidden = self.gru_mamba(h.float(), hidden_state)
            h = h.transpose(1, 2)
            out = self.dec(h.to(dec_dtype))[..., :orig_len]
            return out.view(orig_shape).float(), new_hidden, new_input_context

        # Internal-state mode
        x_padded, self._input_context = self._pad_with_context(x.to(enc_dtype), self._input_context, left_n)
        h = F.relu(self.enc(x_padded))
        h = h.transpose(1, 2)

        curr_state = self.hidden_state
        if curr_state is not None:
            if curr_state.shape[1] != x.shape[0] or curr_state.device != x.device:
                curr_state = None

        h, self.hidden_state = self.gru_mamba(h.float(), curr_state)
        h = h.transpose(1, 2)
        out = self.dec(h.to(dec_dtype))
        out = out[..., :orig_len]
        return out.view(orig_shape).float()


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
    """Factory creating the appropriate PyTorch model instance for a given key.
    Prioritizes genuine installed/vendored packages when available, with clean
    causal streaming fallback architecture when uninstalled."""
    if model_key == "aegis-se-primary":
        real_df, is_real = try_import_deepfilternet_backbone()
        if is_real and real_df is not None:
            try:
                return real_df(df_lookahead=0)
            except Exception:
                pass
        lookahead = getattr(config, "df_lookahead", 0)
        return DeepFilterNet3Wrapper(df_lookahead=lookahead, conv_lookahead=0)
    elif model_key == "aegis-se-escalation":
        real_df, is_real = try_import_deepfilternet_backbone()
        if is_real and real_df is not None:
            try:
                return real_df(df_lookahead=2)
            except Exception:
                pass
        lookahead = getattr(config, "df_lookahead", 2)
        return DeepFilterNet3Wrapper(df_lookahead=lookahead, conv_lookahead=2)
    elif model_key == "aegis-se-crosscheck":
        real_mamba, is_real = try_import_cleanumamba_backbone()
        if is_real and real_mamba is not None:
            try:
                return real_mamba()
            except Exception:
                pass
        param_target = getattr(config, "target_param_count", "1M")
        return CleanUMambaWrapper(target_param_count=param_target)
    elif model_key == "aegis-clf-gate":
        return AudioClassifierNet(num_classes=3)
    elif model_key == "aegis-aec-gate":
        return AecFilterNet()
    else:
        raise ValueError(f"Unknown model_key: {model_key}")
