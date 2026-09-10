"""
inference/engines/onnx_model_adapter.py — Connects the ONNX deployment
engine to the application layer.

THE GAP THIS CLOSES: OnnxRuntimeSession (onnx_runtime_engine.py) and the
stateful export fix (onnx_engine.py, this project's most severe finding)
had no real caller anywhere in the application. Repo-wide search
confirmed `live_mic_anc.py` and `enhance_audio.py` -- the only two real
inference entry points -- always build raw PyTorch models via
`build_model_for_key()` and call them directly; neither ever loads or
runs an exported ONNX model. The ONNX/embedded-deployment layer, despite
being explicitly built for "NVIDIA Jetson AGX Orin, embedded x86/ARM
platforms" per its own docstrings, was an island: real, tested, correct
in isolation, and never actually reachable from a real run of this
system.

THE FIX: rather than duplicate AcousticEscalationRouter's routing logic
in an ONNX-specific copy (a maintenance hazard -- two implementations of
the same decision logic drifting apart is exactly the class of bug this
whole review keeps finding), this adapter makes an OnnxRuntimeSession
LOOK like a PyTorch nn.Module from the router's point of view: same
call signature (`__call__(torch.Tensor) -> torch.Tensor`), same
`reset_state()` method, same `hidden_state` attribute presence check via
hasattr. AcousticEscalationRouter, HybridAncPipeline, and
StatefulHopProcessor all need ZERO changes to work with ONNX-backed
models -- pass instances of this adapter wherever a PyTorch model was
passed before.
"""

from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch

from inference.engines.onnx_runtime_engine import OnnxRuntimeSession


class OnnxModelAdapter:
    """
    Wraps an OnnxRuntimeSession to present the same interface
    AcousticEscalationRouter/HybridAncPipeline already call PyTorch
    models through: `model(tensor) -> tensor`, `model.reset_state()`,
    and a `hidden_state` attribute existence check (used by
    escalation_router.py's `hasattr(self.model_primary, "reset_state")`
    guards).
    """

    def __init__(self, onnx_model_path: Union[str, Path], providers: Optional[list] = None):
        self._session = OnnxRuntimeSession(str(onnx_model_path), execution_providers=providers)
        # Presence of this attribute (even as None) is what
        # escalation_router.py's own model_primary/model_escalation
        # objects expose -- kept here so any code that inspects it
        # (rather than just calling reset_state()) doesn't need a
        # separate ONNX-vs-PyTorch branch.
        self.hidden_state = None if self._session.is_stateful else "n/a-stateless-onnx-model"

    def reset_state(self) -> None:
        """Matches DeepFilterNet3Wrapper/CleanUMambaWrapper's reset_state()
        convention -- delegates to the real session-level reset."""
        self._session.reset_state()
        self.hidden_state = None if self._session.is_stateful else "n/a-stateless-onnx-model"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accepts and returns torch.Tensor, matching every existing caller's
        expectation (escalation_router.py builds `in_t = torch.from_numpy(
        audio_chunk).float().unsqueeze(0)` and calls `self.model_primary(in_t)`
        directly) -- the numpy round-trip ONNX Runtime actually needs
        happens inside this adapter, invisibly to the caller.
        """
        was_batched = x.dim() > 1
        np_input = x.detach().cpu().numpy().astype(np.float32)
        if np_input.ndim == 1:
            np_input = np_input[np.newaxis, :]

        np_output = self._session.forward(np_input)

        out = torch.from_numpy(np_output)
        if not was_batched:
            out = out.squeeze(0)
        return out

    def eval(self) -> "OnnxModelAdapter":
        """No-op, matching nn.Module.eval()'s call signature -- an ONNX
        Runtime session has no train/eval mode distinction, but callers
        (e.g. AcousticEscalationRouter.__init__) call .eval() unconditionally
        on whatever model object they're given."""
        return self

    def to(self, device) -> "OnnxModelAdapter":
        """No-op, matching nn.Module.to(device)'s call signature -- device
        placement for ONNX Runtime is controlled by the `providers` list
        passed at construction (CPU/CUDA/TensorRT execution provider),
        not by a post-hoc .to() call. Accepted here only so router
        construction code that unconditionally calls `.to(self.device)`
        on every model doesn't need an ONNX-specific branch."""
        return self


def build_onnx_backed_router(
    primary_onnx_path: Union[str, Path],
    escalation_onnx_path: Union[str, Path],
    classifier_onnx_path: Union[str, Path],
    providers: Optional[list] = None,
    **router_kwargs,
):
    """
    Convenience constructor: builds an AcousticEscalationRouter backed by
    exported ONNX models instead of raw PyTorch ones -- the actual,
    concrete way to run this project's real-time pipeline on real
    embedded/Jetson hardware via ONNX Runtime, which did not exist
    anywhere in the application before this fix.
    """
    from inference.runtime.escalation_router import AcousticEscalationRouter

    return AcousticEscalationRouter(
        model_primary=OnnxModelAdapter(primary_onnx_path, providers=providers),
        model_escalation=OnnxModelAdapter(escalation_onnx_path, providers=providers),
        classifier=OnnxModelAdapter(classifier_onnx_path, providers=providers),
        **router_kwargs,
    )
