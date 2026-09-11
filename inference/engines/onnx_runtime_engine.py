"""
inference/engines/onnx_runtime_engine.py — Native ONNXRuntime Engine for Edge Deployment

Provides accelerated inference execution using ONNX Runtime with
TensorRT, CUDA, and CPU execution providers.

SOTA Upgrades (Phase 3/5):
- IO Binding: pre-allocates input/output buffers on device memory,
  eliminating host↔device copies per frame (the dominant latency
  source for small-chunk real-time audio on GPU).
- Model Warm-Up: runs N silence frames at session start to stabilize
  hidden states and JIT-compile execution provider kernels before
  real audio arrives.
- Latency Instrumentation: per-chunk timing for RTF monitoring.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import time
import numpy as np
import onnxruntime as ort


class OnnxRuntimeSession:
    """
    High-performance ONNX Runtime inference wrapper for edge AI hardware
    (NVIDIA DGX Spark GB10, Jetson AGX Orin, embedded x86/ARM platforms).
    """

    def __init__(
        self,
        onnx_model_path: Union[str, Path],
        execution_providers: Optional[List[str]] = None,
        intra_op_num_threads: int = 4,
        enable_io_binding: bool = True,
        warmup_frames: int = 10,
        warmup_frame_size: int = 480,
    ):
        self.model_path = Path(onnx_model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        # Configure session options for minimal edge latency
        self.opts = ort.SessionOptions()
        self.opts.intra_op_num_threads = intra_op_num_threads
        self.opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Available execution providers in priority order
        available = ort.get_available_providers()
        if execution_providers is None:
            execution_providers = []
            if "TensorrtExecutionProvider" in available:
                execution_providers.append("TensorrtExecutionProvider")
            if "CUDAExecutionProvider" in available:
                execution_providers.append("CUDAExecutionProvider")
            execution_providers.append("CPUExecutionProvider")
        else:
            # Filter to only providers actually available in this environment
            execution_providers = [p for p in execution_providers if p in available]
            if not execution_providers:
                execution_providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=self.opts,
            providers=execution_providers,
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.active_provider = self.session.get_providers()[0]

        # Stateful model detection (see export_to_onnx's docstring)
        all_inputs = self.session.get_inputs()
        all_outputs = self.session.get_outputs()
        self.state_input_names = [inp.name for inp in all_inputs[1:]]
        self.state_output_names = [out.name for out in all_outputs[1:]]
        self.is_stateful = len(self.state_input_names) > 0
        self._state_shapes = [
            [d if isinstance(d, int) else 1 for d in inp.shape] for inp in all_inputs[1:]
        ]
        self._current_state: Optional[List[np.ndarray]] = None
        if self.is_stateful:
            self.reset_state()

        # IO Binding: pre-allocate device-side buffers to eliminate
        # host↔device copies per frame. Only available when a GPU EP is
        # active — on CPU, standard session.run() is already zero-copy.
        self._use_io_binding = (
            enable_io_binding
            and self.active_provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider")
        )
        self._io_binding = None
        self._io_bound_input_shape: Optional[tuple] = None

        # Latency instrumentation
        self._last_forward_ms: float = 0.0
        self._total_forward_calls: int = 0
        self._total_forward_ms: float = 0.0

        # Model warm-up: stabilize hidden states and JIT-compile EP kernels
        if warmup_frames > 0:
            self._warmup(warmup_frames, warmup_frame_size)

    def _warmup(self, n_frames: int, frame_size: int) -> None:
        """Runs N silence frames through the model to warm up hidden states
        and trigger any lazy kernel compilation in the execution provider.
        Called once at construction — not part of the per-chunk critical path."""
        silence = np.zeros((1, frame_size), dtype=np.float32)
        for _ in range(n_frames):
            self.forward(silence)
        # Reset state after warm-up so real audio starts from a clean slate
        if self.is_stateful:
            self.reset_state()
        # Reset latency counters so warm-up frames don't skew stats
        self._last_forward_ms = 0.0
        self._total_forward_calls = 0
        self._total_forward_ms = 0.0

    def reset_state(self) -> None:
        """Resets persisted state to zeros -- call this at the start of a
        new audio session, exactly like the PyTorch-side reset_state()
        convention this mirrors."""
        if not self.is_stateful:
            return
        self._current_state = [np.zeros(shape, dtype=np.float32) for shape in self._state_shapes]

    def _setup_io_binding(self, input_audio: np.ndarray) -> None:
        """Creates or refreshes IO binding buffers. Only called when shape
        changes (rare in real-time streaming where chunk size is fixed)."""
        self._io_binding = self.session.io_binding()

        # The audio callback owns a host NumPy buffer.  It must be bound as a
        # CPU input; treating its pointer as CUDA memory is invalid and causes
        # ORT to throw on every frame before falling back to session.run().
        # ORT still schedules the graph on the selected execution provider.
        self._io_binding.bind_cpu_input(self.input_name, input_audio)

        # Let ORT allocate the output on the active provider and copy it back
        # once via copy_outputs_to_cpu().
        self._io_binding.bind_output(self.output_name)

        # Bind state inputs/outputs for stateful models
        if self.is_stateful and self._current_state is not None:
            for name, state_arr in zip(self.state_input_names, self._current_state):
                self._io_binding.bind_cpu_input(name, state_arr)
            for name in self.state_output_names:
                self._io_binding.bind_output(name)

        self._io_bound_input_shape = tuple(input_audio.shape)

    def forward(self, input_audio: np.ndarray) -> np.ndarray:
        """
        Runs low-latency inference on input audio array. If the loaded
        graph is stateful, automatically threads state between calls --
        the caller does not need to manage state tensors manually.

        When IO binding is enabled and a GPU EP is active, input/output
        buffers are pre-allocated on device memory — eliminating the
        host↔device copy that otherwise dominates latency for small
        (480-sample / 10ms) audio chunks.

        Args:
            input_audio: 1D or 2D audio array (float32).
        Returns:
            Enhanced audio array (float32).
        """
        input_audio = np.ascontiguousarray(input_audio, dtype=np.float32)
        if input_audio.ndim == 1:
            input_audio = np.expand_dims(input_audio, axis=0)

        t0 = time.perf_counter()

        if self._use_io_binding:
            try:
                # Refresh binding only when input shape changes
                if self._io_bound_input_shape != tuple(input_audio.shape):
                    self._setup_io_binding(input_audio)
                else:
                    # Refresh the host input binding for this callback buffer.
                    self._io_binding.bind_cpu_input(self.input_name, input_audio)
                    if self.is_stateful and self._current_state is not None:
                        for name, state_arr in zip(self.state_input_names, self._current_state):
                            self._io_binding.bind_cpu_input(name, state_arr)

                self.session.run_with_iobinding(self._io_binding)
                outputs = self._io_binding.copy_outputs_to_cpu()

                t1 = time.perf_counter()
                self._record_latency(t0, t1)

                if self.is_stateful and len(outputs) > 1:
                    self._current_state = list(outputs[1:])
                return outputs[0]

            except Exception:
                # Fallback to standard run() if IO binding fails
                # (e.g. ORT version mismatch, unsupported EP feature)
                self._use_io_binding = False

        # Standard session.run() path (CPU, or IO-binding fallback)
        if not self.is_stateful:
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_audio},
            )
            t1 = time.perf_counter()
            self._record_latency(t0, t1)
            return outputs[0]

        feed = {self.input_name: input_audio}
        for name, value in zip(self.state_input_names, self._current_state):
            feed[name] = value

        outputs = self.session.run(
            [self.output_name] + self.state_output_names,
            feed,
        )
        self._current_state = list(outputs[1:])

        t1 = time.perf_counter()
        self._record_latency(t0, t1)
        return outputs[0]

    def _record_latency(self, t0: float, t1: float) -> None:
        """Records per-chunk inference latency for RTF monitoring."""
        ms = (t1 - t0) * 1000.0
        self._last_forward_ms = ms
        self._total_forward_calls += 1
        self._total_forward_ms += ms

    def get_latency_stats(self) -> Dict[str, float]:
        """Returns cumulative latency statistics for performance monitoring."""
        avg_ms = self._total_forward_ms / max(self._total_forward_calls, 1)
        return {
            "last_forward_ms": round(self._last_forward_ms, 3),
            "avg_forward_ms": round(avg_ms, 3),
            "total_calls": self._total_forward_calls,
            "total_ms": round(self._total_forward_ms, 2),
            "active_provider": self.active_provider,
            "io_binding_active": self._use_io_binding,
        }

    def __call__(self, input_audio: np.ndarray) -> np.ndarray:
        return self.forward(input_audio)
