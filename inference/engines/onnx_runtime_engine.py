"""
inference/engines/onnx_runtime_engine.py — Native ONNXRuntime Engine for Edge Deployment

Provides accelerated inference execution using ONNX Runtime with
TensorRT, CUDA, and CPU execution providers.
"""

from pathlib import Path
from typing import List, Optional, Union
import numpy as np
import onnxruntime as ort


class OnnxRuntimeSession:
    """
    High-performance ONNX Runtime inference wrapper for edge AI hardware
    (NVIDIA Jetson AGX Orin, embedded x86/ARM platforms).
    """

    def __init__(
        self,
        onnx_model_path: Union[str, Path],
        execution_providers: Optional[List[str]] = None,
        intra_op_num_threads: int = 4,
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

    def forward(self, input_audio: np.ndarray) -> np.ndarray:
        """
        Runs low-latency inference on input audio array.
        Args:
            input_audio: 1D or 2D audio array (float32).
        Returns:
            Enhanced audio array (float32).
        """
        input_audio = np.ascontiguousarray(input_audio, dtype=np.float32)
        if input_audio.ndim == 1:
            input_audio = np.expand_dims(input_audio, axis=0)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_audio},
        )
        return outputs[0]

    def __call__(self, input_audio: np.ndarray) -> np.ndarray:
        return self.forward(input_audio)
