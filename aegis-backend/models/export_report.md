# Export Report

## ONNX Export
**Attempt 1:** Native `torch.onnx.export`
**Result:** FAILED
**Error:** `RuntimeError: ONNX export failed: Exporting the operator 'aten::stft' to ONNX opset version 14 is not supported.`

**Attempt 2:** Conv1d-based STFT/ISTFT decomposition
**Result:** SUCCESS

## TensorRT Build
**Attempt 1:** Build engine from ONNX
**Result:** FAILED (No CUDA cores on target Raspberry Pi 5 / Unsupported dynamic flow).

## Active Inference Path
**Fallback:** Rust/libDF native CPU inference via `onnxruntime` CPUExecutionProvider or native bindings.
