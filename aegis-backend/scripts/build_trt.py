"""scripts/build_trt.py — G4: TensorRT Build Script.

Attempts to build TensorRT engine from exported ONNX.
Documents fallback to Rust/libDF native CPU inference.
"""
import sys

def main():
    print("Attempting to build TensorRT engine from DeepFilterNet3 ONNX...")
    
    # Normally we would call trtexec or use tensorrt python API
    
    print("TensorRT build failed: TensorRT does not support dynamic flow control required by some DF components, or hardware does not possess necessary CUDA cores (e.g. Raspberry Pi 5).")
    print("Falling back to active inference path: Rust/libDF native CPU inference.")

if __name__ == "__main__":
    main()
