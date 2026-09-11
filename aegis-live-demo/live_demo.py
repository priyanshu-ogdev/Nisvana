"""
Project AEGIS — Phase A Live Demo Pipeline
Status: Minimal Viable Demo (No AEC, No SNR state fusion, just pure DeepFilterNet3)

Requires:
    pip install deepfilternet sounddevice numpy
"""
import sys
import queue
import time
import argparse
import numpy as np
import sounddevice as sd

try:
    from df.enhance import enhance, init_df
    from df.utils import get_norm_alpha
except ImportError:
    print("ERROR: deepfilternet is not installed or failed to load.")
    print("Please install via: pip install deepfilternet")
    sys.exit(1)

# DeepFilterNet operates natively at 48kHz
FS = 48000
FRAME_MS = 10  # 10ms frames are standard for DF
FRAME_SIZE = int(FS * (FRAME_MS / 1000.0))  # 480 samples

def query_devices():
    """Print available audio devices for the user to select from."""
    print(sd.query_devices())
    print("\nUse the device indices listed above for --input-device and --output-device")

def main(args):
    print("=== AEGIS Phase A Live Demo ===")
    print(f"Initializing DeepFilterNet3 (Sample Rate: {FS}Hz, Frame: {FRAME_MS}ms / {FRAME_SIZE} samples)...")
    
    try:
        model, df_state, _ = init_df()
        # Pre-warm the model with a silent frame
        silent_frame = np.zeros(FRAME_SIZE, dtype=np.float32)
        _ = enhance(model, df_state, silent_frame)
    except Exception as e:
        print(f"Failed to initialize DeepFilterNet: {e}")
        return

    print("Model initialized successfully.\n")

    # Audio queues
    q_in = queue.Queue()
    
    def audio_callback(indata, outdata, frames, time_info, status):
        """
        Called by sounddevice for each audio block.
        We must not block in this callback.
        """
        if status:
            print(f"Stream status: {status}", file=sys.stderr)
            
        # We need exactly FRAME_SIZE samples. The stream blocksize handles this.
        # DeepFilterNet expects shape (1, N) or (N,) -> enhance takes torch tensor or numpy array.
        # df.enhance.enhance actually expects a torch tensor of shape (ch, samples),
        # but the wrapper `enhance` might handle numpy. Let's be careful.
        
        # Actually, df.enhance expects a torch.Tensor, but there's a helper or we can just convert.
        # Let's pass the raw indata to the main thread queue so the callback is perfectly fast.
        q_in.put(indata.copy())
        
        # For a truly zero-latency feel, we'd output the processed data from a previous frame here.
        # But for this simple 300-line script, if we block inside the callback it might glitch.
        # Let's do the processing directly in the callback for absolute minimum latency, 
        # as long as the CPU can keep up with <10ms inference.
        pass

    # Actually, a simpler approach for the demo that guarantees sync is a blocking read/write loop,
    # or a callback that does the processing inline (if the Mac CPU is fast enough).
    # Let's try inline callback first for simplicity, falling back to queue if it underruns.

    def inline_callback(indata, outdata, frames, time_info, status):
        if status:
            print(f"Status: {status}", file=sys.stderr)
            
        t0 = time.perf_counter()
        
        # 1. Convert to torch tensor (DeepFilterNet expects float32 torch tensor)
        import torch
        # indata is (frames, channels), usually (480, 1)
        audio_ch0 = indata[:, 0]
        audio_tensor = torch.from_numpy(audio_ch0).unsqueeze(0).to(torch.float32) # Shape: (1, 480)
        
        # 2. Process
        try:
            enhanced = enhance(model, df_state, audio_tensor)
            # 3. Convert back to numpy
            enhanced_np = enhanced.squeeze(0).numpy()
            
            # Write to output (duplicate mono to all output channels if stereo)
            for ch in range(outdata.shape[1]):
                outdata[:, ch] = enhanced_np
                
        except Exception as e:
            print(f"Inference error: {e}")
            outdata[:] = 0.0

        latency_ms = (time.perf_counter() - t0) * 1000
        # Send latency to main thread for printing (so we don't print in audio thread)
        try:
            q_in.put_nowait(latency_ms)
        except queue.Full:
            pass

    print(f"Starting audio stream (Input: {args.input_device}, Output: {args.output_device})...")
    print("Speak into the microphone. You should hear the cleaned audio in your headphones.")
    print("Press Ctrl+C to stop.\n")

    try:
        with sd.Stream(
            device=(args.input_device, args.output_device),
            samplerate=FS,
            blocksize=FRAME_SIZE,
            dtype='float32',
            channels=(1, 2), # 1 channel in, 2 channels out (stereo headphones)
            callback=inline_callback,
            latency='low'
        ):
            frames_processed = 0
            while True:
                # Main thread just prints stats
                try:
                    latency = q_in.get(timeout=1.0)
                    frames_processed += 1
                    if frames_processed % 50 == 0:  # Print every ~0.5 seconds
                        print(f"Running... [Inference latency: {latency:.2f} ms / frame budget: 10.00 ms]")
                except queue.Empty:
                    pass

    except KeyboardInterrupt:
        print("\nStopping demo.")
    except Exception as e:
        print(f"\nStream error: {e}")
        print("Hint: Check your device indices with --list-devices")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AEGIS Live Demo")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("-i", "--input-device", type=int, default=None, help="Input device ID (mic)")
    parser.add_argument("-o", "--output-device", type=int, default=None, help="Output device ID (headphones)")
    
    args = parser.parse_args()
    
    if args.list_devices:
        query_devices()
        sys.exit(0)
        
    main(args)
