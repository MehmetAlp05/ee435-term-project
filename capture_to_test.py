import numpy as np
import adi
import time

# Configuration
fs = 2.5e6
fc = 2e9
sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(fs)
sdr.rx_lo = int(fc)
sdr.rx_buffer_size = 2**16 # Larger buffer is better for saving

raw_samples = []
timestamps = []

print("Recording 10 seconds of raw IQ...")
start_time = time.time()

try:
    while time.time() - start_time < 10:
        samples = sdr.rx()
        raw_samples.append(samples)
        timestamps.append(time.time() - start_time)
finally:
    # Save as a compressed numpy file to save space/time
    np.savez("pluto_capture.npz", samples=raw_samples, times=timestamps)
    print(f"Saved {len(raw_samples)} blocks to pluto_capture.npz")