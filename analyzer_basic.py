import numpy as np
import adi
import time
from scipy.signal import hilbert, find_peaks

# --- 1. CONFIGURATION ---
fs = 2.5e6        # 2.5 MHz to cover the 1.6 MHz transmitter span
fc = 2e9          # Fixed center frequency
sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(fs)
sdr.rx_lo = int(fc)
time.sleep(0.1)  # Allow settings to take effect
sdr.rx_buffer_size = 2**14 # ~6.5ms buffers for 40ms bursts

captured_data = []
threshold = -45 # Adjust based on your lab's noise floor (dB)

print("Capturing 10 seconds of data...")
start_time = time.time()
while time.time() - start_time < 3:
    samples = sdr.rx()
    ts = time.time() - start_time
    captured_data.append((samples, ts))

print(f"Captured {len(captured_data)} buffers. Processing...")
np.savez("pluto_capture.npz", samples=raw_samples, times=timestamps)
print(f"Saved {len(raw_samples)} blocks to pluto_capture.npz")
# --- 2. PROCESSING ---
results = []
for i, (samples, ts) in enumerate(captured_data):
    # Calculate PSD
    psd = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples)))**2 / len(samples))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(samples), 1/fs))
    
    if np.max(psd) > threshold:
        # Find the peak
        peak_idx = np.argmax(psd)
        f_shift = freqs[peak_idx]
        
        # --- 3. MODULATION CLASSIFICATION ---
        # Envelope for AM-class vs FM-class
        env = np.abs(samples)
        env_var = np.var(env / np.mean(env))
        
        # Instantaneous Phase for FM/LFM
        phase = np.unwrap(np.angle(samples))
        inst_freq = np.diff(phase)
        
        mod_type = "Unknown"
        if env_var > 0.1: # High amplitude variation
            # Check SSB by comparing upper/lower sidebands
            left_pwr = np.sum(psd[:len(psd)//2])
            right_pwr = np.sum(psd[len(psd)//2:])
            if abs(10*np.log10(left_pwr/right_pwr)) > 10:
                mod_type = "SSB"
            elif np.mean(env) < 0.2: # Low carrier energy
                mod_type = "DSB-SC"
            else:
                mod_type = "AM"
        else: # Low amplitude variation = FM or LFM
            # Check if frequency is changing linearly (LFM)
            if np.std(np.diff(inst_freq)) < np.std(inst_freq):
                mod_type = "LFM"
            else:
                mod_type = "FM"
        
        results.append({
            'time': ts,
            'f_shift': f_shift,
            'mod': mod_type
        })
#print(results)
# --- 4. CONSOLIDATION (Merge consecutive hits into one burst) ---
# Logic: If hits are within 100ms and have same freq, they are one burst.