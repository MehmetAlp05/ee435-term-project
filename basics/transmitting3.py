import time
import numpy as np
import adi

# -----------------------------
# Configuration
# -----------------------------
sample_rate = 2_000_000          # 2 MS/s
tx_lo = 2_010_000_000             # 2.01 GHz RF center
tx_gain = -10                     # dB (adjust as needed)

tone_freqs = [2.01e9, 2.03e9, 2.05e9]  # Hz (baseband offsets)
tone_duration = 2.0                # seconds
silence_duration = 1.0             # seconds

buffer_size = 262144               # TX buffer size

# -----------------------------
# Initialize PlutoSDR
# -----------------------------
sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = sample_rate
sdr.tx_rf_bandwidth = sample_rate
sdr.tx_lo = tx_lo
sdr.tx_hardwaregain_chan0 = tx_gain
sdr.tx_cyclic_buffer = True
sdr.tx_buffer_size = buffer_size

print("Transmitter initialized")

# -----------------------------
# Helper functions
# -----------------------------
def generate_tone(freq, duration):
    n = int(sample_rate * duration)
    t = np.arange(n) / sample_rate
    tone = np.exp(1j * 2 * np.pi * freq * t)
    return tone.astype(np.complex64)

def transmit_iq(iq):
    sdr.tx(iq)

def transmit_silence(duration):
    zeros = np.zeros(int(sample_rate * duration), dtype=np.complex64)
    sdr.tx(zeros)
    time.sleep(duration)

# -----------------------------
# Transmission sequence
# -----------------------------
try:
    for i, f in enumerate(tone_freqs):
        print(f"TX tone {i+1}: {f/1e3:.1f} kHz for {tone_duration}s")
        iq = generate_tone(f, tone_duration)
        transmit_iq(iq)
        time.sleep(tone_duration)

        if i < len(tone_freqs) - 1:
            print("Silence")
            sdr.tx_destroy_buffer()
            transmit_silence(silence_duration)
            
finally:
    # Stop transmission cleanly
    sdr.tx_destroy_buffer()
    del sdr
    print("Transmission complete")