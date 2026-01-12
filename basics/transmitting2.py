import time
import numpy as np
import adi

# --- Configuration ---
sample_rate = 2e6  # 2 MHz sample rate for the transmitter
center_frequencies = [2.01e9, 2.03e9, 2.05e9] # 2.01, 2.03, and 2.05 GHz
pulse_duration = 2.0  # Seconds
gap_duration = 1.0    # Seconds
amplitude = 2**14     # Signal power

# Initialize Pluto
sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(sample_rate)
sdr.tx_rf_bandwidth = int(sample_rate)
sdr.tx_hardwaregain_chan0 = -10 # Adjust power (0 is max, -80 is min)

# Create a simple Sine Wave buffer
t = np.arange(0, 0.1, 1/sample_rate) # 0.1 seconds of data
# We generate a 100 kHz tone relative to the carrier
iq_data = amplitude * (np.exp(1j * 2 * np.pi * 100000 * t))

print("Starting transmission sequence...")

try:
    for freq in center_frequencies:
        print(f"Transmitting at {freq/1e6:.2f} MHz for {pulse_duration}s...")
        sdr.tx_lo = int(freq)
        
        # Start transmitting the buffer repeatedly
        sdr.tx_cyclic_buffer = True 
        sdr.tx(iq_data)
        
        time.sleep(pulse_duration)
        
        # Stop transmission (Gap)
        print(f"Gap: 1 second silence...")
        sdr.tx_destroy_buffer()
        time.sleep(gap_duration)

    print("Sequence complete.")

except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    sdr.tx_destroy_buffer()
    del sdr