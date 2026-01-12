import numpy as np
import adi
import time
sdr = adi.Pluto('ip:192.168.2.1') # Default IP

# Configure Radio Settings
sdr.sample_rate = int(25e6)          # 25 MHz sample rate
sdr.rx_rf_bandwidth = int(25e6)      # Filter width matches sample rate
sdr.rx_buffer_size = 2**16           # Standard buffer size
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 50.0     # Adjust based on environment


start_freq = 2.0e9
stop_freq = 2.1e9
step_size = 25e6
subbands = np.arange(start_freq + step_size/2, stop_freq, step_size)

# Monitoring Loop for the 10-second window
end_time = time.time() + 10 
while time.time() < end_time:
    for center_f in subbands:
        sdr.rx_lo = int(center_f)     # Retune to sub-band
        samples = sdr.rx()           # Capture IQ samples
        
        # PROCESSING: Insert your detection logic here (Energy Detection, FFT, etc.)
        # spectrum = np.abs(np.fft.fftshift(np.fft.fft(samples)))
print("Monitoring complete.")
print(samples[0:10])