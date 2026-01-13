import numpy as np
from scipy.fft import fft, fftfreq, fftshift
import adi
import time
sdr = adi.Pluto('ip:192.168.2.1') # Default IP

# Configure Radio Settings
sdr.sample_rate = int(25e6)          # 25 MHz sample rate
sdr.rx_rf_bandwidth = int(25e6)      # Filter width matches sample rate
sdr.rx_buffer_size = 2**16           # Standard buffer size
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 50.0     # Adjust based on environment

def detect_bursts(samples, fs, threshold_db=-40):
    # Compute the FFT and convert to dB
    fft_data = np.fft.fftshift(np.fft.fft(samples))
    psd = 20 * np.log10(np.abs(fft_data) / len(samples))
    psd = welch
    # Identify bins above the threshold
    active_bins = psd > threshold_db
    
    if np.any(active_bins):
        # Estimate Bandwidth and Center Frequency
        indices = np.where(active_bins)[0]
        bw_hz = (indices[-1] - indices[0]) * (fs / len(samples))
        fc_offset = (indices.mean() - len(samples)/2) * (fs / len(samples))
        return True, fc_offset, bw_hz
    return False, 0, 0
start_freq = 2.0e9
stop_freq = 2.1e9
step_size = 25e6
subbands = np.arange(start_freq + step_size/2, stop_freq, step_size)

# Monitoring Loop for the 10-second window
end_time = time.time() + 10 
while time.time() < end_time:
    #within 10 seconds approximately 10 scanning
    for center_f in subbands:
        sdr.rx_lo = int(center_f)    # Retune to sub-band
        samples = sdr.rx()           # Capture IQ samples
        
        # PROCESSING: Insert your detection logic here (Energy Detection, FFT, etc.)
        # spectrum = np.abs(np.fft.fftshift(np.fft.fft(samples)))
        detect_bursts(samples, sdr.sample_rate)

print("Monitoring complete.")
print(samples[0:10])


def classify_and_estimate(samples, fs, lo_freq):
    # 1. Frequency Domain Processing
    n = len(samples)
    spec = fftshift(fft(samples))
    psd_db = 20 * np.log10(np.abs(spec) / n + 1e-12)
    freq_axis = fftshift(fftfreq(n, 1/fs))
    
    # 2. Peak Detection (Find the strongest signal in the buffer)
    # distance=n//10 ensures we don't pick multiple peaks for one wide signal
    peaks, props = find_peaks(psd_db, height=-65, distance=n//10)
    
    if len(peaks) == 0:
        return None

    # We'll analyze the primary (strongest) peak for this example
    best_peak = peaks[np.argmax(props['peak_heights'])]
    
    # 3. Parameter Estimation [cite: 28, 29, 32]
    fc_hz = lo_freq + freq_axis[best_peak]
    snr_db = psd_db[best_peak] - np.median(psd_db)
    
    # Bandwidth: Calculate width where power is > -10dB from peak
    mask = psd_db > (psd_db[best_peak] - 10)
    bw_hz = (np.sum(mask) / n) * fs

    # 4. Feature Extraction using SciPy 
    analytic = hilbert(samples)
    envelope = np.abs(analytic)
    # Normalized variance: High for AM, Low for FM [cite: 20]
    env_var = np.var(envelope) / (np.mean(envelope)**2)
    
    # 5. Spectral Symmetry (SSB vs DSB-SC) [cite: 11, 12, 20]
    halfway = n // 2
    lsb_power = np.sum(np.abs(spec[:halfway])**2)
    usb_power = np.sum(np.abs(spec[halfway:])**2)
    sideband_ratio = max(lsb_power, usb_power) / (min(lsb_power, usb_power) + 1e-9)

    # 6. Final Decision Logic [cite: 31]
    if env_var < 0.05: # Threshold for constant envelope
        mod_type = "FM"
        confidence = min(0.99, snr_db / 40)
    elif sideband_ratio > 8: # Significant power in only one sideband
        mod_type = "SSB"
        confidence = min(0.95, sideband_ratio / 20)
    else:
        mod_type = "DSB-SC"
        confidence = 0.85 # Default for symmetric AM

    return {
        'fc_hz': fc_hz,
        'bw_hz': bw_hz,
        'modulation': mod_type,
        'snr_db': snr_db,
        'confidence': confidence
    }