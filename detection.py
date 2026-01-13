
import numpy as np

def classify_modulation(samples, fs):
    # 1. Calculate Envelope (Instantaneous Amplitude)
    envelope = np.abs(samples)
    
    # Normalized Variance of the envelope
    # FM and LFM have constant envelopes (low variance)
    # AM variants have high variance
    env_var = np.var(envelope) / (np.mean(envelope)**2)
    
    # 2. Spectral Analysis (FFT)
    n = len(samples)
    fft_data = np.fft.fftshift(np.fft.fft(samples))
    freqs = np.fft.fftshift(np.fft.fftfreq(n, 1/fs))
    
    # Split spectrum into Lower Sideband (LSB) and Upper Sideband (USB)
    # Assuming the signal is centered in the capture
    halfway = n // 2
    lsb_power = np.sum(np.abs(fft_data[:halfway])**2)
    usb_power = np.sum(np.abs(fft_data[halfway:])**2)
    
    # 3. Decision Logic
    # Thresholds may need tuning based on your SNR 
    if env_var < 0.1:
        # Check for LFM (Chirp) by looking for linear frequency change
        phase = np.unwrap(np.angle(samples))
        inst_freq = np.diff(phase)
        # If the frequency plot is a straight line, it's LFM 
        # Otherwise, classify as FM 
        return "FM" 
        
    else:
        # It is an AM variant 
        # Calculate ratio between sidebands
        sideband_ratio = max(lsb_power, usb_power) / (min(lsb_power, usb_power) + 1e-9)
        
        if sideband_ratio > 10: # If one side is >10x stronger than the other
            return "SSB" 
        else:
            return "DSB-SC" 

##########################Maybe helpful for later###########
##########################
import numpy as np

def extract_features(samples):
    # Amplitude Envelope
    envelope = np.abs(samples)
    env_var = np.var(envelope) / (np.mean(envelope)**2) # Normalized variance

    # Instantaneous Frequency
    phase = np.unwrap(np.angle(samples))
    inst_freq = np.diff(phase)
    freq_var = np.var(inst_freq)

    return env_var, freq_var, inst_freq

