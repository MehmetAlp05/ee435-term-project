import sys
import numpy as np
import adi
import time
from scipy.signal import (firwin, lfilter, kaiserord, find_peaks)
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QHBoxLayout, QLabel, QFileDialog, QStatusBar,
    QLineEdit
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QCursor

##############################################################################
# FIR filter design function
##############################################################################
def design_filter(sample_rate, cutoff_hz=400e3):
    """
    Designs a low-pass FIR filter using a Kaiser window.
    - sample_rate: in Hz
    - cutoff_hz: cutoff frequency in Hz
    """
    nyq_rate = sample_rate / 2.0
    # Transition width for the filter
    width = 10e3 / nyq_rate
    ripple_db = 180
    N_filt, beta_filt = kaiserord(ripple_db, width)
    b_filt = firwin(N_filt, cutoff_hz / nyq_rate, window=('kaiser', beta_filt))
    return b_filt
# Connect to the Pluto SDR
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

def detect_bursts(samples, fs, threshold_db=-40):
    # Compute the FFT and convert to dB
    fft_data = np.fft.fftshift(np.fft.fft(samples))
    psd = 20 * np.log10(np.abs(fft_data) / len(samples))
    
    # Identify bins above the threshold
    active_bins = psd > threshold_db
    
    if np.any(active_bins):
        # Estimate Bandwidth and Center Frequency
        indices = np.where(active_bins)[0]
        bw_hz = (indices[-1] - indices[0]) * (fs / len(samples))
        fc_offset = (indices.mean() - len(samples)/2) * (fs / len(samples))
        return True, fc_offset, bw_hz
    return False, 0, 0

import time

# Data structure to hold detected signals [cite: 36]
detected_signals = [] 
start_monitoring = time.time()

while (time.time() - start_monitoring) < 10:  # 10s window [cite: 61]
    for center_f in subbands:
        sdr.rx_lo = int(center_f)
        samples = sdr.rx()
        
        timestamp = time.time() - start_monitoring
        is_active, offset, bw = detect_bursts(samples, sdr.sample_rate)
        
        if is_active:
            # Record initial detection [cite: 26, 28, 29]
            signal_entry = {
                't_start_s': timestamp,
                'fc_hz': center_f + offset,
                'bw_hz': bw,
                'subband_start_hz': center_f - (sdr.sample_rate / 2)
            }
            detected_signals.append(signal_entry)
            # Logic for modulation classification would follow here [cite: 31]
import pandas as pd
from datetime import datetime

# Generate filename: EE435_TeamXX_YYYY-MM-DDThh-mm-ssZ.csv [cite: 37]
timestamp_str = datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%SZ')
filename = f"EE435_Team01_{timestamp_str}.csv"

df = pd.DataFrame(detected_signals)
# Ensure all required columns are present [cite: 36]
df.to_csv(filename, index=False, encoding='utf-8')

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
    # Thresholds may need tuning based on your SNR [cite: 32, 42]
    if env_var < 0.1:
        # Check for LFM (Chirp) by looking for linear frequency change
        phase = np.unwrap(np.angle(samples))
        inst_freq = np.diff(phase)
        # If the frequency plot is a straight line, it's LFM 
        # Otherwise, classify as FM [cite: 13]
        return "FM" 
        
    else:
        # It is an AM variant [cite: 11, 12]
        # Calculate ratio between sidebands
        sideband_ratio = max(lsb_power, usb_power) / (min(lsb_power, usb_power) + 1e-9)
        
        if sideband_ratio > 10: # If one side is >10x stronger than the other
            return "SSB" #[cite: 12]
        else:
            return "DSB-SC" #[cite: 11]




import pandas as pd
from datetime import datetime
import os

def generate_report(detected_signals, team_id="XX"):
    # Define columns as per project spec 
    columns = [
        "run_id", "burst_id", "t_start_s", "t_end_s", "fc_hz", 
        "bw_hz", "modulation", "confidence", "snr_db", 
        "subband_start_hz", "subband_bw_hz", "notes"
    ]
    
    df = pd.DataFrame(detected_signals, columns=columns)
    
    # Generate filename: EE435_TeamXX_YYYY-MM-DDThh-mm-ssZ.csv 
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%SZ')
    filename = f"EE435_Team{team_id}_{timestamp}.csv"
    
    # Save as UTF-8 [cite: 35]
    df.to_csv(filename, index=False, encoding='utf-8')
    return filename