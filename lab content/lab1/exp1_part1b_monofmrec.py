# METU EE435 Lab. Fall 2025 Experiment 1  (TA:Safa Çelik)
# Part_1b: FM Broadcast Receiver (Mono)
# Updated (Oct. 2025)

# For details:
# FM Modulation: https://en.wikipedia.org/wiki/Frequency_modulation 
# FM broadcasting: https://en.wikipedia.org/wiki/FM_broadcasting

import adi
import numpy as np
import sounddevice as sd
from scipy.signal import lfilter, bilinear, resample_poly, resample, remez, freqz
import matplotlib.pyplot as plt
import time

# Parameters:
fmStation = 102.4e6         # Choose a suitable radio station (ODTÜ Radyo: 103.1) 
sdr_sample_rate = 600e3     # Sample rate of SDR device
sdr_gain = 60               # SDR tuner gain in dB
sdr_frmlen = 20e3           # Number of samples in each capture
sim_time = 10               # Capture for 10 seconds

# Setup the ADALM-Pluto SDR receiver object
sdr_Rx = adi.Pluto("ip:192.168.2.1")   # Replace with your Pluto SDR's IP
sdr_Rx.rx_lo = int(fmStation)
sdr_Rx.sample_rate = int(sdr_sample_rate)
sdr_Rx.rx_buffer_size = int(sdr_frmlen)
sdr_Rx.rx_rf_bandwidth = int(sdr_sample_rate)

# Check if sdr_gain is defined in the global scope
if 'sdr_gain' in globals():
    sdr_Rx.gain_control_mode_chan0 = "manual"
    sdr_Rx.rx_hardwaregain_chan0 = sdr_gain
else:
    sdr_Rx.gain_control_mode_chan0 = "slow_attack" # or "fast_attack"

print(sdr_Rx.rx_lo)
print(sdr_Rx.gain_control_mode_chan0)
print(sdr_Rx.rx_hardwaregain_chan0)
print(sdr_Rx.sample_rate)
print(sdr_Rx.sample_rate)
print(sdr_Rx.rx_buffer_size)

def capture(Fs,dur_sec):

    # Calculate the total number of samples required
    total_samples = int(Fs * dur_sec)
    
    # Initialize an empty array to store the samples
    sdr_data = np.empty(total_samples, dtype=np.complex128)

    # Capture the samples
    samples_captured = 0
    while samples_captured < total_samples:
        # Read a buffer of samples
        samples = sdr_Rx.rx()
        
        # Find how many samples we need to store
        samples_to_store = min(len(samples), total_samples - samples_captured)

        # Store the samples in the array
        sdr_data[samples_captured:samples_captured + samples_to_store] = samples[:samples_to_store]

        # Update the count of captured samples
        samples_captured += samples_to_store

    return sdr_data

print("Capturing the RF signals...")
sdr_data = capture(sdr_sample_rate,sim_time) / 2**11  # 12-bits ADC scaling (1 bit is allocated for sign)

sdr_Rx.rx_destroy_buffer()

# FM receiver parameters
FrequencyDeviation = 75e3;                # Frequency deviation (modulation index)
FilterTimeConstant = 5.0e-5;              # for De-emphasizing operation
AudioSampleRate = 48e3;                   # Sample rate of the audio signal

# FM Demodulation:
def fm_demod(signal, Fs, kf):
    
    # FM demodulation
    # signal:       Input I/Q signal (complex)
    # Fs:           Sampling frequency
    # kf:           Frequency deviation
    # return:       Demodulated FM signal

    # Derivative of phi(t) is approimately same Fs*(phi(t+1)-phi(t))
    # phase = phi(t+1) - phi(t)
    multip_vec = signal * np.conj(np.hstack(([0], signal[:-1])))
    roi = np.where((multip_vec.real == 0) & (multip_vec.imag == 0))
    
    phase = np.angle(signal * np.conj(np.hstack(([0], signal[:-1]))))
    phase[roi] = 0;     # 0 / 0 is ill-defined. Such region set to be 0.     
    
    demodulated_signal = (Fs / (2*np.pi*kf)) * phase

    return demodulated_signal

print("Demodulating...")
m_demod = fm_demod(sdr_data, sdr_sample_rate, FrequencyDeviation)

# De-emphasis Filter:
def deemphasis_filter(signal, Fs, tau):
    
    # tau: Filter time constant
    # The amount of pre-emphasis and de-emphasis used is defined by the time constant of 
    # a simple RC filter circuit. In most of the world a 50 μs time constant is used. 
    # In the Americas and South Korea, 75 μs is used.

    b, a = bilinear([0, 1], [tau, 1], fs=Fs)
    return lfilter(b, a, signal, axis=0)

m_deemph = deemphasis_filter(m_demod, sdr_sample_rate, FilterTimeConstant)

# Low-pass Filter:
def lpfilter(signal, Fs):
    
    # :param signal: Input I/Q signal (complex)
    # :param Fs: Sampling frequency
    # :return: Filtered signal

    Fpass = 15e3                        # Passband frequency (15 kHz)
    Fstop = 20e3                        # Stopband frequency (20 kHz)
    transition_width = Fstop - Fpass

    rs = 60                             # Stopband attenuation in dB

    # Desired frequency bands and response
    cutoffs = [0, Fpass, Fstop, Fs/2]   # Normalized frequencies
    desired = [1, 0]                    # Passband is 1, stopband is 0

    # # Estimated filter order using a rule of thumb
    # # This is a rough estimate based on the transition width
    # # Fred Harris’ filter length approximation [Harris2021, p.59]
    # numtaps = int(np.ceil((rs * Fs) / (22 * transition_width)))
    # # Design FIR filter using remez
    # b = remez(numtaps+1, cutoffs, desired, weight=None, type='bandpass', fs=Fs)

    numtaps = 181
    # Design FIR filter using remez
    b = remez(numtaps, cutoffs, desired, weight=None, type='bandpass', fs=Fs)
    
    # # Plot frequency response
    # F, H = freqz(b, [1], worN=2048, fs=Fs,include_nyquist=True)

    # # Create a figure with subplots
    # plt.figure()
    # plt.plot(F/(Fs/2), 20 * np.log10(np.abs(H)))
    # plt.grid(True)  # Turn on the grid for this subplot
    # plt.title('Magnitude Response (in dB)')
    # plt.ylabel('Magnitude [dB]')
    # plt.xlabel('Frequency [Normalized]')
    # plt.xlim([0, 1])
    # plt.ylim([-100, 5])
    # plt.show()
    
    return lfilter(b, [1], signal, axis=0)

m_lpfilt = lpfilter(m_demod, sdr_sample_rate)

# Decimate to 48 kHz for audio playback
print("Decimating signal to 48 kHz...")
radio_audio = resample_poly(m_lpfilt, AudioSampleRate, sdr_sample_rate)
radio_audio = radio_audio / np.max(np.abs(radio_audio) + 1e-12)

print("Check whether decimating is done")
print(len(m_lpfilt))
print(len(radio_audio))

# Play the audio
print("Playing audio...")
sd.play(radio_audio, int(AudioSampleRate))
sd.wait()