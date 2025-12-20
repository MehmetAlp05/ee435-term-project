# METU EE435 Lab. Fall 2025 Experiment 1  (TA:Safa Çelik)
# Part_2: Spectrum Sweep for Ankara
# Updated (Oct. 2025)

# At the end of the simulation, the recorded data will be processed and plotted in a popup figure

import numpy as np
import os
import adi
import matplotlib.pyplot as plt 
from scipy.signal import firwin, lfilter, filtfilt
from scipy import signal

# PARAMETERS (can change)
location = 'Ankara'        # Location used for figure name
start_freq = 700e6         # Sweep start frequency (Pluto SDR supports tuning range 70 – 6000 MHz)
stop_freq  = 900e6         # Sweep stop frequency
sdr_fs = 2.8e6             # Pluto SDR sampling rate in Hz
sdr_gain = 30              # Pluto SDR tuner gain in dB
sdr_frmlen = 4096          # Pluto SDR output data frame size

# PARAMETERS (can change, but may break code)
nfrmhold = 20              # number of frames to receive
fft_hold = 'avg'           # hold function "max" or "avg"
nfft = 4096                # number of points in FFTs (2^something)
dec_factor = 16            # output plot downsample
overlap = 0.5              # FFT overlap to counter rolloff
nfrmdump = 100             # number of frames to dump after retuning (to clear buffer)

# Create tuner frequency range
sdr_tunerfreq = np.arange(start_freq, stop_freq, sdr_fs * overlap)

# Check if the whole range is covered, if not, add an extra tuner frequency
if max(sdr_tunerfreq) < stop_freq:
    sdr_tunerfreq = np.append(sdr_tunerfreq, max(sdr_tunerfreq) + sdr_fs * overlap)

# Calculate number of retunes required
nretunes = len(sdr_tunerfreq)
# Calculate frequency bin width
freq_bin_width = sdr_fs / nfft
# Create the frequency axis
freq_axis = np.arange(sdr_tunerfreq[0] - sdr_fs/2 * overlap, 
                      (sdr_tunerfreq[-1] + sdr_fs/2 * overlap) - freq_bin_width, 
                      freq_bin_width * dec_factor) / 1e6  # in MHz

# Setup Pluto SDR receiver
sdr_Rx = adi.Pluto('ip:192.168.2.1')  # Replace with your Pluto SDR's IP
sdr_Rx.rx_lo = int(sdr_tunerfreq[0])  # Center frequency
sdr_Rx.sample_rate = int(sdr_fs)      # Sampling rate
sdr_Rx.rx_buffer_size = sdr_frmlen    # Buffer size
sdr_Rx.rx_rf_bandwidth = int(sdr_fs)  # Convert bandwidth to int

# Check if sdr_gain is defined in the global scope
if 'sdr_gain' in globals():
    sdr_Rx.gain_control_mode_chan0 = "manual"
    sdr_Rx.rx_hardwaregain_chan0 = sdr_gain  # Set gain
else:
    sdr_Rx.gain_control_mode_chan0 = "slow_attack" # or "fast_attack"

print(sdr_Rx.rx_lo)
print(sdr_Rx.gain_control_mode_chan0)
print(sdr_Rx.rx_hardwaregain_chan0)
print(sdr_Rx.sample_rate)
print(sdr_Rx.sample_rate)
print(sdr_Rx.rx_buffer_size)

# FIR Decimator
num_taps = 301
fir_filter = firwin(numtaps=num_taps, cutoff = 1/dec_factor, window="hamming")
fir_filter /= np.sum(fir_filter)

sdr_data_fft = np.zeros(int(nfft))                                       # Buffer to hold FFT data
fft_reorder = np.zeros((int(nfrmhold), int(nfft * overlap)))             # Re-ordered FFT buffer
fft_dec = np.zeros((int(nretunes), int( nfft * overlap // dec_factor)))  # Decimated FFT data

# Create progress variable
tune_progress = 0

# For each of the tuner values
for ntune in range(nretunes):
    
    # Simulate tuning the SDR to a new center frequency
    sdr_Rx.rx_lo = int(sdr_tunerfreq[ntune])
    
    # Dump frames to clear software buffer
    for _ in range(nfrmdump):
        sdr_data = np.empty(1, dtype=np.complex128)
        sdr_data = sdr_Rx.rx() / (2**11)
    
    # display current centre frequency
    print(f"            fc = {sdr_tunerfreq[ntune] / 1e6} MHz")
    
    # Loop for nfrmhold frames
    for frm in range(nfrmhold):
        
        filename = os.path.join("received_datas", str(ntune+1) + "_" + str(frm+1) + ".txt")

        sdr_data = np.empty(1, dtype=np.complex128)
        sdr_data = sdr_Rx.rx() / (2**11)  # Normalize the data to be between -1 and 1
        
        # Remove DC component
        sdr_data = sdr_data - np.mean(sdr_data)
        
        # Compute FFT and take absolute value
        sdr_data_fft = np.abs(np.fft.fft(sdr_data, nfft))

        # Rearrange FFT with overlap compensation
        half_nfft = int(nfft / 2)
        fft_reorder[int(frm), : int(overlap * half_nfft)] = sdr_data_fft[int(overlap * half_nfft + half_nfft) :]    # -ve
        fft_reorder[int(frm), int(overlap * half_nfft) :] = sdr_data_fft[: int(overlap * half_nfft)]                # +ve
    
    # Process the FFT data (average or max)
    if fft_hold == 'avg':
        fft_reorder_proc = np.mean(fft_reorder, axis=0)
    elif fft_hold == 'max':
        fft_reorder_proc = np.max(fft_reorder, axis=0)

    # First, pass the signal from a low pass filter
    fft_reorder_proc_lp = filtfilt(fir_filter, 1, fft_reorder_proc)

    # Now apply downsampling
    fft_reorder_proc_downsampled = fft_reorder_proc_lp[::dec_factor]
    fft_dec[ntune, :] = fft_reorder_proc_downsampled

    # Show progress if at a 10% step
    current_progress = int((ntune + 1) * 10 / nretunes)
    if current_progress != tune_progress:
        tune_progress = current_progress
        print(f"      progress = {tune_progress * 10}%")

# Reorder into one matrix
fft_masterreshape = fft_dec.reshape(-1)

# Plot data
freq_axis = np.arange(
    sdr_tunerfreq[0] - (sdr_fs / 2 * overlap),
    (sdr_tunerfreq[-1] + sdr_fs / 2 * overlap) - freq_bin_width,
    freq_bin_width * dec_factor
) / 1e6  # Convert to MHz

# Convert to dBm for plotting
# y_data_dbm = 10 * np.log10((fft_masterreshape**2) / 50)
y_data_dbm = 10 * np.log10((fft_masterreshape**2) / 1)

# Create a figure with two subplots (1 row, 2 columns)
# Format the title string using f-strings
title_str = (f'Pluto SDR Spectrum Sweep   ||   Range = {start_freq / 1e6:.2f} MHz to {stop_freq / 1e6:.2f} MHz   ||   '
             f'Bin Width = {freq_bin_width * dec_factor / 1e3:.2f} kHz   ||   '
             f'Number of Bins = {len(freq_axis)}   ||   Number of Retunes = {nretunes}')
xlims = [int(start_freq/1e6), int(stop_freq/1e6)]

fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
fig.suptitle(title_str, fontsize=9, family='monospace')

axs[0].plot(freq_axis, y_data_dbm, color='blue', linewidth=1.25)
axs[0].set_xlabel('Frequency (MHz)', family='monospace')
axs[0].set_ylabel('Power (dBm) [relative]', family='monospace')
axs[0].grid(True, which='major', linestyle='-', linewidth=1)  # Add detailed grid
axs[0].set_xlim(xlims)

axs[1].plot(freq_axis, fft_masterreshape, color='orange', linewidth=1.25)
axs[1].set_xlabel('Frequency (MHz)', family='monospace')
axs[1].set_ylabel('Power (linear) [relative]', family='monospace')
axs[1].grid(True, which='major', linestyle='-', linewidth=1)  # Add detailed grid
axs[1].set_xlim(xlims)

# Adjust layout for better spacing
plt.tight_layout()

# Save data
filename = f'sdr_Rx_specsweep_{start_freq / 1e6:.2f}MHz_{stop_freq / 1e6:.2f}MHz_{location}.jpeg'

# Save the figure using plt.savefig()
plt.savefig(filename)

# Display the plots
plt.show()