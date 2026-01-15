import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.load("pluto_capture.npz")
all_samples = data['samples'] # This is a 2D array [block_index, samples]
fs = 2.5e6

# Flatten the blocks into one long continuous signal for visualization
continuous_signal = all_samples.flatten()

plt.figure(figsize=(12, 8))

# --- Plot 1: Time Domain (Magnitude) ---
plt.subplot(2, 1, 1)
t = np.arange(len(continuous_signal)) / fs
plt.plot(t, np.abs(continuous_signal))
plt.title("Time Domain: Signal Magnitude (See the 15 Bursts)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

# --- Plot 2: Spectrogram (Frequency vs Time) ---
plt.subplot(2, 1, 2)
plt.specgram(continuous_signal, NFFT=2048, Fs=fs, noverlap=1024, cmap='viridis')
plt.title("Spectrogram: Frequency vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Frequency Offset (Hz)")
plt.ylim(-1e6, 1e6) # Zoom into the +/- 800kHz range

plt.tight_layout()
plt.show()