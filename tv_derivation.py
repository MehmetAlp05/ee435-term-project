import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# --- 1. Load Data ---
try:
    data = np.load("pluto_capture.npz", allow_pickle=True)
    samples_list = data['samples']
    fs = 2.5e6  # Adjust if your capture rate was different
except FileNotFoundError:
    print("Error: pluto_capture.npz not found.")
    exit()

# --- 2. Setup Figure ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
plt.subplots_adjust(hspace=0.3, bottom=0.15)

index = 0

def update_plot(idx):
    ax1.clear()
    ax2.clear()
    
    samples = samples_list[idx]
    N = len(samples)
    
    # --- 3. PSD Calculation ---
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    # Linear PSD
    psd_lin = np.abs(np.fft.fftshift(np.fft.fft(samples)))**2 / N
    # Logarithmic PSD (dB)
    psd_db = 10 * np.log10(psd_lin + 1e-12)
    
    # --- 4. Smoothing & Derivation ---
    # Window length must be odd. 51 is good for 2.5MHz span.
    window_len = 51 
    psd_smoothed = savgol_filter(psd_db, window_length=window_len, polyorder=3)
    
    # Derivation: Change in dB per kHz
    # np.gradient calculates the slope between neighboring points
    df = (freqs[1] - freqs[0]) / 1e3  # freq step in kHz
    deriv = np.gradient(psd_smoothed, df)

    # --- 5. Plot Top: PSD ---
    ax1.plot(freqs / 1e3, psd_db, color='silver', alpha=0.5, label='Raw PSD')
    ax1.plot(freqs / 1e3, psd_smoothed, color='red', linewidth=1.5, label='Smoothed PSD')
    ax1.set_ylabel("Power (dB)")
    ax1.set_title(f"Buffer {idx} | Power Spectral Density")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    
    # --- 6. Plot Bottom: Derivation ---
    ax2.plot(freqs / 1e3, deriv, color='purple', linewidth=1.2, label='d(PSD)/df')
    ax2.axhline(0, color='black', linestyle='--', alpha=0.6) # Zero Crossing line
    ax2.set_ylabel("Gradient (dB/kHz)")
    ax2.set_xlabel("Frequency Offset (kHz)")
    ax2.set_title("PSD Derivation (Rate of Change)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')

    # Highlight peaks (where derivative crosses zero from positive to negative)
    # This is a high-precision peak detection trick
    plt.draw()

print("Use Left/Right arrow keys to browse buffers.")

def on_key(event):
    global index
    if event.key == 'right':
        index = (index + 1) % len(samples_list)
    elif event.key == 'left':
        index = (index - 1) % len(samples_list)
    update_plot(index)

fig.canvas.mpl_connect('key_press_event', on_key)
update_plot(0)
plt.show()