import numpy as np
import matplotlib.pyplot as plt

# --- 1. Load Data ---
try:
    data = np.load("pluto_capture.npz", allow_pickle=True)
    samples_list = data['samples']
    fs = 2.5e6 
except FileNotFoundError:
    print("Error: pluto_capture.npz not found.")
    exit()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
plt.subplots_adjust(hspace=0.3)

index = 0

def update_plot(idx):
    ax1.clear()
    ax2.clear()
    
    samples = samples_list[idx]
    N = len(samples)
    
    # --- 2. Calculate PSD (Linear scale is best for integration) ---
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    psd_lin = np.abs(np.fft.fftshift(np.fft.fft(samples)))**2 / N
    
    # --- 3. Calculate Integral (Cumulative Sum) ---
    # We normalize it so the maximum value is 1.0 (100% of power)
    psd_integral = np.cumsum(psd_lin)
    psd_integral_norm = psd_integral / psd_integral[-1]

    # --- 4. Plot Top: Standard PSD (dB) ---
    psd_db = 10 * np.log10(psd_lin + 1e-12)
    ax1.plot(freqs / 1e3, psd_db, color='blue', label='PSD (dB)')
    ax1.set_ylabel("Power (dB)")
    ax1.set_title(f"Buffer {idx} | Power Spectral Density")
    ax1.grid(True, alpha=0.3)
    
    # --- 5. Plot Bottom: PSD Integral (Cumulative Power) ---
    ax2.plot(freqs / 1e3, psd_integral_norm, color='green', linewidth=2, label='Cumulative Power')
    
    # Add 99% threshold lines (useful for OBW)
    ax2.axhline(0.995, color='red', linestyle='--', alpha=0.6, label='99% Boundary')
    ax2.axhline(0.005, color='red', linestyle='--', alpha=0.6)
    
    ax2.set_ylabel("Normalized Integral (0 to 1)")
    ax2.set_xlabel("Frequency Offset (kHz)")
    ax2.set_title("Integral of PSD (Cumulative Power Distribution)")
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right')

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