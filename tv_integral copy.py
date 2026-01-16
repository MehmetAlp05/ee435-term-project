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
    
    # --- 2. Calculate PSD ---
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    psd_lin = np.abs(np.fft.fftshift(np.fft.fft(samples)))**2 / N
    
    # --- 3. Calculate Normalized Integral ---
    psd_integral = np.cumsum(psd_lin)
    psd_integral_norm = psd_integral / (psd_integral[-1] + 1e-12)

    # --- 4. 50% Jump Detection Logic ---
    # We check if there is any 100kHz window where the integral increases > 0.5
    window_size_hz = 100e3
    # Number of bins that make up 100kHz
    bins_in_100k = int(window_size_hz / (fs / N))
    
    # Calculate the difference over the sliding window
    # diff_array[i] = integral[i + bins] - integral[i]
    if len(psd_integral_norm) > bins_in_100k:
        jumps = psd_integral_norm[bins_in_100k:] - psd_integral_norm[:-bins_in_100k]
        max_jump = np.max(jumps)
        max_jump_idx = np.argmax(jumps) + (bins_in_100k // 2)
        is_detected = max_jump > 0.5
    else:
        max_jump = 0
        is_detected = False

    # --- 5. Plotting ---
    psd_db = 10 * np.log10(psd_lin + 1e-12)
    color = 'red' if is_detected else 'blue'
    
    # Top Plot
    ax1.plot(freqs / 1e3, psd_db, color=color)
    ax1.set_ylabel("Power (dB)")
    ax1.set_title(f"Buffer {idx} | Signal Detected: {is_detected} (Max Jump: {max_jump*100:.1f}%)")
    ax1.grid(True, alpha=0.3)
    
    # Bottom Plot
    ax2.plot(freqs / 1e3, psd_integral_norm, color='green', linewidth=2)
    if is_detected:
        # Highlight the detected jump area
        ax2.axvspan((freqs[max_jump_idx] - 50), (freqs[max_jump_idx] + 50), 
                    color='yellow', alpha=0.3, label='50% Jump Zone')
    
    ax2.set_ylabel("Normalized Integral")
    ax2.set_xlabel("Frequency Offset (kHz)")
    ax2.axhline(0.5, color='black', linestyle=':', alpha=0.5)
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.draw()

print("Use Left/Right arrow keys to navigate.")

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