import numpy as np
import matplotlib.pyplot as plt

# Load your recorded data
data = np.load("pluto_capture.npz", allow_pickle=True)
samples_list = data['samples']
times_list = data['times']
fs = 2.5e6

fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.2)

index = 0

def update_plot(idx):
    ax.clear()
    samples = samples_list[idx]
    
    # 1. Calculate PSD
    N = len(samples)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    # Use Welch-like scaling for a more stable PSD
    psd_linear = np.abs(np.fft.fftshift(np.fft.fft(samples)))**2 / N
    psd_db = 10 * np.log10(psd_linear + 1e-12)
    
    # 2. Noise Floor Calculation
    # We use median to ensure the signal itself doesn't pull the floor up
    noise_floor_db = np.median(psd_db)
    
    # 3. Peak Detection
    peak_idx = np.argmax(psd_db)
    peak_val = psd_db[peak_idx]
    peak_freq = freqs[peak_idx]
    
    # 4. Detection Logic (Threshold: Noise Floor + 10dB)
    is_detected = peak_val > (noise_floor_db + 10)
    plot_color = 'red' if is_detected else 'blue'
    
    # 5. Frequency Masking Logic
    # If detected, keep 100kHz range (±50kHz) around peak, set rest to noise floor
    if is_detected:
        mask_bw = 100e3
        mask = (freqs >= (peak_freq - mask_bw/2)) & (freqs <= (peak_freq + mask_bw/2))
        # Create a modified PSD for visualization
        display_psd = np.full_like(psd_db, noise_floor_db)
        display_psd[mask] = psd_db[mask]
    else:
        display_psd = psd_db

    # 6. Plotting
    ax.plot(freqs / 1e3, display_psd, color=plot_color, linewidth=1.2)
    
    # Visual cues for the threshold and peak
    ax.axhline(noise_floor_db + 10, color='green', linestyle='--', alpha=0.5, label='Threshold')
    if is_detected:
        ax.scatter(peak_freq/1e3, peak_val, color='red', s=40, zorder=5)
        ax.annotate(f"{peak_val:.1f} dB", (peak_freq/1e3, peak_val + 2))

    ax.set_ylim([noise_floor_db - 10, noise_floor_db + 50]) # Dynamic Y-axis
    ax.set_title(f"Buffer {idx} | Signal Detected: {is_detected} | Color: {plot_color}")
    ax.set_xlabel("Frequency Offset (kHz)")
    ax.set_ylabel("Power (dB)")
    ax.legend(loc='upper right')
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.draw()

print("Use Left/Right arrow keys to navigate buffers.")

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