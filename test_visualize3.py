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
def find_signal_bandwidth(freqs, psd_db, peak_freq, search_range=100e3):
    """
    Finds the 99% OBW within the 100kHz masked region.
    """
    # 1. Isolate the 100kHz region in linear scale
    mask = (freqs >= (peak_freq - search_range/2)) & (freqs <= (peak_freq + search_range/2))
    f_segment = freqs[mask]
    # Convert dB back to linear for power integration
    psd_linear = 10**(psd_db[mask] / 10)
    
    # 2. Calculate Cumulative Sum of Power
    total_pow = np.sum(psd_linear)
    if total_pow == 0:
        return 0, peak_freq, peak_freq
        
    cum_pow = np.cumsum(psd_linear)
    
    # 3. Find the points where 0.5% and 99.5% of power are reached
    # (Leaving 1% total as 'out of band' noise)
    idx_low = np.where(cum_pow >= total_pow * 0.005)[0][0]
    idx_high = np.where(cum_pow >= total_pow * 0.995)[0][0]
    
    f_low = f_segment[idx_low]
    f_high = f_segment[idx_high]
    bandwidth = f_high - f_low
    
    return bandwidth, f_low, f_high
# Constants for calculation
samples_per_buffer = len(samples_list[0])
seconds_per_buffer = samples_per_buffer / fs

def get_burst_info(current_idx, samples_list, fs, threshold_db):
    """
    Looks forward and backward from current_idx to find how long 
    this specific burst lasts.
    """
    detection_status = []
    
    # 1. Pre-calculate detection for all buffers (or a window around current)
    for s in samples_list:
        psd_lin = np.abs(np.fft.fft(s))**2 / len(s)
        psd_db = 10 * np.log10(np.fft.fftshift(psd_lin) + 1e-12)
        noise_floor = np.median(psd_db)
        detection_status.append(np.max(psd_db) > (noise_floor + 10))
    
    # 2. If current isn't a signal, duration is 0
    if not detection_status[current_idx]:
        return 0, 0, 0

    # 3. Find start of burst
    start_idx = current_idx
    while start_idx > 0 and detection_status[start_idx - 1]:
        start_idx -= 1
        
    # 4. Find end of burst
    end_idx = current_idx
    while end_idx < len(detection_status) - 1 and detection_status[end_idx + 1]:
        end_idx += 1
        
    num_buffers = (end_idx - start_idx) + 1
    duration_ms = num_buffers * seconds_per_buffer * 1000
    
    return duration_ms, start_idx, end_idx


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
    is_detected = peak_val > (noise_floor_db + 40)
    plot_color = 'red' if is_detected else 'blue'
        
    # 5. Frequency Masking Logic
    # If detected, keep 100kHz range (±50kHz) around peak, set rest to noise floor
    if is_detected:
        mask_bw = 100e3
        mask = (freqs >= (peak_freq - mask_bw/2)) & (freqs <= (peak_freq + mask_bw/2))
        # Create a modified PSD for visualization
        display_psd = np.full_like(psd_db, noise_floor_db)
        display_psd[mask] = psd_db[mask]
        # Calculate real BW
        bw, f_low, f_high = find_signal_bandwidth(freqs, psd_db, peak_freq)
        
        # Highlight the detected bandwidth on the plot
        ax.axvspan(f_low/1e3, f_high/1e3, color='yellow', alpha=0.2, label=f'BW: {bw/1e3:.1f} kHz')
        ax.text(peak_freq/1e3, peak_val + 5, f"{bw/1e3:.1f} kHz", 
                color='red', fontweight='bold', ha='center')
    else:
        display_psd = psd_db

    # 6. Plotting
    ax.plot(freqs / 1e3, display_psd, color=plot_color, linewidth=1.2)
    
    # Visual cues for the threshold and peak
    ax.axhline(noise_floor_db + 10, color='green', linestyle='--', alpha=0.5, label='Threshold')
    if is_detected:
        ax.scatter(peak_freq/1e3, peak_val, color='red', s=40, zorder=5)
        ax.annotate(f"{peak_val:.1f} dB", (peak_freq/1e3, peak_val + 2))

    ax.set_ylim([noise_floor_db - 10, noise_floor_db + 100]) # Dynamic Y-axis
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