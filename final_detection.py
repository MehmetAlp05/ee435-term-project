import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Load your recorded data
data = np.load("pluto_capture.npz", allow_pickle=True)
samples_list = data['samples']
time_list = data['times'] 
fs = 2.5e6

# --- 1. GLOBAL BURST PRE-ANALYSIS ---
print("Scanning all buffers to calculate burst durations...")
samples_per_buffer = len(samples_list[0])
buffer_duration = samples_per_buffer / fs
window_size_hz = 100e3
##detectionsss
detection_map = []
for samples in samples_list:
    N = len(samples)
    psd_lin = np.abs(np.fft.fft(samples))**2 / N
    psd_int = np.cumsum(psd_lin)
    psd_int_norm = psd_int / (psd_int[-1] + 1e-12)
    bins_in_100k = int(window_size_hz / (fs / N))
    
    if len(psd_int_norm) > bins_in_100k:
        jumps = psd_int_norm[bins_in_100k:] - psd_int_norm[:-bins_in_100k]
        detection_map.append(np.max(jumps) > 0.5)
    else:
        detection_map.append(False)

detection_map = np.array(detection_map).astype(int)
diff = np.diff(np.concatenate(([0], detection_map, [0])))
starts = np.where(diff == 1)[0]
ends = np.where(diff == -1)[0] - 1

burst_info = {}
for i in range(len(starts)):
    t_start = time_list[starts[i]]
    t_end = time_list[ends[i]]
    duration_ms = (t_end - t_start + buffer_duration) * 1000
    for idx in range(starts[i], ends[i] + 1):
        burst_info[idx] = {'id': i + 1, 'duration': duration_ms, 'range': (starts[i], ends[i])}

# --- 2. SETUP FIGURE ---
fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
plt.subplots_adjust(bottom=0.15)
index = 0

def burst_analyze(idx):
    ax1.clear()
    samples = samples_list[idx]
    N = len(samples)
    
    # --- 3. PSD CALCULATION ---
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    psd_lin = np.abs(np.fft.fftshift(np.fft.fft(samples)))**2 / N
    psd_db = 10 * np.log10(psd_lin + 1e-12)
    psd_smoothed = savgol_filter(psd_db, window_length=51, polyorder=3)
    
    # --- 4. NOISE FLOOR CALCULATION ---
    # Median is used to prevent the signal peaks from skewing the floor value
    noise_floor = np.median(psd_db)
    
    # --- 5. DETECTION LOGIC ---
    is_detected = detection_map[idx]
    
    # --- 6. PLOTTING ---
    # Plot the noise floor as a reference line
    ax1.axhline(y=noise_floor, color='black', linestyle='--', alpha=0.7, label=f'Noise Floor ({noise_floor:.1f} dB)')
    
    if is_detected:
        ax1.plot(freqs / 1e3, psd_smoothed, color='green', linewidth=1.5, label='Signal Detected')
        info = burst_info[idx]
        title_str = (f"Buffer {idx} | BURST #{info['id']} | "
                     f"Duration: {info['duration']:.2f} ms")
        ax1.set_title(title_str, color='red', fontweight='bold', fontsize=14)
    else:
        ax1.plot(freqs / 1e3, psd_db, color='silver', alpha=0.5, label='Raw PSD')
        ax1.plot(freqs / 1e3, psd_smoothed, color='blue', linewidth=1.2, label='Smoothed PSD')
        ax1.set_title(f"Buffer {idx} | NO SIGNAL", color='black')

    # Add a threshold line (Noise Floor + 10dB) to show why signals are detected
    ax1.axhline(y=noise_floor + 10, color='orange', linestyle=':', alpha=0.6, label='Detection Threshold')

    ax1.set_ylabel("Power (dB)")
    ax1.set_xlabel("Frequency Offset (kHz)")
    
    # Set dynamic Y-limits based on noise floor for better visibility
    ax1.set_ylim([noise_floor - 10, noise_floor + 80])
    
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')
    plt.draw()

print("Use Left/Right arrow keys to browse buffers.")

def on_key(event):
    global index
    if event.key == 'right':
        index = (index + 1) % len(samples_list)
    elif event.key == 'left':
        index = (index - 1) % len(samples_list)
    burst_analyze(index)

fig.canvas.mpl_connect('key_press_event', on_key)
burst_analyze(0)
plt.show()