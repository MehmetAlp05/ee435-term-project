import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Load your recorded data
data = np.load("pluto_capture.npz", allow_pickle=True)
samples_list = data['samples']
time_list = data['times'] 
fs = 2.5e6

# --- 1. GLOBAL BURST ANALYSIS (For Duration) ---
print("Scanning all buffers to calculate burst durations...")
samples_per_buffer = len(samples_list[0])
buffer_duration = samples_per_buffer / fs
window_size_hz = 100e3
##detectionsss
detection_map = []
for samples in samples_list:
    N=len(samples)
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
starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0] - 1

burst_info = {}
for i in range(len(starts)):
    dur = (time_list[ends[i]] - time_list[starts[i]] + buffer_duration) * 1000
    for idx in range(starts[i], ends[i] + 1):
        burst_info[idx] = {'id': i + 1, 'duration': dur}

# --- 2. SETUP FIGURE ---
fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
plt.subplots_adjust(bottom=0.15)
index = 0

def burst_analyze(idx):
    ax1.clear()
    samples = samples_list[idx]
    N = len(samples)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    psd_db = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples)))**2 / N + 1e-12)
    psd_smoothed = savgol_filter(psd_db, window_length=51, polyorder=3)
    
    noise_floor = np.median(psd_db)
    base_threshold = noise_floor + 9
    
    if detection_map[idx]:
        # 1. Find Highest Peak
        peak_idx = np.argmax(psd_smoothed)
        
        # 2. Find Boundaries (Noise + 15dB)
        is_above = psd_smoothed > base_threshold
        left_side = np.where(is_above[:peak_idx] == False)[0]
        f_low_idx = left_side[-1] + 1 if len(left_side) > 0 else 0
        right_side = np.where(is_above[peak_idx:] == False)[0]
        f_high_idx = right_side[0] + peak_idx - 1 if len(right_side) > 0 else N - 1
        
        f_low, f_high = freqs[f_low_idx], freqs[f_high_idx]
        bw_khz = (f_high - f_low) / 1e3
        
        # 3. Calculate Center of Bandwidth
        f_center = (f_low + f_high) / 2
        f_center_khz = f_center / 1e3

        # 4. Plotting
        ax1.plot(freqs / 1e3, psd_smoothed, color='green', linewidth=2)
        ax1.axhline(y=base_threshold, color='orange', linestyle='--', alpha=0.6, label='Threshold')
        
        # Bandwidth Span
        ax1.axvspan(f_low/1e3, f_high/1e3, color='yellow', alpha=0.2, label=f'BW: {bw_khz:.1f} kHz')
        
        # Mark Center of Bandwidth
        ax1.axvline(x=f_center_khz, color='red', linestyle='-.', linewidth=1.5, label=f'BW Center: {f_center_khz:.1f} kHz')
        ax1.scatter(f_center_khz, base_threshold, color='red', marker='x', s=100, zorder=5)

        # Labels and Title
        info = burst_info[idx]
        ax1.set_title(f"Buffer {idx} | Burst #{info['id']} | Center: {f_center_khz:.1f} kHz| Duration: {info['duration']:.2f} ms", 
                     fontsize=14, fontweight='bold')
    else:
        ax1.plot(freqs / 1e3, psd_db, color='silver', alpha=0.4)
        ax1.set_title(f"Buffer {idx} | NO SIGNAL")

    ax1.axhline(y=noise_floor, color='black', linestyle=':', alpha=0.5, label='Noise Floor')
    ax1.set_ylim([noise_floor - 10, noise_floor + 80])
    ax1.set_xlabel("Frequency (kHz)")
    ax1.set_ylabel("Power (dB)")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.2)
    plt.draw()

def on_key(event):
    global index
    if event.key == 'right': index = (index + 1) % len(samples_list)
    elif event.key == 'left': index = (index - 1) % len(samples_list)
    burst_analyze(index)

fig.canvas.mpl_connect('key_press_event', on_key)
burst_analyze(0)
plt.show()