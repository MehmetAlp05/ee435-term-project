import numpy as np
import matplotlib.pyplot as plt

# Load your recorded data
data = np.load("pluto_capture.npz", allow_pickle=True)
samples_list = data['samples']
times_list = data['times']
fs = 2.5e6
# --- GLOBAL BURST ANALYSIS ---
samples_per_buffer = len(samples_list[0])
seconds_per_buffer = samples_per_buffer / fs
threshold_offset = 40  

detection_map = []
for s in samples_list:
    psd_lin = np.abs(np.fft.fft(s))**2 / len(s)
    psd_db = 10 * np.log10(np.fft.fftshift(psd_lin) + 1e-12)
    noise_floor = np.median(psd_db)
    detection_map.append(np.max(psd_db) > (noise_floor + threshold_offset))

detection_map = np.array(detection_map).astype(int)

# Identify the start and end indices of every discrete burst
diff = np.diff(np.concatenate(([0], detection_map, [0])))
burst_starts = np.where(diff == 1)[0]
burst_ends = np.where(diff == -1)[0] # This index is one past the last signal buffer

# Calculate individual durations
# duration = (end_idx - start_idx) * time_per_buffer
burst_durations = (burst_ends - burst_starts) * seconds_per_buffer * 1000 
num_bursts = len(burst_starts)

print(f"Detected {num_bursts} bursts.")
for i, dur in enumerate(burst_durations):
    print(f"Burst {i+1}: Start Buffer {burst_starts[i]}, Duration {dur:.2f} ms")
# ------------------------------

fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.2)
index = 0

def find_signal_bandwidth(freqs, psd_db, peak_freq, search_range=100e3):
    mask = (freqs >= (peak_freq - search_range/2)) & (freqs <= (peak_freq + search_range/2))
    f_segment = freqs[mask]
    psd_linear = 10**(psd_db[mask] / 10)
    total_pow = np.sum(psd_linear)
    if total_pow == 0: return 0, peak_freq, peak_freq
    cum_pow = np.cumsum(psd_linear)
    idx_low = np.where(cum_pow >= total_pow * 0.005)[0][0]
    idx_high = np.where(cum_pow >= total_pow * 0.995)[0][0]
    return f_segment[idx_high] - f_segment[idx_low], f_segment[idx_low], f_segment[idx_high]

def update_plot(idx):
    ax.clear()
    samples = samples_list[idx]
    N = len(samples)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    psd_linear = np.abs(np.fft.fftshift(np.fft.fft(samples)))**2 / N
    psd_db = 10 * np.log10(psd_linear + 1e-12)
    noise_floor_db = np.median(psd_db)
    
    peak_idx = np.argmax(psd_db)
    peak_val = psd_db[peak_idx]
    peak_freq = freqs[peak_idx]
    
    is_detected = detection_map[idx]
    plot_color = 'red' if is_detected else 'blue'
        
    if is_detected:
        mask_bw = 300e3
        mask = (freqs >= (peak_freq - mask_bw/2)) & (freqs <= (peak_freq + mask_bw/2))
        display_psd = np.full_like(psd_db, noise_floor_db)
        display_psd[mask] = psd_db[mask]
        bw, f_low, f_high = find_signal_bandwidth(freqs, psd_db, peak_freq)
        ax.axvspan(f_low/1e3, f_high/1e3, color='yellow', alpha=0.2)
        ax.text(peak_freq/1e3, peak_val + 5, f"BW: {bw/1e3:.1f} kHz", color='red', ha='center')
    else:
        display_psd = psd_db

    ax.plot(freqs / 1e3, display_psd, color=plot_color, linewidth=1.2)
    ax.axhline(noise_floor_db + threshold_offset, color='green', linestyle='--', alpha=0.5)
    
# Find if the current index belongs to a burst
    current_burst_id = None
    for i in range(num_bursts):
        if burst_starts[i] <= idx < burst_ends[i]:
            current_burst_id = i
            break

    if current_burst_id is not None:
        this_dur = burst_durations[current_burst_id]
        info_text = (f"BURST DETECTED (#{current_burst_id + 1})\n"
                     f"Burst Duration: {this_dur:.2f} ms\n"
                     f"Total Bursts Found: {num_bursts}")
    else:
        info_text = f"NO SIGNAL\nTotal Bursts Found: {num_bursts}"

    ax.text(0.02, 0.95, info_text, transform=ax.transAxes, va='top', 
            bbox=dict(facecolor='white', alpha=0.8), color='red' if current_burst_id is not None else 'blue')
    
    ax.set_ylim([noise_floor_db - 10, noise_floor_db + 80])
    ax.set_title(f"Buffer {idx} | Detected: {bool(is_detected)}")
    ax.set_xlabel("Frequency (kHz)")
    ax.set_ylabel("Power (dB)")
    ax.grid(True, alpha=0.3)
    plt.draw()

def on_key(event):
    global index
    if event.key == 'right': index = (index + 1) % len(samples_list)
    elif event.key == 'left': index = (index - 1) % len(samples_list)
    update_plot(index)

fig.canvas.mpl_connect('key_press_event', on_key)
update_plot(0)
plt.show()