import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, hilbert, butter, filtfilt

# Load your recorded data
data = np.load("pluto_capture.npz", allow_pickle=True)
samples_list = data['samples']
time_list = data['times'] 
fs = 2.5e6

# --- 1. GLOBAL BURST ANALYSIS ---
print("Scanning all buffers for burst durations and LFM sweep...")
samples_per_buffer = len(samples_list[0])
buffer_duration = samples_per_buffer / fs
window_size_hz = 100e3

detection_map = []
center_freqs_all = [] 

for samples in samples_list:
    N = len(samples)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    psd_lin = np.abs(np.fft.fft(samples))**2 / N
    psd_db = 10 * np.log10(np.fft.fftshift(psd_lin) + 1e-12)
    psd_smoothed = savgol_filter(psd_db, window_length=51, polyorder=3)
    
    psd_int = np.cumsum(np.fft.fftshift(psd_lin))
    psd_int_norm = psd_int / (psd_int[-1] + 1e-12)
    bins_in_100k = int(window_size_hz / (fs / N))

    is_det = False
    if len(psd_int_norm) > bins_in_100k:
        jumps = psd_int_norm[bins_in_100k:] - psd_int_norm[:-bins_in_100k]
        is_det = np.max(jumps) > 0.5
    detection_map.append(is_det)

    if is_det:
        noise_floor = np.median(psd_db)
        base_threshold = noise_floor + 9
        peak_idx = np.argmax(psd_smoothed)
        is_above = psd_smoothed > base_threshold
        left = np.where(is_above[:peak_idx] == False)[0]
        l_idx = left[-1] + 1 if len(left) > 0 else 0
        right = np.where(is_above[peak_idx:] == False)[0]
        r_idx = right[0] + peak_idx - 1 if len(right) > 0 else N - 1
        center_freqs_all.append((freqs[l_idx] + freqs[r_idx]) / 2)
    else:
        center_freqs_all.append(None)

detection_map = np.array(detection_map).astype(int)
diff = np.diff(np.concatenate(([0], detection_map, [0])))
starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0] - 1

burst_info = {}
for i in range(len(starts)):
    s_idx, e_idx = starts[i], ends[i]
    dur = (time_list[e_idx] - time_list[s_idx] + buffer_duration) * 1000
    f_start, f_end = center_freqs_all[s_idx], center_freqs_all[e_idx]
    total_sweep_khz = np.abs(f_end - f_start) / 1e3 if (f_start is not None and f_end is not None) else 0
    is_lfm = total_sweep_khz > 40.0 
    for idx in range(s_idx, e_idx + 1):
        burst_info[idx] = {'id': i + 1, 'duration': dur, 'is_lfm': is_lfm, 'total_sweep': total_sweep_khz}

# --- HIGH PRECISION SLIDING WINDOW VARIANCE FUNCTION ---
def get_min_window_metrics(samples, fs, window_ms=5):
    analytic_signal = (samples)
    envelope = np.abs(analytic_signal)
    
    window_size = int(window_ms * 1e-3 * fs)
    step_size = window_size // 4 
    
    variances = []
    for start in range(0, len(envelope) - window_size, step_size):
        end = start + window_size
        chunk = envelope[start:end]
        mean_val = np.mean(chunk)
        if mean_val > 1e-6:
            v = np.var(chunk) / (mean_val**2)
            variances.append((v, start, end))
            
    if not variances:
        return 1.0, envelope, (0, window_size)
    
    min_v, b_start, b_end = min(variances, key=lambda x: x[0])
    return min_v, envelope, (b_start, b_end)

# --- 2. SETUP FIGURE ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
plt.subplots_adjust(hspace=0.4, bottom=0.1)
index = 0

def burst_analyze(idx):
    ax1.clear()
    ax2.clear()
    samples = samples_list[idx]
    N = len(samples)
    
    # Frequency Domain Prep
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    psd_db = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples)))**2 / N + 1e-12)
    psd_smoothed = savgol_filter(psd_db, window_length=51, polyorder=3)
    noise_floor = np.median(psd_db)
    base_threshold = noise_floor + 9
    
    # --- TIME DOMAIN & HIGH PRECISION ENVELOPE (ax2 Zoomed) ---
    env_var, envelope, (b_start, b_end) = get_min_window_metrics(samples, fs)
    t_full = np.arange(N) / fs * 1e3  # Full time array in ms
    
    # Slice arrays to plot ONLY the 5ms min-variance window
    t_window = t_full[b_start:b_end]
    samples_window = np.real(samples[b_start:b_end])
    envelope_window = envelope[b_start:b_end]
    
    ax2.plot(t_window, samples_window, color='blue', alpha=0.3, label='Real (I)')
    ax2.plot(t_window, envelope_window, color='red', linewidth=2, label='Envelope')
    ax2.plot(t_window, -envelope_window, color='red', linewidth=1, alpha=0.2)
    
    ax2.set_title(f"Zoomed 5ms Analysis Window | Precision Env Var: {env_var:.6f}")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Amplitude")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.2)

    # --- FREQUENCY DOMAIN PLOTTING (ax1) ---
    if detection_map[idx]:
        peak_idx = np.argmax(psd_smoothed)
        peak_val = psd_smoothed[peak_idx]
        is_above = psd_smoothed > base_threshold
        left_side = np.where(is_above[:peak_idx] == False)[0]
        f_l_idx = left_side[-1] + 1 if len(left_side) > 0 else 0
        right_side = np.where(is_above[peak_idx:] == False)[0]
        f_h_idx = right_side[0] + peak_idx - 1 if len(right_side) > 0 else N - 1
        
        ten_db_threshold = peak_val - 10
        is_above_10 = psd_smoothed > ten_db_threshold
        l_10 = np.where(is_above_10[:peak_idx] == False)[0]
        r_10 = np.where(is_above_10[peak_idx:] == False)[0]
        l_idx_10 = l_10[-1] + 1 if len(l_10) > 0 else 0
        r_idx_10 = r_10[0] + peak_idx - 1 if len(r_10) > 0 else N - 1
        width_10db_khz = (freqs[r_idx_10] - freqs[l_idx_10]) / 1e3
        
        info = burst_info[idx]
        
        if info['is_lfm']:
            mod_type, color = f"LFM (Type 5) | Sweep: {info['total_sweep']:.1f}kHz", 'magenta'
        elif width_10db_khz < 5.0:
            mod_type, color = f"AM (Type 1) | 10dB Width: {width_10db_khz:.1f}kHz", 'orange'
        elif env_var < 0.005:
            mod_type, color = f"FM (Type 4) | Constant Envelope", 'cyan'
        else:
            mod_type, color = f"DSB/SSB (Type 2/3) | Varying Envelope", 'green'

        ax1.plot(freqs / 1e3, psd_smoothed, color=color, linewidth=2)
        ax1.axvspan(freqs[f_l_idx]/1e3, freqs[f_h_idx]/1e3, color='yellow', alpha=0.2)
        ax1.hlines(ten_db_threshold, freqs[l_idx_10]/1e3, freqs[r_idx_10]/1e3, color='blue', linewidth=3)
        ax1.set_title(f"Buffer {idx} | {mod_type} | Burst #{info['id']}", color=color, fontweight='bold')
    else:
        ax1.plot(freqs / 1e3, psd_db, color='silver', alpha=0.4)
        ax1.set_title(f"Buffer {idx} | NO SIGNAL")

    ax1.axhline(y=noise_floor, color='black', linestyle=':', alpha=0.5, label='Noise Floor')
    ax1.set_ylim([noise_floor - 10, noise_floor + 80])
    ax1.set_ylabel("Power (dB)")
    ax1.set_xlabel("Frequency (kHz)")
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