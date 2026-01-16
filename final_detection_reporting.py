import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, hilbert, butter, filtfilt
from collections import Counter
import pandas as pd  # Added for Excel export

# Load your recorded data
data = np.load("pluto_capture.npz", allow_pickle=True)
samples_list = data['samples']
time_list = data['times'] 
fs = 2.5e6

# --- 1. GLOBAL BURST ANALYSIS ---
print("Scanning all buffers for burst durations and frequency hops...")
samples_per_buffer = len(samples_list[0])
buffer_duration = samples_per_buffer / fs
window_size_hz = 100e3
# Threshold for splitting a burst if the center frequency jumps > 250kHz
FREQ_HOP_THRESHOLD = 250e3 

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

# --- UPDATED BURST ASSIGNMENT WITH FREQUENCY HOP LOGIC ---
detection_map = np.array(detection_map).astype(int)
burst_ids = np.zeros(len(detection_map), dtype=int)
current_id = 0

for i in range(len(detection_map)):
    if detection_map[i] == 1:
        if i == 0 or detection_map[i-1] == 0:
            # New signal after a gap
            current_id += 1
        else:
            # Continuous detection: check for central frequency change
            f_prev = center_freqs_all[i-1]
            f_curr = center_freqs_all[i]
            if f_prev is not None and f_curr is not None:
                # If center point moves > 250kHz, divide into separate burst
                if np.abs(f_curr - f_prev) > FREQ_HOP_THRESHOLD:
                    current_id += 1
        burst_ids[i] = current_id

unique_bursts = np.unique(burst_ids[burst_ids > 0])
burst_starts = {b: np.where(burst_ids == b)[0][0] for b in unique_bursts}
burst_ends = {b: np.where(burst_ids == b)[0][-1] for b in unique_bursts}

# --- SYMMETRY CALCULATION ---
def calculate_symmetry(psd, f_l_idx, f_h_idx):
    """Checks spectral mirroring around the center of the bandwidth."""
    mid = (f_l_idx + f_h_idx) // 2
    span = (f_h_idx - f_l_idx) // 2
    if span < 10: return 0
    left_half = psd[mid - span : mid]
    right_half = psd[mid : mid + span]
    right_half_rev = right_half[::-1]
    correlation = np.corrcoef(left_half, right_half_rev)[0, 1]
    return correlation if not np.isnan(correlation) else 0

# --- SLIDING WINDOW VARIANCE ---
def get_min_window_metrics(samples, fs, window_ms=5):
    analytic_signal = samples
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

# --- CONSENSUS CLASSIFICATION ---
print("Calculating consensus modulation types for bursts...")
burst_votes = {b: [] for b in unique_bursts}
burst_durations = {}
burst_sweeps = {}
burst_bandwidths = {b: [] for b in unique_bursts} # Store BW for Excel

for b_id in unique_bursts:
    s_idx, e_idx = burst_starts[b_id], burst_ends[b_id]
    dur = (time_list[e_idx] - time_list[s_idx] + buffer_duration) * 1000
    f_start, f_end = center_freqs_all[s_idx], center_freqs_all[e_idx]
    sweep = np.abs(f_end - f_start) / 1e3
    is_lfm_sweep = sweep > 40.0
    
    burst_durations[b_id] = dur
    burst_sweeps[b_id] = sweep

    for idx in range(s_idx, e_idx + 1):
        samples = samples_list[idx]
        N = len(samples)
        freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
        psd_lin = np.abs(np.fft.fftshift(np.fft.fft(samples)))**2 / N
        psd_db = 10 * np.log10(psd_lin + 1e-12)
        psd_smoothed = savgol_filter(psd_db, window_length=51, polyorder=3)
        peak_idx = np.argmax(psd_smoothed)
        
        # --- FIXED BANDWIDTH LOGIC TO MATCH YELLOW HIGHLIGHT ---
        noise_floor = np.median(psd_db)
        base_threshold = noise_floor + 20
        is_above = psd_smoothed > base_threshold
        
        if np.any(is_above):
            left_side = np.where(is_above[:peak_idx] == False)[0]
            f_l_idx_burst = left_side[-1] + 1 if len(left_side) > 0 else 0
            right_side = np.where(is_above[peak_idx:] == False)[0]
            f_h_idx_burst = right_side[0] + peak_idx - 1 if len(right_side) > 0 else N - 1
            
            # This is the exact value visualized in yellow
            bw_val = (freqs[f_h_idx_burst] - freqs[f_l_idx_burst]) / 1e3
            burst_bandwidths[b_id].append(bw_val)

        env_var, _, _ = get_min_window_metrics(samples, fs)
        is_above_10 = psd_smoothed > (psd_smoothed[peak_idx] - 10)
        idx_10 = np.where(is_above_10)[0]
        width_10db_khz = (freqs[idx_10[-1]] - freqs[idx_10[0]]) / 1e3
        sym_score = calculate_symmetry(psd_smoothed, idx_10[0], idx_10[-1])

        if is_lfm_sweep and env_var < 0.01: label = "LFM (Type 5)"
        elif width_10db_khz < 5.0: label = "AM (Type 1)"
        elif env_var < 0.01: label = "FM (Type 4)"
        elif sym_score > 0.7: label = "DSB-SC (Type 2)"
        else: label = "SSB (Type 3)"
        burst_votes[b_id].append(label)

burst_consensus = {b: Counter(votes).most_common(1)[0][0] for b, votes in burst_votes.items()}

# --- NEW: EXPORT TO EXCEL (.xlsx) ---
print("Generating Faculty Report...")
report_data = []
mod_map = {
    "AM (Type 1)": 1,
    "DSB-SC (Type 2)": 2,
    "SSB (Type 3)": 3,
    "FM (Type 4)": 4,
    "LFM (Type 5)": 5
}

for b_id in unique_bursts:
    s_idx = burst_starts[b_id]
    mod_str = burst_consensus[b_id]
    
    # Calculate most frequent or average bandwidth of the visual highlights for this burst
    avg_bw = np.mean(burst_bandwidths[b_id]) if burst_bandwidths[b_id] else 0
    
    row = {
        'modType': mod_map.get(mod_str, 0),
        'Tmsg_ms': burst_durations[b_id],
        'f_Shift_kHz': center_freqs_all[s_idx] / 1e3, # Shift from DC in kHz
        'B_rf_kHz': avg_bw 
    }
    report_data.append(row)

df = pd.DataFrame(report_data)
df.to_excel("signal_report.xlsx", index=False)
print("Report saved as 'signal_report.xlsx'")

# --- 2. SETUP FIGURE ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
plt.subplots_adjust(hspace=0.4, bottom=0.1)
index = 0

def burst_analyze(idx):
    ax1.clear()
    ax2.clear()
    samples = samples_list[idx]
    N = len(samples)
    
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    psd_lin = np.abs(np.fft.fftshift(np.fft.fft(samples)))**2 / N
    psd_db = 10 * np.log10(psd_lin + 1e-12)
    psd_smoothed = savgol_filter(psd_db, window_length=51, polyorder=3)
    noise_floor = np.median(psd_db)
    base_threshold = noise_floor + 9
    
    env_var, envelope, (b_start, b_end) = get_min_window_metrics(samples, fs)
    t_full = np.arange(N) / fs * 1e3 
    
    ax2.plot(t_full[b_start:b_end], np.real(samples[b_start:b_end]), color='blue', alpha=0.3)
    ax2.plot(t_full[b_start:b_end], envelope[b_start:b_end], color='red', linewidth=2)
    ax2.set_title(f"Buffer {idx} | Env Var: {env_var:.6f}")
    ax2.grid(True, alpha=0.2)

    if burst_ids[idx] > 0:
        b_id = burst_ids[idx]
        mod_type = burst_consensus[b_id]
        color_map = {"LFM (Type 5)": "magenta", "AM (Type 1)": "orange", "FM (Type 4)": "cyan", "DSB-SC (Type 2)": "green", "SSB (Type 3)": "lime"}
        color = color_map.get(mod_type, "blue")

        peak_idx = np.argmax(psd_smoothed)
        is_above = psd_smoothed > base_threshold
        left_side = np.where(is_above[:peak_idx] == False)[0]
        f_l_idx = left_side[-1] + 1 if len(left_side) > 0 else 0
        right_side = np.where(is_above[peak_idx:] == False)[0]
        f_h_idx = right_side[0] + peak_idx - 1 if len(right_side) > 0 else N - 1
        
        ax1.plot(freqs / 1e3, psd_smoothed, color=color, linewidth=2)
        ax1.axvspan(freqs[f_l_idx]/1e3, freqs[f_h_idx]/1e3, color='yellow', alpha=0.2)
        ax1.set_title(f"Buffer {idx} | CONSENSUS: {mod_type} | Burst ID: {b_id}", color=color, fontweight='bold')
    else:
        ax1.plot(freqs / 1e3, psd_db, color='silver', alpha=0.4)
        ax1.set_title(f"Buffer {idx} | NO SIGNAL")

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