import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# --- 1. SETTINGS & DATA LOAD ---
data = np.load("pluto_capture.npz", allow_pickle=True)
samples_list = data['samples']
fs = 2.5e6
threshold_offset = 40 
from scipy.signal import welch
from scipy.signal import savgol_filter
def get_smoothed_psd(samples, fs):
    # nperseg: larger = more resolution, smaller = smoother/less noise
    f, psd_lin = welch(samples, fs=fs, nperseg=1024, scaling='density', 
                       return_onesided=False)
    
    f = np.fft.fftshift(f)
    psd_lin = np.fft.fftshift(psd_lin)
    psd_db = 10 * np.log10(psd_lin + 1e-12)
    return f, psd_db
# --- 2. THE MASTER CLASSIFICATION FUNCTION ---
def detect_modulation(samples, fs, bw, peak_freq):
    # Time-domain features
    amplitude = np.abs(samples)
    phase = np.unwrap(np.angle(samples))
    inst_freq = np.diff(phase) / (2 * np.pi) * fs
    
    # Normalized Variance (Lower = Constant Envelope)
    env_var = np.var(amplitude) / (np.mean(amplitude)**2 + 1e-12)
    
    # LFM Check (Linear Regression on frequency)
    t = np.linspace(0, len(inst_freq)/fs, len(inst_freq))
    slope, intercept, r_value, _, _ = linregress(t, inst_freq)
    is_linear = r_value**2 > 0.85 

    # Spectral features for AM/DSB/SSB
    N = len(samples)
    freqs, psd_lin = get_smoothed_psd(samples, fs)
    psd_lin = savgol_filter(psd_lin, window_length=51, polyorder=3)
    # Symmetry Check
    left_power = np.sum(psd_lin[freqs < peak_freq])
    right_power = np.sum(psd_lin[freqs > peak_freq])
    symmetry_ratio = min(left_power, right_power) / (max(left_power, right_power) + 1e-12)

    # Carrier Spike Check (AM vs DSB)
    center_idx = np.argmin(np.abs(freqs - peak_freq))
    carrier_to_noise = psd_lin[center_idx] / (np.mean(psd_lin) + 1e-12)

    # Decision Logic
    if bw<20e3: return "AM (Type 1)"
    if bw<40e3: return "SSB (Type 3)"
    if env_var < 0.1: return "FM (Type 4)"
    if is_linear and bw > 35e3: return "LFM (Type 5)"
    return "DSB-SC (Type 2)"

# --- 3. PRE-ANALYSIS FOR BURST CONCATENATION ---
detection_map = []
for s in samples_list:
    p_db = 10 * np.log10(np.abs(np.fft.fft(s))**2 / len(s) + 1e-12)
    detection_map.append(np.max(p_db) > (np.median(p_db) + threshold_offset))
detection_map = np.array(detection_map).astype(int)

# --- 4. PLOTTING & UI ---
fig, ax = plt.subplots(figsize=(10, 7))
index = 0

def find_signal_bandwidth(freqs, psd_db, peak_freq):
    mask = (freqs >= peak_freq - 50e3) & (freqs <= peak_freq + 50e3)
    f_seg, p_lin = freqs[mask], 10**(psd_db[mask]/10)
    c = np.cumsum(p_lin)
    if c[-1] == 0: return 0, peak_freq, peak_freq
    i_l, i_h = np.where(c >= c[-1]*0.005)[0][0], np.where(c >= c[-1]*0.995)[0][0]
    return f_seg[i_h] - f_seg[i_l], f_seg[i_l], f_seg[i_h]

def update_plot(idx):
    ax.clear()
    samples = samples_list[idx]
    N = len(samples)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    psd_db = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples)))**2 / N + 1e-12)
    noise_floor = np.median(psd_db)
    
    is_detected = detection_map[idx]
    
    if is_detected:
        peak_idx = np.argmax(psd_db)
        peak_f = freqs[peak_idx]
        bw, f_l, f_h = find_signal_bandwidth(freqs, psd_db, peak_f)
        
        # --- Run Classification ---
        mod_type = detect_modulation(samples, fs, bw, peak_f)
        
        # Visual Masking
        mask_bw= 300e3
        mask = (freqs >= peak_f - mask_bw/2) & (freqs <= peak_f + mask_bw/2)
        display_psd = np.full_like(psd_db, noise_floor)
        display_psd[mask] = psd_db[mask]
        
        ax.plot(freqs/1e3, display_psd, color='red')
        ax.axvspan(f_l/1e3, f_h/1e3, color='yellow', alpha=0.2)
        ax.set_title(f"Buffer {idx} | DETECTED: {mod_type}", fontsize=14, color='red', fontweight='bold')
    else:
        ax.plot(freqs/1e3, psd_db, color='blue')
        ax.set_title(f"Buffer {idx} | NO SIGNAL", fontsize=12)

    ax.set_ylim([noise_floor-10, noise_floor+70])
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