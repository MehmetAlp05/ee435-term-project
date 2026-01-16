import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt, savgol_filter

# Load your recorded data
data = np.load("pluto_capture.npz", allow_pickle=True)
samples_list = data['samples']
time_list = data['times'] 
fs = 2.5e6

# --- 1. HIGH PRECISION ENVELOPE TOOL ---
def get_min_window_metrics(samples, fs, window_ms=5):
    """
    Finds the 5ms window with the lowest normalized variance to ensure 
    the plot shows the most stable/representative part of the signal.
    """
    # Analytic signal for high-precision magnitude
    analytic_signal = (samples)
    envelope = np.abs(analytic_signal)
    
    # Butterworth LPF (25kHz) to remove thermal noise fuzz
    nyq = 0.5 * fs
    b, a = butter(4, 25000 / nyq, btype='low')
    smooth_env = filtfilt(b, a, envelope)
    
    # Sliding window parameters (5ms)
    window_size = int(window_ms * 1e-3 * fs)
    step_size = window_size // 4
    
    variances = []
    for start in range(0, len(smooth_env) - window_size, step_size):
        end = start + window_size
        chunk = smooth_env[start:end]
        mean_val = np.mean(chunk)
        if mean_val > 1e-6:
            # Normalized Variance: Var / Mean^2
            v = np.var(chunk) / (mean_val**2)
            variances.append((v, start, end))
            
    if not variances:
        return 1.0, smooth_env, (0, window_size)
    
    # Return the minimum variance found and the indices for the 'best' window
    min_v, b_start, b_end = min(variances, key=lambda x: x[0])
    return min_v, smooth_env, (b_start, b_end)

# --- 2. SETUP FIGURE ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
plt.subplots_adjust(hspace=0.4, bottom=0.1)
index = 0

def plot_analysis(idx):
    ax1.clear()
    ax2.clear()
    
    samples = samples_list[idx]
    N = len(samples)
    t_full = np.arange(N) / fs * 1e3 # Time in ms
    
    # Calculate Precision Metrics
    min_var, smooth_env, (b_start, b_end) = get_min_window_metrics(samples, fs)
    
    # --- TOP PLOT: FULL BUFFER ENVELOPE ---
    ax1.plot(t_full, np.abs(samples), color='blue', alpha=0.15, label='Raw Magnitude')
    ax1.plot(t_full, smooth_env, color='red', linewidth=1.5, label='Smooth Envelope')
    
    # Highlight the 5ms window where variance was measured
    ax1.axvspan(t_full[b_start], t_full[b_end], color='yellow', alpha=0.3, label='5ms Analysis Window')
    
    ax1.set_title(f"Buffer {idx} | Full Buffer Overview", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.2)

    # --- BOTTOM PLOT: ZOOMED TIME DOMAIN & ENVELOPE ---
    # Zoom into the high-precision window found above
    t_zoom = t_full[b_start:b_end]
    samples_zoom = samples[b_start:b_end]
    env_zoom = smooth_env[b_start:b_end]
    
    ax2.plot(t_zoom, np.real(samples_zoom), color='blue', alpha=0.4, label='Real (Carrier)')
    ax2.plot(t_zoom, env_zoom, color='red', linewidth=2.5, label='Precision Envelope')
    
    # Classification for the Title
    if min_var < 0.001:
        status, color = "Constant (FM/LFM)", 'cyan'
    elif min_var < 0.02:
        status, color = "Stable Carrier", 'green'
    else:
        status, color = "Varying (AM/DSB/SSB)", 'orange'
        
    ax2.set_title(f"Zoomed 5ms Window | Min Precision Variance: {min_var:.8f} -> {status}", 
                 color=color, fontsize=13, fontweight='bold')
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Magnitude")
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim([0, np.max(env_zoom) * 1.3]) # Ensure visual space for the envelope

    plt.draw()

# --- 3. INTERACTIVE NAVIGATION ---
print("Scanning Complete. Use Arrow Keys to navigate buffers.")

def on_key(event):
    global index
    if event.key == 'right': index = (index + 1) % len(samples_list)
    elif event.key == 'left': index = (index - 1) % len(samples_list)
    plot_analysis(index)

fig.canvas.mpl_connect('key_press_event', on_key)
plot_analysis(0)
plt.show()