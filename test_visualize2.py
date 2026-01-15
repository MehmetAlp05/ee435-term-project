import numpy as np
import matplotlib.pyplot as plt

# Load your recorded data
data = np.load("pluto_capture.npz", allow_pickle=True)
samples_list = data['samples']
times_list = data['times']
fs = 2.5e6
band_step = 20e3  # 10 kHz target bandwidth

fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.2)

index = 0

def update_plot(idx):
    ax.clear()
    samples = samples_list[idx]
    
    # 1. Standard FFT Processing
    N = len(samples)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    # Get linear power (not dB yet)
    power_linear = np.abs(np.fft.fftshift(np.fft.fft(samples)))**2 / N
    
    # 2. Binning Logic (Every 10kHz)
    # Define the bin edges from min frequency to max frequency
    bin_edges = np.arange(freqs.min(), freqs.max(), band_step)
    
    bin_centers = []
    bin_averages_db = []
    
    for i in range(len(bin_edges) - 1):
        # Find indices of frequencies that fall within this 10kHz window
        mask = (freqs >= bin_edges[i]) & (freqs < bin_edges[i+1])
        
        if np.any(mask):
            # Average the linear power in this band
            avg_power = np.mean(power_linear[mask])
            # Convert the average to dB
            bin_averages_db.append(10 * np.log10(avg_power + 1e-12)) # added epsilon to avoid log(0)
            bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)

    # 3. Plotting
    ax.step(np.array(bin_centers) / 1e3, bin_averages_db, where='mid', color='orange', label='10kHz Average')
    
    ax.set_title(f"Binned PSD (10kHz Bands) | Buffer {idx} | Time: {times_list[idx]:.3f}s")
    ax.set_xlabel("Frequency Offset (kHz)")
    ax.set_ylabel("Average Power (dB)")
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.draw()

# Navigation instructions
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