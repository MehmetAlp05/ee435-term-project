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
    
    # Calculate PSD
    N = len(samples)
    freqs = np.fft.fftshift(np.fft.fftfreq(N, 1/fs))
    psd = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(samples)))**2 / N)
    
    ax.plot(freqs / 1e3, psd) # Plot in kHz for better readability
    ax.set_ylim([20, 100])      # Adjust based on your signal strength
    ax.set_title(f"Buffer {idx} | Time: {times_list[idx]:.3f}s")
    ax.set_xlabel("Frequency Offset (kHz)")
    ax.set_ylabel("Power (dB)")
    ax.grid(True)
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