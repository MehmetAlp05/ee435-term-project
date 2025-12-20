# METU EE435 Lab. Fall 2025 Experiment 4  (TA:Safa Çelik)
# Part 3: FM noise audio
# Updated (Nov. 2025)

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import sounddevice as sd
from FM_discriminator_audio import FM_discriminator_audio  # Ensure the function is defined and available

# Load data
data = loadmat('exp4_data.mat')
fm_sig  = data['data_array'].flatten()
time    = data['time_array'].flatten()

margin = 100
SNR_dB = np.array([40, 30, 20, 15, 10, 5, 0, -5])
SNR_out = np.zeros_like(SNR_dB, dtype=float)

for ii, snr in enumerate(SNR_dB):

    dec_nf, dec_sig, noise, fs = FM_discriminator_audio(fm_sig, time, snr)
    
    N = dec_sig.size
    f = np.linspace(-fs/2, fs/2 - fs/N, N) / 1e3
    
    mag_resp = np.abs(np.fft.fftshift(np.fft.fft(dec_sig, N)))
    mag_resp /= np.max(mag_resp)
    
    plt.figure()
    plt.plot(f, mag_resp, 'b-', linewidth=1.3)
    plt.title(f'Magnitude Response of Demodulated FM Speech Signal for CNR = {snr} dB')
    plt.xlabel('f (kHz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.xlim(0.5 * np.array([-fs/2, fs/2]) / 1e3)
    plt.tight_layout()
    
    SNR_out[ii] = 10 * np.log10(np.var(dec_nf[margin : N - margin]) / np.var(noise[margin : N - margin]))
    
    print(f"Playing audio for CNR = {snr} dB")
    sd.play(dec_sig, fs, blocking=True)

plt.figure()
plt.plot(SNR_dB, SNR_out, marker='*', linewidth=2)
plt.title('Output SNR vs Input SNR')
plt.xlabel('Input SNR (dB)')
plt.ylabel('Output SNR (dB)')
plt.grid(True)
plt.xlim(SNR_dB[-1], SNR_dB[0])
plt.tight_layout()
plt.show()
