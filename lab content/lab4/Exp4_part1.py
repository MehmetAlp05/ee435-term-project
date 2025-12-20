# METU EE435 Lab. Fall 2025 Experiment 4  (TA:Safa Çelik)
# Part 1: FM Noise PSD
# Updated (Nov. 2025)

import numpy as np
import matplotlib.pyplot as plt
from FM_discriminator import FM_discriminator

# --- Parameters
SNR_dB = 12             # Signal-to-noise ratio
fm = 3e3                # Modulating signal frequency [Hz]
Delta_f = 15e3          # Max frequency deviation [Hz]
T = 0.5                 # Duration [s]

MC = 100                # Monte Carlo runs
margin = 10

signal, dec_signal, noise, fm_sig, t, f, BW, fs = FM_discriminator(T, fm, Delta_f, SNR_dB)
N = len(signal)
Fm_sig = np.abs(np.fft.fftshift(np.fft.fft(fm_sig, N)/N))

plt.figure(figsize=(10, 4))
plt.plot(f/1e3, Fm_sig, linewidth=2)
plt.grid(True)
plt.xlabel(r'$f~(\mathrm{kHz})$')
plt.ylabel(r'Baseband FM Signal')
plt.xlim(np.array([-fs/2, fs/2]) / 1e3)
plt.title(
    "Noise-free FM signal at the input of the receiver\n"
    + rf"$\Delta f = {Delta_f/1e3:.0f}\,\mathrm{{kHz}},\; f_m = {fm/1e3:.0f}\,\mathrm{{kHz}},\; SNR = {SNR_dB:.0f}\,\mathrm{{dB}}$"
)
plt.tight_layout()

# Monte Carlo simulation for noise PSD
PSD_fm = np.zeros(N)

for kk in range(MC):
    signal, dec_signal, noise, fm_sig, t, f, BW, fs = FM_discriminator(T, fm, Delta_f, SNR_dB)

    dec_signal = dec_signal[margin : -margin]

    Dec_signal = np.abs(np.fft.fftshift(np.fft.fft(dec_signal, N)) / N)

    PSD_fm += Dec_signal

    print(f"Iteration {kk+1}")

PSD_fm = PSD_fm / MC

plt.figure(figsize=(10, 4))
plt.plot(f/1e3, PSD_fm, linewidth=1.3)
plt.grid(True)
plt.xlabel(r'$f~(\mathrm{kHz})$')
plt.ylabel(r'Noise PSD')
plt.xlim(1.1 * np.array([-fm, fm]) / 1e3)
plt.title(
    "Noise Spectrum at the Output of Limiter-Discriminator\n"
    + rf"$\Delta f = {Delta_f/1e3:.0f}\,\mathrm{{kHz}},\; f_m = {fm/1e3:.0f}\,\mathrm{{kHz}},\; SNR = {SNR_dB:.0f}\,\mathrm{{dB}}$"
)
plt.tight_layout()
plt.show()