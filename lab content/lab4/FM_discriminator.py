# METU EE435 Lab. Fall 2025 Experiment 4  (TA:Safa Çelik)
# FM Discriminator
# Updated (Nov. 2025)

import numpy as np
from scipy.signal import firwin, lfilter

def gradient_six_point(x):
    """
    6-point central difference stencil.
    v_diff = (x[n-3] - 9x[n-2] + 45x[n-1] - 45x[n+1] + 9x[n+2] - x[n+3]) / 60
    Zero-padding is used at the edges, then the middle part (4:end-3) is kept.
    """
    x = np.asarray(x).ravel()           # 1D array
    x_zp = np.concatenate([np.zeros(3), x, np.zeros(3)])

    x_diff = (np.roll(x_zp, -3)
              - 9*np.roll(x_zp, -2)
              + 45*np.roll(x_zp, -1)
              - 45*np.roll(x_zp,  1)
              + 9*np.roll(x_zp,   2)
              -    np.roll(x_zp,  3)) / 60

    y = x_diff[3:-3]
    return y.reshape(x.shape)

def FM_discriminator(T, fm, Delta_f, SNR_dB):
    """
    T         : Simulation duration [s]
    fm        : Frequency of the modulating signal [Hz]
    Delta_f   : Max frequency deviation [Hz]
    SNR_dB    : Signal-to-noise ratio

    """
    # --- Parameters
    beta = Delta_f / fm                     # Modulation index
    BW = 2 * (Delta_f + fm)                 # Carson rule BW
    fs = 10 * BW                            # Sampling rate [Hz]
    N = int(round(fs * T))                  # Number of samples
    t = np.arange(N) / fs                   # Time array
    f = np.linspace(-fs/2, fs/2 - fs/N, N)  # Frequency array
    fir_order = 64
    b_lpf = firwin(fir_order + 1, min(0.98, fm / (fs/2)))

    # --- FM signal: s(t) = Ac * exp(j*phi(t)),  phi = beta * sin(2π fm t)
    Ac  = np.sqrt(10**(SNR_dB/10))
    phi = beta * np.sin(2*np.pi*fm*t)
    fm_sig = Ac * np.exp(1j*phi)

    # --- Received signal (only noise)
    input_noise = (np.sqrt(0.5) *
                   (np.random.randn(N) + 1j*np.random.randn(N)))
    b_noise = firwin(fir_order + 1, min(0.98, (BW/2) / (fs/2)))
    input_noise = lfilter(b_noise, 1, input_noise)
    input_noise = input_noise / np.std(input_noise)

    y = input_noise
    n_I = np.real(y)
    n_Q = np.imag(y)

    v_r = n_I * np.cos(phi) + n_Q * np.sin(phi)
    v_i = n_Q * np.cos(phi) - n_I * np.sin(phi)
    v_r_diff = fs * gradient_six_point(v_r)
    v_i_diff = fs * gradient_six_point(v_i)

    R_t_2 = (Ac + v_r)**2 + v_i**2

    noise = (1/(2*np.pi)) * ((Ac + v_r) * v_i_diff - v_i * v_r_diff) / R_t_2
    noise = lfilter(b_lpf, 1, noise)

    # --- Received signal (noisy case)
    y = fm_sig + input_noise

    v_r = np.real(y)
    v_i = np.imag(y)
    v_r_diff = fs * gradient_six_point(v_r)
    v_i_diff = fs * gradient_six_point(v_i)

    # --- Decoded signal
    dec_signal = (1/(2*np.pi)) * (v_r * v_i_diff - v_i * v_r_diff) / (v_r**2 + v_i**2 + 1e-12)
    dec_signal = lfilter(b_lpf, 1, dec_signal)

    signal = dec_signal - noise

    return signal, dec_signal, noise, fm_sig, t, f, BW, fs