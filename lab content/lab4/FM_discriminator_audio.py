# METU EE435 Lab. Fall 2025 Experiment 4  (TA:Safa Çelik)
# FM Discriminator for audio
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


def FM_discriminator_audio(fm_sig, time, SNR_dB):
    """
    fm_sig    : FM signal (complex baseband)
    time      : Time array
    SNR_dB    : Signal-to-noise ratio

    """
    # --- Parameters
    W = 6e3                              # Frequency of the modulating signal [Hz]
    Delta_f = 16050                      # Max frequency deviation [Hz]
    BW = 2 * (Delta_f + W)               # Carson'rule BW [Hz]
    fs = 1.0 / (time[1] - time[0])       # Sampling rate [Hz]
    T = fm_sig.size / fs                 # Duration [s]
    N = int(round(fs * T))               # Number of samples
   
    fir_order = 64
    b_lpf = firwin(fir_order + 1, min(0.98, W / (fs/2)))

    # --- Received signal (noisy free case)
    y = fm_sig

    v_r = np.real(y)
    v_i = np.imag(y)
    v_r_diff = gradient_six_point(v_r)
    v_i_diff = gradient_six_point(v_i)

    dec_sig_noisefree = (1/(2*np.pi)) * (v_r * v_i_diff - v_i * v_r_diff) / (
        v_r**2 + v_i**2 + 1e-12)
    dec_sig_noisefree = lfilter(b_lpf, 1, dec_sig_noisefree)

    # --- Received signal (noisy case)
    SNR_lin = 10**(SNR_dB/10)
    input_noise = (np.sqrt(0.5) *
                   (np.random.randn(*fm_sig.shape) + 1j*np.random.randn(*fm_sig.shape)))
    b_noise = firwin(fir_order + 1, min(0.98, (BW/2) / (fs/2)))
    input_noise = lfilter(b_noise, 1, input_noise)
    input_noise = np.sqrt(1/SNR_lin) * input_noise / np.std(input_noise)

    y = fm_sig + input_noise

    v_r = np.real(y)
    v_i = np.imag(y)
    v_r_diff = gradient_six_point(v_r)
    v_i_diff = gradient_six_point(v_i)

    dec_sig = (1/(2*np.pi)) * (v_r * v_i_diff - v_i * v_r_diff) / (
        v_r**2 + v_i**2 + 1e-12)
    dec_sig = lfilter(b_lpf, 1, dec_sig)

    noise = dec_sig - dec_sig_noisefree

    dec_sig_noisefree = dec_sig_noisefree[1:]
    dec_sig = dec_sig[1:]
    noise = noise[1:]

    return dec_sig_noisefree, dec_sig, noise, fs