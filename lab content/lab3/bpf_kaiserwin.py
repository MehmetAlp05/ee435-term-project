# METU EE435 Lab. Fall 2025 Experiment 3  (TA:Safa Çelik)
# Updated (Oct. 2025)

import numpy as np
from scipy.signal import kaiserord, firwin
from scipy.signal.windows import kaiser

def bpf_kaiserwin(fs, fstop, fpass, sb_attenuation, pb_ripple):
    """
    Bandpass FIR filter design using the Kaiser window.
    
    Parameters:
    fs              : Sampling frequency (Hz)
    fstop           : Stopband frequencies [fstop1, fstop2] in Hz
    fpass           : Passband frequencies [fpass1, fpass2] in Hz
    sb_attenuation  : Stopband attenuation in dB
    pb_ripple       : Passband ripple in dB
    
    Returns:
    bFilt           : FIR filter coefficients
    """
    # Define cutoff frequencies for Kaiser order
    fcuts = [fstop[0], fpass[0], fpass[1], fstop[1]]
    band_amp = [0, 1, 0]        # Desired amplitudes in each band (stop-pass-stop)
    
    # Maximum allowable deviation in each band
    # dev = [10**(-sb_attenuation / 20), (10**(pb_ripple / 20) - 1) / (10**(pb_ripple / 20) + 1), 10**(-sb_attenuation / 20)]
    dev = [10**(-sb_attenuation / 20), 10**(pb_ripple / 20), 10**(-sb_attenuation / 20)]

    # Calculate order and beta for Kaiser window
    # N: Number of taps
    # beta: Kaiser window beta parameter
    N, beta = kaiserord("Fill in", "Fill in")

    # Design the bandpass filter
    bFilt = firwin(N, [fpass[0] / (0.5 * fs), fpass[1] / (0.5 * fs)], pass_zero=False, window=('kaiser', beta))

    return bFilt