# METU EE435 Lab. Fall 2025 Experiment 2  (TA:Safa Çelik)
# Updated (Oct. 2025)

import numpy as np 

def Pre_Q2_func(AM_signal, fc, fs, RC_const_multiplier):
    
    BW = 10e3
    """
    Parameters:
        AM_signal           : numpy array, Amplitude modulated signal
        fc                  : Carrier frequency (Hz)
        fs                  : Sampling rate (Hz)
        RC_const_multiplier : Multiplier for RC constant

    Returns:
        y : numpy array, Envelope detected signal
    """
    # Diode input voltage
    diode_in = AM_signal

    # Geometric mean of fc and BW (1/fc << RC << 1/fm)
    RC_const = RC_const_multiplier / np.sqrt(fc * BW)
    
    # Number of samples
    frame_size = len(AM_signal)
    
    # Total duration of the signal
    voice_dur = frame_size / fs
    
    # Sampling interval (infinitesimal time duration)
    dt = voice_dur / frame_size
    
    # Initialize previous voltage and output array
    prev_voltage = 0                    # Previos voltage level at the diode output should be
                                        # remembered and should exponentially decay
                                        # within each infinitesimal time interval dt
    RC_filter_out = np.zeros(frame_size)
    
    # When diode is on the output of envelope detector is directly voltage on
    # diode and when it is off the output is previous diode voltage sample
    # which decayed exponentially by exp(-dt/RC_const).
    # Diode detector simulation
    for t_index in range(frame_size):
        RC_filter_out[t_index] = max(prev_voltage * np.exp(-dt / RC_const), "Fill in here"])
        prev_voltage = RC_filter_out[t_index]

    return RC_filter_out