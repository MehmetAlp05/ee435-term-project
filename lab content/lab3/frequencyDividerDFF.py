# METU EE435 Lab. Fall 2025 Experiment 3  (TA:Safa Çelik)
# Updated (Oct. 2025)

import numpy as np

def frequencyDividerDFF(CLK, not_CLR=None):
    """
    Frequency Divider using D Flip-Flop with !Q Feedback.

    Parameters:
    CLK : numpy array
        Clock signal array (high-frequency input).
    not_CLR : numpy array, optional
        Clear signal (1: active, 0: inactive). If not provided, defaults to 1 (active).

    Returns:
    Q_output : numpy array
        The output array after frequency division.
    """
    N = len(CLK)            # Length of the input signal

    # Default not_CLR signal to all ones if not provided
    if not_CLR is None:
        not_CLR = np.ones(N)

    # Initialize parameters
    Qprev = 0                           # Initial state of Q output
    not_Qprev = int(not Qprev)          # Initial state of not Q output
    CLK_prev = -1                       # Initial state for previous clock
    Q_output = np.zeros(N, dtype=int)   # Array to store the Q output

    # D Flip-Flop with feedback logic
    for ii in range(N):
        D = not_Qprev  # D input takes the value of not_Qprev

        # Check the not_clear signal
        if not_CLR[ii] != 1:
            Qcurrent = 0  # Reset the Q output
            not_Qcurrent = int(not Qcurrent)  # Reset the not Q output
        else:
            # Edge detection (rising edge)
            if CLK[ii] == 1 and CLK_prev == -1:
                Qcurrent = D  # Update Q on rising edge
                not_Qcurrent = int(not Qcurrent)
            else:
                Qcurrent = "Fill in"
                not_Qcurrent = "Fill in"

        # Store current Q output
        Q_output[ii] = Qcurrent

        # Update previous states
        Qprev = Qcurrent
        not_Qprev = not_Qcurrent
        CLK_prev = CLK[ii]

    return Q_output
