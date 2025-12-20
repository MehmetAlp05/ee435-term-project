# METU EE435 Lab. Fall 2025 Experiment 4  (TA:Safa Çelik)
# Part 2: Single tone FM noise
# Updated (Nov. 2025)

import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from FM_discriminator import FM_discriminator

# --- Parameters
SNR_dB = np.concatenate([
    np.arange(-5, 1, 1),
    np.arange(1, 4.0, 0.5),
    np.arange(4.5, 9.0, 0.25),
    np.arange(9.5, 12.0, 0.5),
    np.arange(13, 31, 1)
])                                      # Signal-to-noise ratio
fm = 1e3                                # Frequency of the modulating signal [Hz]
T = 1                                   # Duration [s]
MC = 100                                # Number of Monte Carlo simulations
margin = 10

# ---- Do not change ---
def parfor_block(Delta_f):

    return Parallel(n_jobs=-1)(
        delayed(lambda SNR_in: np.mean([

            (lambda signal, noise:
                10*np.log10(
                    np.var(signal[margin:-margin]) /
                    np.var(noise[margin:-margin])
                )
            )(
                *FM_discriminator(T, fm, Delta_f, SNR_in)[0:3:2]
            )

            for _ in range(MC)
        ]))(float(snr))
        for snr in SNR_dB
    )
# ---- Do not change ---

if __name__ == "__main__":

    # --- Δf = 5 kHz
    Delta_f1 = 5e3
    SNR_out_5k = np.array(parfor_block(Delta_f1))

    plt.figure(figsize=(8,5))
    plt.grid(True)

    plt.plot(SNR_dB, SNR_out_5k, 'b-', linewidth=2,
             label=r'$\Delta f = 5~\mathrm{kHz}$')

    plt.xlabel(r'Input SNR (dB)')
    plt.ylabel(r'Output SNR (dB)')
    plt.title(r'SNR Improvement of FM Discriminator')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
