# METU EE435 Lab. Fall 2025 Experiment 1  (TA:Safa Çelik)
# Part_1a: Spectra of RF signals
# Updated (Oct. 2025)

import exp1_part1a_spectrumAnalyzer
import time
import threading

# Simulation time in seconds
sdr_Rx = exp1_part1a_spectrumAnalyzer.Receiver()

sdr_Rx.tunerfreq = 2e9         # Pluto SDR center frequency in Hz
sdr_Rx.sample_rate = 2.5e6         # Pluto SDR sampling rate in Hz
sdr_Rx.gain = 60                 # Pluto SDR tuner gain in dB
sdr_Rx.frmlen = 4096             # Number of samples per frame

sdr_Rx.loadParameters()

def sdr_Rx_loop():
    while (not sdr_Rx.simulation_ended):
        sdr_Rx.receiveData()

def visualizer_loop():
    while(not sdr_Rx.simulation_ended):
        sdr_Rx.visualize()

# sdr_Rx keeps receiving new data in a seperate thread
sdr_Rx_thread = threading.Thread(target=sdr_Rx_loop, args=())
visualizer_thread = threading.Thread(target=visualizer_loop, args=())

sdr_Rx_thread.start()

time.sleep(0.1)

visualizer_thread.start()

visualizer_thread.join()





