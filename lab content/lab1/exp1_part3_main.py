# METU EE435 Lab. Fall 2025 Experiment 1  (TA:Safa Çelik)
# Part_3: Spectrum of SSB-SC modulated signal
# Updated (Oct. 2025)

import exp1_part3_spectrumAnalyzer
import time
import threading

# Receiver Settings
sdr = exp1_part3_spectrumAnalyzer.Receiver()

# Parameters
sdr.fc = 2e9                 # Pluto SDR tuner frequency in Hz
sdr.gain = 60                # Pluto SDR tuner gain in dB

sdr.loadParameters()
time.sleep(0.1)

def sdr_loop():
    while (not sdr.simulation_ended):
        sdr.receiveData()

def visualizer_loop():
    while(not sdr.simulation_ended):
        sdr.visualize()

# sdr keeps receiving new data in a seperate thread
sdr_thread = threading.Thread(target=sdr_loop, args=())
visualizer_thread = threading.Thread(target=visualizer_loop, args=())

sdr_thread.start()
time.sleep(0.1)
visualizer_thread.start()

visualizer_thread.join()