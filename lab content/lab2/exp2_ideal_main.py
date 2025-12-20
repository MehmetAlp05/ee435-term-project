# METU EE435 Lab. Fall 2025 Experiment 2  (TA:Safa Çelik)
# Part_1: Amplitude Modulation with Ideal Envelope Detector
# Updated (Oct. 2025)

import exp2_ideal_Visualizer
import time
import threading
import cv2

# Receiver Settings
sdr = exp2_ideal_Visualizer.Receiver()

# Parameters
sdr.fc = 2e9                    # Modulated signal frequency
sdr.gain = 40                   # Pluto SDR tuner gain in dB

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
time.sleep(1.0)
visualizer_thread.start()
time.sleep(0.1)

sdr.timeScope()
visualizer_thread.join()
cv2.destroyAllWindows()

