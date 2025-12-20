# METU EE435 Lab. Fall 2025 Experiment 2  (TA:Safa Çelik)
# Part_2: Amplitude Modulation with RC Detector
# Updated (Oct. 2025)

import exp2_rc_Visualizer_audio
import time
import threading
import cv2
import sounddevice as sd

# Receiver Settings
sdr = exp2_rc_Visualizer_audio.Receiver()

# Parameters
sdr.RC_const_multiplier = 0.1     # Time constant multiplier
sdr.fc = 2e9                      # Modulated signal frequency
sdr.gain = 40                     # Pluto SDR tuner gain in dB
sdr.audio_fs = 44.1e3             # Sampling rate of audio signal 
sdr.listen_time = 10.0

sdr.loadParameters()
time.sleep(0.1)

def sdr_loop():
    while (not sdr.simulation_ended):
        sdr.receiveData()
    
def visualizer_loop():
    while(not sdr.simulation_ended):
        sdr.visualize()

def audio_loop():
    audio = sdr.prepare_audio()
    sd.play(audio, sdr.audio_fs)
    sd.wait()

# sdr keeps receiving new data in a seperate thread
sdr_thread = threading.Thread(target=sdr_loop, args=())
visualizer_thread = threading.Thread(target=visualizer_loop, args=())
audio_thread = threading.Thread(target=audio_loop, args=())

sdr_thread.start()
time.sleep(1.0)
visualizer_thread.start()
time.sleep(0.1)
audio_thread.start()
time.sleep(0.1)

sdr.timeScope()
visualizer_thread.join()
cv2.destroyAllWindows()