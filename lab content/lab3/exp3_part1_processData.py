# METU EE435 Lab. Fall 2025 Experiment 3  (TA:Safa Çelik)
# Part 1: Frequency Synchronization with Squaring Loop
# Updated (Oct. 2025)

import numpy as np
from scipy.fftpack import fft, fftshift
from scipy.signal import resample_poly, lfilter
import scipy.io as sio
import sounddevice as sd
import sys
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import time
from bpf_kaiserwin import bpf_kaiserwin
from frequencyDividerDFF import frequencyDividerDFF
import threading

class parameters:
    def __init__(self):

        self.sim_time = None                      # Simulation time in seconds
        self.switch = None                        # Simulation mode
        self.fc = None                            # Carrier frequency
        
        self.fstop = [None, None]                 # Stopband frequencies of BPF at 2fc
        self.fpass = [None, None]                 # Passband frequencies of BPF at 2fc

        self.fstop2 = [None, None]                # Stopband frequencies of BPF at fc
        self.fpass2 = [None, None]                # Passband frequencies of BPF at fc


class dataManager:
    def __init__(self,params:parameters):

        self.params = params

        # System parameters (do not change!)
        self.fs = int(1.5e6)                                                       # Sample rate
        self.fs_audio = int(48e3)                                                  # Sample rate of audio signal
               
        self.exp3_data = sio.loadmat('Exp3_data2.mat')                             # Load the data
        self.dsb_sc_all = self.exp3_data['dsb_sc']                                 # DSB-SC modulated data
        self.t_dsb_sc_all = self.exp3_data['t_dsb_sc']                             # Time array of the DSB-SC modulated data
        self.dsb_sc_rows, self.dsb_sc_cols = self.dsb_sc_all.shape                 # Size of DSB-SC modulated data

        self.L_data = self.dsb_sc_rows * self.dsb_sc_cols                                  
        self.total_duration = self.L_data / self.fs
        
        if self.params.sim_time > self.total_duration:
            print("Error : Simulation duration should be less then 18.3 seconds")
            sys.exit()

        self.data_audio = self.dsb_sc_all[:int(self.fs*self.params.sim_time)]   # Truncated data for the sim_time

    def case0(self):
        print("Processing : Squaring Loop...Wait please...")
        self.stop_atten = 30                                                    # Stopband attenuation of BPF at 2fc (dB)
        self.pass_ripple = 1                                                    # Passband ripple of BPF at 2fc (dB)
        # Filter coefficients of BPF at 2fc
        self.bFilt = bpf_kaiserwin(self.fs, self.params.fstop, self.params.fpass, self.stop_atten, self.pass_ripple)
        self.aFilt = 1

        self.stop_atten2 = 40                                                   # Stopband attenuation of BPF at fc (dB)
        self.pass_ripple2 = 1                                                   # Passband ripple of BPF at fc (dB)
        # Filter coefficients of BPF at fc
        self.bFilt2 = bpf_kaiserwin(self.fs, self.params.fstop2, self.params.fpass2, self.stop_atten2, self.pass_ripple2)
        self.aFilt2 = 1                                                              

        self.animation_loop = 20
        self.buffer_size = 4096
        self.data_animation = self.dsb_sc_all[:self.animation_loop*self.buffer_size]
        self.data_animation = self.data_animation.reshape((self.animation_loop, self.buffer_size))
        
        self.Nfft = int(1e1*4096)                                                # FFT size for spectrum
        self.f = np.linspace(-self.fs/2, self.fs/2, self.Nfft, endpoint=False)   # Frequency vector for power spectra

        self.Dsb_sc_dict = {}
        self.Squared_dict = {}
        self.Bpf_2fc_dict = {}
        self.HardLim_dict = {}
        self.FreqDiv_dict = {}
        
        self.Bpf_fc_dict = {}

        for idx in range(self.animation_loop):
            self.dsb_sc_loop = 10 * self.data_animation[idx, :]
            self.Dsb_sc = fftshift(fft(self.dsb_sc_loop, self.Nfft))
            self.Dsb_sc = 30 + 10 * np.log10(np.abs(self.Dsb_sc / self.Nfft) ** 2 + 1e-16)
            self.squared = self.dsb_sc_loop ** 2
            self.Squared = fftshift(fft(self.squared, self.Nfft))
            self.Squared = 30 + 10 * np.log10(np.abs(self.Squared / self.Nfft) ** 2 + 1e-16)
            self.bpf_2fc = lfilter(self.bFilt, self.aFilt, self.squared, axis=-1, zi=None)
            self.Bpf_2fc = fftshift(fft(self.bpf_2fc, self.Nfft))
            self.Bpf_2fc = 30 + 10 * np.log10(np.abs(self.Bpf_2fc / self.Nfft) ** 2 + 1e-16)

            self.hardLim = np.sign(self.bpf_2fc)
            self.Hardlim = fftshift(fft(self.hardLim, self.Nfft))
            self.Hardlim = 30 + 10 * np.log10(np.abs(self.Hardlim / self.Nfft) ** 2 + 1e-16)
            self.freqDiv = frequencyDividerDFF(self.hardLim, not_CLR=None)
            self.FreqDiv = fftshift(fft(self.freqDiv, self.Nfft))
            self.FreqDiv = 30 + 10 * np.log10(np.abs(self.FreqDiv / self.Nfft) ** 2 + 1e-16)

            self.bpf_fc = lfilter(self.bFilt2, self.aFilt2, self.freqDiv, axis=-1, zi=None)
            self.Bpf_fc = fftshift(fft(self.bpf_fc, self.Nfft))
            self.Bpf_fc = 30 + 10 * np.log10(np.abs(self.Bpf_fc / self.Nfft) ** 2 + 1e-16)

            self.Dsb_sc_dict.update({idx:self.Dsb_sc})
            self.Squared_dict.update({idx:self.Squared})
            self.Bpf_2fc_dict.update({idx:self.Bpf_2fc})
            self.HardLim_dict.update({idx:self.Hardlim})
            self.FreqDiv_dict.update({idx:self.FreqDiv})

            self.Bpf_fc_dict.update({idx:self.Bpf_fc})

        # Apply all process to the all data
        squared_all = 10*self.data_audio.reshape(-1) ** 2
        bpf_2fc_all = lfilter(self.bFilt, self.aFilt, squared_all, axis=0, zi=None)
        hardlim_all = np.sign(bpf_2fc_all)
        freqDiv_all = frequencyDividerDFF(hardlim_all, not_CLR=None)
        bpf_fc_all = lfilter(self.bFilt2, self.aFilt2, freqDiv_all, axis=0, zi=None)
        self.dsb_sc_demod_all = bpf_fc_all * self.data_audio.reshape(-1)

    def case1(self):
        print("Processing : Direct...")
        t_array = np.linspace(0, self.params.sim_time, int(self.fs * self.params.sim_time), endpoint=False)
        self.dsb_sc_demod_all = np.cos(2*np.pi*self.params.fc*t_array) * self.data_audio.reshape(-1)
        
    
    def listen_audio(self):
        # Resample the demodulated signal to audio sampling rate
        m_hat = resample_poly(self.dsb_sc_demod_all, self.fs_audio, self.fs)
        m_hat = m_hat / np.max(np.abs(m_hat))                           # Normalize to [-1, 1]

        # Play the audio (ensure sounddevice is properly configured for fs_audio rate)
        sd.play(m_hat, self.fs_audio)
        sd.wait()

class Visualizer:
    def __init__(self,data:dataManager):
        # Figure 1: Spectrum Analyzer 1
        self.fig1, self.ax1 = plt.subplots()
        self.fig1.canvas.manager.window.move(0, 100)  # Position the first figure at (100, 100)
        self.ax1.set_title("Spectrum Analyzer 1")
        self.ax1.set_xlabel("Frequency (kHz)")
        self.ax1.set_ylabel("Power (dBm)")
        self.ax1.set_xlim(data.f[0]/1e3, data.f[-1]/1e3)
        self.ax1.set_ylim(-100, 20)
        self.line_dsb_sc, = self.ax1.plot([], [], label="DSB-SC", color="k")
        self.line_squared, = self.ax1.plot([], [], label="Squared", color="b")
        self.line_bpf_2fc, = self.ax1.plot([], [], label="BPF at 2fc", color="r")
        self.ax1.legend()
        self.ax1.grid(True)

        # Figure 2: Spectrum Analyzer 2
        self.fig2, self.ax2 = plt.subplots()
        self.fig2.canvas.manager.window.move(600, 100)  # Position the second figure at (700, 100)
        self.ax2.set_title("Spectrum Analyzer 2")
        self.ax2.set_xlabel("Frequency (kHz)")
        self.ax2.set_ylabel("Power (dBm)")
        self.ax2.set_xlim(data.f[0]/1e3, data.f[-1]/1e3)
        self.ax2.set_ylim(-100, 10)
        self.line_hardLim, = self.ax2.plot([], [], label="Hard Limiter", color="b")
        self.line_freqDiv, = self.ax2.plot([], [], label="Frequency Divider", color="r")
        self.ax2.legend()
        self.ax2.grid(True)

        self.line_dsb_sc.set_xdata(data.f/1e3)
        self.line_squared.set_xdata(data.f/1e3)
        self.line_bpf_2fc.set_xdata(data.f/1e3)
        self.line_hardLim.set_xdata(data.f/1e3)
        self.line_freqDiv.set_xdata(data.f/1e3)

        # Figure 3: Spectrum Analyzer 3
        self.fig3, self.ax3 = plt.subplots()
        self.fig3.canvas.manager.window.move(1200, 100)  # Position the second figure at (700, 100)
        self.ax3.set_title("Spectrum Analyzer 3")
        self.ax3.set_xlabel("Frequency (kHz)")
        self.ax3.set_ylabel("Power (dBm)")
        self.ax3.set_xlim(data.f[0]/1e3, data.f[-1]/1e3)
        self.ax3.set_ylim(-100, 10)
        self.line_bpf_fc, = self.ax3.plot([], [], label="BPF at fc", color="b")
        self.ax3.legend()
        self.ax3.grid(True)

        self.line_bpf_fc.set_xdata(data.f/1e3)
    
    def set_data(self,data:dataManager):
        # Loop through the data and update the plot
        self.finished = False
        for idx in range(data.animation_loop):
            self.line_dsb_sc.set_ydata(data.Dsb_sc_dict[idx])
            self.line_squared.set_ydata(data.Squared_dict[idx])
            self.line_bpf_2fc.set_ydata(data.Bpf_2fc_dict[idx])

            self.line_hardLim.set_ydata(data.HardLim_dict[idx])
            self.line_freqDiv.set_ydata(data.FreqDiv_dict[idx])

            self.line_bpf_fc.set_ydata(data.Bpf_fc_dict[idx])

            time.sleep(0.2)
        self.finished = True    


    def show(self):
        # Loop through the data and update the plot
        self.fig1.canvas.draw()
        self.fig1.canvas.flush_events()
        self.fig2.canvas.draw()
        self.fig2.canvas.flush_events()

        self.fig3.canvas.draw()
        self.fig3.canvas.flush_events()
        plt.pause(0.001)  # Pause for 1 second to simulate real-time update


def case0(data:dataManager):
    data.case0()
    Vis = Visualizer(data=data)

    def data_set_loop():
        Vis.set_data(data=data)

    def data_listen_loop():
        data.listen_audio()

    data_update_thread = threading.Thread(target=data_set_loop, args=())
    audio_listen_thread = threading.Thread(target=data_listen_loop, args=())
    data_update_thread.start()
    audio_listen_thread.start()

    while Vis.finished is False:
        Vis.show()
    plt.show()


def case1(data:dataManager):
    data.case1()
    print("Proccessing the audio...")
    data.listen_audio()