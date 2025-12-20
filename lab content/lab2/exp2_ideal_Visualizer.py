# METU EE435 Lab. Fall 2025 Experiment 2  (TA:Safa Çelik)
# Part_1: Amplitude Modulation with Ideal Envelope Detector
# Updated (Oct. 2025)

import adi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.signal import welch
from scipy.signal import remez, filtfilt 
import cv2

class timeScope:
    def __init__(self):
        self.fig, self.ax_spectrum = plt.subplots(1, 1, figsize=(8, 4), sharex=True, num='Time Scope')
        
        # Time scope plot configuration
        self.ax_spectrum.set_title(r'Time Scope', family='monospace')
        self.ax_spectrum.set_xlabel(r'Time [sec]', family='monospace')
        self.ax_spectrum.set_ylabel(r'Amplitude [mV]', family='monospace')
        
        # Time scope for bandpass filtered signal and demodulated signal
        self.real_part, = self.ax_spectrum.plot([], [], color='orange', linewidth=1.0, label='Filtered (real)')
        self.imag_part, = self.ax_spectrum.plot([], [], color='blue', linewidth=1.0, label='Filtered (imag)')
        self.demod, = self.ax_spectrum.plot([], [], color='red', linewidth=1.0, label='Demodulated')
        self.ax_spectrum.legend()

        # Enable both major and minor ticks
        self.ax_spectrum.minorticks_on()

        # Customize the grid for major and minor ticks
        self.ax_spectrum.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.7)

    def addNewData(self, t, real, imag, demod):
        self.ax_spectrum.plot(t, real, color='orange', linewidth=1.0)
        self.ax_spectrum.plot(t, imag, color='blue', linewidth=1.0)
        self.ax_spectrum.plot(t, demod, color='red', linewidth=1.0)

    def visualize(self):
        fig = self.fig 
        canvas = FigureCanvas(fig)
        canvas.draw()
        # img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        # img = img.reshape(canvas.get_width_height()[::-1] + (3,))
        img = np.frombuffer(canvas.tostring_argb(), dtype='uint8')
        img = img.reshape(canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, 1:4]

        # Convert the image to OpenCV format (BGR)
        img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("Time Scope", img_cv)
        key = cv2.waitKey(1)


class spectrumAnalyzerModulated:
    def __init__(self):
        # Set up the plots for real-time spectrum
        self.fig, self.ax_spectrum = plt.subplots(1, 1, figsize=(7, 5))
        plt.subplots_adjust(top=0.85, bottom=0.1, hspace=0.4)

        # These will be changed automatically
        self.pow_max = -100
        self.pow_min = 100

        # Spectrum plot configuration
        self.ax_spectrum.set_title(r'Spectrum Analyzer Modulated', family='monospace')
        self.ax_spectrum.set_xlabel(r'Frequency [kHz]', family='monospace')
        self.ax_spectrum.set_ylabel(r'Power [dBm]', family='monospace')

        # Spectrum data for incoming signal and bandpass filtered signals
        self.rec_signal, = self.ax_spectrum.plot([], [], color='orange', linewidth=1.5, label='Received')
        self.bpf_signal, = self.ax_spectrum.plot([], [], color='blue', linewidth=1.5, label='Filtered')
        self.ax_spectrum.legend()
        
        self.top_right_dot, = self.ax_spectrum.plot([], [], 'ro')                  # Red dot for the maximum value
        self.top_right_text = self.ax_spectrum.text(0, 0, '', family='monospace')  # Text placeholder for maximum value
        
        # Enable both major and minor ticks
        self.ax_spectrum.minorticks_on()
        
        # Customize the grid for major and minor ticks
        self.ax_spectrum.grid(which='major', color='gray', linestyle='-', linewidth=0.8)         # Major grid
        self.ax_spectrum.grid(which='minor', color='lightgray', linestyle='--', linewidth=0.5)   # Minor grid

    def visualizeNewData(self, Pxx_dB, Pxx_bpf_dB, f):
        # Set the x axis limits
        f_min = np.min(f)
        f_max = np.max(f)
        # # Zoom in ## scaled to kHz. To see 100 kHz, type 100.
        # f_min = 
        # f_max = 
        # # Zoom in ##
        self.ax_spectrum.set_xlim(f_min, f_max)
        self.ax_spectrum.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x: .2f}'))

        # Set the y axis limits
        pow_max = np.max(Pxx_bpf_dB)
        pow_min = np.min(Pxx_bpf_dB)

        if pow_max > self.pow_max:
            self.pow_max = pow_max + 10

        if pow_min < self.pow_min:
            self.pow_min = pow_min - 5

        self.ax_spectrum.set_ylim(self.pow_min, self.pow_max)     

        # Update the spectrum plot
        self.rec_signal.set_ydata(Pxx_dB)       # Update y-data
        self.rec_signal.set_xdata(f)            # Update x-data (frequency)

        self.bpf_signal.set_ydata(Pxx_bpf_dB)   # Update y-data
        self.bpf_signal.set_xdata(f)            # Update x-data (frequency)
        
        self.show_max = True
        if self.show_max:
            half_data_len = int(0.5*len(f))

            # Get the right side data
            x_tmp = f[half_data_len:]
            y_tmp = Pxx_bpf_dB[half_data_len:]

            # Find the maximum y value and its corresponding x value
            max_index = np.argmax(y_tmp)     # Index of the maximum y value
            max_x = x_tmp[max_index]         # X value at the maximum
            max_y = y_tmp[max_index]         # Maximum y value

            # Update the dot to be at the maximum point
            self.top_right_dot.set_data([max_x], [max_y]) 
            self.top_right_text.set_position((max_x, max_y))
            self.top_right_text.set_text(f'({max_x: .3f}, {max_y:.3f})')            
        
        canvas = FigureCanvas(self.fig)
        canvas.draw()
        # img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        # img = img.reshape(canvas.get_width_height()[::-1] + (3,))
        img = np.frombuffer(canvas.tostring_argb(), dtype='uint8')
        img = img.reshape(canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, 1:4]

        # Convert the image to OpenCV format (BGR)
        img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("Modulated", img_cv)
        key = cv2.waitKey(1)

        # If user preasses "q", end the simulation
        if key == ord("q") or key == ord("Q"):
            return True
        else:
            return False
        
 
class spectrumAnalyzerDemodulated:
    def __init__(self):
        # Set up the plots for real-time spectrum
        self.fig, self.ax_spectrum = plt.subplots(1, 1, figsize=(7, 5))
        plt.subplots_adjust(top=0.85, bottom=0.1, hspace=0.4)  # Adjust space and margins

        # These will be changed automatically
        self.pow_max = -100
        self.pow_min = 100

        # Spectrum plot configuration
        self.ax_spectrum.set_title(r'Spectrum Analyzer', family='monospace')
        self.ax_spectrum.set_xlabel(r'Frequency [kHz]', family='monospace')
        self.ax_spectrum.set_ylabel(r'Power [dBm]', family='monospace')

        # Spectrum data for demodulated signal
        self.demod_signal, = self.ax_spectrum.plot([], [], color='black', linewidth=1.5, label='Demodulated')
        self.ax_spectrum.legend()

        self.top_all_dot, = self.ax_spectrum.plot([], [], 'ro')                  # Red dot for the maximum value
        self.top_all_text = self.ax_spectrum.text(0, 0, '', family='monospace')  # Text placeholder for maximum value
        
        # Enable both major and minor ticks
        self.ax_spectrum.minorticks_on()
        
        # Customize the grid for major and minor ticks
        self.ax_spectrum.grid(which='major', color='gray', linestyle='-', linewidth=0.8)        # Major grid
        self.ax_spectrum.grid(which='minor', color='lightgray', linestyle='--', linewidth=0.5)  # Minor grid

    def visualizeNewData(self, Pxx_demod_dB, f):
        # Set the x axis limits
        f_min = np.min(f)
        f_max = np.max(f)
        # # Zoom in ## scaled to kHz. To see 100 kHz, type 100.
        # f_min = 
        # f_max = 
        # # Zoom in ##
        self.ax_spectrum.set_xlim(f_min, f_max)
        self.ax_spectrum.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x: .2f}'))      
        self.ax_spectrum.set_xlim(f_min, f_max)
        self.ax_spectrum.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x: .2f}'))

        # Set the y axis limits
        pow_max = np.max(Pxx_demod_dB)
        pow_min = np.min(Pxx_demod_dB)

        if pow_max > self.pow_max:
            self.pow_max = pow_max + 10

        if pow_min < self.pow_min:
            self.pow_min = pow_min - 5

        self.ax_spectrum.set_ylim(self.pow_min, self.pow_max)     

        # Update the spectrum plot
        self.demod_signal.set_ydata(Pxx_demod_dB)   # Update y-data
        self.demod_signal.set_xdata(f)              # Update x-data (frequency)

        self.show_max = True
        if self.show_max:
            half_data_len = int(len(f))

            # Get the left side data
            x_tmp = f[:half_data_len]
            y_tmp = Pxx_demod_dB[:half_data_len]

            # Find the maximum y value and its corresponding x value
            max_index = np.argmax(y_tmp)     # Index of the maximum y value
            max_x = x_tmp[max_index]         # X value at the maximum
            max_y = y_tmp[max_index]         # Maximum y value

            # Update the dot to be at the maximum point
            self.top_all_dot.set_data([max_x], [max_y]) 
            
            self.top_all_text.set_position((max_x, max_y))
            self.top_all_text.set_text(f'({max_x: .3f}, {max_y:.3f})')
            
        canvas = FigureCanvas(self.fig)
        canvas.draw()
        # img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        # img = img.reshape(canvas.get_width_height()[::-1] + (3,))
        img = np.frombuffer(canvas.tostring_argb(), dtype='uint8')
        img = img.reshape(canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, 1:4]

        # Convert the image to OpenCV format (BGR)
        img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        cv2.imshow("Demodulated", img_cv)
        key = cv2.waitKey(1)

        # If user preasses "q", end the simulation
        if key == ord("q") or key == ord("Q"):
            return True
        else:
            return False


class Receiver:
    def __init__(self):
        # Visualizer
        self.modulated_visualizer = spectrumAnalyzerModulated()
        self.demodulated_visualizer = spectrumAnalyzerDemodulated()
        self.timeScopeVis = timeScope()

        # Simulation parameters
        self.simulation_ended = False

        # Objects that holds the saved data initialized as None
        self.sdr_data = None
        self.f = None
        self.f_MHz = None
        self.Pxx = None
        self.Pxx_dB = None
        self.Pxx_bandpassed = None
        self.fc = None
        self.gain = None                   # Pluto SDR tuner gain in dB
        
        self.Deltaf = 400e3                # IF frequency to visualize the modulation
        self.passband = 40e3               # Bandwidth of bandpass filter  
        self.sample_rate = 1e6             # Pluto SDR sampling rate in Hz
        self.frmlen = 20e3                 # Number of samples per frame
        self.fftsize = 4096                # FFT size for spectrum

    def loadParameters(self):
        # Pluto SDR configuration
        self.sdr = adi.Pluto('ip:192.168.2.1')
        self.tunerfreq = self.fc - self.Deltaf
        self.sdr.rx_lo = int(self.tunerfreq)
        self.sdr.sample_rate = int(self.sample_rate)

        if self.gain is not None:
            self.sdr.gain_control_mode_chan0 = "manual"
            self.sdr.rx_hardwaregain_chan0 = self.gain  # Set gain
        else:
            self.sdr.gain_control_mode_chan0 = "slow_attack" # or "fast_attack"

        self.sdr.rx_rf_bandwidth = int(self.sample_rate)
        
        if self.frmlen is not None:
            self.sdr.rx_buffer_size = int(self.frmlen)
        else:
            self.frmlen = int(self.sdr.rx_buffer_size)

        # Bandpass filter parameters
        self.stopband1 = (self.Deltaf - self.passband/2)
        self.passband1 = (self.Deltaf - self.passband/2 + 5e3)
        self.passband2 = (self.Deltaf + self.passband/2 - 5e3)
        self.stopband2 = (self.Deltaf + self.passband/2)
        # Create bandpass FIR filter using remez
        self.freqs = [0, self.stopband1, self.passband1, self.passband2, self.stopband2, self.sample_rate/2]
        self.gains = [0, 1, 0]
        self.weights = [30, 1, 30]
        self.numtaps = 501
        self.bpFilt = remez(self.numtaps, self.freqs, self.gains, weight=self.weights, fs=self.sample_rate)

    def receiveData(self):
        self.sdr_data = self.sdr.rx()                               # Receive samples from Pluto SDR

        self.sdr_data = self.sdr_data / 2**11                       # 12-bit ADC scaling (one bit is sign)

        self.bpf_sig = filtfilt(self.bpFilt, 1.0, self.sdr_data)    # Filtered signal
        
        self.demod  = np.abs(self.bpf_sig)                          # Envelope detection (absolute value)

        self.f, self.Pxx = welch(self.sdr_data, fs=self.sdr.sample_rate, window='hann',
                    nperseg=self.fftsize, noverlap=None, 
                    nfft=None, detrend=None, return_onesided=False,
                    scaling='spectrum', axis=0, average='mean')
        
        self.f, self.Pxx_bpf = welch(self.bpf_sig, fs=self.sdr.sample_rate, window='hann',
                    nperseg=self.fftsize, noverlap=None, 
                    nfft=None, detrend=None, return_onesided=False,
                    scaling='spectrum', axis=0, average='mean')
        
        self.f, self.Pxx_demod = welch(self.demod, fs=self.sdr.sample_rate, window='hann',
                    nperseg=self.fftsize, noverlap=None, 
                    nfft=None, detrend=None, return_onesided=False,
                    scaling='spectrum', axis=0, average='mean')

        self.f = np.fft.fftshift(self.f)
        self.f_kHz = self.f / 1e3
        
        self.Pxx = np.fft.fftshift(self.Pxx)
        self.Pxx_dB = 10 * np.log10(self.Pxx + 1e-12)                           # Avoid log of zero with small epsilon
        self.Pxx_dB = self.Pxx_dB + 10 * np.log10(1e3)                          # dB to dBm

        self.Pxx_bpf = np.fft.fftshift(self.Pxx_bpf)
        self.Pxx_bpf_dB = 10 * np.log10(self.Pxx_bpf + 1e-12)                   # Avoid log of zero with small epsilon
        self.Pxx_bpf_dB = self.Pxx_bpf_dB + 10 * np.log10(1e3)                  # dB to dBm

        self.Pxx_demod = np.fft.fftshift(self.Pxx_demod)
        self.Pxx_demod_dB = 10 * np.log10(self.Pxx_demod + 1e-12)               # Avoid log of zero with small epsilon
        self.Pxx_demod_dB = self.Pxx_demod_dB + 10 * np.log10(1e3)              # dB to dBm

    def visualize(self):
        self.simulation_ended = self.modulated_visualizer.visualizeNewData(self.Pxx_dB, self.Pxx_bpf_dB, self.f_kHz)
        self.simulation_ended = self.demodulated_visualizer.visualizeNewData(self.Pxx_demod_dB, self.f_kHz)

    def timeScope(self):
        simulation_duration = 0.1

        self.timeScopeVis.ax_spectrum.set_xlim([0, simulation_duration])
        frame_time = self.frmlen / self.sample_rate

        t_all = np.linspace(0, simulation_duration, int(simulation_duration * self.sample_rate), endpoint=False)
        
        number_of_intervals = simulation_duration / frame_time
        data_bpf_real = []
        data_bpf_imag = []
        data_demod = []
        for idx in range(int(number_of_intervals)):
            # Bandpass filtered (real)
            data, = self.timeScopeVis.ax_spectrum.plot([], [], color='orange', linewidth=1.5)
            start_idx = idx * int(self.frmlen)
            stop_idx = (idx + 1) * int(self.frmlen)
            t = t_all[start_idx:stop_idx]
            y = np.zeros_like(t)
            data.set_xdata(t) 
            data.set_ydata(y) 
            data_bpf_real.append(data)

            # Bandpass filtered (imag)
            data, = self.timeScopeVis.ax_spectrum.plot([], [], color='blue', linewidth=1.5)
            start_idx = idx * int(self.frmlen)
            stop_idx = (idx + 1) * int(self.frmlen)
            t = t_all[start_idx:stop_idx]
            y = np.zeros_like(t)
            data.set_xdata(t) 
            data.set_ydata(y) 
            data_bpf_imag.append(data)

            # Demodulated
            data, = self.timeScopeVis.ax_spectrum.plot([], [], color='red', linewidth=1.5)
            start_idx = idx * int(self.frmlen)
            stop_idx = (idx + 1) * int(self.frmlen)
            t = t_all[start_idx:stop_idx]
            y = np.zeros_like(t)
            data.set_xdata(t) 
            data.set_ydata(y) 
            data_demod.append(data)

        run_time = 0.0
        index_counter = 0

        plt.close(self.modulated_visualizer.fig)
        plt.close(self.demodulated_visualizer.fig)
        while run_time<simulation_duration and index_counter<len(data_bpf_real):
            # Time vector for the current frame
            t = np.linspace(run_time, run_time + frame_time, int(self.frmlen), endpoint=False)

            data_bpf_real[index_counter].set_ydata(self.bpf_sig.real * 1e3) 
            data_bpf_imag[index_counter].set_ydata(self.bpf_sig.imag * 1e3) 
            data_demod[index_counter].set_ydata(self.demod * 1e3) 
            
            self.timeScopeVis.ax_spectrum.relim()           # Recalculate limits
            self.timeScopeVis.ax_spectrum.autoscale_view()  # Autoscale based on current dat

            plt.show(block = False)                         # Show the plot but do not block 
            run_time = run_time + frame_time
            index_counter += 1
        plt.show()                                          # Show the plot but do not block