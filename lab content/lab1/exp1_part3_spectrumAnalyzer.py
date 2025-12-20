# METU EE435 Lab. Fall 2025 Experiment 1  (TA:Safa Çelik)
# Updated (Oct. 2025)

import adi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.signal import welch
import cv2

class Visualizer:
    def __init__(self):
        # Set up the plots for real-time spectrum
        self.fig, self.ax_spectrum = plt.subplots(1, 1, figsize=(8, 6))
        plt.subplots_adjust(top=0.85, bottom=0.1, hspace=0.4)

        # These will be changed automatically
        self.pow_max = -100
        self.pow_min = 100

        # Spectrum plot configuration
        self.ax_spectrum.set_title(r'Spectrum Analyzer', family='monospace')
        self.ax_spectrum.set_xlabel(r'Frequency [GHz]', family='monospace')
        self.ax_spectrum.set_ylabel(r'Power [dBm]', family='monospace')

        self.rec_signal, = self.ax_spectrum.plot([], [], color='blue', linewidth=1.5)
        
        self.top_left_dot, = self.ax_spectrum.plot([], [], 'ro')                    # Red dot for the maximum value
        self.top_left_text = self.ax_spectrum.text(0, 0, '')                        # Text placeholder for maximum value

        self.top_right_dot, = self.ax_spectrum.plot([], [], 'ro')                   # Red dot for the maximum value
        self.top_right_text = self.ax_spectrum.text(0, 0, '', family='monospace')   # Text placeholder for maximum value

        # Enable both major and minor ticks
        self.ax_spectrum.minorticks_on()

        # Customize the grid for major and minor ticks
        self.ax_spectrum.grid(which='major', color='gray', linestyle='-', linewidth=0.8)
        self.ax_spectrum.grid(which='minor', color='lightgray', linestyle='--', linewidth=0.5)

    def visualizeNewData(self, Pxx_dB, f):
        # Set the x axis limits
        center = np.mean(f)
        f_min = center - 60e3/1e9
        f_max = center + 60e3/1e9
        
        self.ax_spectrum.set_xlim(f_min, f_max)
        
        self.ax_spectrum.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x: .4f}'))

        # Set the y axis limits
        pow_max = np.max(Pxx_dB)
        pow_min = np.min(Pxx_dB)

        if pow_max > self.pow_max:
            self.pow_max = pow_max + 10

        if pow_min < self.pow_min:
            self.pow_min = pow_min - 5

        self.ax_spectrum.set_ylim(self.pow_min, self.pow_max)     

        # Update the spectrum plot
        self.rec_signal.set_ydata(Pxx_dB)  # Update y-data
        self.rec_signal.set_xdata(f)       # Update x-data (frequency)

        self.show_max = True
        if self.show_max:
            half_data_len = int(0.5*len(f))

            ###  Visualize the Left Side Top
            # Get the left side data
            x_tmp = f[:half_data_len]
            y_tmp = Pxx_dB[:half_data_len]

            # Find the maximum y value and its corresponding x value
            max_index = np.argmax(y_tmp)            # Index of the maximum y value
            max_x = x_tmp[max_index]                # x value at the maximum
            max_y = y_tmp[max_index]                # Maximum y value

            # Update the dot to be at the maximum point
            self.top_left_dot.set_data([max_x], [max_y]) 
            
            offset_kHz = (max_x - center) * 1e6
            self.top_left_text.set_position((max_x, max_y))
            self.top_left_text.set_text(f'({offset_kHz:+.3f} kHz, {max_y:.2f} dBm)')

            ###  Visualize the Right Side Top
            # Get the right side data
            x_tmp = f[half_data_len:]
            y_tmp = Pxx_dB[half_data_len:]

            # Find the maximum y value and its corresponding x value
            max_index = np.argmax(y_tmp)            # Index of the maximum y value
            max_x = x_tmp[max_index]                # x value at the maximum
            max_y = y_tmp[max_index]                # Maximum y value

            # Update the dot to be at the maximum point
            self.top_right_dot.set_data([max_x], [max_y]) 
            
            offset_kHz = (max_x - center) * 1e6
            self.top_right_text.set_position((max_x, max_y))
            self.top_right_text.set_text(f'({offset_kHz:+.3f} kHz, {max_y:.2f} dBm)')

        canvas = FigureCanvas(self.fig)
        canvas.draw()
        # img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        # img = img.reshape(canvas.get_width_height()[::-1] + (3,))
        img = np.frombuffer(canvas.tostring_argb(), dtype='uint8')
        img = img.reshape(canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, 1:4]

        # Convert the image to OpenCV format (BGR)
        img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imshow("Spectrum Analyzer", img_cv)
        key = cv2.waitKey(1)

        # If user preasses "q", end the simulation
        if key == ord("q") or key == ord("Q"):
            return True
        else:
            return False


class Receiver:
    def __init__(self):
        # Visualizer
        self.visualizer = Visualizer()

        # Simulation parameters
        self.simulation_ended = False

        # Objects that holds the saved data initialized as None
        self.sdr_data = None
        self.f = None
        self.f_MHz = None
        self.Pxx = None
        self.Pxx_dB = None
        self.Pxx_bandpassed = None
        self.fc = None                   # Pluto SDR tuner frequency in Hz
        self.gain = None                 # Pluto SDR tuner gain in dB
        
        self.sample_rate = 1e6           # Pluto SDR sampling rate in Hz
        self.frmlen = 20e3               # Number of samples per frame
        
    def loadParameters(self):
        # Pluto SDR configuration
        self.sdr = adi.Pluto('ip:192.168.2.1')
        self.sdr.rx_lo = int(self.fc)
        self.sdr.sample_rate = int(self.sample_rate)

        if self.gain is not None:
            self.sdr.gain_control_mode_chan0 = "manual"
            self.sdr.rx_hardwaregain_chan0 = self.gain              # Set gain
        else:
            self.sdr.gain_control_mode_chan0 = "slow_attack"        # or "fast_attack"
        
        self.sdr.rx_rf_bandwidth = int(self.sample_rate)
                
        if self.frmlen is not None:
            self.sdr.rx_buffer_size = int(self.frmlen)
        else:
            self.frmlen = int(self.sdr.rx_buffer_size)

    def receiveData(self):
        self.sdr_data = self.sdr.rx()               # Receive samples from Pluto SDR

        self.sdr_data = self.sdr_data / 2**11       # 12-bit ADC scaling (one bit is sign)

        self.f, self.Pxx = welch(self.sdr_data, fs=self.sdr.sample_rate, window='hann',
                    nperseg=self.sdr.rx_buffer_size, noverlap=None, 
                    nfft=None, detrend=None, return_onesided=False,
                    scaling='spectrum', axis=0, average='mean')
        
        self.f = np.fft.fftshift(self.f)
        self.f_kHz = (self.f + self.fc) / (1e9)

        self.Pxx = np.fft.fftshift(self.Pxx)
        self.Pxx_dB = 10 * np.log10(self.Pxx + 1e-12)      # Avoid log of zero with small epsilon
        self.Pxx_dB = self.Pxx_dB + 10 * np.log10(1e3)     # dB to dBm

    def visualize(self):
        self.simulation_ended = self.visualizer.visualizeNewData(self.Pxx_dB, self.f_kHz)        