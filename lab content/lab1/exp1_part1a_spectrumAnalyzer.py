# METU EE435 Lab. Fall 2025 Experiment 1  (TA:Safa Çelik)
# Updated (Oct. 2025)

import adi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib as mpl
from scipy.signal import welch
import cv2
import time

class Visualizer:
    def __init__(self, x_lims, waterfall_data, n_waterfall_lines, sdr_frmtime):
        self.firstFrame = True
        self.xlims = x_lims

        # Set up the plots for real-time spectrum and waterfall
        self.fig, (self.ax_spectrum, self.ax_waterfall) = plt.subplots(2, 1, figsize=(8, 6))
        plt.subplots_adjust(top=0.85, bottom=0.1, hspace=0.4)
        
        # Chose a colormap as your choice 
        # Check https://matplotlib.org/stable/users/explain/colors/colormaps.html
        self.colormap = 'rainbow' # You can see your options in the above link

        # Spectrum plot configuration
        self.ax_spectrum.set_title(r'Spectrum Analyzer FFT', family='monospace')
        self.ax_spectrum.set_xlabel(r'Frequency [MHz]', family='monospace')
        self.ax_spectrum.set_ylabel(r'Power [dBm]', family='monospace')

        self.ax_spectrum.set_xlim(self.xlims[0], self.xlims[1])

        self.line_spectrum, = self.ax_spectrum.plot([], [], color='blue', linewidth=1.0)

        # Enable both major and minor ticks
        self.ax_spectrum.minorticks_on()

        # Customize the grid for major and minor ticks
        self.ax_spectrum.grid(which='major', color='gray', linestyle='-', linewidth=0.8)
        self.ax_spectrum.grid(which='minor', color='lightgray', linestyle='--', linewidth=0.5)

        # Waterfall plot configuration
        self.img = self.ax_waterfall.imshow(waterfall_data, extent=[self.xlims[0], self.xlims[1], 0, n_waterfall_lines * sdr_frmtime],
                          aspect='auto', cmap=mpl.colormaps[self.colormap], origin='lower')

        self.ax_waterfall.set_title(r'Spectrum Analyzer Waterfall', family='monospace')
        self.ax_waterfall.set_xlabel(r'Frequency [MHz]', family='monospace')
        self.ax_waterfall.set_ylabel(r'Time [s]', family='monospace')

        # Visualization parameters
        self.lower_threshold = 0.05  # 5%
        self.upper_threshold = 0.95  # 95%

    def visualizeNewData(self, waterfall_data, Pxx_dB, f, new_extent):
        if self.firstFrame:
            pow_mean = np.mean(Pxx_dB)
            self.pow_lims = [pow_mean-50, pow_mean+50]
            self.ax_spectrum.set_ylim(self.pow_lims[0], self.pow_lims[1])     

            # Visualization parameters
            self.histogram_lower_limit = int(self.pow_lims[0])
            self.histogram_upper_limit = int(self.pow_lims[1])
            self.histogram_num_of_bins = self.histogram_upper_limit - self.histogram_lower_limit

            self.firstFrame = False
        
        # Update the spectrum plot
        self.line_spectrum.set_ydata(Pxx_dB)        # Update y-data
        self.line_spectrum.set_xdata(f)             # Update x-data (frequency)

        self.ax_spectrum.relim()                    # Recompute the limits of the axes
        self.ax_spectrum.autoscale_view()           # Update the view
        
        t_start = time.monotonic_ns()

        # We modify the original data for a better visualization
        waterfall_data_tmp = waterfall_data.astype(np.float32)

        # Calculate histogram
        hist = cv2.calcHist([waterfall_data_tmp], [0], None, [self.histogram_num_of_bins], [self.histogram_lower_limit, self.histogram_upper_limit])

        # Normalize the histogram to get the probability density function (PDF)
        pdf = hist / hist.sum()

        # Calculate the CDF
        cdf = np.cumsum(pdf)

        # Get the intensity value corresponding to the 5% limit
        lower_limit = np.interp(self.lower_threshold, cdf, np.linspace(self.histogram_lower_limit, self.histogram_upper_limit, len(cdf)))
        upper_limit = np.interp(self.upper_threshold, cdf, np.linspace(self.histogram_lower_limit, self.histogram_upper_limit, len(cdf)))

        # Normalize the image for better visualization
        waterfall_data_tmp_normalized = (waterfall_data_tmp - lower_limit) / (upper_limit - lower_limit)

        # those smaller than 0 is mapped to 0
        cold_roi = np.where(waterfall_data_tmp_normalized<0)                    
        waterfall_data_tmp_normalized[cold_roi] = 0

        # those bigger than 1 is mapped to 1
        hot_roi = np.where(waterfall_data_tmp_normalized>1)                     
        waterfall_data_tmp_normalized[hot_roi] = 1 

        # Update the waterfall plot
        self.img.set_data(waterfall_data_tmp_normalized)
        self.img.set_clim(vmin=0, vmax=1)  # Set color limits
        self.img.set_extent(new_extent)

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
        # Simulation parameters
        self.run_time = 0
        self.simulation_ended = False

        # Objects that holds the saved data initialized as None
        self.sdr_data = None
        self.f = None
        self.f_MHz = None
        self.Pxx = None
        self.Pxx_dB = None

        # SDR and signal configuration
        self.tunerfreq = 105e6           # Pluto SDR tuner frequency in Hz
        self.sample_rate = 3e6           # Pluto SDR sampling rate in Hz
        self.gain = None                 # Pluto SDR tuner gain in dB
        self.frmlen = 4096               # Number of samples per frame
        self.rx_bw = 3e6                 # Pluto SDR receiver analog bandwidth
        self.sim_time = 5                # Simulation time in seconds

        self.n_waterfall_lines = 100     # Number of lines to display in waterfall

        self.loadParameters()

    def loadParameters(self):
        # Pluto SDR configuration
        self.sdr = adi.Pluto('ip:192.168.2.1')
        self.sdr.rx_lo = int(self.tunerfreq)
        self.sdr.sample_rate = int(self.sample_rate)

        if self.gain is not None:
            self.sdr.gain_control_mode_chan0 = "manual"
            self.sdr.rx_hardwaregain_chan0 = self.gain
        else:
            self.sdr.gain_control_mode_chan0 = "slow_attack" # or "fast_attack"

        if self.rx_bw is not None:
            self.sdr.rx_rf_bandwidth = int(self.rx_bw)
        
        if self.frmlen is not None:
            self.sdr.rx_buffer_size = int(self.frmlen)
        else:
            self.frmlen = self.sdr.rx_buffer_size
        
        self.waterfall_data = np.zeros((self.n_waterfall_lines, self.frmlen))

        # Calculate frame time (time per buffer capture)
        self.sdr_frmtime = self.sdr.rx_buffer_size / self.sdr.sample_rate

        # Visualizer
        x_lims = [(-self.sample_rate / 2 + self.sdr.rx_lo)/(1e6), (self.sample_rate / 2 + self.sdr.rx_lo)/(1e6)]        # Unit is converted to MegaHertz
        self.visualizer = Visualizer(x_lims, self.waterfall_data, self.n_waterfall_lines, self.sdr_frmtime)
    
    def receiveData(self):
        # Receive samples from Pluto SDR
        self.sdr_data = self.sdr.rx()

        self.sdr_data = self.sdr_data / 2**11           # 12-bit ADC scaling (one bit is sign)

        # Compute the power spectral density (PSD) using Welch's method
        self.f, self.Pxx = welch(self.sdr_data, fs=self.sdr.sample_rate, window='hann',
                    nperseg=self.sdr.rx_buffer_size, noverlap=None, 
                    nfft=None, detrend='constant', return_onesided=False,
                    scaling='spectrum', axis=0, average='mean')
        
        self.f = np.fft.fftshift(self.f)
        self.Pxx = np.fft.fftshift(self.Pxx)
        self.f_MHz = (self.f + self.sdr.rx_lo)/(1e6)
        
        self.Pxx_dB = 10 * np.log10(self.Pxx + 1e-12)   # Avoid log of zero with small epsilon
        self.Pxx_dB = self.Pxx_dB + 10 * np.log10(1e3)  # dB to dBm

        # Append the new arrived data into our waterfall_data
        # Shift the waterfall data upwards (old data moves up)
        self.waterfall_data = np.roll(self.waterfall_data, -1, axis=0)
        
        # Update the last row with the current spectrum
        self.waterfall_data[-1, :] = self.Pxx_dB

        self.run_time += self.sdr_frmtime

    def visualize(self):
        # Update the extent to reflect real time in the Y-axis
        new_extent = [(-self.sample_rate / 2 + self.sdr.rx_lo)/(1e6), (self.sample_rate / 2 + self.sdr.rx_lo)/(1e6), self.run_time, self.run_time + self.n_waterfall_lines * self.sdr_frmtime]
        self.simulation_ended = self.visualizer.visualizeNewData(self.waterfall_data.copy(), self.Pxx_dB, self.f_MHz, new_extent)        

