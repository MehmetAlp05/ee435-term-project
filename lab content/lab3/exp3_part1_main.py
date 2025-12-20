# METU EE435 Lab. Fall 2025 Experiment 3  (TA:Safa Çelik)
# Part 1: Frequency Synchronization with Squaring Loop
# Updated (Oct. 2025)

from exp3_part1_processData import*

params = parameters()
params.switch = 0                                 # Valid values are 0 and 1. "0": Demodulation with Squaring Loop and "1": Demodulation with sinusoidal carrier signal (cos(2*pi*fc*t))
params.fc = int(240e3)                            # Carrier frequency
params.sim_time = 3                               # Simulation duration

params.fstop = [None, None]                       # Stopband frequencies of BPF at 2fc
params.fpass = [None, None]                       # Passband frequencies of BPF at 2fc

params.fstop2 = [None, None]                      # Stopband frequencies of BPF at fc
params.fpass2 = [None, None]                      # Passband frequencies of BPF at fc

exp3_data = dataManager(params)
exec(f"case{str(int(params.switch))}(data=exp3_data)")