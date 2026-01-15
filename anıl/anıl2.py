import numpy as np
import scipy.signal as sig
from scipy.signal import hilbert
import adi
import time

# ============================================================
# PARAMETERS (MATCH TX)
# ============================================================
fs = 2e6
fc = 2e9
rx_gain = 30
run_time_s = 10.0

# FFT / Detection
FFT_LEN = 4096
DECIM = 2                  # downsample factor
CFAR_TRAIN = 24
CFAR_GUARD = 6
P_FA = 1e-4

# Burst detection
POWER_THRESH_DB = 6        # above noise floor
MIN_BURST_MS = 30

# ============================================================
# FAST VECTOR CFAR
# ============================================================
def ca_cfar_fast(psd, num_train, num_guard, p_fa):
    T = 2 * num_train
    alpha = T * (p_fa ** (-1 / T) - 1)

    kernel = np.ones(2*num_train + 2*num_guard + 1)
    kernel[num_train:num_train + 2*num_guard + 1] = 0
    kernel /= T

    noise = np.convolve(psd, kernel, mode="same")
    threshold = alpha * noise

    return psd > threshold, threshold

# ============================================================
# MODULATION CLASSIFIER (MATCHES TX EXACTLY)
# ============================================================
def classify_modulation(iq, fs):
    env = np.abs(iq)
    env_var = np.var(env)

    # PSD bandwidth
    f, Pxx = sig.welch(iq, fs, nperseg=2048)
    bw = f[Pxx > 0.1 * np.max(Pxx)]
    bw_est = bw[-1] - bw[0] if len(bw) > 0 else 0

    # FM / LFM (constant envelope)
    if env_var < 1:
        phase = np.unwrap(np.angle(iq))
        inst_freq = np.diff(phase)
        slope = np.polyfit(np.arange(len(inst_freq)), inst_freq, 1)[0]
        return ("LFM" if abs(slope) > 1e-6 else "FM"), bw_est

    # AM family
    spectrum = np.abs(np.fft.fftshift(np.fft.fft(iq)))
    if np.max(spectrum) / np.mean(spectrum) > 30:
        return "AM", bw_est

    # SSB vs DSB-SC
    analytic = hilbert(np.real(iq))
    if np.allclose(analytic, iq, atol=0.15):
        return "SSB", bw_est

    return "DSB-SC", bw_est

# ============================================================
# PLUTO RX
# ============================================================
print("[RX] Initializing Pluto...")
sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(fs)
sdr.rx_rf_bandwidth = int(fs)
sdr.rx_lo = int(fc)
sdr.rx_hardwaregain_chan0 = rx_gain
sdr.rx_buffer_size = 262144


# ============================================================
# ACQUIRE IQ
# ============================================================
print("[RX] Capturing samples...")
try:
    # RX processing here
    iq = []
    start = time.time()
    while time.time() - start < run_time_s:
        iq.append(sdr.rx())
    iq = np.concatenate(iq)

finally:
    # CLEAN EXIT (prevents crash)
    try:
        sdr.rx_destroy_buffer()
    except:
        pass
    del sdr
# ============================================================
# DECIMATE FOR DETECTION (HUGE SPEED WIN)
# ============================================================
iq = sig.decimate(iq, DECIM, zero_phase=True)
fs = fs / DECIM

# ============================================================
# BLOCK FFT POWER DETECTION
# ============================================================
num_blocks = len(iq) // FFT_LEN
psd_time = []
fft_blocks = []

for i in range(num_blocks):
    blk = iq[i*FFT_LEN:(i+1)*FFT_LEN]
    fft_blk = np.fft.fftshift(np.fft.fft(blk))
    psd = np.abs(fft_blk)**2
    psd_time.append(np.mean(psd))
    fft_blocks.append(psd)

psd_time = np.array(psd_time)
fft_blocks = np.array(fft_blocks).T  # freq × time

# ============================================================
# TIME BURST DETECTION (CHEAP)
# ============================================================
noise_floor = np.median(psd_time)
burst_mask = 10*np.log10(psd_time/noise_floor) > POWER_THRESH_DB

bursts = []
active = False

for i, v in enumerate(burst_mask):
    if v and not active:
        t0 = i
        active = True
    elif not v and active:
        t1 = i
        if (t1 - t0)*FFT_LEN/fs*1e3 > MIN_BURST_MS:
            bursts.append((t0, t1))
        active = False

# ============================================================
# FREQUENCY CFAR (ON AVERAGED PSD)
# ============================================================
psd_avg = np.mean(fft_blocks, axis=1)
detections, threshold = ca_cfar_fast(
    psd_avg,
    CFAR_TRAIN,
    CFAR_GUARD,
    P_FA
)

freqs = np.linspace(-fs/2, fs/2, FFT_LEN)

bands = []
in_band = False
for i, d in enumerate(detections):
    if d and not in_band:
        f0 = freqs[i]
        in_band = True
    elif not d and in_band:
        f1 = freqs[i]
        bands.append((f0, f1))
        in_band = False

# ============================================================
# ANALYZE BURSTS
# ============================================================
results = []

for bid, (t0, t1) in enumerate(bursts):
    s0 = int(t0 * FFT_LEN)
    s1 = int(t1 * FFT_LEN)
    burst_iq = iq[s0:s1]

    mod, bw = classify_modulation(burst_iq, fs)

    for (f0, f1) in bands:
        results.append({
            "burst_id": bid,
            "t_start_s": s0/fs,
            "t_end_s": s1/fs,
            "fc_hz": fc + (f0 + f1)/2,
            "bw_hz": abs(f1 - f0),
            "modulation": mod
        })

# ============================================================
# OUTPUT
# ============================================================
print("\n=== DETECTED SIGNALS ===")
for r in results:
    print(r)