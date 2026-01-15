import numpy as np
import scipy.signal as sig
from scipy.signal import hilbert
import adi
import time

############################################
# PARAMETERS (MATCH TX)
############################################
fs = 2e6
fc = 2e9
rx_gain = 30
duration_s = 10.0

# STFT
nfft = 2048
hop = 512
window = sig.windows.hann(nfft)

# CFAR
NUM_TRAIN = 12
NUM_GUARD = 4
P_FA = 1e-4

############################################
# PLUTO RX
############################################
sdr = adi.Pluto("ip:192.168.2.1")
sdr.sample_rate = int(fs)
sdr.rx_rf_bandwidth = int(fs)
sdr.rx_lo = int(fc)
sdr.rx_hardwaregain_chan0 = rx_gain
sdr.rx_buffer_size = 262144

############################################
# CFAR FUNCTION (2D: frequency only)
############################################
def ca_cfar(psd, num_train, num_guard, p_fa):
    N = len(psd)
    detections = np.zeros(N, dtype=bool)
    T = 2 * num_train
    alpha = T * (p_fa ** (-1 / T) - 1)

    for k in range(num_train + num_guard, N - num_train - num_guard):
        noise = np.mean(
            np.r_[psd[k-num_guard-num_train:k-num_guard],
                  psd[k+num_guard+1:k+num_guard+num_train+1]]
        )
        if psd[k] > alpha * noise:
            detections[k] = True
    return detections

############################################
# MODULATION CLASSIFIER
############################################
def classify_burst(iq, fs):
    # Envelope
    env = np.abs(iq)
    env_var = np.var(env)

    # PSD
    f, Pxx = sig.welch(iq, fs, nperseg=2048)
    bw = f[Pxx > 0.1 * np.max(Pxx)]
    bw_est = bw[-1] - bw[0] if len(bw) > 0 else 0

    # FM check: constant envelope
    if env_var < 1e-3:
        # LFM vs FM
        inst_freq = np.diff(np.unwrap(np.angle(iq)))
        slope = np.polyfit(np.arange(len(inst_freq)), inst_freq, 1)[0]
        if abs(slope) > 1e-6:
            return "LFM", bw_est
        else:
            return "FM", bw_est

    # AM family
    # Carrier check
    fft = np.abs(np.fft.fftshift(np.fft.fft(iq)))
    if np.max(fft) / np.mean(fft) > 10:
        return "AM", bw_est

    # SSB vs DSB
    analytic = hilbert(np.real(iq))
    if np.allclose(analytic, iq, atol=0.1):
        return "SSB", bw_est

    return "DSB-SC", bw_est

############################################
# ACQUIRE DATA
############################################
print("Receiving...")
iq = []
start = time.time()
while time.time() - start < duration_s:
    iq.append(sdr.rx())
    print(time.time())
iq = np.concatenate(iq)

############################################
# STFT
############################################
f, t, Sxx = sig.stft(
    iq, fs,
    window=window,
    nperseg=nfft,
    noverlap=nfft-hop,
    return_onesided=False
)
Sxx = np.fft.fftshift(Sxx, axes=0)
f = np.fft.fftshift(f)

PSD = np.abs(Sxx)**2

############################################
# CFAR PER TIME SLICE
############################################
detections = np.zeros_like(PSD, dtype=bool)

for ti in range(PSD.shape[1]):
    detections[:, ti] = ca_cfar(
        PSD[:, ti],
        NUM_TRAIN,
        NUM_GUARD,
        P_FA
    )

############################################
# BURST EXTRACTION
############################################
bursts = []
active = False

for ti in range(detections.shape[1]):
    if detections[:, ti].any() and not active:
        t_start = t[ti]
        active = True
    elif not detections[:, ti].any() and active:
        t_end = t[ti]
        bursts.append((t_start, t_end))
        active = False

############################################
# ANALYZE BURSTS
############################################
results = []

for i, (ts, te) in enumerate(bursts):
    idx = np.where((np.arange(len(iq))/fs >= ts) &
                   (np.arange(len(iq))/fs <= te))[0]
    burst_iq = iq[idx]

    mod, bw = classify_burst(burst_iq, fs)
    fc_est = np.mean(f[np.any(detections[:, (t>=ts)&(t<=te)], axis=1)])

    results.append({
        "burst_id": i,
        "t_start": ts,
        "t_end": te,
        "fc_offset_hz": fc_est,
        "bw_hz": bw,
        "modulation": mod
    })

############################################
# PRINT RESULTS
############################################
for r in results:
    print(r)