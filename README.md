# EE 435 Term Project Phase 2 Report

#### Lab 1J – Mehmet Alp Demircioğlu, 2575082, Anıl Budak, 25 74812

## 1. Introduction

In this report, we provide a reliable solution for the acquisition and identification of five modulated

signals, namely AM-DSB-SC (Amplitude Modulation Double Sideband Suppressed Carrier), AM-

Conventional, AM-SSB (Amplitude Modulation Single Sideband), FM (Frequency Modulation),

and LFM (Linear Frequency Modulation – chirp signal). Throughout the report, we first investigate

the data acquisition process. Then, based on the acquired data, we discuss how to process this raw

data. We explain what kinds of filters and pre-processing techniques are used to reduce the effect

of noise. Moreover, we propose a methodology for modulated signal classification based on their

signal statistics. After that, we explain how we handle reporting and decision logic. In the last step,

we conclude the report with our observations.

## 2. Data Acquisition

The data used in this work consist of complex baseband samples recorded using a wideband
software-defined radio and stored for offline analysis. The samples are loaded from a NumPy
archive file (pluto_capture.npz), which contains multiple consecutive buffers along with their
corresponding time stamps.

Each buffer represents a fixed-duration snapshot of the received signal, sampled at a rate of 2.
MHz. This buffered acquisition approach allows time-localized spectral analysis and reliable
detection of short-duration bursts while preserving amplitude and phase information required for
modulation classification.

The total recording spans approximately 10 seconds, during which the frequency band of interest
is repeatedly observed. Time stamps associated with each buffer are used later to estimate burst
durations and track signal activity over time. Table 1 summarizes the main data acquisition
parameters.

```
Table 1 : Parameters and values used in Data Acquisition
```
```
Parameter Value Description
Sampling frequency (fs) 2.5 MHz SDR sampling rate
Data format Complex baseband (I/Q) Amplitude and phase preserved
Buffer duration N / fs Fixed per buffer
Total capture time ≈ 10 s Full observation window
Storage format .npz file Offline processing
```

## 3. Data Analysis

In this section it is explained that the processing of raw digital IQ (In-phase and Quadrature) data.
In the data acquisiton part we obtained this raw data from ADALM-Pluto with a given 2.0 GHz
carrier frequency and 2.1 MHz sampling frequency. We conduct spectral pre-processing and noise
reduction right after the data obtained to operate signal processing on a less faulty data. After that
we worked on signal detection via energy concentration to decide whether we should further
process the data in the current buffer. In the case of signal detection we used such techniques that
frequency hopping detection, spectral symmetry calculation, envelope variance calculation and
peak detection to classify the signals among the corresponding signal types which is AM, DSB-
SC, SSB, FM and LFM.

Another issue we worked on is relation between consecutive buffers and burst detection. A signal
may appear in multiple buffers and we should detect these consecutive buffers to analyze the data
accordingly. Multiple buffer information may give us a more precise classification and feature
extraction abilities.

### 3.1 Spectral Pre-processing and Noise Reduction

We obtained raw complex baseband signal which corresponds to a time domain sequence of
discrete IQ samples. We used both the information contained in the time domain and frequency
domain for feature extraction and modulation classification. To use the information in the
frequency domain we must obtain it from the time domain sequence. For that we used Fast Fourier
Transform(FFT).

Such high resolution is critical in distinguishing narrow band signals (such as AM carriers) from
wide band signals. After obtaining the Power Spectral Density, we proceed to convert the linear
power to logarithmic scale (dB) for easier signal interpretation with varying power levels
compared to the noise floor.

However, raw PSD data is frequently contaminated by thermal noise and spectral leakage, causing
"false peaks." To counter this problem, we applied a Savitzky-Golay digital filter. Unlike a
conventional moving average filter, this filter uses a local polynomial regression to smooth the
data. This enables the algorithm to reject noise-induced peaks while retaining the characteristics
of the signal's occupied bandwidth, which is a key requirement for accurate measurement.

### 3.2 Signal Detection via Energy Concentration

To detect a signal in a buffer we analyzed the PSD of it. We calculated the cumulative sum of the
normalized linear PSD. If there is not a signal in the buffer there is just obtained noise. Which
corresponds to a linear increase in the calculated cumulative sum through the 2.1 MHz span we
investigate. If there is a signal in the current buffer most of the signal should contained in a small
frequecny range. We set a worst case threshold as 100 kHz range should contain 50% of the total
power in the 2.1MHz span. This case corresponds to a sharp jump in the cumulative sum. If there


is a signal in the buffer algorithm will look upon modification classification through feature
extraction. Also calculated PSD helps us to calculate noise floor by calculating the median of it.

### 3.3 Adaptive Burst Segmentation and Hop Logic

One of the most complex part in this project to distinguish bursts. Burst is a continuous period of
signal activity. In our environment bursts are contained in multiple buffers and may some buffer
contain multiple bursts. For different bursts they mostly have different central frequencies.

To identify bursts we need to define central frequency. We calculate center frequency by the
midpoint of the detected bandwidth. Calculation of bandwidth held by finding the energy space
which contains the 99% of the energy in the PSD (effective bandwidth). We used the cumulative
sum to find the 99% of energy. We simply take the average of start and stop frequency of the found
bandwidth to decide center frequency.

If the center frequency of the current buffer shifts by more than 250 kHz compared to the previous
buffer, the system concludes that a frequency hop has occurred. In the event of such a hop, the
system assigns a new Unique Burst ID to the subsequent samples. This allows the system to treat
each frequency-stable segment as a separate entity for the majority-voting classification logic,
ensuring that a frequency-hopping FM signal isn't mistakenly classified as LFM (Linear Frequency
Modulation) due to the large frequency change over time.

### 3.4 Feature Extraction and Classification Metrics

Once the signals are segmented into discrete bursts, the system enters the feature extraction phase.
The goal here is to identify unique statistical and spectral characteristics that allow the algorithm
to distinguish between constant envelope (FM, LFM) and varying envelope (AM, DSB, SSB)
signals, as well as mirrored (DSB) vs. non-mirrored (SSB) spectra.

#### 3.4.1 High-Precision Envelope Analysis via Sliding Window

The primary differentiator between analog angle modulation (FM/LFM) and amplitude-based
modulation (AM/DSB/SSB) is the signal envelope. Theoretically, FM and LFM signals possess a
constant envelope. However, in real-world SDR captures, the end of the transitions of a burst often
contain transient noise and amplitude fluctuations that can disrupt statistical averages.

To achieve high-precision classification, we implemented a minimum variance sliding 5ms
window. Instead of calculating the variance of the entire buffer, the algorithm performs the
following steps:

1. Analytic Signal Generation: We compute the magnitude of the complex IQ samples to
    obtain the envelope A[n]=&I[n]!+Q[n]!
2. Windowing: We define a window of 5 ms (which, at f"= 2. 5 MSPS, corresponds to
    12,500 samples).


3. Sliding Metric Calculation: The window slides across the buffer with a 75% overlap. For
    each position, the Normalized Envelope Variance (σ#$%&! ) is calculated as:

```
σ#$%&! =
```
```
E[(A−μ')!]
μ'!
```
```
where A refers to signal envelope amplitude and 𝜇( is the mean amplitude A.
```
4. Minimum Selection: The algorithm selects the specific 5 ms window with the minimum
    variance.

By selecting the most stable portion of the signal, we effectively ignore the noisy edges of the
transmission. A threshold of is σ#$%&! < 0. 01 is used to classify the signal as Constant Envelope
(FM or LFM). If the variance exceeds this value, the signal is categorized as Varying Envelope
(AM, DSB, or SSB).

#### 3.4.2 Spectral Symmetry for Sideband Discrimination

Distinguishing between Double Sideband Suppressed Carrier (DSB-SC) and Single Sideband
(SSB) modulation requires a frequency-domain analysis of the sidebands relative to the carrier
frequency.

For signals identified as having varying envelopes, we perform a Spectral Correlation Check:

- Sideband Isolation: The algorithm identifies the bandwidth center frequency (f)) and
    splits the occupied bandwidth into a Left Half (S*) and a Right Half (S+).
- Mirror Comparison: We reverse the Right Half spectrum (S+,%-.) to align the indices for
    mirroring.
- Correlation Coefficient: We calculate the Pearson correlation coefficient (ρ) between S*
    and S+,%-.:

```
ρ=
```
```
cov=S*,S/,%-.?
σ/!σ/",$%&
```
A high correlation score (typically) indicates that the power spectral density is mirrored around
the center, which is the defining characteristic of DSB-SC (Type 2). A low correlation score
indicates power is concentrated on only one side, identifying the signal as SSB (Type 3).

#### 3.4.3 LFM Identification via Temporal Center-Shift

The uniqueness of Linear Frequency Modulated (LFM) signals is that their centre frequency
periodically adjusts in a linear fashion. As each buffer for each individual LFM signal appears to
maintain the same envelope (like FM signals), the Global Analysis Pass provides an indication on
the type of LFM signal.


The algorithm compares the center frequency of the first buffer in a burst (f"01%0) to the center
frequency of the last buffer (f-#2). If the total sweep Δf0$013=|f-#2−f"01%0| exceeds 40 kHz and
the envelope remains constant, the burst is classified as LFM (Type 5).

## 4. Reporting and Decision Logic

The final phase of the system is responsible for aggregating the features extracted from individual
buffers into a single, cohesive decision for each detected signal and exporting these results into the
required format.

### 4.1 Majority-Vo t i n g C o n s e n s u s M e c h a n i s m

Because signals are captured in real-time and processed in discrete 20,000-sample buffers, a single
long-duration transmission can span multiple buffers. Environmental noise, momentary fading, or
spectral interference can cause a single buffer within a burst to be misclassified.

To prevent these misclassifications from affecting the final output, we implemented a Consensus
Classification logic based on majority voting.

- Tallying: For every unique Burst ID identified during the segmentation phase, the system
    stores the predicted modulation type for every buffer belonging to that ID.
- Voting: A Counter object from the Python collections library is used to identify the Most
    Common modulation type within the burst.
- Assignment: The most frequent prediction is applied to the entire duration of the burst
    (T_msg), ensuring a stable and reliable classification for the final report.

### 4.2 Parameter Measurement for Faculty Compliance

As per the project requirements, four specific parameters must be reported for every signal. The
accuracy of these measurements is vital for the integrity of the data analysis.

#### 4.2.1 modType (Modulation Category)

The system maps the qualitative consensus results to the required integers:

1. AM: Identified by a sharp spectral carrier spike and varying envelope.
2. DSB-SC: Identified by varying envelope and high spectral symmetry (ρ> 0. 7 ).
3. SSB-SC: Identified by varying envelope and low spectral symmetry (ρ< 0. 7 ).
4. FM: Identified by constant envelope (σ#$%&! < 0. 01 ) and static center frequency.
5. LFM: Identified by constant envelope and significant frequency shift (>40 kHz) across
    the burst.


#### 4.2.2 Tmsg_ms (Message Duration)

The duration is calculated by comparing the start timestamp of the first buffer in a burst to the end
timestamp of the final buffer. Since each buffer represents a fixed temporal window based on the
sampling rate (f"= 2 .5MHz), we ensure that the duration accounts for the full length of the
samples processed.

#### 4.2.3 f_Shift_kHz (Frequency Offset)

The f"4560 represents the signal's center frequency relative to the DC component of the baseband.
This is calculated as the midpoint of the detected bandwidth:

```
f"4560=
```
```
f 4 −f 3
2
```
where f 4 and f 3 are the highest and lowest frequencies where the power exceeds the noise floor by
9dB.

The final value reported in the .xls file is the mean bandwidth calculated across all buffers within
that specific burst ID to provide a representative figure of the occupied spectrum.

### 4.3 Automated Data Export

The system utilizes the pandas library to organize the collected metrics into a structured dataframe.
This dataframe is automatically exported to an Excel file (signal_report.xlsx) upon completion of
the 10-second capture analysis. This automation eliminates human error in data entry and ensures
that the formatting strictly stick to the requested column order: modType, Tmsg_ms, f_Shift_kHz,
and B_rf_kHz.

## 5. Performance Metrics and System Evaluation

A critical aspect of the project is quantifying its reliability and efficiency. We evaluated the system
across four key performance domains: detection reliability, classification accuracy, computational
timing, and spectral resolution.

### 5.1 Detection Performance

The detection stage uses a cumulative sum jump detection algorithm. To evaluate this, we define
the following metrics based on our 10-second capture window:

- Probability of Detection: This represents the ratio of correctly identified signal bursts to
    the total number of actual bursts.


- Probability of False Alarm: This measures how often background noise triggered a
    "signal detected" state.
- Sensitivity: The system successfully detected signals with a Signal-to-Noise Ratio (SNR)
    as low as approximately 9 dB (our base threshold).

### 5.2 Classification Accuracy

The accuracy of the system was verified by manually inspecting the created .xls file by the faculty
to the ground truth values.

### 5.3 Timing and Computational Performance

A requirement of the project was to summarize timing performance. We measured the processing
time on a 4 - core intel i5 processor and 6 GB of ram.

### 5.4 Spectral and Temporal Resolution

This section quantifies the physical limits of what our algorithm can observe.

- Frequency Resolution: With an FFT size of and , the resolution is 125 Hz per bin. This
    allowed us to detect with extreme precision.
- Minimum Detectable Burst Duration: Our segmentation logic requires at least one full
    buffer to register a burst. Therefore, the minimum detectable is 8 ms.
- Minimum Detectable Hop: The frequency-hop logic is configured to split bursts if the
    center shifts by 250 kHz. This ensures that wide-band frequency-hopping signals are
    accurately separated.

## 6. Conclusion

The developed system successfully demonstrates the integration of Software Defined Radio

(SDR) with robust signal processing algorithms. By combining time-domain statistical analysis

(5ms sliding window variance) with frequency-domain correlation techniques (spectral

symmetry), the system achieved distinguishing between complex analog modulation types.

Through the project we experienced system design with project phase 1. After phase 1 we started

to develop on SDR project and deal with real world acquisiton and signal processing problems.
Our system design have changed through this real world application. We tried to keep our

algorithm simple as using more convenient and statistical point of view. We understand what key

features of modulation types and abstract concepts as bandwidth represent in this application.

The algorithm and project we developed has some insufficiency to detect all of the signals in more

challenging scenarios. These threshold based approaches struggled to perform under edge cases or

some extended bandwidth and speed limitations. Also, the algorithm fail to capture time


information. We observed high deviations in time information even though we were able to

classify most of the modulated signals. This is expected because in order to get better frequency

resolution we used high number of points to take FFT. We know that frequency bin = fs/N, meaning

that as numer of points in FFT increases we increase the frequency resolution. It is known that as

resolution in frequency increases, resolution in time decreases by Heisenberg Uncertainty

Principle. As a result of, we experience non-optimal results in terms of timing. To overcome this,

it may be benefical to use Short-Time Fourier Transform (STFT). However, this would also require

higher computational cost which would decrease real-time performance, having more parameters

to optimize, and reduce in frequency resolution. In conclusion, the suggested approach in this paper

is more optimized for classification of modulated signals and may fail in time analysis.


