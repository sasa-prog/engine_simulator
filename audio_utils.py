import numpy as np
import librosa
from scipy.signal import iirpeak, butter, lfilter, sosfilt

def smoothing_rpm(rpm_interp: np.ndarray, sr: int | float, limit: int = 2700):
    max_drop = 2700  # rpm/s

    for i in range(1, len(rpm_interp)):
        dt = 1 / sr

        allowed_drop = max_drop * dt

        if rpm_interp[i] < rpm_interp[i-1] - allowed_drop:
            rpm_interp[i] = rpm_interp[i-1] - allowed_drop

    return rpm_interp

def calculate_rpm(f0: np.ndarray, y: np.ndarray):
    # RPMの計算
    rpm_f0 = 1500 + 35 * f0
    hop_length = 512
    rms = librosa.feature.rms(
        y=y,
        hop_length=hop_length
    )[0]
    rms *= 1.2

    rpm_rms = 1100 + 4000 * rms / np.max(rms)
    return (0.3 * rpm_f0 + 0.7 * rpm_rms, rms)

def lowpass(x, cutoff, sr, order=4):
    sos = butter(
        order,
        cutoff,
        fs=sr,
        btype='low',
        output='sos'
    )
    return sosfilt(sos, x)


def resonator(x, freq, decay, sr):
    y = np.zeros_like(x)

    r = np.exp(-decay/sr)
    theta = 2*np.pi*freq/sr

    a1 = 2*r*np.cos(theta)
    a2 = -r*r

    for n in range(2, len(x)):
        y[n] = (
            x[n]
            + a1*y[n-1]
            + a2*y[n-2]
        )

    return y

def highpass(x, cutoff, sr, order=4):
    sos = butter(
        order,
        cutoff,
        btype="high",
        fs=sr,
        output="sos"
    )
    return sosfilt(sos, x)

def bandpass(x, lowcut, highcut, sr, order=4):
    sos = butter(
        order,
        [lowcut, highcut],
        btype="band",
        fs=sr,
        output="sos"
    )
    return sosfilt(sos, x)
