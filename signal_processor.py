import matplotlib.pyplot as plt
import numpy as np
from audio_utils import smoothing_rpm, lowpass, resonator
from scipy.signal import lfilter, iirpeak, butter, sosfilt
from audio_utils import bandpass
from effects import exhaust_reflect
import soundfile as sf

def process_engine(
    rpm_interp: np.ndarray,
    rms_interp: np.ndarray,
    sr: int | float
):
    t = np.arange(len(rpm_interp)) / sr

    window = int(sr * 0.02)  # 20msくらい
    kernel = np.ones(window) / window
    rpm_interp = np.convolve(
        rpm_interp,
        kernel,
        mode="same"
    )

    engine = np.zeros_like(rpm_interp)

    limiter = 4000
    rpm_interp = smoothing_rpm(rpm_interp, sr, limiter)


    cycle_freq = rpm_interp / 120      # 720°サイクルの周波数
    cycle_phase = 4*np.pi * np.cumsum(cycle_freq) / sr
    cycle_phase = np.mod(cycle_phase, 4*np.pi)

    left_angles = np.deg2rad([
        0,     # 1
        180,  # 7
        270,  # 3
        450   # 5
    ])

    right_angles = np.deg2rad([
        90,   # 8
        360,  # 6
        540,  # 4
        630   # 2
    ])

    width = np.deg2rad(
        np.interp(
            rpm_interp,
            [800, 7300],
            [13, 7]
        )
    )
    width = np.maximum(width, np.deg2rad(0.5))  # 最小0.5度

    left_impulse = np.zeros_like(cycle_phase)

    for ang in left_angles:
        d = np.angle(np.exp(1j*(cycle_phase-ang)))
        left_impulse += np.exp(-(d/width)**2)

    right_impulse = np.zeros_like(cycle_phase)

    for ang in right_angles:
        d = np.angle(np.exp(1j*(cycle_phase-ang)))
        right_impulse += np.exp(-(d/width)**2)


    cutoff_left = 1000

    cutoff_right = 1200
    left = lowpass(
        left_impulse,
        cutoff_left,
        sr
    )

    right = lowpass(
        right_impulse,
        cutoff_right + 200,
        sr
    )

    left = exhaust_reflect(left, sr, rpm_interp)
    right = exhaust_reflect(right, sr, rpm_interp)

    left += 0.5 * np.asarray(bandpass(
        left,
        500,
        1200,
        sr
    ))
    right += 0.5 * np.asarray(bandpass(
        right,
        700,
        1500,
        sr
    ))
    right = np.roll(
        right,
        int(0.0020 * sr)
    )

    engine = left + right

    gain = np.interp(
        rpm_interp,
        [800, 7300],
        [0.3, 1.0]
    )

    engine *= gain

    sf.write("left.wav", left, int(sr))
    sf.write("right.wav", left, int(sr))

    return (engine, rpm_interp, cycle_phase)

def effect_engine(engine: np.ndarray, rpm_interp: np.ndarray, phase: np.ndarray, sr: int | float):
    peak = np.max(np.abs(engine))
    if peak > 0:
        engine /= peak

    b1, a1 = iirpeak(120/(sr/2), Q=3)

    # 300Hz付近を強調
    b2, a2 = iirpeak(300/(sr/2), Q=5)

    engine = np.asarray(lfilter(b1, a1, engine))
    engine += np.asarray(lfilter(b2, a2, engine)) * 0.5

    engine = np.asarray(lfilter(
        [1],
        [1, -0.9],
        engine
    ))

    intake = (
        np.sin(2*phase)
        + 0.5*np.sin(3*phase)
    )

    intake_gain = np.interp(
        rpm_interp,
        [800, 4000, 5500, 7300],
        [0.0, 0.05, 0.15, 0.25]
    )

    engine += intake_gain * intake

    fade_ms = 100
    fade_samples = int(sr * fade_ms / 1000)


    # フェードイン
    engine[:fade_samples] *= np.linspace(0, 1, fade_samples)

    # フェードアウト
    engine[-fade_samples:] *= np.linspace(1, 0, fade_samples)
    return engine

def normalize_engine(engine: np.ndarray, rpm_interp: np.ndarray, sr: int | float):

    sos = butter(
        4,
        320 / (sr / 2),
        btype='low',
        output='sos'
    )

    engine = np.asarray(sosfilt(sos, engine))
    return engine
