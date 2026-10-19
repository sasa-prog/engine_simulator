import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from audio_utils import calculate_rpm
from signal_processor import process_engine


def load_audio(input_path: str) -> tuple[np.ndarray, int]:
    """Load an audio file while preserving its original sample rate."""
    return librosa.load(input_path, sr=None)


def estimate_controls(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Estimate RPM and RMS control signals from an audio waveform."""
    audio_duration_sec = librosa.get_duration(y=y, sr=sr)
    f0, _, _ = librosa.pyin(y, fmin=80, fmax=400)
    f0 = np.nan_to_num(f0, nan=0)
    rpm, rms = calculate_rpm(f0, y)
    return rpm, rms, audio_duration_sec


def interpolate_controls(
    rpm: np.ndarray,
    rms: np.ndarray,
    sample_count: int,
    sr: int,
    audio_duration_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate frame-level controls to one value per audio sample."""
    times_rpm = np.arange(len(rpm)) / 60
    t_audio = np.arange(sample_count) / sr
    rpm_interp = np.interp(t_audio, times_rpm, rpm)
    rms_interp = np.interp(
        t_audio,
        np.arange(len(rms)) / (len(rms) / audio_duration_sec),
        rms,
    )
    return rpm_interp, rms_interp


def generate_engine_audio(
    rpm_interp: np.ndarray,
    rms_interp: np.ndarray,
    sr: int,
) -> np.ndarray:
    """Generate and normalize engine audio from interpolated controls."""
    engine, _, _ = process_engine(rpm_interp, rms_interp, sr)
    engine /= np.max(np.abs(engine))
    return engine


def plot_rpm(rpm: np.ndarray) -> None:
    """Display the estimated RPM curve."""
    plt.plot(rpm)
    plt.ylabel("RPM")
    plt.xlabel("Time (frames)")
    plt.show()


def main() -> None:
    """Run the audio-driven engine synthesis workflow."""
    input_path = "./silen2.mp3"
    output_path = "engine.wav"
    y, sr = load_audio(input_path)
    print(f"Audio duration: {librosa.get_duration(y=y, sr=sr):.2f} sec")

    rpm, rms, audio_duration_sec = estimate_controls(y, sr)
    rpm_interp, rms_interp = interpolate_controls(
        rpm,
        rms,
        len(y),
        sr,
        audio_duration_sec,
    )
    plot_rpm(rpm)
    engine = generate_engine_audio(rpm_interp, rms_interp, sr)
    print(f"Generated audio duration: {len(engine) / sr:.2f} sec")
    sf.write(output_path, engine, int(sr))


if __name__ == "__main__":
    main()
