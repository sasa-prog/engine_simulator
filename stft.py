import librosa
import numpy as np
import soundfile as sf


def load_stft(input_path: str) -> tuple[np.ndarray, int, np.ndarray]:
    """Load an audio file and return its waveform, sample rate, and STFT."""
    y, sr = librosa.load(input_path, sr=None)
    return y, sr, librosa.stft(y)


def add_harmonics(spectrum: np.ndarray, harmonic_count: int = 5) -> np.ndarray:
    """Add harmonics to an STFT magnitude while preserving its phase."""
    magnitude = np.abs(spectrum)
    phase = spectrum / (magnitude + 1e-8)
    modified_magnitude = magnitude.copy()

    for harmonic in range(2, harmonic_count + 1):
        target = modified_magnitude[harmonic::harmonic]
        source = magnitude[:target.shape[0]]
        target += 0.3 * source

    return modified_magnitude * phase


def synthesize(input_path: str = "voice.mp3", output_path: str = "engine.wav") -> None:
    """Create a harmonic-enhanced waveform from an input audio file."""
    y, sr, spectrum = load_stft(input_path)
    print(sr)
    print(y.shape)
    modified_spectrum = add_harmonics(spectrum)
    y_new = librosa.istft(modified_spectrum)
    sf.write(output_path, y_new, int(sr))


def main() -> None:
    """Run the default STFT harmonic synthesis experiment."""
    synthesize()


if __name__ == "__main__":
    main()
