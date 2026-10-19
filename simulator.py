import matplotlib.pyplot as plt
import numpy as np
import pygame
import soundfile as sf
from scipy.signal import spectrogram

from signal_processor import effect_engine, process_engine

SAMPLE_RATE = 44100
FPS = 60
DURATION = 10
MIN_RPM = 800
MAX_RPM = 7300


def update_rpm(rpm: float, throttle: float, dt: float) -> float:
    """Advance the simulated RPM by one frame."""
    inertia = 0.045
    torque = throttle * np.interp(
        rpm,
        [800, 2000, 4800, 7300],
        [320, 420, 540, 460]
    )
    engine_brake = 0.015 * rpm
    drpm = (torque - engine_brake) / inertia
    return float(np.clip(rpm + drpm * dt, MIN_RPM, MAX_RPM))


def simulate_interactive(duration: float = DURATION, fps: int = FPS) -> np.ndarray:
    """Collect an RPM curve while the space key controls the throttle."""
    pygame.init()
    pygame.display.set_mode((300, 100))
    clock = pygame.time.Clock()

    rpm = float(MIN_RPM)
    rpm_history: list[float] = []
    running = True
    elapsed = 0.0

    try:
        while running and elapsed < duration:
            dt = clock.tick(fps) / 1000.0
            elapsed += dt

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            throttle = 1.0 if pygame.key.get_pressed()[pygame.K_SPACE] else 0.0
            rpm = update_rpm(rpm, throttle, dt)
            rpm_history.append(rpm)
            print(f"\rRPM: {rpm:.0f}", end="")
    finally:
        pygame.quit()

    return np.asarray(rpm_history)


def synthesize_engine(
    rpm_history: np.ndarray,
    duration: float = DURATION,
    sr: int = SAMPLE_RATE,
    fps: int = FPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the RPM curve and synthesize the engine sound."""
    times_rpm = np.arange(len(rpm_history)) / fps
    t_audio = np.arange(int(duration * sr)) / sr
    rpm_interp = np.interp(t_audio, times_rpm, rpm_history)
    rms = np.interp(rpm_interp, [MIN_RPM, MAX_RPM], [0.3, 1.0])

    engine, rpm_interp, phase = process_engine(rpm_interp, rms, sr)
    engine = effect_engine(np.asarray(engine), rpm_interp, phase, sr)
    engine /= np.max(np.abs(engine))
    return engine, rpm_interp


def plot_spectrogram(engine: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Display the generated engine sound's spectrogram."""
    frequencies, times, spectrum = spectrogram(engine, sr, nperseg=2048)
    plt.pcolormesh(
        times,
        frequencies,
        10 * np.log10(spectrum + 1e-12),
        shading="gouraud"
    )
    plt.ylim(0, 5000)
    plt.ylabel("Hz")
    plt.xlabel("Time")
    plt.colorbar()
    plt.show()


def main() -> None:
    """Run the interactive simulation and save its generated audio."""
    rpm_history = simulate_interactive()
    engine, _ = synthesize_engine(rpm_history)
    sf.write("simulator.wav", engine, SAMPLE_RATE)
    plot_spectrogram(engine)


if __name__ == "__main__":
    main()
