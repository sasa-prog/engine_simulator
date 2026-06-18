from engine_simulator.engine_simulator import simulate_rpm
import numpy as np
from engine_simulator.signal_processor import process_engine, effect_engine, normalize_engine
import soundfile as sf
import pygame
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

sr = 44100
fps = 60
duration = 10  # 10秒録音

pygame.init()
screen = pygame.display.set_mode((300, 100))
clock = pygame.time.Clock()

rpm = 800.0
rpm_history = []

running = True
elapsed = 0.0

while running and elapsed < duration:
    dt = clock.tick(fps) / 1000.0
    elapsed += dt

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    throttle = 1.0 if keys[pygame.K_SPACE] else 0.0

    target = 800 + throttle * (7300 - 800)

    # tau_up = np.interp(
    #     rpm,
    #     [800, 3000, 7300],
    #     [0.40, 0.30, 0.20]
    # )

    # tau_down = np.interp(
    #     rpm,
    #     [800, 7300],
    #     [0.50, 0.70]
    # )

    # tau = tau_up if target > rpm else tau_down

    # rpm += (target - rpm) * dt / tau
    inertia = 0.045

    torque = throttle * np.interp(
        rpm,
        [800, 2000, 4800, 7300],
        [320, 420, 540, 460]
    )

    engine_brake = 0.015 * rpm

    drpm = (torque - engine_brake) / inertia

    rpm += drpm * dt
    rpm = np.clip(rpm, 800, 7300)
    rpm_history.append(rpm)

    print(f"\rRPM: {rpm:.0f}", end="")

pygame.quit()
rpm_history = np.array(rpm_history)

times_rpm = np.arange(len(rpm_history)) / fps

t_audio = np.arange(int(duration * sr)) / sr

rpm_interp = np.interp(
    t_audio,
    times_rpm,
    rpm_history
)
sr = 44100

rms = np.interp(
    rpm_interp,
    [800, 7300],
    [0.3, 1.0]
)
dummy = np.zeros_like(rpm_interp)

jitter = (
    8*np.sin(2*np.pi*6*times_rpm)
    + 4*np.sin(2*np.pi*13*times_rpm)
)

rpm += jitter

engine, rpm_interp, phase = process_engine(
    rpm_interp,
    rms,
    sr
)

engine = effect_engine(np.asarray(engine), rpm_interp, phase, sr)

engine /= np.max(np.abs(engine))

# print(np.max(np.abs(engine)))
# print(np.percentile(np.abs(engine), 99))
# print(np.percentile(np.abs(engine), 99.9))

sf.write("simulator.wav", engine, int(sr))

f, t, S = spectrogram(
    engine,
    sr,
    nperseg=2048
)

plt.pcolormesh(
    t,
    f,
    10*np.log10(S + 1e-12),
    shading="gouraud"
)

plt.ylim(0,5000)
plt.ylabel("Hz")
plt.xlabel("Time")
plt.colorbar()
plt.show()
