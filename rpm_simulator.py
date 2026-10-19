import numpy as np
import pygame

FPS = 60
MIN_RPM = 800
MAX_RPM = 7300


def update_rpm(rpm: float, throttle: float, dt: float) -> float:
    """Advance RPM using the asymmetric rise and fall time constants."""
    target = MIN_RPM + throttle * (MAX_RPM - MIN_RPM)
    tau_up = 0.15
    tau_down = 0.4
    tau = tau_up if target > rpm else tau_down
    return rpm + (target - rpm) * dt / tau


def simulate_interactive(fps: int = FPS) -> np.ndarray:
    """Collect RPM values while the space key controls the throttle."""
    pygame.init()
    pygame.display.set_mode((300, 100))
    clock = pygame.time.Clock()
    rpm = float(MIN_RPM)
    rpm_history: list[float] = []
    running = True

    try:
        while running:
            dt = clock.tick(fps) / 1000.0

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


def main() -> None:
    """Run the standalone interactive RPM simulator."""
    simulate_interactive()


if __name__ == "__main__":
    main()
