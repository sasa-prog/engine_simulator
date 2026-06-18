import pygame
import numpy as np

pygame.init()

screen = pygame.display.set_mode((300, 100))
clock = pygame.time.Clock()

sr = 44100
rpm = 800.0
throttle = 0.0

running = True
rpm_history = []

while running:
    dt = clock.tick(60) / 1000.0   # 秒

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    if keys[pygame.K_SPACE]:
        throttle = 1.0
    else:
        throttle = 0.0

    target = 800 + throttle * (7300 - 800)

    tau_up = 0.15
    tau_down = 0.4

    tau = tau_up if target > rpm else tau_down

    rpm += (target - rpm) * dt / tau
    rpm_history.append(rpm)
    print(f"\rRPM: {rpm:.0f}", end="")

pygame.quit()
