import numpy as np

def simulate_rpm(duration, sr):
    t = np.arange(int(duration * sr)) / sr

    throttle = np.zeros_like(t)

    # デモ用アクセル操作
    throttle[(t > 1) & (t < 4)] = 1.0
    throttle[(t > 6) & (t < 8)] = 0.5

    rpm = np.zeros_like(t)
    rpm[0] = 800

    tau = 0.2   # 慣性

    for i in range(1, len(rpm)):
        target = 800 + throttle[i] * (7300 - 800)

        rpm[i] = rpm[i-1] + (
            target - rpm[i-1]
        ) / (tau * sr)

    return rpm, throttle
