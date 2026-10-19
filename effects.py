import numpy as np

def exhaust_reflect(x, sr, rpm_interp):
    """Add RPM-dependent delayed reflections to an exhaust signal."""
    y = x.copy()
    gain = np.interp(
        rpm_interp.mean(),
        [800, 7300],
        [0.25, 0.55]
    )
    d1 = int(0.0010 * sr)
    d2 = int(0.0023 * sr)
    d3 = int(0.0045 * sr)

    y += gain * np.roll(x, d1)
    y += gain * 0.5 * np.roll(x, d2)
    y += gain * 0.25 * np.roll(x, d3)


    return y
