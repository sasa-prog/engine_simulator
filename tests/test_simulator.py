import numpy as np

from engine import interpolate_controls
from simulator import update_rpm


def test_update_rpm_stays_within_engine_limits():
    rpm = 800.0
    for _ in range(120):
        rpm = update_rpm(rpm, throttle=1.0, dt=1 / 60)

    assert 800 <= rpm <= 7300


def test_interpolate_controls_returns_audio_length():
    rpm = np.array([800.0, 1200.0, 2000.0])
    rms = np.array([0.3, 0.5, 0.8])

    rpm_interp, rms_interp = interpolate_controls(
        rpm,
        rms,
        sample_count=12,
        sr=12,
        audio_duration_sec=1.0,
    )

    assert rpm_interp.shape == (12,)
    assert rms_interp.shape == (12,)
    assert np.all(np.isfinite(rpm_interp))
    assert np.all(np.isfinite(rms_interp))
