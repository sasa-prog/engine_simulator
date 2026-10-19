import numpy as np

from rpm_simulator import update_rpm
from stft import add_harmonics


def test_rpm_update_responds_to_throttle():
    """Opening the throttle increases RPM more than idle input."""
    idle_rpm = update_rpm(800.0, throttle=0.0, dt=1 / 60)
    accelerated_rpm = update_rpm(800.0, throttle=1.0, dt=1 / 60)

    assert accelerated_rpm > idle_rpm


def test_add_harmonics_preserves_spectrum_shape():
    """Harmonic processing preserves the STFT shape and finite values."""
    spectrum = np.ones((16, 4), dtype=np.complex128)

    result = add_harmonics(spectrum)

    assert result.shape == spectrum.shape
    assert np.all(np.isfinite(result))
