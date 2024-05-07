import numpy as np
import pytest

from qubecalib.neopulse import DEFAULT_SAMPLING_PERIOD, Arbit, Waveform


def test_inheritance():
    """Arbit should inherit from Waveform."""
    assert issubclass(Arbit, Waveform)


def test_empty_init():
    """Arbit should initialize with no arguments."""
    wf = Arbit()
    assert wf.duration is None


def test_init():
    """Arbit should initialize with arguments."""
    dt = DEFAULT_SAMPLING_PERIOD
    wf = Arbit(duration=5 * dt)
    assert wf.duration == 10
    assert (wf.iq == np.array([0, 0, 0, 0, 0])).all()


def test_default_sampling_period():
    """Default sampling period should be 2 ns."""
    assert DEFAULT_SAMPLING_PERIOD == 2


def test_set_iq():
    """Arbit should set the IQ data."""
    dt = DEFAULT_SAMPLING_PERIOD
    wf = Arbit(duration=5 * dt)
    wf.iq[:] = np.array([1, 2, 3, 4, 5])
    assert (wf.iq == np.array([1, 2, 3, 4, 5])).all()


def test_set_iq_wrong_length():
    """Arbit should raise an error if the IQ data is the wrong length."""
    dt = DEFAULT_SAMPLING_PERIOD
    wf = Arbit(duration=5 * dt)
    wf.begin = 0
    with pytest.raises(ValueError):
        wf.iq[:] = np.array([1, 2, 3, 4, 5, 6])


def test_func():
    """Arbit should return the correct samples."""
    dt = DEFAULT_SAMPLING_PERIOD
    wf = Arbit(duration=5 * dt)
    wf.begin = 0
    wf.iq[:] = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert wf.func(-1) == 0.0
    assert wf.func(0) == 0.1
    assert wf.func(1) == 0.1
    assert wf.func(2) == 0.2
    assert wf.func(3) == 0.2
    assert wf.func(4) == 0.3
    assert wf.func(5) == 0.3
    assert wf.func(6) == 0.4
    assert wf.func(7) == 0.4
    assert wf.func(8) == 0.5
    assert wf.func(9) == 0.5
    assert wf.func(10) == 0.0
