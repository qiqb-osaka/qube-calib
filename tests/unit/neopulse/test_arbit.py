import pytest
import numpy as np
from qubecalib.neopulse import Arbit, Waveform


def test_inheritance():
    """Arbit should inherit from Waveform."""
    assert issubclass(Arbit, Waveform)


def test_empty_init():
    """Arbit should initialize with no arguments."""
    wf = Arbit()
    assert wf.__iq__ is None
    assert wf.duration is None
    assert wf.cmag == 1 + 0j  # TODO
    assert wf.init == 0 + 0j  # TODO


def test_init():
    """Arbit should initialize with arguments."""
    wf = 0.5j * Arbit(duration=10.0, init=1.0)
    assert wf.__iq__ is None
    assert wf.duration == 10.0
    assert wf.cmag == 0 + 0.5j  # TODO
    assert wf.init == 1.0 + 0j  # TODO


def test_default_sampling_period():
    """Default sampling period should be 2 ns."""
    assert Arbit.DEFAULT_SAMPLING_PERIOD == 2e-9
    assert Arbit().DEFAULT_SAMPLING_PERIOD == 2e-9


def test_set_iq():
    """Arbit should set the IQ data."""
    dt = Arbit.DEFAULT_SAMPLING_PERIOD
    wf = Arbit(duration=5 * dt)
    wf.begin = 0
    wf.__iq__ = np.array([1, 2, 3, 4, 5])
    assert (wf.iq == np.array([1, 2, 3, 4, 5])).all()


def test_set_iq_wrong_length():
    """Arbit should raise an error if the IQ data is the wrong length."""
    pass  # TODO


def test_func():
    """Arbit should return the correct samples."""
    dt = Arbit.DEFAULT_SAMPLING_PERIOD
    wf = Arbit(duration=5 * dt)
    wf.begin = 0
    wf.__iq__ = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert wf.func(-1e-9) == 0.0
    assert wf.func(0e-9) == 0.1
    assert wf.func(1e-9) == 0.1
    assert wf.func(2e-9) == 0.2
    assert wf.func(3e-9) == 0.2
    assert wf.func(4e-9) == 0.3
    assert wf.func(5e-9) == 0.3
    assert wf.func(6e-9) == 0.4
    assert wf.func(7e-9) == 0.4
    assert wf.func(8e-9) == 0.5
    assert wf.func(9e-9) == 0.5
    assert wf.func(10e-9) == 0.0
