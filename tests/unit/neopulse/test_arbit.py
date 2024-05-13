import numpy as np
import pytest

from qubecalib.neopulse import DEFAULT_SAMPLING_PERIOD, Arbit, Waveform


def test_inheritance():
    """Arbit should inherit from Waveform."""
    assert issubclass(Arbit, Waveform)


def test_empty_init():
    """Arbit should not initialize with no arguments."""
    with pytest.raises(TypeError):
        Arbit()  # type: ignore


def test_init():
    """Arbit should initialize with arguments."""
    wf_list = Arbit([1, 1j])
    wf_array = Arbit(np.array([1, 1j]))
    assert wf_list.duration == 4
    assert wf_array.duration == 4
    assert (wf_list.iq == np.array([1, 1j])).all()
    assert (wf_array.iq == np.array([1, 1j])).all()


def test_default_sampling_period():
    """Default sampling period should be 2 ns."""
    assert DEFAULT_SAMPLING_PERIOD == 2


def test_func():
    """Arbit should return the correct samples."""
    wf = Arbit([0.1, 0.2, 0.3, 0.4, 0.5])
    wf.begin = 0
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
