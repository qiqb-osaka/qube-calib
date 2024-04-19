import pytest
import numpy as np
from qubecalib.neopulse import RaisedCosFlatTop, Waveform


def test_inheritance():
    """RaisedCosFlatTop should inherit from Waveform."""
    assert issubclass(RaisedCosFlatTop, Waveform)


def test_empty_init():
    """RaisedCosFlatTop should initialize with no arguments."""
    wf = RaisedCosFlatTop()
    assert wf.__iq__ is None
    assert wf.duration is None
    assert wf.cmag == 1 + 0j  # TODO
    assert wf.rise_time is None


def test_init():
    """RaisedCosFlatTop should initialize with arguments."""
    wf = 0.5j * RaisedCosFlatTop(duration=10.0, rise_time=2.0)
    assert wf.__iq__ is None
    assert wf.duration == 10.0
    assert wf.cmag == 0 + 0.5j  # TODO
    assert wf.rise_time == 2.0


def test_func():
    """RaisedCosFlatTop should return the correct values."""
    duration = 30
    rise_time = 10
    wf = 0.1 * RaisedCosFlatTop(duration=duration, rise_time=rise_time)
    assert wf.func(-5.0) == 0.0
    assert wf.func(0.0) == 0.0
    assert wf.func(2.5) == pytest.approx(0.1464466094067262)
    assert wf.func(5.0) == pytest.approx(0.5)
    assert wf.func(10.0) == 1.0
    assert wf.func(15.0) == 1.0
    assert wf.func(20.0) == 1.0
    assert wf.func(25.0) == pytest.approx(0.5)
    assert wf.func(27.5) == pytest.approx(0.1464466094067262)
    assert wf.func(30.0) == 0.0
    assert wf.func(35.0) == 0.0


def test_ufunc():
    """RaisedCosFlatTop should return the correct samples."""
    duration = 8
    rise_time = 4
    wf = 2 * RaisedCosFlatTop(duration=duration, rise_time=rise_time)

    wf.begin = 0
    samples_begin0 = wf.ufunc(np.array([0, 2, 4, 6, 8, 10]))
    assert samples_begin0 == pytest.approx([0, 1, 2, 1, 0, 0])

    wf.begin = 2
    samples_begin2 = wf.ufunc(np.array([0, 2, 4, 6, 8, 10]))
    assert samples_begin2 == pytest.approx([0, 0, 1, 2, 1, 0])
