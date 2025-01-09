import numpy as np
import pytest
from qubecalib.neopulse import RaisedCosFlatTop, Waveform


def test_inheritance() -> None:
    """RaisedCosFlatTop should inherit from Waveform."""
    assert issubclass(RaisedCosFlatTop, Waveform)


def test_empty_init() -> None:
    """RaisedCosFlatTop should initialize with no arguments."""
    wf = RaisedCosFlatTop()
    assert wf.duration is None
    assert wf.amplitude == 1.0
    assert wf.rise_time == 0.0


def test_init() -> None:
    """RaisedCosFlatTop should initialize with arguments."""
    wf = RaisedCosFlatTop(duration=10.0, amplitude=0.5, rise_time=2.0)
    assert wf.duration == 10.0
    assert wf.amplitude == 0.5
    assert wf.rise_time == 2.0


def test_func() -> None:
    """RaisedCosFlatTop should return the correct values."""
    duration = 30
    amplitude = 0.1
    rise_time = 10
    wf = RaisedCosFlatTop(duration=duration, amplitude=amplitude, rise_time=rise_time)
    assert wf.func(-5.0) == 0.0
    assert wf.func(0.0) == 0.0
    assert wf.func(2.5) == pytest.approx(0.01464466094067262)
    assert wf.func(5.0) == pytest.approx(0.05)
    assert wf.func(10.0) == 0.1
    assert wf.func(15.0) == 0.1
    assert wf.func(20.0) == 0.1
    assert wf.func(25.0) == pytest.approx(0.05)
    assert wf.func(27.5) == pytest.approx(0.01464466094067262)
    assert wf.func(30.0) == 0.0
    assert wf.func(35.0) == 0.0


def test_ufunc() -> None:
    """RaisedCosFlatTop should return the correct samples."""
    duration = 8
    amplitude = 2
    rise_time = 4
    wf = RaisedCosFlatTop(duration=duration, amplitude=amplitude, rise_time=rise_time)

    wf.begin = 0
    samples_begin0 = wf.ufunc(np.array([0, 2, 4, 6, 8, 10]))
    assert samples_begin0 == pytest.approx([0, 1, 2, 1, 0, 0])

    wf.begin = 2
    samples_begin2 = wf.ufunc(np.array([0, 2, 4, 6, 8, 10]))
    assert samples_begin2 == pytest.approx([0, 0, 1, 2, 1, 0])
