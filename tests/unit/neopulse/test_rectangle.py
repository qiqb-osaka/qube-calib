import pytest
import numpy as np
from qubecalib.neopulse import Rectangle, Waveform


def test_inheritance():
    """RaisedCosFlatTop should inherit from Waveform."""
    assert issubclass(Rectangle, Waveform)


def test_empty_init():
    """Rectangle should initialize with no arguments."""
    wf = Rectangle()
    assert wf.__iq__ is None
    assert wf.duration is None
    assert wf.amplitude == 1.0


def test_init():
    """Rectangle should initialize with arguments."""
    wf = Rectangle(duration=10.0, amplitude=0.5)
    assert wf.__iq__ is None
    assert wf.duration == 10.0
    assert wf.amplitude == 0.5


def test_func():
    """Rectangle should return the correct values."""
    duration = 10
    wf = Rectangle(duration=duration, amplitude=0.1)
    assert wf.func(-5.0) == 0.0
    assert wf.func(0.0) == 0.1
    assert wf.func(5.0) == 0.1
    assert wf.func(10.0) == 0.0
    assert wf.func(15.0) == 0.0


def test_ufunc():
    """Rectangle should return the correct samples."""
    duration = 8
    amplitude = 2
    wf = Rectangle(duration=duration, amplitude=amplitude)

    wf.begin = 0
    samples_begin0 = wf.ufunc(np.array([0, 2, 4, 6, 8, 10]))
    assert samples_begin0 == pytest.approx([2, 2, 2, 2, 0, 0])

    wf.begin = 2
    samples_begin2 = wf.ufunc(np.array([0, 2, 4, 6, 8, 10]))
    assert samples_begin2 == pytest.approx([0, 2, 2, 2, 2, 0])
