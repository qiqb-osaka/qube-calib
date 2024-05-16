import numpy as np

from qubecalib.neopulse import Slot, Waveform


def test_inheritance():
    """Waveform should inherit from Slot."""
    assert issubclass(Waveform, Slot)


def test_empty_init():
    """Waveform should initialize with no arguments."""
    wf = Waveform()
    assert wf.duration is None
    assert wf.cmag == 1 + 0j


def test_init():
    """Waveform should initialize with arguments."""
    wf = Waveform(duration=10.0)
    assert wf.duration == 10.0
    assert wf.cmag == 1 + 0j


def test_scaled():
    """scaled should return the amplitude-scaled waveform."""
    wf1 = Waveform(duration=10.0)
    wf2 = wf1.scaled(2).scaled(3)
    assert wf2.cmag == 6 + 0j
    assert wf1.cmag == 1 + 0j


def test_shifted():
    """shifted should return the phase-shifted waveform."""
    wf1 = Waveform(duration=10.0)
    wf2 = wf1.shifted(np.pi / 2).shifted(np.pi / 2)
    assert wf2.cmag == np.exp(1j * np.pi)
    assert wf1.cmag == 1 + 0j
