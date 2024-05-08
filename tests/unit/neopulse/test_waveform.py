from qubecalib.neopulse import Waveform


def test_empty_init():
    """Waveform should initialize with no arguments."""
    wf = Waveform()
    assert wf.duration is None


def test_init():
    """Waveform should initialize with arguments."""
    wf = Waveform(duration=10.0)
    assert wf.duration == 10.0


def test_target():
    """target should set the target(s) of the waveform."""
    wf1 = Waveform()
    wf1.target("RQ00")
    assert wf1.targets == ("RQ00",)
    wf2 = Waveform()
    wf2.target("RQ00", "RQ01")
    assert wf2.targets == ("RQ00", "RQ01")
