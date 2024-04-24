from qubecalib.neopulse import Waveform


def test_empty_init():
    """Waveform should initialize with no arguments."""
    wf = Waveform()
    assert wf.__iq__ is None
    assert wf.duration is None
    assert wf.cmag == 1 + 0j  # TODO


def test_init():
    """Waveform should initialize with arguments."""
    wf = Waveform(duration=10.0)
    assert wf.__iq__ is None
    assert wf.duration == 10.0
    assert wf.cmag == 1 + 0j  # TODO


def test_set_target():
    """set_target should set the target(s) of the waveform."""
    wf1 = Waveform()
    wf1.set_target("RQ00")
    assert wf1.targets == ("RQ00",)
    wf2 = Waveform()
    wf2.set_target("RQ00", "RQ01")
    assert wf2.targets == ("RQ00", "RQ01")
