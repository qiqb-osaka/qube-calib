import qubecalib.neopulse as pls
from qubecalib.backend.qube import Qube
from qubecalib.qubecalib import QubeCalib


def test_backend_append_sequence() -> None:
    qc = QubeCalib()
    qc.backend.append_sequence(pls.Sequence(), time_offset={}, time_to_start={})


def test_backend() -> None:
    qc = QubeCalib()
    assert isinstance(qc.backend, Qube) is True
