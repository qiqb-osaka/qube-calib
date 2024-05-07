import numpy as np
import pytest

from qubecalib.neopulse import Magnifier, Rectangle, Sequence, VirtualZ


def test_magnifier():
    target = "TARGET"

    with Sequence() as seq:
        Rectangle(duration=4, amplitude=0.1).set_target(target)
        Magnifier(magnitude=2).set_target(target)
        Rectangle(duration=4, amplitude=0.1).set_target(target)
        Rectangle(duration=4, amplitude=0.2).set_target(target)

    gen, _ = seq.convert_to_sampled_sequence()
    subseq = gen[target].sub_sequences[0]

    assert (subseq.real == np.array([0.1, 0.1, 0.2, 0.2, 0.4, 0.4])).all()


def test_virtual_z():
    target = "TARGET"

    with Sequence() as seq:
        Rectangle(duration=4, amplitude=1.0).set_target(target)
        VirtualZ(theta=np.pi / 2).set_target(target)
        Rectangle(duration=4, amplitude=1.0).set_target(target)
        VirtualZ(theta=np.pi / 4).set_target(target)
        Rectangle(duration=4, amplitude=1.0).set_target(target)

    gen, _ = seq.convert_to_sampled_sequence()
    subseq = gen[target].sub_sequences[0]

    assert subseq.real == pytest.approx(
        np.array([1.0, 1.0, 0.0, 0.0, -1.0 / np.sqrt(2), -1.0 / np.sqrt(2)])
    )
    assert subseq.imag == pytest.approx(
        np.array([0.0, 0.0, 1.0, 1.0, 1.0 / np.sqrt(2), 1.0 / np.sqrt(2)])
    )
