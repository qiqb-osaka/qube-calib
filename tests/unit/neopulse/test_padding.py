import numpy as np
import pytest

from qubecalib.neopulse import Flushleft, Item, Padding, Rectangle, Sequence


def test_inheritance():
    """Padding should inherit from Item."""
    assert issubclass(Padding, Item)


def test_empty_init():
    """Padding should not initialize with no arguments."""
    with pytest.raises(TypeError):
        Padding()  # type: ignore


def test_init():
    """Padding should initialize with arguments."""
    wf = Padding(duration=10)
    assert wf.duration == 10


def test_padding():
    """Padding should take up the correct duration."""
    target = "RQ00"
    with Sequence() as seq:
        with Flushleft():
            Padding(duration=6)
            Rectangle(duration=2, amplitude=1).target(target)

    gen_sampled_sequence, _ = seq.convert_to_sampled_sequence()
    gen_sub_sequence = gen_sampled_sequence[target].sub_sequences[0]
    assert (gen_sub_sequence.real == np.array([1.0, 0.0, 0.0])).all()
