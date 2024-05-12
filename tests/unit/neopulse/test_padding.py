import numpy as np

from qubecalib.neopulse import Flushleft, Rectangle, Sequence, padding


def test_padding():
    """padding should take up the correct duration."""
    target = "RQ00"
    with Sequence() as seq:
        with Flushleft():
            padding(duration=6)
            Rectangle(duration=2, amplitude=1).target(target)

    gen_sampled_sequence, _ = seq.convert_to_sampled_sequence()
    gen_sub_sequence = gen_sampled_sequence[target].sub_sequences[0]
    assert (gen_sub_sequence.real == np.array([1.0, 0.0, 0.0])).all()
