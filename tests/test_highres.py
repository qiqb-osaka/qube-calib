def test_highres() -> None:
    import numpy as np
    from qubecalib.neopulse import Arbit, RaisedCosFlatTop, Rectangle, Sequence, Series
    from qubecalib.units import Units as U

    with Sequence() as sequence:
        with Series():
            RaisedCosFlatTop(duration=10.5 * U.nS, rise_time=4 * U.nS).set_target("CQ1")
            RaisedCosFlatTop(duration=10.5 * U.nS, rise_time=4 * U.nS).set_target("CQ1")
            Rectangle(duration=10.5 * U.nS).set_target("CQ1")
            a = Arbit(duration=10.5 * U.nS).set_target("CQ1")

    a.iq[:] = np.random.rand(*a.iq.shape)

    sampled_subsequences = sequence.convert_to_sampled_sequence()
