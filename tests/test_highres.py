def test_highres() -> None:
    # from matplotlib import pyplot as plt
    import numpy as np
    import qubecalib
    from qubecalib.units import nS

    # %matplotlib inline

    pls = qubecalib.neopulse

    with pls.Sequence() as sequence:
        with pls.Series():
            pls.RaisedCosFlatTop(duration=10.5 * nS, rise_time=4 * nS)
            pls.RaisedCosFlatTop(duration=10.5 * nS, rise_time=4 * nS)
            pls.Rectangle(duration=10.5 * nS)
            a = pls.Arbit(duration=10.5 * nS)

    a.iq_array[:] = np.random.rand(*a.iq_array.shape)

    sequence[0].begin = 0

    s = sequence.flatten().slots[None]
    x, y = qubecalib.plot.get_sampling_data(s)
    # plt.stairs(np.real(y[:-1]), edges = x)
    # plt.locator_params(axis='x', nbins=25)

    x, y = qubecalib.plot.get_sampling_data(s, oversampling_ratio=500)
    # plt.plot(x, np.real(y))
