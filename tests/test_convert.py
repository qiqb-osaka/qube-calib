def test_convert() -> None:
    # from matplotlib import pyplot as plt
    from typing import Final, cast

    import numpy as np
    import qubecalib as qc
    import qubecalib.backendqube as backend
    from qubecalib.backendqube import ChannelMap
    from qubecalib.neopulse import (
        Arbit,
        Control,
        RaisedCosFlatTop,
        Readout,
        Rectangle,
        Sequence,
        Series,
    )
    from qubecalib.units import MHz, nS, uS

    box: Final = cast(
        qc.qcbox.QubeRikenTypeAQcBox, qc.QcBoxFactory("qube_riken_1-08.yml").produce()
    )

    # %matplotlib inline
    # define Physical Port
    MUX1TX: Final = box.port0
    MUX1RX: Final = box.port1
    TXQ1: Final = box.port5
    TXQ2: Final = box.port6
    TXQ3: Final = box.port7
    TXQ4: Final = box.port8
    MUX2RX: Final = box.port12
    MUX2TX: Final = box.port13

    # define Logical Channel
    RQ1: Final[Readout] = Readout(10000 * MHz)
    RQ2: Final[Readout] = Readout(10000 * MHz)
    CQ1: Final[Control] = Control(9990 * MHz)
    CQ2: Final[Control] = Control(10100 * MHz)
    CQ3: Final[Control] = Control(10100 * MHz)
    CQ4: Final[Control] = Control(9990 * MHz)
    RQ3: Final[Readout] = Readout(10000 * MHz)
    RQ4: Final[Readout] = Readout(10000 * MHz)

    with ChannelMap() as m:
        m.map(MUX1TX.channel, RQ1, RQ2)
        m.map(MUX1RX.channel0, RQ1)
        m.map(MUX1RX.channel1, RQ2)
        m.map(TXQ1.channel0, CQ1)
        m.map(TXQ2.channel0, CQ2)
        m.map(TXQ3.channel0, CQ3)
        m.map(TXQ4.channel0, CQ4)
        m.map(MUX2RX.channel0, RQ3)
        m.map(MUX2RX.channel1, RQ4)
        m.map(MUX2TX.channel, RQ3, RQ4)
    channel_map = m

    with Sequence() as sequence:
        with Series():
            RaisedCosFlatTop(duration=10.5 * nS, rise_time=4 * nS)
            RaisedCosFlatTop(duration=10.5 * nS, rise_time=4 * nS)
            Rectangle(duration=10.5 * nS)
            a = Arbit(duration=10.5 * nS)

    a.iq_array[:] = np.random.rand(*a.iq_array.shape)

    sequence[0].begin = 0

    section = backend.acquire_section(sequence, channel_map)
    period = backend.quantize_sequence_duration(sequence_duration=400 * uS)

    backend.convert(sequence.flatten(), section, channel_map, period)
