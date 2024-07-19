from collections import defaultdict

import numpy as np
import numpy.typing as npt
from tqdm.auto import tqdm

from .. import neopulse as pls
from ..qubecalib import QubeCalib, find_primary_component


def rabi_sequence(
    qubit: str,
    readout: str,
    duration: int,
    *,
    qubit_amp: float = 0.03,
    readout_amp: float = 0.01,
    readout_duration: int = 1024,
    padding_duration: int = 1024,
) -> pls.Sequence:
    control_pulse = pls.Rectangle(duration=duration, amplitude=1.0)
    readout_pulse = pls.RaisedCosFlatTop(
        duration=readout_duration, amplitude=readout_amp, rise_time=32
    )
    blank = pls.Blank(duration=padding_duration)
    capture = pls.Capture(duration=2 * readout_duration)

    with pls.Sequence() as sequence:
        with pls.Flushright():
            control_pulse.scaled(qubit_amp).target(qubit)
            blank.target()
        with pls.Flushleft():
            readout_pulse.target(readout)
            capture.target(readout)
    return sequence


def detect_skew_value_by_rabi(
    qubecalib: QubeCalib,
    qubit: str,
    readout: str,
    *,
    time_to_start: int = 0,
    time_offset_range: tuple[int, int] = (-3, 3),
    sweep_range: tuple[int, int, int] = (0, 201, 8),
    qubit_amp: float = 0.03,
    readout_amp: float = 0.01,
    readout_duration: int = 1024,
) -> npt.NDArray[np.float32]:
    info = qubecalib.get_target_info(qubit)
    box_name = next(iter(info["box_name"]))
    primary_component = []
    for offset in tqdm(range(*time_offset_range)):
        for duration in np.arange(*sweep_range):
            qubecalib.add_sequence(
                rabi_sequence(
                    qubit,
                    readout,
                    duration,
                    qubit_amp=qubit_amp,
                    readout_amp=readout_amp,
                    readout_duration=readout_duration,
                    padding_duration=sweep_range[1],
                ),
                time_offset={box_name: offset},
                time_to_start={box_name: time_to_start},
            )
        signals = defaultdict(list)
        for _, (status, data, config) in tqdm(
            enumerate(
                qubecalib.step_execute(
                    repeats=1024,
                    interval=15 * 10240,
                )
            ),
            total=len(np.arange(*sweep_range)),
            desc=f"offset={offset}",
        ):
            for i, (target, iqs) in enumerate(data.items()):
                iq = iqs[0].squeeze()
                signals[target].append(iq.mean())

        signal = np.array(signals[readout])
        d, e, x = find_primary_component(np.real(signal), np.imag(signal))
        primary_component.append((offset, d[-1]))

    return np.array(primary_component)
