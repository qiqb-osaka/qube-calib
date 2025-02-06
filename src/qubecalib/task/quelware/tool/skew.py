from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt
from e7awgsw import CaptureParam, WaveSequence
from quel_clock_master import QuBEMasterClient

from .... import QubeCalib
from ....utils.pca import principal_axis_rotation
from ..direct import (
    Action,
    AwgId,
    AwgSetting,
    Quel1System,
    RunitId,
    RunitSetting,
    TriggerSetting,
)

REPETITION_PERIOD = 1280 * 128  # words


class Skew:
    def __init__(
        self,
        system: Quel1System,
        *,
        recv_port: tuple[str, int],
        send_port: tuple[str, int],
        trig_port: tuple[str, int],
    ) -> None:
        self._system: Final[Quel1System] = system
        self._recv_port: Final[tuple[str, int]] = recv_port
        self._send_port: Final[tuple[str, int]] = send_port
        self._trig_port: Final[tuple[str, int]] = trig_port

    @classmethod
    def create_with_qubecalib(
        cls,
        qubecalib: QubeCalib,
        *,
        recv_port: tuple[str, int],
        send_port: tuple[str, int],
        trig_port: tuple[str, int],
    ) -> Skew:
        if qubecalib.system_config_database._clockmaster_setting is None:
            raise ValueError("Clockmaster setting is not found")

        boxes = set([recv_port[0], send_port[0], trig_port[0]])

        system = Quel1System.create(
            clockmaster=QuBEMasterClient(
                qubecalib.system_config_database._clockmaster_setting.ipaddr
            ),
            boxes=[qubecalib.create_named_box(box) for box in boxes],
        )
        return cls.create(
            system,
            recv_port=recv_port,
            send_port=send_port,
            trig_port=trig_port,
        )

    @classmethod
    def create(
        cls,
        system: Quel1System,
        *,
        recv_port: tuple[str, int],
        send_port: tuple[str, int],
        trig_port: tuple[str, int],
    ) -> Skew:
        return cls(
            system,
            recv_port=recv_port,
            send_port=send_port,
            trig_port=trig_port,
        )

    def frequency_setting(
        self,
        cnco_offset: float = 0,  # multiple of 15.625e6
        cnco_freq: float = 1500e6,  # Hz multiple of 15.625e6
        lo_freq: float = 11000e6,  # Hz multiple of 500e6
    ) -> None:
        recv_box, recv_port_num = self._recv_port
        send_box, send_port_num = self._send_port

        self._system.box[send_box].config_port(
            send_port_num,
            lo_freq=lo_freq,
            cnco_freq=cnco_freq - cnco_offset,
            sideband="L",
        )
        self._system.box[recv_box].config_port(
            recv_port_num,
            lo_freq=lo_freq,
            cnco_freq=cnco_freq - cnco_offset,
        )

    def _measure(
        self,
        capture_delay: int = 7 * 16,
        number_of_iterations: int = 1000,
        repetition_period: int = REPETITION_PERIOD,
        rfswitch: str = "open",
    ) -> tuple[
        dict[tuple[str, int, int], npt.NDArray[np.complex64]],
        dict[tuple[str, int, int], float],
    ]:
        w = WaveSequence(
            num_wait_words=0,  # words
            num_repeats=number_of_iterations,  # times
        )
        w.add_chunk(
            # iq_samples must be a multiple of 64
            iq_samples := 64 * [(32767, 0)],  # samples
            num_blank_words=repetition_period - len(iq_samples) // 4,
            num_repeats=1,
        )

        w0 = WaveSequence(
            num_wait_words=0,  # words
            num_repeats=number_of_iterations,  # times
        )
        w0.add_chunk(
            # iq_samples must be a multiple of 64
            iq_samples := 64 * [(0, 0)],  # samples
            num_blank_words=repetition_period - len(iq_samples) // 4,
            num_repeats=1,
        )

        c = CaptureParam()
        c.capture_delay = capture_delay  # words
        c.num_integ_sections = number_of_iterations
        c.add_sum_section(
            num_words := 3 * w.chunk(0).num_wave_words,  # words
            num_post_blank_words=repetition_period - num_words,
        )
        # c.sel_dsp_units_to_enable(DspUnit.INTEGRATION)

        system = self._system

        recv_box, recv_port_num = self._recv_port
        send_box, send_port_num = self._send_port
        trig_box, trig_port_num = self._trig_port

        if not recv_box == trig_box:
            raise ValueError("recv box and trig box must be the same")

        if send_box == trig_box and send_port_num == trig_port_num:
            a = Action.build(
                system=system,
                settings=[
                    RunitSetting(
                        cprm=c,
                        runit=RunitId(
                            box=recv_box,
                            port=recv_port_num,
                            runit=0,
                        ),
                    ),
                    TriggerSetting(
                        triggerd_port=recv_port_num,
                        trigger_awg=AwgId(
                            trig_box,
                            port=trig_port_num,
                            channel=0,
                        ),
                    ),
                    AwgSetting(
                        wseq=w,
                        awg=AwgId(
                            box=send_box,
                            port=send_port_num,
                            channel=0,
                        ),
                    ),
                ],
            )
        else:
            a = Action.build(
                system=system,
                settings=[
                    RunitSetting(
                        cprm=c,
                        runit=RunitId(
                            box=recv_box,
                            port=recv_port_num,
                            runit=0,
                        ),
                    ),
                    TriggerSetting(
                        triggerd_port=recv_port_num,
                        trigger_awg=AwgId(
                            trig_box,
                            port=trig_port_num,
                            channel=0,
                        ),
                    ),
                    AwgSetting(
                        wseq=w0,
                        awg=AwgId(
                            box=trig_box,
                            port=trig_port_num,
                            channel=0,
                        ),
                    ),
                    AwgSetting(
                        wseq=w,
                        awg=AwgId(
                            box=send_box,
                            port=send_port_num,
                            channel=0,
                        ),
                    ),
                ],
            )

        self._system.box[recv_box].config_port(recv_port_num, rfswitch=rfswitch)

        _, data = a.action()

        iqs, angle = {}, {}
        for runit, iq in data.items():
            iq = iq.reshape(
                number_of_iterations,
                iq.shape[0] // number_of_iterations,
            ).sum(axis=0)
            iqs[runit], angle[runit] = principal_axis_rotation(iq)
        return iqs, angle


def measure(
    qubecalib: QubeCalib,
    *,
    recv_port: tuple[str, int],  # (box_name, receive port)
    send_port: tuple[str, int],  # (box_name, send port)
    cnco_offset: float = 0,  # multiple of 15.625e6
    cnco_freq: float = 1500e6,  # Hz multiple of 15.625e6
    lo_freq: float = 11000e6,  # Hz multiple of 500e6
    capture_delay: int = 7 * 16,  # multiple of 16 words
    number_of_iterations: int = 1000,
    repetition_period: int = REPETITION_PERIOD,
    clock_master: str = "10.3.0.255",
) -> tuple[
    dict[tuple[str, int, int], npt.NDArray[np.complex64]],
    dict[tuple[str, int, int], float],
]:
    w = WaveSequence(
        num_wait_words=0,  # words
        num_repeats=number_of_iterations,  # times
    )

    w.add_chunk(
        # iq_samples must be a multiple of 64
        iq_samples := 64 * [(32767, 0)],  # samples
        num_blank_words=repetition_period - len(iq_samples) // 4,
        num_repeats=1,
    )

    w0 = WaveSequence(
        num_wait_words=0,  # words
        num_repeats=number_of_iterations,  # times
    )

    w0.add_chunk(
        # iq_samples must be a multiple of 64
        iq_samples := 64 * [(0, 0)],  # samples
        num_blank_words=repetition_period - len(iq_samples) // 4,
        num_repeats=1,
    )

    c = CaptureParam()
    c.capture_delay = capture_delay  # words
    c.num_integ_sections = number_of_iterations
    c.add_sum_section(
        num_words := 3 * w.chunk(0).num_wave_words,  # words
        num_post_blank_words=repetition_period - num_words,
    )
    # c.sel_dsp_units_to_enable(DspUnit.INTEGRATION)

    boxes = set([recv_port[0], send_port[0]])

    system = Quel1System.create(
        clockmaster=QuBEMasterClient(clock_master),
        boxes=[qubecalib.create_named_box(box) for box in boxes],
    )

    recv_box, recv_port_num = recv_port
    send_box, send_port_num = send_port

    system.box[send_box].config_port(
        send_port_num,
        lo_freq=lo_freq,
        # cnco_freq=1500e6 - 2 * 8 * 15.625e6,
        cnco_freq=cnco_freq - cnco_offset,
        sideband="L",
    )
    system.box[recv_box].config_port(
        recv_port_num,
        lo_freq=lo_freq,
        # cnco_freq=1500e6 - 2 * 8 * 15.625e6,
        cnco_freq=cnco_freq - cnco_offset,
        rfswitch="open",
    )

    trig_port_num: int | tuple[int, int] = next(
        iter(system.box[recv_box].get_loopbacks_of_port(recv_port_num))
    )
    if isinstance(trig_port_num, tuple):
        raise ValueError("fogi port is not supported yet")

    if send_box == recv_box and send_port_num == recv_port_num:
        a = Action.build(
            system=system,
            settings=[
                RunitSetting(
                    cprm=c,
                    runit=RunitId(
                        box=recv_box,
                        port=recv_port_num,
                        runit=0,
                    ),
                ),
                TriggerSetting(
                    triggerd_port=recv_port_num,
                    trigger_awg=AwgId(
                        recv_box,
                        port=trig_port_num,
                        channel=0,
                    ),
                ),
                AwgSetting(
                    wseq=w,
                    awg=AwgId(
                        box=recv_box,
                        port=trig_port_num,
                        channel=0,
                    ),
                ),
            ],
        )
    else:
        a = Action.build(
            system=system,
            settings=[
                RunitSetting(
                    cprm=c,
                    runit=RunitId(
                        box=recv_box,
                        port=recv_port_num,
                        runit=0,
                    ),
                ),
                TriggerSetting(
                    triggerd_port=recv_port_num,
                    trigger_awg=AwgId(
                        recv_box,
                        port=trig_port_num,
                        channel=0,
                    ),
                ),
                AwgSetting(
                    wseq=w0,
                    awg=AwgId(
                        box=recv_box,
                        port=trig_port_num,
                        channel=0,
                    ),
                ),
                AwgSetting(
                    wseq=w,
                    awg=AwgId(
                        box=send_box,
                        port=send_port_num,
                        channel=0,
                    ),
                ),
            ],
        )

    _, data = a.action()
    iqs, angle = {}, {}
    for runit, iq in data.items():
        iq = iq.reshape(
            number_of_iterations,
            iq.shape[-1] // number_of_iterations,
        ).sum(axis=0)
        iqs[runit], angle[runit] = principal_axis_rotation(iq)
    return iqs, angle
