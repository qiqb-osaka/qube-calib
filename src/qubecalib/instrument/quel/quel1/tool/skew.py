from __future__ import annotations

# from ......qubecalib import QubeCalib
from typing import Final

import numpy as np
import numpy.typing as npt
from tqdm.notebook import tqdm

from .....instrument.quel.quel1.driver import Quel1System
from .....neopulse import Capture, Flushleft, Rectangle, Sequence, Series
from .....qubecalib import QubeCalib

REPETITION_PERIOD = 1280 * 128  # words


class Skew:
    def __init__(
        self, system: Quel1System, qubecalib: QubeCalib
    ) -> None:  # TODO ここは多分変わります
        self._system: Final[Quel1System] = system
        self._qubecalib: Final[QubeCalib] = qubecalib
        self._monitor_port: tuple[str, int] = ("", 0)
        self._trigger_port: int = 0
        self._reference_port: tuple[str, int] = ("", 0)
        self._scale: dict[tuple[str, int], float] = {}

    @property
    def monitor_port(self) -> tuple[str, int]:
        return self._monitor_port

    @monitor_port.setter
    def monitor_port(self, monitor_port: tuple[str, int]) -> None:
        self._monitor_port = monitor_port

    @property
    def trigger_port(self) -> int:
        return self._trigger_port

    @trigger_port.setter
    def trigger_port(self, trigger_port: int) -> None:
        self._trigger_port = trigger_port

    @property
    def reference_port(self) -> tuple[str, int]:
        return self._reference_port

    @reference_port.setter
    def reference_port(self, reference_port: tuple[str, int]) -> None:
        self._reference_port = reference_port

    def set_scale(self, port: tuple[str, int], scale: float) -> None:
        self._scale[port] = scale

    def read_setup(
        self,
        read_port: tuple[str, int],
        trig_port_number: int,
        *,
        lo_freq: float = 11000e6,
        cnco_freq: float = 1250e6,
        sideband: str = "L",
        rfswitch: str = "open",
        vatt: int = 0x900,
    ) -> None:
        box_name, read_port_number = read_port
        box = self._system.box[box_name]
        box.config_port(
            trig_port_number,
            lo_freq=lo_freq,
            cnco_freq=cnco_freq,
            sideband=sideband,
            vatt=vatt,
        )
        for channel in box.get_channels_of_port(trig_port_number):
            box.config_channel(trig_port_number, channel, fnco_freq=0.0)
        box.config_port(
            read_port_number,
            lo_freq=lo_freq,
            cnco_freq=cnco_freq,
            rfswitch=rfswitch,
        )

    def ctrl_setup(
        self,
        ctrl_port: tuple[str, int],
        *,
        lo_freq: float = 11000e6,
        cnco_freq: float = 1250e6,
        sideband: str = "L",
        rfswitch: str = "pass",
        vatt: int = 0x900,
    ) -> None:
        ctrl_box, ctrl_port_number = ctrl_port
        box = self._system.box[ctrl_box]
        box.config_port(
            ctrl_port_number,
            lo_freq=lo_freq,
            cnco_freq=cnco_freq,
            sideband=sideband,
            rfswitch=rfswitch,
            vatt=vatt,
        )
        for channel in box.get_channels_of_port(ctrl_port_number):
            box.config_channel(ctrl_port_number, channel, fnco_freq=0.0)

    @classmethod
    def acquire_target(
        cls,
        qc: QubeCalib,
        port: tuple[str, int],
        channel: int = 0,
    ) -> str:
        channel_id: tuple[str, int, int] = (*port, channel)
        return qc.sysdb.get_target_by_channel(*channel_id)

    def measure_skew(
        self,
        target_ports: list[tuple[str, int]],
        *,
        show_reference: bool = False,
        extra_capture_range: int = 1024,  # multiple of 64 ns
    ) -> dict[str, npt.NDArray]:
        system = self._system
        qc = self._qubecalib
        reference_port = self._reference_port
        monitor_port = self._monitor_port
        monitor_box_name, _ = monitor_port
        trigger_port: tuple[str, int] = (monitor_box_name, self._trigger_port)
        for p in [monitor_port, trigger_port, reference_port]:
            qc.sysdb._target_settings[self.acquire_target(qc, p)] = dict(frequency=9.75)
        for p in target_ports:
            qc.sysdb._target_settings[self.acquire_target(qc, p)] = dict(frequency=9.75)
        trigger_box_name, trigger_port_number = trigger_port
        trigger_channel: tuple[str, int, int] = (
            trigger_box_name,
            trigger_port_number,
            0,
        )
        qc.sysdb.trigger = {monitor_port: trigger_channel}
        self.read_setup(monitor_port, trigger_port_number, rfswitch="open")
        for ctrl_box, ctrl_nport in [reference_port] + [p for p in target_ports]:
            self.ctrl_setup(
                (ctrl_box, ctrl_nport),
                sideband="L",
                rfswitch="pass",
                vatt=0xC00,
            )
            m = system.box[ctrl_box].get_monitor_input_ports()
            if m:
                for port in m:
                    if isinstance(port, tuple):
                        raise ValueError("fogi port is not supported yet")
                    system.box[ctrl_box].config_rfswitch(port, rfswitch="open")

        pulse = Rectangle(duration=128, amplitude=1.0)
        capture = Capture(
            duration=2 * len(target_ports) * 128 + 128 + extra_capture_range
        )

        result: dict[str, npt.NDArray] = {}
        reference_scale = (
            self._scale[reference_port] if reference_port in self._scale else 1
        )
        for j in tqdm(range(len(target_ports))):
            focused_port = target_ports[j]

            with Sequence() as seq:
                with Flushleft():
                    with Series():
                        # if j == 0:
                        #     pulse.scaled(reference_scale).target(
                        #         self.acquire_target(qc, reference_port)
                        #     )
                        #     for target_port in target_ports[1:]:
                        #         pulse.scaled(0).target(
                        #             self.acquire_target(qc, target_port)
                        #         )
                        # else:
                        #     pulse.scaled(reference_scale).target(
                        #         self.acquire_target(qc, reference_port)
                        #     )
                        #     for target_port in target_ports[1:]:
                        #         target_scale = (
                        #             self._scale[target_port]
                        #             if target_port in self._scale
                        #             else 1
                        #         )
                        #         scale = (
                        #             target_scale if target_port == focused_port else 0
                        #         )
                        #         pulse.scaled(scale).target(
                        #             self.acquire_target(qc, target_port)
                        #         )
                        pulse.scaled(reference_scale).target(
                            self.acquire_target(qc, reference_port)
                        )
                        for target_port in target_ports[1:]:
                            target_scale = (
                                self._scale[target_port]
                                if target_port in self._scale
                                else 1
                            )
                            scale = target_scale if target_port == focused_port else 0
                            pulse.scaled(scale).target(
                                self.acquire_target(qc, target_port)
                            )
                    pulse.scaled(0).target(self.acquire_target(qc, trigger_port))
                    capture.target(self.acquire_target(qc, monitor_port))

            qc.add_sequence(seq)

            for _, data, _ in qc.step_execute(
                repeats=100,
                interval=1280 * 128,
                integral_mode="single",
                dsp_demodulation=False,
                software_demodulation=True,
            ):
                for _, iqs in data.items():
                    iqs = iqs[0].sum(axis=1).squeeze()
                    result[focused_port] = iqs

        return result

    def measure_single_target(
        self,
        target_port: tuple[str, int],
        *,
        show_reference: bool = False,
        extra_capture_range: int = 1024,  # multiple of 64 ns
    ) -> dict[str, npt.NDArray]:
        system = self._system
        qc = self._qubecalib
        reference_port = self._reference_port
        monitor_port = self._monitor_port
        monitor_box_name, _ = monitor_port
        trigger_port: tuple[str, int] = (monitor_box_name, self._trigger_port)
        target_ports = [target_port]
        for p in [monitor_port, trigger_port, reference_port]:
            qc.sysdb._target_settings[self.acquire_target(qc, p)] = dict(frequency=9.75)
        for p in target_ports:
            qc.sysdb._target_settings[self.acquire_target(qc, p)] = dict(frequency=9.75)
        trigger_box_name, trigger_port_number = trigger_port
        trigger_channel: tuple[str, int, int] = (
            trigger_box_name,
            trigger_port_number,
            0,
        )
        qc.sysdb.trigger = {monitor_port: trigger_channel}
        self.read_setup(monitor_port, trigger_port_number, rfswitch="open")
        for ctrl_box, ctrl_nport in [reference_port] + [p for p in target_ports]:
            self.ctrl_setup(
                (ctrl_box, ctrl_nport),
                sideband="L",
                rfswitch="pass",
                vatt=0xC00,
            )
            m = system.box[ctrl_box].get_monitor_input_ports()
            if m:
                for port in m:
                    if isinstance(port, tuple):
                        raise ValueError("fogi port is not supported yet")
                    system.box[ctrl_box].config_rfswitch(port, rfswitch="open")

        pulse = Rectangle(duration=128, amplitude=1.0)
        capture = Capture(
            duration=2 * len(target_ports) * 128 + 128 + extra_capture_range
        )

        result: dict[str, npt.NDArray] = {}
        reference_scale = (
            self._scale[reference_port] if reference_port in self._scale else 1
        )
        for j in tqdm(range(len(target_ports))):
            focused_port = target_ports[j]

            with Sequence() as seq:
                with Flushleft():
                    with Series():
                        # if j == 0:
                        #     pulse.scaled(reference_scale).target(
                        #         self.acquire_target(qc, reference_port)
                        #     )
                        #     for target_port in target_ports[1:]:
                        #         pulse.scaled(0).target(
                        #             self.acquire_target(qc, target_port)
                        #         )
                        # else:
                        #     pulse.scaled(reference_scale).target(
                        #         self.acquire_target(qc, reference_port)
                        #     )
                        #     for target_port in target_ports[1:]:
                        #         target_scale = (
                        #             self._scale[target_port]
                        #             if target_port in self._scale
                        #             else 1
                        #         )
                        #         scale = (
                        #             target_scale if target_port == focused_port else 0
                        #         )
                        #         pulse.scaled(scale).target(
                        #             self.acquire_target(qc, target_port)
                        #         )
                        scale = reference_scale if show_reference else 0
                        pulse.scaled(scale).target(
                            self.acquire_target(qc, reference_port)
                        )
                        for target_port in target_ports:
                            target_scale = (
                                self._scale[target_port]
                                if target_port in self._scale
                                else 1
                            )
                            scale = target_scale if target_port == focused_port else 0
                            pulse.scaled(scale).target(
                                self.acquire_target(qc, target_port)
                            )
                    pulse.scaled(0).target(self.acquire_target(qc, trigger_port))
                    capture.target(self.acquire_target(qc, monitor_port))

            qc.add_sequence(seq)

            for _, data, _ in qc.step_execute(
                repeats=100,
                interval=1280 * 128,
                integral_mode="single",
                dsp_demodulation=False,
                software_demodulation=True,
            ):
                for _, iqs in data.items():
                    iqs = iqs[0].sum(axis=1).squeeze()
                    result[focused_port] = iqs

        return result

    @classmethod
    def fit_pulse(cls, iqs: npt.NDArray) -> tuple[int, int, npt.NDArray]:
        # ref_waveform = np.ones(64)
        # abs_iqs = np.abs(iqs).reshape(-1, 64)
        # corr = ref_waveform.reshape(-1, 64) / len(ref_waveform) * abs_iqs
        # slot = corr.sum(axis=1).argmax()
        dif_waveform = np.array([-1, 1])[::-1]
        # conv = np.convolve(
        #     dif_waveform, np.abs(iqs)[slot * 64 : (slot + 1) * 64], "valid"
        # )
        conv = np.convolve(dif_waveform, np.abs(iqs), "valid")
        iloc = int(conv.argmax()) + 1
        slot = iloc // 64
        idx = iloc % 64
        estimated = np.zeros(iqs.size)
        estimated[slot * 64 + idx : slot * 64 + idx + 64] = np.ones(64)
        estimated *= np.sqrt((np.abs(iqs).var())) / np.sqrt(estimated.var())
        estimated += np.abs(iqs).mean() - estimated.mean()
        wait = idx
        # dif_waveform = np.array([])
        return slot, wait, estimated


# class Skew:
#     def __init__(
#         self,
#         system: Quel1System,
#         *,
#         recv_port: tuple[str, int],
#         send_port: tuple[str, int],
#         trig_port: tuple[str, int],
#     ) -> None:
#         self._system: Final[Quel1System] = system
#         self._recv_port: Final[tuple[str, int]] = recv_port
#         self._send_port: Final[tuple[str, int]] = send_port
#         self._trig_port: Final[tuple[str, int]] = trig_port

#     @classmethod
#     def create_with_qubecalib(
#         cls,
#         qubecalib: QubeCalib,
#         *,
#         recv_port: tuple[str, int],
#         send_port: tuple[str, int],
#         trig_port: tuple[str, int],
#     ) -> Skew:
#         if qubecalib.system_config_database._clockmaster_setting is None:
#             raise ValueError("Clockmaster setting is not found")

#         boxes = set([recv_port[0], send_port[0], trig_port[0]])

#         system = Quel1System.create(
#             clockmaster=QuBEMasterClient(
#                 qubecalib.system_config_database._clockmaster_setting.ipaddr
#             ),
#             boxes=[qubecalib.create_named_box(box) for box in boxes],
#         )
#         return cls.create(
#             system,
#             recv_port=recv_port,
#             send_port=send_port,
#             trig_port=trig_port,
#         )

#     @classmethod
#     def create(
#         cls,
#         system: Quel1System,
#         *,
#         recv_port: tuple[str, int],
#         send_port: tuple[str, int],
#         trig_port: tuple[str, int],
#     ) -> Skew:
#         return cls(
#             system,
#             recv_port=recv_port,
#             send_port=send_port,
#             trig_port=trig_port,
#         )

#     def frequency_setting(
#         self,
#         cnco_offset: float = 0,  # multiple of 15.625e6
#         cnco_freq: float = 1500e6,  # Hz multiple of 15.625e6
#         lo_freq: float = 11000e6,  # Hz multiple of 500e6
#     ) -> None:
#         recv_box, recv_port_num = self._recv_port
#         send_box, send_port_num = self._send_port

#         self._system.box[send_box].config_port(
#             send_port_num,
#             lo_freq=lo_freq,
#             cnco_freq=cnco_freq - cnco_offset,
#             sideband="L",
#         )
#         self._system.box[recv_box].config_port(
#             recv_port_num,
#             lo_freq=lo_freq,
#             cnco_freq=cnco_freq - cnco_offset,
#         )

#     def _measure(
#         self,
#         capture_delay: int = 7 * 16,
#         number_of_iterations: int = 1000,
#         repetition_period: int = REPETITION_PERIOD,
#         rfswitch: str = "open",
#     ) -> tuple[
#         dict[tuple[str, int, int], npt.NDArray[np.complex64]],
#         dict[tuple[str, int, int], float],
#     ]:
#         w = WaveSequence(
#             num_wait_words=0,  # words
#             num_repeats=number_of_iterations,  # times
#         )
#         w.add_chunk(
#             # iq_samples must be a multiple of 64
#             iq_samples := 64 * [(32767, 0)],  # samples
#             num_blank_words=repetition_period - len(iq_samples) // 4,
#             num_repeats=1,
#         )

#         w0 = WaveSequence(
#             num_wait_words=0,  # words
#             num_repeats=number_of_iterations,  # times
#         )
#         w0.add_chunk(
#             # iq_samples must be a multiple of 64
#             iq_samples := 64 * [(0, 0)],  # samples
#             num_blank_words=repetition_period - len(iq_samples) // 4,
#             num_repeats=1,
#         )

#         c = CaptureParam()
#         c.capture_delay = capture_delay  # words
#         c.num_integ_sections = number_of_iterations
#         c.add_sum_section(
#             num_words := 3 * w.chunk(0).num_wave_words,  # words
#             num_post_blank_words=repetition_period - num_words,
#         )
#         # c.sel_dsp_units_to_enable(DspUnit.INTEGRATION)

#         system = self._system

#         recv_box, recv_port_num = self._recv_port
#         send_box, send_port_num = self._send_port
#         trig_box, trig_port_num = self._trig_port

#         if not recv_box == trig_box:
#             raise ValueError("recv box and trig box must be the same")

#         if send_box == trig_box and send_port_num == trig_port_num:
#             a = Action.build(
#                 system=system,
#                 settings=[
#                     RunitSetting(
#                         cprm=c,
#                         runit=RunitId(
#                             box=recv_box,
#                             port=recv_port_num,
#                             runit=0,
#                         ),
#                     ),
#                     TriggerSetting(
#                         triggerd_port=recv_port_num,
#                         trigger_awg=AwgId(
#                             trig_box,
#                             port=trig_port_num,
#                             channel=0,
#                         ),
#                     ),
#                     AwgSetting(
#                         wseq=w,
#                         awg=AwgId(
#                             box=send_box,
#                             port=send_port_num,
#                             channel=0,
#                         ),
#                     ),
#                 ],
#             )
#         else:
#             a = Action.build(
#                 system=system,
#                 settings=[
#                     RunitSetting(
#                         cprm=c,
#                         runit=RunitId(
#                             box=recv_box,
#                             port=recv_port_num,
#                             runit=0,
#                         ),
#                     ),
#                     TriggerSetting(
#                         triggerd_port=recv_port_num,
#                         trigger_awg=AwgId(
#                             trig_box,
#                             port=trig_port_num,
#                             channel=0,
#                         ),
#                     ),
#                     AwgSetting(
#                         wseq=w0,
#                         awg=AwgId(
#                             box=trig_box,
#                             port=trig_port_num,
#                             channel=0,
#                         ),
#                     ),
#                     AwgSetting(
#                         wseq=w,
#                         awg=AwgId(
#                             box=send_box,
#                             port=send_port_num,
#                             channel=0,
#                         ),
#                     ),
#                 ],
#             )

#         self._system.box[recv_box].config_port(recv_port_num, rfswitch=rfswitch)

#         _, data = a.action()

#         iqs, angle = {}, {}
#         for runit, iq in data.items():
#             iq = iq.reshape(
#                 number_of_iterations,
#                 iq.shape[0] // number_of_iterations,
#             ).sum(axis=0)
#             iqs[runit], angle[runit] = principal_axis_rotation(iq)
#         return iqs, angle


# def measure(
#     qubecalib: QubeCalib,
#     *,
#     recv_port: tuple[str, int],  # (box_name, receive port)
#     send_port: tuple[str, int],  # (box_name, send port)
#     cnco_offset: float = 0,  # multiple of 15.625e6
#     cnco_freq: float = 1500e6,  # Hz multiple of 15.625e6
#     lo_freq: float = 11000e6,  # Hz multiple of 500e6
#     capture_delay: int = 7 * 16,  # multiple of 16 words
#     number_of_iterations: int = 1000,
#     repetition_period: int = REPETITION_PERIOD,
#     clock_master: str = "10.3.0.255",
# ) -> tuple[
#     dict[tuple[str, int, int], npt.NDArray[np.complex64]],
#     dict[tuple[str, int, int], float],
# ]:
#     w = WaveSequence(
#         num_wait_words=0,  # words
#         num_repeats=number_of_iterations,  # times
#     )

#     w.add_chunk(
#         # iq_samples must be a multiple of 64
#         iq_samples := 64 * [(32767, 0)],  # samples
#         num_blank_words=repetition_period - len(iq_samples) // 4,
#         num_repeats=1,
#     )

#     w0 = WaveSequence(
#         num_wait_words=0,  # words
#         num_repeats=number_of_iterations,  # times
#     )

#     w0.add_chunk(
#         # iq_samples must be a multiple of 64
#         iq_samples := 64 * [(0, 0)],  # samples
#         num_blank_words=repetition_period - len(iq_samples) // 4,
#         num_repeats=1,
#     )

#     c = CaptureParam()
#     c.capture_delay = capture_delay  # words
#     c.num_integ_sections = number_of_iterations
#     c.add_sum_section(
#         num_words := 3 * w.chunk(0).num_wave_words,  # words
#         num_post_blank_words=repetition_period - num_words,
#     )
#     # c.sel_dsp_units_to_enable(DspUnit.INTEGRATION)

#     boxes = set([recv_port[0], send_port[0]])

#     system = Quel1System.create(
#         clockmaster=QuBEMasterClient(clock_master),
#         boxes=[qubecalib.create_named_box(box) for box in boxes],
#     )

#     recv_box, recv_port_num = recv_port
#     send_box, send_port_num = send_port

#     system.box[send_box].config_port(
#         send_port_num,
#         lo_freq=lo_freq,
#         # cnco_freq=1500e6 - 2 * 8 * 15.625e6,
#         cnco_freq=cnco_freq - cnco_offset,
#         sideband="L",
#     )
#     system.box[recv_box].config_port(
#         recv_port_num,
#         lo_freq=lo_freq,
#         # cnco_freq=1500e6 - 2 * 8 * 15.625e6,
#         cnco_freq=cnco_freq - cnco_offset,
#         rfswitch="open",
#     )

#     trig_port_num: int | tuple[int, int] = next(
#         iter(system.box[recv_box].get_loopbacks_of_port(recv_port_num))
#     )
#     if isinstance(trig_port_num, tuple):
#         raise ValueError("fogi port is not supported yet")

#     if send_box == recv_box and send_port_num == recv_port_num:
#         a = Action.build(
#             system=system,
#             settings=[
#                 RunitSetting(
#                     cprm=c,
#                     runit=RunitId(
#                         box=recv_box,
#                         port=recv_port_num,
#                         runit=0,
#                     ),
#                 ),
#                 TriggerSetting(
#                     triggerd_port=recv_port_num,
#                     trigger_awg=AwgId(
#                         recv_box,
#                         port=trig_port_num,
#                         channel=0,
#                     ),
#                 ),
#                 AwgSetting(
#                     wseq=w,
#                     awg=AwgId(
#                         box=recv_box,
#                         port=trig_port_num,
#                         channel=0,
#                     ),
#                 ),
#             ],
#         )
#     else:
#         a = Action.build(
#             system=system,
#             settings=[
#                 RunitSetting(
#                     cprm=c,
#                     runit=RunitId(
#                         box=recv_box,
#                         port=recv_port_num,
#                         runit=0,
#                     ),
#                 ),
#                 TriggerSetting(
#                     triggerd_port=recv_port_num,
#                     trigger_awg=AwgId(
#                         recv_box,
#                         port=trig_port_num,
#                         channel=0,
#                     ),
#                 ),
#                 AwgSetting(
#                     wseq=w0,
#                     awg=AwgId(
#                         box=recv_box,
#                         port=trig_port_num,
#                         channel=0,
#                     ),
#                 ),
#                 AwgSetting(
#                     wseq=w,
#                     awg=AwgId(
#                         box=send_box,
#                         port=send_port_num,
#                         channel=0,
#                     ),
#                 ),
#             ],
#         )

#     _, data = a.action()
#     iqs, angle = {}, {}
#     for runit, iq in data.items():
#         iq = iq.reshape(
#             number_of_iterations,
#             iq.shape[-1] // number_of_iterations,
#         ).sum(axis=0)
#         iqs[runit], angle[runit] = principal_axis_rotation(iq)
#     return iqs, angle
