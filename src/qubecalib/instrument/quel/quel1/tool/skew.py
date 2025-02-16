from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# from ......qubecalib import QubeCalib
from typing import Final

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import yaml
from tqdm.auto import tqdm

from .....instrument.quel.quel1.driver import Quel1System
from .....neopulse import Blank, Capture, Flushleft, Rectangle, Sequence, Series
from .....qubecalib import QubeCalib, SystemConfigDatabase

DEFAULT_FREQUENCY = 9.75
DEFAULT_LO_FREQ = 11000e6
DEFAULT_CNCO_FREQ = 1250e6
DEFAULT_FNCO_FREQ = 0.0
DEFAULT_SIDEBAND = "L"
DEFAULT_VATT = 0x900

REPETITION_PERIOD = 1280 * 128  # words
EXTRA_CAPTURE_RANGE = 1024  # words


def str2port(v: str) -> tuple[str, int]:
    box_name, nport = v.split("-")[:2]
    return box_name, int(nport)


def port2str(v: tuple[str, int]) -> str:
    box_name, nport = v
    return f"{box_name}-{nport}"


@dataclass
class SkewAdjust:
    sysdb: SystemConfigDatabase
    target_ports: set[tuple[str, int]] = field(default_factory=set)
    slot: dict[tuple[str, int], int] = field(default_factory=dict)
    wait: dict[tuple[str, int], int] = field(default_factory=dict)
    time_to_start: int = 0

    def push(self) -> None:
        for (box_name, _), slot in self.slot.items():
            self.sysdb.timing_shift[box_name] = slot * 16
        for (box_name, _), wait in self.wait.items():
            self.sysdb.skew[box_name] = wait
        self.sysdb.time_to_start = self.time_to_start

    def pull(self) -> None:
        for port in self.target_ports:
            box_name, _ = port
            self.slot[port] = self.sysdb.timing_shift[box_name] // 16
            self.wait[port] = self.sysdb.skew[box_name]
        self.time_to_start = self.sysdb.time_to_start

    def backup(self) -> SkewAdjust:
        o = SkewAdjust(self.sysdb, target_ports=self.target_ports)
        o.pull()
        return o


class Skew:
    def __init__(
        self,
        system: Quel1System,
        *,
        qubecalib: QubeCalib,
        monitor_port: tuple[str, int] = ("", 0),
        trigger_nport: int = 0,
        reference_port: tuple[str, int] = ("", 0),
    ) -> None:  # TODO ここは多分変わります
        self._system: Final[Quel1System] = system
        self._qubecalib: Final[QubeCalib] = qubecalib
        self._monitor_port: tuple[str, int] = monitor_port
        self._trigger_nport: int = trigger_nport
        self._reference_port: tuple[str, int] = reference_port
        self._scale: dict[tuple[str, int], float] = {}
        self._measured: dict[tuple[str, int], npt.NDArray] = {}
        self._estimated: dict[tuple[str, int], npt.NDArray] = {}
        self._offset: dict[tuple[str, int], int] = {}
        self._target_port: set[tuple[str, int]] = set()
        self._skew_adjust: SkewAdjust = SkewAdjust(qubecalib.sysdb)

    @property
    def monitor_port(self) -> tuple[str, int]:
        return self._monitor_port

    @monitor_port.setter
    def monitor_port(self, monitor_port: tuple[str, int]) -> None:
        self._monitor_port = monitor_port

    @property
    def trigger_nport(self) -> int:
        return self._trigger_nport

    @trigger_nport.setter
    def trigger_nport(self, trigger_nport: int) -> None:
        self._trigger_nport = trigger_nport

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
        lo_freq: float = DEFAULT_LO_FREQ,
        cnco_freq: float = DEFAULT_CNCO_FREQ,
        sideband: str = DEFAULT_SIDEBAND,
        rfswitch: str = "open",
        vatt: int = DEFAULT_VATT,
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
            box.config_channel(trig_port_number, channel, fnco_freq=DEFAULT_FNCO_FREQ)
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
        lo_freq: float = DEFAULT_LO_FREQ,
        cnco_freq: float = DEFAULT_CNCO_FREQ,
        sideband: str = DEFAULT_SIDEBAND,
        rfswitch: str = "pass",
        vatt: int = DEFAULT_VATT,
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
            box.config_channel(ctrl_port_number, channel, fnco_freq=DEFAULT_FNCO_FREQ)

    @classmethod
    def acquire_target(
        cls,
        qc: QubeCalib,
        port: tuple[str, int],
        channel: int = 0,
    ) -> str:
        channel_id: tuple[str, int, int] = (*port, channel)
        return qc.sysdb.get_target_by_channel(*channel_id)

    def target_from_box(self, box_names: list[str]) -> set[tuple[str, int]]:
        return {
            (box_name, nport)
            for box_name, nport in self._target_port
            if box_name in box_names  #  and (box_name, nport) != self._reference_port
        }

    def measure(
        self,
        *,
        offset: int = 1,  # multiplied by 128 ns
        extra_capture_range: int = EXTRA_CAPTURE_RANGE,  # multiple of 128 ns
        show_reference: bool = True,
        reset_skew_parameter: bool = False,
    ) -> None:
        target_ports = self.target_from_box(list(self._system.boxes))
        for target_box, target_nport in tqdm(target_ports):
            target_port = (target_box, target_nport)
            self._measure(
                target_port,
                offset=offset,
                extra_capture_range=extra_capture_range,
                show_reference=show_reference,
                reset_skew_parameter=reset_skew_parameter,
            )

    def _measure(
        self,
        target_port: tuple[str, int],
        *,
        offset: int = 0,  # multiplied by 128 ns
        extra_capture_range: int = EXTRA_CAPTURE_RANGE,  # multiple of 128 ns
        reset_skew_parameter: bool = True,
        show_reference: bool = False,
    ) -> npt.NDArray:
        system = self._system
        qc = self._qubecalib
        reference_port = self._reference_port
        monitor_port = self._monitor_port
        monitor_box_name, _ = monitor_port
        trigger_port: tuple[str, int] = (monitor_box_name, self._trigger_nport)
        target_ports = set([target_port])
        for p in [monitor_port, trigger_port, reference_port]:
            qc.sysdb._target_settings[self.acquire_target(qc, p)] = dict(
                frequency=DEFAULT_FREQUENCY
            )
        for p in target_ports:
            qc.sysdb._target_settings[self.acquire_target(qc, p)] = dict(
                frequency=DEFAULT_FREQUENCY
            )
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
        backup = (
            self._skew_adjust.slot[reference_port],
            self._skew_adjust.wait[reference_port],
            self._skew_adjust.slot[target_port],
            self._skew_adjust.wait[target_port],
        )
        if reset_skew_parameter:
            self._skew_adjust.slot[reference_port] = 0
            self._skew_adjust.wait[reference_port] = 0
            self._skew_adjust.slot[target_port] = 0
            self._skew_adjust.wait[target_port] = 0
            self._skew_adjust.push()

        pulse = Rectangle(duration=128, amplitude=1.0)
        blank = Blank(
            duration=offset * 128
        )  # reference と target の間に offset * 128ns の待ち時間を入れる
        capture = Capture(
            duration=2 * len(target_ports) * 128 + 128 + extra_capture_range
        )

        reference_scale = (
            self._scale[reference_port] if reference_port in self._scale else 1
        )
        with Sequence() as seq:
            with Flushleft():
                with Series():
                    scale = reference_scale if show_reference else 0
                    pulse.scaled(scale).target(self.acquire_target(qc, reference_port))
                    if offset != 0:
                        blank.target()
                    scale = (
                        self._scale[target_port] if target_port in self._scale else 1
                    )
                    pulse.scaled(scale).target(self.acquire_target(qc, target_port))
                pulse.scaled(0).target(self.acquire_target(qc, trigger_port))
                capture.target(self.acquire_target(qc, monitor_port))

        qc.add_sequence(seq)

        for _, data, _ in qc.step_execute(
            repeats=100,
            interval=REPETITION_PERIOD,
            integral_mode="single",
            dsp_demodulation=False,
            software_demodulation=True,
        ):
            for _, iqs in data.items():
                iqs = iqs[0].sum(axis=1).squeeze()
                self._measured[target_port] = iqs
                self._offset[target_port] = offset

        if reset_skew_parameter:
            adj = self._skew_adjust
            (
                adj.slot[reference_port],
                adj.wait[reference_port],
                adj.slot[target_port],
                adj.wait[target_port],
            ) = backup

        return iqs

    def _adjust(
        self,
        target_port: tuple[str, int],
        reference_idx: int,
    ) -> tuple[int, int, npt.NDArray]:
        measured = self._measured[target_port]
        offset = self._offset[target_port]
        idx, estimated = self._fit_pulse(measured)
        delta = idx - reference_idx
        slot = delta // 64 + 1
        wait = 64 - delta % 64
        self._skew_adjust.slot[target_port] = -slot
        self._skew_adjust.wait[target_port] = wait
        self._estimated[target_port] = estimated
        return slot, wait, estimated

    def adjust(
        self,
        *,
        offset: int = 1,  # multiplied by 128 ns
        extra_capture_range: int = EXTRA_CAPTURE_RANGE,  # multiple of 128 ns
    ) -> dict[tuple[str, int], dict[str, int]]:
        target_ports = self.target_from_box(list(self._system.boxes))
        for target_box, target_nport in tqdm(target_ports):
            target_port = (target_box, target_nport)
            iqs = self._measure(
                target_port,
                offset=offset,
                extra_capture_range=extra_capture_range,
                show_reference=False,
                reset_skew_parameter=True,
            )
            if target_port == self._reference_port:
                reference_idx, _ = self._fit_pulse(iqs)
        for target_port in target_ports:
            self._adjust(target_port, reference_idx)
        min_slot = min([self._skew_adjust.slot[p] for p in target_ports])
        for p in target_ports:
            self._skew_adjust.slot[p] -= min_slot
        self._skew_adjust.push()
        return {
            p: {
                "slot": self._skew_adjust.slot[p],
                "wait": self._skew_adjust.wait[p],
            }
            for p in target_ports
        }

    @classmethod
    def fit_pulse(cls, iqs: npt.NDArray) -> tuple[int, int, npt.NDArray]:
        idx, estimated = cls._fit_pulse(iqs)
        slot, wait = idx // 64, idx % 64
        return slot, wait, estimated

    @classmethod
    def _fit_pulse(cls, iqs: npt.NDArray) -> tuple[int, npt.NDArray]:
        # ref_waveform = np.ones(64)
        # abs_iqs = np.abs(iqs).reshape(-1, 64)
        # corr = ref_waveform.reshape(-1, 64) / len(ref_waveform) * abs_iqs
        # slot = corr.sum(axis=1).argmax()
        dif_waveform = np.array([-1, 0, 1])[::-1]
        # conv = np.convolve(
        #     dif_waveform, np.abs(iqs)[slot * 64 : (slot + 1) * 64], "valid"
        # )
        conv = np.convolve(dif_waveform, np.abs(iqs), "valid")
        idx = int(conv.argmax()) + 1
        # slot = iloc // 64
        # idx = iloc % 64
        estimated = np.zeros(iqs.size)
        estimated[idx : idx + 64] = np.ones(64)
        estimated *= np.sqrt((np.abs(iqs).var())) / np.sqrt(estimated.var())
        estimated += np.abs(iqs).mean() - estimated.mean()
        # wait = idx
        # dif_waveform = np.array([])
        return idx, estimated

    def plot(self) -> None:
        measured = self._measured
        estimated = self._estimated
        fig = go.FigureWidget(
            layout=go.Layout(
                height=len(measured) * 100 + 100,
                width=None,
                autosize=True,
                showlegend=True,
            ),
        ).set_subplots(
            rows=len(measured),
            cols=1,
            shared_xaxes=True,
        )
        for i, (focused_port, iqs) in enumerate(measured.items()):
            fig.add_trace(
                go.Scatter(
                    x=2 * np.arange(len(iqs)),
                    y=np.abs(iqs),
                    mode="lines",
                    name=f"{focused_port}: measured",
                ),
                row=1 + i,
                col=1,
            )
        for i, (focused_port, iqs) in enumerate(estimated.items()):
            fig.add_trace(
                go.Scatter(
                    x=2 * np.arange(len(iqs)),
                    y=iqs,
                    mode="lines",
                    name=f"{focused_port}: estimated",
                ),
                row=1 + i,
                col=1,
            )
        fig.show()

    def load(self, filename: str) -> None:
        self._skew_adjust, _ = self._load(filename)
        self._skew_adjust.push()

    def _load(self, filename: str) -> tuple[SkewAdjust, dict]:
        with open(Path(os.getcwd()) / Path(filename), "r") as file:
            config = yaml.safe_load(file)
        # sysdb = self._qubecalib.sysdb
        self._reference_port = str2port(config["reference_port"])
        self._monitor_port = str2port(config["monitor_port"])
        self._trigger_nport = config["trigger_nport"]
        self._target_port = {str2port(v) for v in config["target_port"]}
        self._scale = {str2port(p): v for p, v in config["scale"].items()}
        skew_adjust = SkewAdjust(self._qubecalib.sysdb, target_ports=self._target_port)
        box_setting = config["box_setting"]
        for port in self._target_port:
            box_name, _ = port
            skew_adjust.slot[port] = box_setting[box_name]["slot"]
            skew_adjust.wait[port] = box_setting[box_name]["wait"]
        skew_adjust.time_to_start = config["time_to_start"]
        return skew_adjust, config

    def save(self, filename: str) -> None:
        sysdb = self._qubecalib.sysdb
        config = {
            "time_to_start": sysdb.time_to_start,
            "box_setting": {
                box_name: {"slot": v // 16, "wait": sysdb.skew[box_name]}
                for box_name, v in sysdb.timing_shift.items()
            },
            "reference_port": port2str(self._reference_port),
            "monitor_port": port2str(self._monitor_port),
            "trigger_nport": self._trigger_nport,
            "target_port": {port2str(v) for v in self._target_port},
            "scale": {port2str(p): v for p, v in self._scale.items()},
        }
        with open(Path(os.getcwd()) / Path(filename), "w") as file:
            yaml.safe_dump(config, file)
