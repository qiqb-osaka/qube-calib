from __future__ import annotations

import os
from copy import copy
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path

# from ......qubecalib import QubeCalib
from typing import Final, cast

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import yaml
from tqdm.auto import tqdm

from .....instrument.quel.quel1.driver import Quel1System
from .....neopulse import Capture, Flushleft, Rectangle, Sequence
from .....qubecalib import Converter, Executor, QubeCalib, SystemConfigDatabase

DEFAULT_FREQUENCY = 9.75
DEFAULT_LO_FREQ = 11000e6
DEFAULT_CNCO_FREQ = 1250e6
DEFAULT_FNCO_FREQ = 0.0
DEFAULT_SIDEBAND = "L"
DEFAULT_VATT = 0x900
DEFAULT_CHANNEL_NUM = 0

REPETITION_PERIOD = 1280 * 128  # words
EXTRA_CAPTURE_RANGE = 1024  # words

DEFAULT_FNCO_LIMIT = 500e6


PORT = tuple[str, int]

logger = getLogger(__name__)


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

    @staticmethod
    def load(
        config: dict[str, str | int | set[str] | dict[str, dict[str, int]]],
        sysdb: SystemConfigDatabase,
        target_ports: set[tuple[str, int]],
    ) -> SkewAdjust:
        box_setting = cast(dict[str, dict[str, int]], config["box_setting"])
        slot = {
            (bname, nport): box_setting[bname]["slot"]
            for (bname, nport) in target_ports
        }
        wait = {
            (bname, nport): box_setting[bname]["wait"]
            for (bname, nport) in target_ports
        }
        time_to_start = cast(int, config["time_to_start"])
        return SkewAdjust(
            sysdb,
            target_ports=target_ports,
            slot=slot,
            wait=wait,
            time_to_start=time_to_start,
        )

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


@dataclass
class SkewSetting:
    reference_port: tuple[str, int]
    monitor_port: tuple[str, int]
    trigger_nport: int
    target_port: set[tuple[str, int]]
    scale: dict[tuple[str, int], float]

    @staticmethod
    def load(
        config: dict[str, str | int | set[str] | dict[str, dict[str, int]]],
    ) -> "SkewSetting":
        reference_port = str2port(cast(str, config["reference_port"]))
        monitor_port = str2port(cast(str, config["monitor_port"]))
        trigger_nport = cast(int, config["trigger_nport"])
        target_port = {str2port(v) for v in cast(set[str], config["target_port"])}
        scale = {
            str2port(p): v for p, v in cast(dict[str, float], config["scale"]).items()
        }
        return SkewSetting(
            reference_port=reference_port,
            monitor_port=monitor_port,
            trigger_nport=trigger_nport,
            target_port=target_port,
            scale=scale,
        )


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
        self._sysdb: Final[SystemConfigDatabase] = qubecalib.sysdb
        self._executor: Final[Executor] = qubecalib._executor
        self._monitor_port: tuple[str, int] = monitor_port
        self._trigger_nport: int = trigger_nport
        self._reference_port: tuple[str, int] = reference_port
        self._scale: dict[tuple[str, int], float] = {}
        self._measured: dict[tuple[str, int], npt.NDArray] = {}
        self._estimated: dict[tuple[str, int], npt.NDArray] = {}
        self._offset: dict[tuple[str, int], int] = {}
        self._target_port: set[tuple[str, int]] = set()
        self._skew_adjust: SkewAdjust = SkewAdjust(qubecalib.sysdb)
        self._setting: SkewSetting | None = None

    @property
    def setting(self) -> SkewSetting | None:
        return self._setting

    @setting.setter
    def setting(self, setting: SkewSetting) -> None:
        self._setting = setting
        self._reference_port = setting.reference_port
        self._monitor_port = setting.monitor_port
        self._trigger_nport = setting.trigger_nport
        self._target_port = setting.target_port
        self._scale = setting.scale

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

    @classmethod
    def acquire_freq_setting(cls, target_freq: float) -> dict[str, float | str]:
        # !!! CAUTION !!! Frequency is in HMz, not in GHz
        target_freq = target_freq * 1e3
        MINIMUM_LO_FREQ = 7500
        # MINIMUM_LO_FREQ = 2000
        MINIMUM_CNCO_FREQ = 2000
        LO_STEP_SIZE = 500
        CNCO_STEP_SIZE = 125
        if MINIMUM_LO_FREQ + MINIMUM_CNCO_FREQ < target_freq:
            # target_freq = lo_freq + (cnco_freq + delta)
            sideband = "U"
            lo_freq = (
                MINIMUM_LO_FREQ
                + (target_freq - (MINIMUM_LO_FREQ + MINIMUM_CNCO_FREQ))
                // LO_STEP_SIZE
                * LO_STEP_SIZE
            )
            cnco_freq = (target_freq - lo_freq) // CNCO_STEP_SIZE * CNCO_STEP_SIZE
            fnco_freq = 0
        else:
            # target_freq = lo_freq - (cnco_freq + delta)
            sideband = "L"
            lo_freq = (
                MINIMUM_LO_FREQ
                + (
                    (target_freq - (MINIMUM_LO_FREQ - MINIMUM_CNCO_FREQ))
                    // LO_STEP_SIZE
                    + 1
                )
                * LO_STEP_SIZE
            )
            cnco_freq = (lo_freq - target_freq) // CNCO_STEP_SIZE * CNCO_STEP_SIZE
            fnco_freq = 0

        # # target_freq = lo_freq + (cnco_freq + delta)
        # sideband = "U"
        # lo_freq = (
        #     MINIMUM_LO_FREQ
        #     + (target_freq - (MINIMUM_LO_FREQ + MINIMUM_CNCO_FREQ))
        #     // LO_STEP_SIZE
        #     * LO_STEP_SIZE
        # )
        # cnco_freq = (target_freq - lo_freq) // CNCO_STEP_SIZE * CNCO_STEP_SIZE
        # fnco_freq = 0
        return {
            "lo_freq": lo_freq * 1e-3,
            "cnco_freq": cnco_freq * 1e-3,
            "fnco_freq": fnco_freq * 1e-3,
            "sideband": sideband,
        }

    @classmethod
    def acquire_target(
        cls,
        sysdb: SystemConfigDatabase,
        port: tuple[str, int],
        channel: int = 0,
    ) -> str:
        channel_id: tuple[str, int, int] = (*port, channel)
        return sysdb.get_target_by_channel(*channel_id)

    def target_from_box(self, box_names: list[str]) -> set[tuple[str, int]]:
        return {
            (box_name, nport)
            for box_name, nport in self._target_port
            if box_name in box_names  #  and (box_name, nport) != self._reference_port
        }

    @classmethod
    def _setup_monitor_port(
        cls,
        *,
        target_port: PORT,
        monitor_port: PORT,
        system: Quel1System,
        sysdb: SystemConfigDatabase,
    ) -> None:
        name, nport = target_port
        box = system.box[name]
        if not box.is_output_port(nport):
            raise ValueError(f"{target_port} is not output port")
        target = cls.acquire_target(sysdb, target_port)
        dump_port = box.dump_port(nport)
        lo_freq = dump_port["lo_freq"]
        cnco_freq = dump_port["cnco_freq"]
        fnco_freq = dump_port["channels"][DEFAULT_CHANNEL_NUM]["fnco_freq"]
        sideband = dump_port["sideband"]
        frequency = (
            sysdb._target_settings[target]["frequency"]
            if target in sysdb._target_settings
            else DEFAULT_FREQUENCY
        )
        mod_freq = Converter._calc_modulation_frequency(
            target_freq=frequency,
            lo_freq=lo_freq * 1e-9,
            cnco_freq=cnco_freq * 1e-9,
            fnco_freq=fnco_freq * 1e-9,
            sideband=sideband,
        )
        logger.debug(
            f"Target {target}:{target_port}, lo_freq={lo_freq * 1e-9}, cnco_freq={cnco_freq * 1e-9}, fnco_freq={fnco_freq * 1e-9}, target_freq={frequency}, mod_freq={mod_freq}, sideband={sideband}"
        )

        name, nport = monitor_port
        box = system.box[name]
        if not box.is_input_port(nport):
            raise ValueError(f"{monitor_port} is not input port")

        freq_setting = cls.acquire_freq_setting(frequency)
        lo_freq = cast(float, freq_setting["lo_freq"]) * 1e9
        cnco_freq = cast(float, freq_setting["cnco_freq"]) * 1e9
        fnco_freq = cast(float, freq_setting["fnco_freq"]) * 1e9
        sideband = cast(str, freq_setting["sideband"])

        logger.debug(
            f"Monitor {monitor_port}, lo_freq={lo_freq * 1e-9}, cnco_freq={cnco_freq * 1e-9}, fnco_freq={fnco_freq * 1e-9}, target_freq={frequency}, mod_freq={mod_freq}"
        )

        if abs(fnco_freq) >= DEFAULT_FNCO_LIMIT:
            sign = fnco_freq / abs(fnco_freq)
            fnco_freq = sign * (DEFAULT_FNCO_LIMIT - 1)

        box.config_port(nport, lo_freq=lo_freq, cnco_freq=cnco_freq)
        box.config_runit(nport, runit=0, fnco_freq=fnco_freq)
        m = cls.acquire_target(sysdb, monitor_port)
        sysdb._target_settings[m] = dict(frequency=frequency)
        logger.debug(
            f"Monitor {monitor_port}, lo_freq={lo_freq * 1e-9}, cnco_freq={cnco_freq * 1e-9}, fnco_freq={fnco_freq * 1e-9}, target_freq={frequency}, mod_freq={mod_freq}"
        )

    @classmethod
    def _setup_trigger_port(
        cls,
        *,
        target_port: PORT,
        trigger_port: PORT,
        system: Quel1System,
        sysdb: SystemConfigDatabase,
    ) -> None:
        name, nport = target_port
        box = system.box[name]
        if not box.is_output_port(nport):
            raise ValueError(f"{target_port} is not output port")
        target = cls.acquire_target(sysdb, target_port)
        dump_port = box.dump_port(nport)
        lo_freq = dump_port["lo_freq"]
        cnco_freq = dump_port["cnco_freq"]
        fnco_freq = dump_port["channels"][DEFAULT_CHANNEL_NUM]["fnco_freq"]
        sideband = dump_port["sideband"]
        frequency = (
            sysdb._target_settings[target]["frequency"]
            if target in sysdb._target_settings
            else DEFAULT_FREQUENCY
        )

        name, nport = trigger_port
        box = system.box[name]
        if not box.is_output_port(nport):
            raise ValueError(f"{trigger_port} is not output port")

        freq_setting = cls.acquire_freq_setting(frequency)
        lo_freq = cast(float, freq_setting["lo_freq"]) * 1e9
        cnco_freq = cast(float, freq_setting["cnco_freq"]) * 1e9
        fnco_freq = cast(float, freq_setting["fnco_freq"]) * 1e9
        sideband = freq_setting["sideband"]

        box.config_port(
            nport,
            lo_freq=lo_freq,
            cnco_freq=cnco_freq,
            sideband=sideband,
            vatt=DEFAULT_VATT,
        )
        box.config_channel(
            nport,
            channel=0,
            fnco_freq=fnco_freq,
        )
        sysdb._target_settings[
            cls.acquire_target(
                sysdb,
                trigger_port,
            )
        ] = dict(frequency=frequency)
        mod_freq = Converter._calc_modulation_frequency(
            target_freq=frequency,
            lo_freq=lo_freq * 1e-9,
            cnco_freq=cnco_freq * 1e-9,
            fnco_freq=fnco_freq * 1e-9,
            sideband=sideband,
        )
        logger.debug(
            f"Trigger {trigger_port}, lo_freq={lo_freq * 1e-9}, cnco_freq={cnco_freq * 1e-9}, fnco_freq={fnco_freq * 1e-9}, target_freq={frequency}, mod_freq={mod_freq}, sideband={sideband}"
        )

    def measure(
        self,
        *,
        show_reference: bool = False,
        reset_skew_parameter: bool = False,
        extra_capture_range: int = EXTRA_CAPTURE_RANGE,  # multiple of 128 ns
    ) -> None:
        target_ports = self.target_from_box(list(self._system.boxes))
        for target_box, target_nport in tqdm(target_ports):
            target_port = (target_box, target_nport)
            self._measure(
                target_port,
                extra_capture_range=extra_capture_range,
                show_reference=show_reference,
                reset_skew_parameter=reset_skew_parameter,
            )

    def _measure(
        self,
        target_port: tuple[str, int],
        *,
        show_reference: bool = False,
        reset_skew_parameter: bool = False,
        extra_capture_range: int = EXTRA_CAPTURE_RANGE,  # multiple of 128 ns
    ) -> npt.NDArray:
        system = self._system
        sysdb = self._sysdb
        reference_port = self._reference_port
        monitor_port = self._monitor_port
        monitor_box_name, _ = monitor_port
        trigger_port: tuple[str, int] = (monitor_box_name, self._trigger_nport)
        target_ports = set([target_port])
        trigger_box_name, trigger_port_number = trigger_port
        trigger_channel: tuple[str, int, int] = (
            trigger_box_name,
            trigger_port_number,
            0,
        )
        sysdb.trigger = {monitor_port: trigger_channel}
        self._setup_monitor_port(
            target_port=target_port,
            monitor_port=monitor_port,
            system=system,
            sysdb=sysdb,
        )
        self._setup_trigger_port(
            target_port=target_port,
            trigger_port=trigger_port,
            system=system,
            sysdb=sysdb,
        )
        for ctrl_box, _ in [reference_port] + [p for p in target_ports]:
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
        capture = Capture(
            duration=2 * len(target_ports) * 128 + 128 + extra_capture_range
        )

        reference_scale = (
            self._scale[reference_port] if reference_port in self._scale else 1
        )
        with Sequence() as seq:
            with Flushleft():
                scale = reference_scale if show_reference else 0
                pulse.scaled(scale).target(self.acquire_target(sysdb, trigger_port))
                scale = self._scale[target_port] if target_port in self._scale else 1
                pulse.scaled(scale).target(self.acquire_target(sysdb, target_port))
                capture.target(self.acquire_target(sysdb, monitor_port))
                # with Series():
                #     scale = reference_scale if show_reference else 0
                #     pulse.scaled(scale).target(self.acquire_target(sysdb, trigger_port))
                #     if offset != 0:
                #         blank.target()
                #     scale = (
                #         self._scale[target_port] if target_port in self._scale else 1
                #     )
                #     pulse.scaled(scale).target(self.acquire_target(sysdb, target_port))
                # capture.target(self.acquire_target(sysdb, monitor_port))

        self._executor.add_sequence(seq)

        for _, data, _ in self._executor.step_execute(
            repeats=100,
            interval=REPETITION_PERIOD,
            integral_mode="single",
            dsp_demodulation=False,
            software_demodulation=True,
        ):
            for _, iqs in data.items():
                iqs = iqs[0].sum(axis=1).squeeze()
                self._measured[target_port] = iqs
                # self._offset[target_port] = offset

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
        extra_capture_range: int = EXTRA_CAPTURE_RANGE,  # multiple of 128 ns
    ) -> dict[tuple[str, int], dict[str, int]]:
        target_ports = self.target_from_box(list(self._system.boxes))
        for target_box, target_nport in tqdm(target_ports):
            target_port = (target_box, target_nport)
            iqs = self._measure(
                target_port,
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
        with open(Path(os.getcwd()) / Path(filename), "r") as file:
            config = yaml.safe_load(file)
        self.setting = SkewSetting.load(config)
        sysdb = self._sysdb
        target = self._target_port
        self._skew_adjust = SkewAdjust.load(config, sysdb=sysdb, target_ports=target)
        self._skew_adjust.push()

    def load_setting(self, filename: str) -> None:
        with open(Path(os.getcwd()) / Path(filename), "r") as file:
            config = yaml.safe_load(file)
        self.setting = SkewSetting.load(config)
        self._skew_adjust.target_ports = copy(self.setting.target_port)
        self._skew_adjust.pull()

    def save(self, filename: str) -> None:
        sysdb = self._sysdb
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
