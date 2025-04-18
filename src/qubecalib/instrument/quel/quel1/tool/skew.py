from __future__ import annotations

import os
from copy import copy
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
from types import TracebackType
from typing import Final, Type, cast

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import yaml
from tqdm.auto import tqdm

from .....instrument.quel.quel1.driver import Quel1System
from .....neopulse import Capture, Flushleft, Rectangle, Sequence
from .....qubecalib import Executor, PortSetting, QubeCalib, SystemConfigDatabase

DEFAULT_FREQUENCY = 9.75
DEFAULT_LO_FREQ = 11000e6
DEFAULT_CNCO_FREQ = 1250e6
DEFAULT_FNCO_FREQ = 0.0
DEFAULT_SIDEBAND = "L"
DEFAULT_VATT = 0x900
DEFAULT_CHANNEL_NUM = 0
DEFAULT_CAPTURE_RANGE = 8 * 128  # ns

REPETITION_PERIOD = 1280 * 128  # words
EXTRA_CAPTURE_RANGE = 1024  # words

DEFAULT_FNCO_LIMIT = 500e6


PORT = tuple[str, int]

logger = getLogger(__name__)


def str2port(v: str) -> PORT:
    box_name, nport = v.split("-")[:2]
    return box_name, int(nport)


def port2str(v: PORT) -> str:
    box_name, nport = v
    return f"{box_name}-{nport}"


@dataclass
class BoxSkewData:
    target_port: PORT
    slot: int
    wati: int


@dataclass
class MeasuredPulseWaveform:
    waveform: npt.NDArray[np.complex64]
    offset: int


@dataclass
class EstimatedPulseParams:
    waveform: npt.NDArray[np.float64]
    idx: int
    scale: int
    mean: int
    # slot: int = field(init=False)
    # wait: int = field(init=False)

    # def __post_init__(self) -> None:
    #     self.slot = self.idx // 64
    #     self.wait = self.idx % 64


@dataclass
class SkewData:
    sysdb: SystemConfigDatabase
    boxes: dict[str, BoxSkewData] = field(default_factory=dict)
    time_to_start: int = 0


@dataclass
class SkewAdjust:
    sysdb: SystemConfigDatabase
    target_ports: set[PORT] = field(default_factory=set)
    slot: dict[PORT, int] = field(default_factory=dict)
    wait: dict[PORT, int] = field(default_factory=dict)
    time_to_start: int = 0

    @staticmethod
    def load(
        config: dict[str, str | int | set[str] | dict[str, dict[str, int]]],
        sysdb: SystemConfigDatabase,
        target_ports: set[PORT],
    ) -> SkewAdjust:
        return SkewAdjust.from_yaml_dict(config, sysdb, target_ports)

    @staticmethod
    def from_yaml_dict(
        yaml_dict: dict[str, str | int | set[str] | dict[str, dict[str, int]]],
        sysdb: SystemConfigDatabase,
        target_ports: set[PORT],
    ) -> SkewAdjust:
        box_setting = cast(dict[str, dict[str, int]], yaml_dict["box_setting"])
        slot = {
            (bname, nport): box_setting[bname]["slot"]
            for (bname, nport) in target_ports
        }
        wait = {
            (bname, nport): box_setting[bname]["wait"]
            for (bname, nport) in target_ports
        }
        time_to_start = cast(int, yaml_dict["time_to_start"])
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
    reference_port: PORT
    monitor_port: PORT
    trigger_nport: int
    target_port: set[PORT]
    scale: dict[PORT, float]

    @staticmethod
    def load(
        config: dict[str, str | int | set[str] | dict[str, dict[str, int]]],
    ) -> "SkewSetting":
        return SkewSetting.from_yaml_dict(config)

    @staticmethod
    def from_yaml(filename: str) -> "SkewSetting":
        with open(Path(os.getcwd()) / Path(filename), "r") as file:
            config = yaml.safe_load(file)
        return SkewSetting.from_yaml_dict(config)

    @staticmethod
    def from_yaml_dict(
        yaml_dict: dict[str, str | int | set[str] | dict[str, dict[str, int]]],
    ) -> "SkewSetting":
        reference_port = str2port(cast(str, yaml_dict["reference_port"]))
        monitor_port = str2port(cast(str, yaml_dict["monitor_port"]))
        trigger_nport = cast(int, yaml_dict["trigger_nport"])
        target_port = {str2port(v) for v in cast(set[str], yaml_dict["target_port"])}
        scale = {
            str2port(p): v
            for p, v in cast(dict[str, float], yaml_dict["scale"]).items()
        }
        return SkewSetting(
            reference_port=reference_port,
            monitor_port=monitor_port,
            trigger_nport=trigger_nport,
            target_port=target_port,
            scale=scale,
        )

    @property
    def monitor_box_name(self) -> str:
        box_name, _ = self.monitor_port
        return box_name

    @property
    def target_box_names(self) -> set[str]:
        return set({box_name for box_name, _ in self.target_port})


class SkewAdjustResetter:
    def __init__(
        self,
        skew_adjust: SkewAdjust,
        *,
        reference_port: PORT,
        target_ports: set[PORT],
    ) -> None:
        self._skew_adjust = skew_adjust
        self._reference_port = reference_port
        self._target_ports = target_ports
        self._backup_reference: tuple[int, int] | None = None
        self._backup_targets: dict[PORT, tuple[int, int]] = {}

    def __enter__(self) -> None:
        self._backup_reference = (
            self._skew_adjust.slot[self._reference_port],
            self._skew_adjust.wait[self._reference_port],
        )
        self._skew_adjust.slot[self._reference_port] = 0
        self._skew_adjust.wait[self._reference_port] = 0
        for target_port in self._target_ports:
            self._backup_targets[target_port] = (
                self._skew_adjust.slot[target_port],
                self._skew_adjust.wait[target_port],
            )
            self._skew_adjust.slot[target_port] = 0
            self._skew_adjust.wait[target_port] = 0
        self._skew_adjust.push()

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        adj = self._skew_adjust
        (
            adj.slot[self._reference_port],
            adj.wait[self._reference_port],
        ) = cast(tuple[int, int], self._backup_reference)
        for target_port in self._target_ports:
            (
                adj.slot[target_port],
                adj.wait[target_port],
            ) = self._backup_targets[target_port]
        self._skew_adjust.push()


class Skew:
    def __init__(
        self,
        system: Quel1System,
        *,
        sysdb: SystemConfigDatabase | None = None,
        # executor: Executor | None = None,
        qubecalib: QubeCalib
        | None = None,  # TODO: qubex の experiment.py:138 を修正してもらう
        monitor_port: PORT = ("", 0),
        trigger_nport: int = 0,
        reference_port: PORT = ("", 0),
    ) -> None:  # TODO ここは多分変わります
        if qubecalib is not None:
            sysdb = qubecalib.sysdb
        if sysdb is None:
            raise ValueError("sysdb and executor must be provided")
        self._system: Final[Quel1System] = system
        self._sysdb: Final[SystemConfigDatabase] = sysdb
        self._executor: Final[Executor] = Executor(self.sysdb)
        self._monitor_port: PORT = monitor_port
        self._trigger_nport: int = trigger_nport
        self._reference_port: PORT = reference_port
        self._scale: dict[PORT, float] = {}
        self._measured_waveform: dict[PORT, npt.NDArray] = {}
        # self._estimated_waveform: dict[PORT, npt.NDArray] = {}
        self._offset: dict[PORT, int] = {}
        self._target_port: set[PORT] = set()
        self._skew_adjust: SkewAdjust = SkewAdjust(self.sysdb)
        self._setting: SkewSetting | None = None
        # self._estimated_idx: dict[PORT, int] = {}
        # self._estimated_slot: dict[PORT, int] = {}
        # self._estimated_wait: dict[PORT, int] = {}
        self._estimated: dict[PORT, EstimatedPulseParams] = {}
        for box_name in self._system.boxes:
            if not self.is_channel_defined(box_name, sysdb=self.sysdb):
                self._define_channel_names(
                    box_name, system=self._system, sysdb=self.sysdb
                )

    @classmethod
    def create(
        cls,
        *,
        setting: SkewSetting,
        system: Quel1System,
        sysdb: SystemConfigDatabase,
    ) -> Skew:
        skew = Skew(system=system, sysdb=sysdb)
        skew.setting = setting
        return skew

    @property
    def sysdb(self) -> SystemConfigDatabase:
        return self._sysdb

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
    def monitor_port(self) -> PORT:
        return self._monitor_port

    @monitor_port.setter
    def monitor_port(self, monitor_port: PORT) -> None:
        self._monitor_port = monitor_port

    @property
    def trigger_nport(self) -> int:
        return self._trigger_nport

    @trigger_nport.setter
    def trigger_nport(self, trigger_nport: int) -> None:
        self._trigger_nport = trigger_nport

    @property
    def reference_port(self) -> PORT:
        return self._reference_port

    @reference_port.setter
    def reference_port(self, reference_port: PORT) -> None:
        self._reference_port = reference_port

    def set_scale(self, port: PORT, scale: float) -> None:
        self._scale[port] = scale

    @classmethod
    def acquire_freq_setting(cls, target_freq: float) -> dict[str, float | str]:
        # !!! CAUTION !!! Frequency is in HMz, not in GHz
        target_freq = target_freq * 1e3
        MINIMUM_LO_FREQ = 7500
        MINIMUM_CNCO_FREQ = 2000
        LO_STEP_SIZE = 500
        CNCO_STEP_SIZE = 125
        if MINIMUM_LO_FREQ + MINIMUM_CNCO_FREQ < target_freq:
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

        return {
            "lo_freq": lo_freq * 1e-3,
            "cnco_freq": cnco_freq * 1e-3,
            "fnco_freq": fnco_freq * 1e-3,
            "sideband": sideband,
        }

    @classmethod
    def get_target_by_channel(
        cls,
        sysdb: SystemConfigDatabase,
        port: PORT,
        channel: int | None = None,
    ) -> str:
        channel = DEFAULT_CHANNEL_NUM if channel is None else channel
        channel_id: tuple[str, int, int] = (*port, channel)
        targets = sysdb.get_targets_by_channel(*channel_id)
        target = next(iter({t for t in targets if len(t.split("-")) == 1}))
        return target

    @classmethod
    def acquire_target(
        cls,
        sysdb: SystemConfigDatabase,
        port: PORT,
        channel: int | None = None,
    ) -> str:
        # default value
        # channel = DEFAULT_CHANNEL_NUM if channel is None else channel
        return cls.get_target_by_channel(sysdb, port, channel)

    def _acquire_target(
        self,
        port: PORT,
        channel: int | None = None,
        sysdb: SystemConfigDatabase | None = None,
    ) -> str:
        sysdb = self._sysdb if sysdb is None else sysdb
        return self.acquire_target(sysdb, port, channel)

    def target_from_box(self, box_names: list[str]) -> set[PORT]:
        return {
            (box_name, nport)
            for box_name, nport in self._target_port
            if box_name in box_names  #  and (box_name, nport) != self._reference_port
        }

    def sync_lo_nco(
        self,
        *,
        src_port: PORT,
        dest_port: PORT,
    ) -> dict[str, str | float | dict[int, dict[str, float]]]:
        """src_port の周波数を dest_port に合わせる"""
        return self._sync_lo_nco(
            src_port=src_port,
            dest_port=dest_port,
            system=self._system,
        )

    @classmethod
    def _sync_lo_nco(
        cls,
        *,
        src_port: PORT,
        dest_port: PORT,
        system: Quel1System,
    ) -> dict[str, str | float | dict[int, dict[str, float]]]:
        # src_port: (box_name, port_number)
        # dest_port: (box_name, port_number)
        boxname, nport = src_port
        box = system.box[boxname]
        dump_port = box.dump_port(nport)
        kws = ["lo_freq", "cnco_freq", "runits", "channels", "sideband"]
        freqs = {k: v for k, v in dump_port.items() if k in kws}
        ch_freqs = freqs["channels"] if "channels" in freqs else freqs["runits"]
        lo_freq, cnco_freq = freqs["lo_freq"], freqs["cnco_freq"]
        kw = "fnco_freq"
        usb_freqs = {i: lo_freq + (cnco_freq + v[kw]) for i, v in ch_freqs.items()}
        lsb_freqs = {i: lo_freq - (cnco_freq + v[kw]) for i, v in ch_freqs.items()}
        if "sideband" in freqs:
            sideband = freqs["sideband"]
        else:
            sideband = "L" if lo_freq + cnco_freq < 9500e6 else "U"
            freqs["sideband"] = sideband
        if sideband == "U":
            for i, v in usb_freqs.items():
                freqs["channels"][i]["target_freq"] = v
        else:
            for i, v in lsb_freqs.items():
                freqs["channels"][i]["target_freq"] = v

        logger.debug(
            f"{src_port} ->: lo_freq={lo_freq}, cnco_freq={cnco_freq}, channels={freqs['channels']}, sideband={sideband}"
        )

        boxname, nport = dest_port
        box = system.box[boxname]
        lo_freq, cnco_freq = (
            freqs["lo_freq"],
            freqs["cnco_freq"] + ch_freqs[0]["fnco_freq"],
        )
        for v in ch_freqs.values():
            v["fnco_freq"] = 0
        dump_port = box.dump_port(nport)
        if "channels" in dump_port:
            ch = dump_port["channels"]
            n = len(ch_freqs) if len(ch_freqs) < len(ch) else len(ch)
            box.config_port(
                port=nport,
                lo_freq=lo_freq,
                cnco_freq=cnco_freq,
            )
            for i in range(n):
                box.config_channel(
                    port=nport,
                    channel=i,
                    fnco_freq=ch_freqs[i]["fnco_freq"],
                )
        elif "runits" in dump_port:
            ch = dump_port["runits"]
            n = len(ch_freqs) if len(ch_freqs) < len(ch) else len(ch)
            box.config_port(
                port=nport,
                lo_freq=lo_freq,
                cnco_freq=cnco_freq,
            )
            for i in range(n):
                box.config_runit(
                    port=nport,
                    runit=i,
                    fnco_freq=ch_freqs[i]["fnco_freq"],
                )

        logger.debug(
            f"-> {dest_port}: lo_freq={lo_freq}, cnco_freq={cnco_freq}, channels={freqs['channels']}, sideband={sideband}"
        )

        return freqs

    def setup_monitor_port(
        self,
        *,
        target_port: PORT,
        monitor_port: PORT,
    ) -> None:
        """target に合わせて周波数を設定する"""
        self._setup_monitor_port(
            target_port=target_port,
            monitor_port=monitor_port,
            system=self._system,
            sysdb=self._sysdb,
        )

    @classmethod
    def is_channel_defined(
        cls,
        box_name: str,
        *,
        sysdb: SystemConfigDatabase,
    ) -> bool:
        channels = {
            k: v for k, v in sysdb._port_settings.items() if v.box_name == box_name
        }
        return len(channels) != 0

    @classmethod
    def _define_channel_names(
        cls,
        box_name: str,
        *,
        system: Quel1System,
        sysdb: SystemConfigDatabase,
    ) -> None:
        box = system.box[box_name]
        ports = box.dump_box()["ports"]
        for nport, v in ports.items():
            nport = cast(int, nport)
            port_name = f"{box_name}.PORT{nport}"
            channels = v["channels" if "channels" in v else "runits"]
            io = "IN" if v["direction"] == "in" else "OUT"
            port_setting = PortSetting(
                port_name=port_name,
                box_name=box_name,
                port=nport,
                lo_freq=None,
                cnco_freq=None,
                sideband="U",
                vatt=2048,
                fnco_freq=None,
                ndelay_or_nwait=tuple(
                    len(channels) * [7 if io == "IN" else 0]
                ),  # TODO: hard coding
            )
            sysdb._port_settings[port_name] = port_setting
            for nchannel, v in channels.items():
                channel_name = f"{box_name}.PORT{nport}.{io}{nchannel}"
                sysdb._relation_channel_port.append(
                    (channel_name, dict(port_name=port_name, channel_number=nchannel))
                )

    @classmethod
    def _setup_monitor_port(
        cls,
        *,
        target_port: PORT,
        monitor_port: PORT,
        system: Quel1System,
        sysdb: SystemConfigDatabase,
    ) -> None:
        """target に合わせて周波数を設定する"""
        name, nport = target_port
        box = system.box[name]
        if not box.is_output_port(nport):
            raise ValueError(f"{target_port} is not output port")
        freqs = cls._sync_lo_nco(
            src_port=target_port,
            dest_port=monitor_port,
            system=system,
        )
        ch_freqs = cast(dict[int, dict[str, float]], freqs["channels"])
        DEFAULT_CHANNEL = 0
        lo_freq, cnco_freq, fnco_freq, target_freq, sideband = (
            cast(float, freqs["lo_freq"]) * 1e-9,
            cast(float, freqs["cnco_freq"]) * 1e-9,
            ch_freqs[DEFAULT_CHANNEL]["fnco_freq"] * 1e-9,
            ch_freqs[DEFAULT_CHANNEL]["target_freq"] * 1e-9,
            cast(str, freqs["sideband"]),
        )
        target = cls.acquire_target(sysdb, target_port)
        sysdb._target_settings[target] = dict(frequency=target_freq)
        monitor = cls.acquire_target(sysdb, monitor_port)
        sysdb._target_settings[monitor] = dict(frequency=target_freq)
        logger.debug(
            f"Target {target}:{target_port}, lo_freq={lo_freq}, cnco_freq={cnco_freq}, fnco_freq={fnco_freq}, target_freq={target_freq}, sideband={sideband}"
        )

    def setup_trigger_port(
        self,
        *,
        target_port: PORT,
        trigger_port: PORT,
    ) -> None:
        """target に合わせて周波数を設定する"""
        self._setup_trigger_port(
            target_port=target_port,
            trigger_port=trigger_port,
            system=self._system,
            sysdb=self._sysdb,
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
        """target に合わせて周波数を設定する"""
        name, nport = target_port
        box = system.box[name]
        if not box.is_output_port(nport):
            raise ValueError(f"{target_port} is not output port")
        freqs = cls._sync_lo_nco(
            src_port=target_port,
            dest_port=trigger_port,
            system=system,
        )
        ch_freqs = cast(dict[int, dict[str, float]], freqs["channels"])
        DEFAULT_CHANNEL = 0
        lo_freq, cnco_freq, fnco_freq, target_freq, sideband = (
            cast(float, freqs["lo_freq"]),
            cast(float, freqs["cnco_freq"]),
            ch_freqs[DEFAULT_CHANNEL]["fnco_freq"],
            ch_freqs[DEFAULT_CHANNEL]["target_freq"] * 1e-9,
            cast(str, freqs["sideband"]),
        )
        target = cls.acquire_target(sysdb, target_port)
        sysdb._target_settings[target] = dict(frequency=target_freq)
        trigger = cls.acquire_target(sysdb, trigger_port)
        sysdb._target_settings[trigger] = dict(frequency=target_freq)
        logger.debug(
            f"Trigger {trigger_port}, lo_freq={lo_freq * 1e-9}, cnco_freq={cnco_freq * 1e-9}, fnco_freq={fnco_freq * 1e-9}, target_freq={target_freq}, sideband={sideband}"
        )

    def reset_skew_parameter(self) -> SkewAdjustResetter:
        # with reset_skew_parameter():
        #     skew.measure()
        return SkewAdjustResetter(
            self._skew_adjust,
            reference_port=self._reference_port,
            target_ports=self._target_port,
        )

    def measure(
        self,
        *,
        target_ports: set[PORT] | None = None,
        show_reference: bool | None = None,
        extra_capture_range: int | None = None,  # multiple of 128 ns
        reset_skew_parameter: bool = False,
    ) -> None:
        target_ports = (
            self.target_from_box(list(self._system.boxes))
            if target_ports is None
            else target_ports
        )
        self.define_targets(target_ports=target_ports)
        if reset_skew_parameter:
            with self.reset_skew_parameter():
                self._measure_targets(
                    target_ports,
                    show_reference=show_reference,
                    extra_capture_range=extra_capture_range,
                )
        else:
            self._measure_targets(
                target_ports,
                show_reference=show_reference,
                extra_capture_range=extra_capture_range,
            )

    def _measure_targets(
        self,
        target_ports: set[PORT],
        *,
        show_reference: bool | None = None,
        extra_capture_range: int | None = None,
    ) -> None:
        target_ports = self.target_from_box(list(self._system.boxes))
        for target_port in tqdm(target_ports):
            self._measure(
                target_port,
                show_reference=show_reference,
                extra_capture_range=extra_capture_range,
            )

    def _create_check_sequence(
        self,
        target_port: PORT,
        *,
        monitor_port: PORT | None = None,
        trigger_nport: int | None = None,
        capture_range: int | None = None,  # multiple of 128 ns
    ) -> Sequence:
        # default values
        monitor_port = self._monitor_port if monitor_port is None else monitor_port
        monitor_box_name, _ = monitor_port
        trigger_nport = self._trigger_nport if trigger_nport is None else trigger_nport
        trigger_port: PORT = (monitor_box_name, trigger_nport)
        capture_range = DEFAULT_CAPTURE_RANGE
        # create pulse sequence
        pulse = Rectangle(duration=128, amplitude=1.0)
        capture = Capture(duration=capture_range)
        with Sequence() as seq:
            with Flushleft():
                pulse.scaled(0).target(
                    self._acquire_target(trigger_port),
                )
                pulse.scaled(
                    self._scale[target_port] if target_port in self._scale else 1,
                ).target(
                    self._acquire_target(target_port),
                )
                capture.target(
                    self._acquire_target(monitor_port),
                )
        return seq

    def _execute(self, sequence: Sequence) -> npt.NDArray:
        """Executes the measurement, assuming that the sequence contains only a single capture."""
        self._executor.add_sequence(sequence)
        for _, data, _ in self._executor.step_execute(
            repeats=100,
            interval=REPETITION_PERIOD,
            integral_mode="single",
            dsp_demodulation=False,
            software_demodulation=True,
        ):
            for _, iqs in data.items():
                iqs = iqs[0].sum(axis=1).squeeze()

        return iqs

    def _store(self, target_port: PORT, iqs: npt.NDArray) -> None:
        self._measured_waveform[target_port] = iqs

    def _config_ports(
        self,
        target_port: PORT,
        *,
        reference_port: PORT | None = None,
        monitor_port: PORT | None = None,
        trigger_nport: int | None = None,
        show_reference: bool | None = None,
        extra_capture_range: int | None = None,  # multiple of 128 ns
    ) -> None:
        # create alias
        system = self._system
        sysdb = self._sysdb
        # default values
        reference_port = (
            self._reference_port if reference_port is None else reference_port
        )
        monitor_port = self._monitor_port if monitor_port is None else monitor_port
        trigger_nport = self._trigger_nport if trigger_nport is None else trigger_nport
        show_reference = show_reference if show_reference is not None else False
        extra_capture_range = (
            extra_capture_range
            if extra_capture_range is not None
            else EXTRA_CAPTURE_RANGE
        )

        monitor_box_name, _ = monitor_port
        trigger_port: PORT = (monitor_box_name, trigger_nport)
        target_ports = set([target_port])
        trigger_box_name, trigger_port_number = trigger_port
        trigger_channel: tuple[str, int, int] = (
            trigger_box_name,
            trigger_port_number,
            0,
        )
        sysdb.trigger = {monitor_port: trigger_channel}
        # monitor の周波数を target に合わせる
        self._setup_monitor_port(
            target_port=target_port,
            monitor_port=monitor_port,
            system=system,
            sysdb=sysdb,
        )
        # trigger の周波数を target に合わせる
        self._setup_trigger_port(
            target_port=target_port,
            trigger_port=trigger_port,
            system=system,
            sysdb=sysdb,
        )
        # target の ctrl の rfswitch を open にする？
        for ctrl_box, _ in [reference_port] + [p for p in target_ports]:
            m = system.box[ctrl_box].get_monitor_input_ports()
            if m:
                for port in m:
                    if isinstance(port, tuple):
                        raise ValueError("fogi port is not supported yet")
                    system.box[ctrl_box].config_rfswitch(port, rfswitch="open")

    @classmethod
    def _define_targets(
        cls,
        *,
        target_ports: set[PORT],
        reference_port: PORT,
        monitor_port: PORT,
        trigger_nport: int,
        sysdb: SystemConfigDatabase,
    ) -> None:
        box_name, _ = monitor_port
        trigger_port = (box_name, trigger_nport)
        defined_channel = set([c for c, _ in sysdb._relation_channel_target])
        channel_names_by_channel: dict[tuple[str, int | tuple[int, int], int], str] = {}
        for k, v in sysdb._relation_channel_port:
            p = sysdb._port_settings[cast(str, v["port_name"])]
            channel = (p.box_name, p.port, cast(int, v["channel_number"]))
            channel_names_by_channel[channel] = k
        for target_port in set(
            list(target_ports) + [reference_port, monitor_port, trigger_port]
        ):
            channel = target_port + (0,)
            target_name = channel_name = channel_names_by_channel[channel]
            if channel_name not in defined_channel:
                sysdb._relation_channel_target.append((target_name, channel_name))
                sysdb._target_settings[target_name] = dict(frequency=0)
            # sysdb._relation_channel_target.append((target_name, channel_name))
            # sysdb._target_settings[target_name] = dict(frequency=0)

    def define_targets(
        self,
        *,
        target_ports: set[PORT],
        reference_port: PORT | None = None,
        monitor_port: PORT | None = None,
        trigger_nport: int | None = None,
    ) -> None:
        # default values
        reference_port = (
            self._reference_port if reference_port is None else reference_port
        )
        monitor_port = self._monitor_port if monitor_port is None else monitor_port
        trigger_nport = self._trigger_nport if trigger_nport is None else trigger_nport
        # call the main function
        self._define_targets(
            target_ports=target_ports,
            reference_port=reference_port,
            monitor_port=monitor_port,
            trigger_nport=trigger_nport,
            sysdb=self._sysdb,
        )

    def _measure(
        self,
        target_port: PORT,
        *,
        reference_port: PORT | None = None,
        monitor_port: PORT | None = None,
        trigger_nport: int | None = None,
        show_reference: bool | None = None,
        extra_capture_range: int | None = None,  # multiple of 128 ns
    ) -> None:
        self._config_ports(
            target_port,
            reference_port=reference_port,
            monitor_port=monitor_port,
            trigger_nport=trigger_nport,
            show_reference=show_reference,
            extra_capture_range=extra_capture_range,
        )
        seq = self._create_check_sequence(target_port)
        iqs = self._execute(seq)
        self._store(target_port, iqs)

    def estimate(self) -> None:
        for focused_port, iqs in self._measured_waveform.items():
            self._estimated[focused_port] = self._estimate(iqs)

    @classmethod
    def _estimate(cls, iqs: npt.NDArray) -> EstimatedPulseParams:
        # A filter used to identify the position of a pulse by maximizing the energy
        # within a fixed-length window. The filter detects the region where the pulse
        # energy is highest.
        pulse_window_filter = np.array(64 * [1])
        conv = np.convolve(pulse_window_filter, np.abs(iqs), "valid")
        # Allocate a container for estimated pulses.
        idx = int(conv.argmax())
        estimated = np.zeros(iqs.size).astype(np.float64)
        estimated[idx : idx + 64] = np.ones(64)
        scale = np.sqrt(np.abs(iqs).var()) / np.sqrt(estimated.var())
        estimated *= scale
        mean = np.abs(iqs).mean() - estimated.mean()
        estimated += mean
        return EstimatedPulseParams(
            waveform=estimated,
            idx=idx,
            scale=scale,
            mean=mean,
        )

    def plot(self) -> go.FigureWidget:
        return self._plot(
            measured_waveform=self._measured_waveform,
            estimated=self._estimated,
            reference_port=self._reference_port,
        )

    @classmethod
    def _plot(
        cls,
        *,
        measured_waveform: dict[PORT, npt.NDArray],
        estimated: dict[PORT, EstimatedPulseParams],
        reference_port: PORT,
    ) -> go.FigureWidget:
        fig = go.FigureWidget(
            layout=go.Layout(
                height=len(measured_waveform) * 100 + 100,
                width=None,
                autosize=True,
                showlegend=False,
            ),
        ).set_subplots(
            rows=len(measured_waveform),
            cols=1,
            shared_xaxes=True,
        )
        for i, focused_port in enumerate(measured_waveform):
            iqs = measured_waveform[focused_port]
            fig.add_trace(
                go.Scatter(
                    x=2 * np.arange(len(iqs)),
                    y=np.abs(iqs),
                    mode="lines",
                    # name=f"{focused_port}: measured",
                ),
                row=1 + i,
                col=1,
            )
            if focused_port not in estimated:  # if estimate is empty
                continue
            iqs = estimated[focused_port].waveform
            fig.add_trace(
                go.Scatter(
                    x=2 * np.arange(len(iqs)),
                    y=iqs,
                    mode="lines",
                    # name=f"{focused_port}: estimated",
                ),
                row=1 + i,
                col=1,
            )
            e = estimated[focused_port]
            fig.add_annotation(
                text=("Reference " if focused_port == reference_port else "")
                + f"{focused_port}"
                + f", idx={e.idx}",
                xref="paper",
                yref="y domain" if i == 0 else f"y{i + 1} domain",
                xanchor="right",
                yanchor="top",
                x=(2 * np.arange(len(iqs)))[-1],
                y=1,
                showarrow=False,
                row=1 + i,
                col=1,
            )
        return fig

    def load(self, filename: str) -> None:
        with open(Path(os.getcwd()) / Path(filename), "r") as file:
            config = yaml.safe_load(file)
        self.setting = SkewSetting.from_yaml_dict(config)
        sysdb = self._sysdb
        target = self._target_port
        self._skew_adjust = SkewAdjust.from_yaml_dict(
            config,
            sysdb=sysdb,
            target_ports=target,
        )
        self._skew_adjust.push()

    def load_setting(self, filename: str) -> None:
        with open(Path(os.getcwd()) / Path(filename), "r") as file:
            config = yaml.safe_load(file)
        self.setting = SkewSetting.from_yaml_dict(config)
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
