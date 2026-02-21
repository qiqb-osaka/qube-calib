from __future__ import annotations

import os
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import yaml
from tqdm.auto import tqdm

import datetime
import shutil

from .....instrument.quel.quel1.driver import Quel1System
from .....instrument.quel.quel1.driver.single import Quel1PortType
from .....neopulse import Capture, Flushleft, Rectangle, Sequence
from .....qubecalib import Executor, PortSetting, SystemConfigDatabase

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

DEFAULT_CNCO_LIMIT = 3e9
DEFAULT_FNCO_LIMIT = 500e6


PORT = tuple[str, int]

logger = getLogger(__name__)


def str2port(v: str) -> PORT:
    box_name, nport = v.split("-")[:2]
    return box_name, int(nport)

    # def port2str(v: PORT) -> str:
    #     box_name, nport = v
    #     return f"{box_name}-{nport}"

    # @dataclass
    # class BoxSkewData:
    #     target_port: PORT
    #     slot: int
    #     wait: int

    # @dataclass
    # class MeasuredPulseWaveform:
    #     waveform: npt.NDArray[np.complex64]
    #     offset: int


def _parse_port_type(k: Any) -> Quel1PortType:
    """
    Normalize rf_switches keys loaded from YAML into Quel1PortType
    (int | tuple[int, int]).

    Notes
    -----
    - Quel1PortType is a *type alias*, not a runtime class.
      Therefore:
        * isinstance(k, Quel1PortType) is invalid
        * Quel1PortType[...] (Enum-style access) is invalid
    - Runtime checks must be performed against concrete types
      (int / tuple), and the result is cast to Quel1PortType.
    """

    # Case 1: already an integer port index
    if isinstance(k, int):
        return cast(Quel1PortType, k)

    # Case 2: already a tuple[int, int]
    if isinstance(k, tuple) and len(k) == 2 and all(isinstance(x, int) for x in k):
        return cast(Quel1PortType, k)

    # Case 3: string representations coming from YAML or user edits
    if isinstance(k, str):
        s = k.strip()

        # "3" -> 3
        if s.isdecimal():
            return cast(Quel1PortType, int(s))

        # "(3, 4)" or "3,4" -> (3, 4)
        s2 = s
        if s2.startswith("(") and s2.endswith(")"):
            s2 = s2[1:-1].strip()

        parts = [p.strip() for p in s2.split(",")]
        if len(parts) == 2 and all(p.isdecimal() for p in parts):
            return cast(Quel1PortType, (int(parts[0]), int(parts[1])))

    # Any other form is considered invalid
    raise TypeError(f"Invalid rf_switch port key: {k!r}")


@dataclass
class EstimatedPulseParams:
    waveform: npt.NDArray[np.float64]
    idx: int
    scale: int
    mean: int


# @dataclass
# class SkewData:
#     sysdb: SystemConfigDatabase
#     boxes: dict[str, BoxSkewData] = field(default_factory=dict)
#     time_to_start: int = 0


# @dataclass
# class SkewAdjust:
#     sysdb: SystemConfigDatabase
#     target_ports: set[PORT] = field(default_factory=set)
#     slot: dict[PORT, int] = field(default_factory=dict)
#     wait: dict[PORT, int] = field(default_factory=dict)
#     time_to_start: int = 0

#     @staticmethod
#     def load(
#         config: dict[str, str | int | set[str] | dict[str, dict[str, int]]],
#         sysdb: SystemConfigDatabase,
#         target_ports: set[PORT],
#     ) -> SkewAdjust:
#         return SkewAdjust.from_yaml_dict(config, sysdb, target_ports)

#     @staticmethod
#     def from_yaml_dict(
#         yaml_dict: dict[str, str | int | set[str] | dict[str, dict[str, int]]],
#         sysdb: SystemConfigDatabase,
#         target_ports: set[PORT],
#     ) -> SkewAdjust:
#         box_setting = cast(dict[str, dict[str, int]], yaml_dict["box_setting"])
#         slot = {
#             (bname, nport): box_setting[bname]["slot"]
#             for (bname, nport) in target_ports
#         }
#         wait = {
#             (bname, nport): box_setting[bname]["wait"]
#             for (bname, nport) in target_ports
#         }
#         time_to_start = cast(int, yaml_dict["time_to_start"])
#         return SkewAdjust(
#             sysdb,
#             target_ports=target_ports,
#             slot=slot,
#             wait=wait,
#             time_to_start=time_to_start,
#         )

#     def push(self) -> None:
#         for (box_name, _), slot in self.slot.items():
#             self.sysdb.timing_shift[box_name] = slot * 16
#         for (box_name, _), wait in self.wait.items():
#             self.sysdb.skew[box_name] = wait
#         self.sysdb.time_to_start = self.time_to_start

#     def pull(self) -> None:
#         for port in self.target_ports:
#             box_name, _ = port
#             self.slot[port] = self.sysdb.timing_shift[box_name] // 16
#             self.wait[port] = self.sysdb.skew[box_name]
#         self.time_to_start = self.sysdb.time_to_start

#     def backup(self) -> SkewAdjust:
#         o = SkewAdjust(self.sysdb, target_ports=self.target_ports)
#         o.pull()
#         return o


@dataclass
class SkewSetting:
    reference_port: PORT
    monitor_port: PORT
    trigger_nport: int
    target_port: set[PORT]
    scale: dict[PORT, float]
    repeats: dict[PORT, int]
    rf_switches: dict[str, dict[Quel1PortType, str]]
    clockmaster_ip: str

    # @staticmethod
    # def load(
    #     config: dict[str, str | int | set[str] | dict[str, dict[str, int]]],
    # ) -> "SkewSetting":
    #     return SkewSetting.from_yaml_dict(config)

    @staticmethod
    def from_yaml(filename: str, *, clockmaster_ip: str | None = None) -> "SkewSetting":
        with open(Path(os.getcwd()) / Path(filename), "r") as file:
            config = yaml.safe_load(file)
        return SkewSetting.from_yaml_dict(
            config,
            clockmaster_ip=clockmaster_ip,
        )

    @staticmethod
    def from_yaml_dict(
        yaml_dict: dict[str, str | int | set[str] | dict[str, dict[str, int]]],
        *,
        clockmaster_ip: str | None = None,
    ) -> "SkewSetting":
        reference_port = str2port(cast(str, yaml_dict["reference_port"]))
        monitor_port = str2port(cast(str, yaml_dict["monitor_port"]))
        trigger_nport = cast(int, yaml_dict["trigger_nport"])
        raw_targets = yaml_dict["target_port"]
        if isinstance(raw_targets, (list, tuple, set)):
            target_port = {str2port(v) for v in raw_targets}
        else:
            raise TypeError("target_port must be a list/tuple/set of strings")
        target_port = {str2port(v) for v in cast(set[str], yaml_dict["target_port"])}
        scale = {
            str2port(p): v
            for p, v in cast(dict[str, float], yaml_dict["scale"]).items()
        }
        if "repeats" in yaml_dict:
            repeats = {
                str2port(p): v
                for p, v in cast(dict[str, int], yaml_dict["repeats"]).items()
            }
        else:
            repeats = {}
        if "rf_switches" in yaml_dict:
            rf_switches = {
                box_name: {_parse_port_type(k): str(v) for k, v in box_setting.items()}
                for box_name, box_setting in cast(
                    dict[str, dict[str, Any]], yaml_dict["rf_switches"]
                ).items()
            }
        else:
            rf_switches = {}
        if clockmaster_ip is not None:
            pass
        elif "clockmaster_ip" in yaml_dict:
            clockmaster_ip = cast(str, yaml_dict["clockmaster_ip"])
        else:
            raise ValueError("clockmaster_ip is required")
        return SkewSetting(
            reference_port=reference_port,
            monitor_port=monitor_port,
            trigger_nport=trigger_nport,
            target_port=target_port,
            scale=scale,
            repeats=repeats,
            rf_switches=rf_switches,
            clockmaster_ip=clockmaster_ip,
        )

    @property
    def monitor_box_name(self) -> str:
        box_name, _ = self.monitor_port
        return box_name

    @property
    def target_box_names(self) -> set[str]:
        return set({box_name for box_name, _ in self.target_port})


class Skew:
    DEFAULT_CHANNEL = 0
    DEFAULT_REPEATS = 100

    def __init__(
        self,
        system: Quel1System,
        *,
        sysdb: SystemConfigDatabase,
        skew_yaml_path: str | None = None,
    ) -> None:
        self._system: Final[Quel1System] = system
        self._sysdb: Final[SystemConfigDatabase] = sysdb
        self._executor: Final[Executor] = Executor(self.sysdb, quel1system=system)
        self._monitor_port: PORT = ("", 0)
        self._trigger_nport: int = 0
        self._reference_port: PORT = ("", 0)
        self._scale: dict[PORT, float] = {}
        self._measured_waveform: dict[PORT, npt.NDArray] = {}
        self._target_port: set[PORT] = set()
        # self._skew_adjust: SkewAdjust = SkewAdjust(self.sysdb)
        self._setting: SkewSetting | None = None
        self._skew_yaml_path: str | None = skew_yaml_path
        self._estimated: dict[PORT, EstimatedPulseParams] = {}
        for box_name in self._system.boxes:
            if not self.is_channel_defined(box_name, sysdb=self.sysdb):
                self._define_channel_names(
                    box_name, system=self._system, sysdb=self.sysdb
                )

    def reload(self, skew_yaml: str | None = None) -> None:
        """
        Reload skew parameters from skew.yaml without recreating system objects.
        This is intended for quick trial iterations.
        """
        path = skew_yaml if skew_yaml is not None else self._skew_yaml_path
        if path is None:
            raise ValueError("skew_yaml path must be provided at least once")
        clockmaster_ip = self.setting.clockmaster_ip if self.setting else None
        self._skew_yaml_path = path
        self._sysdb.load_skew_yaml(path)
        self._sysdb.refresh_quel1system(self._system)
        new_setting = SkewSetting.from_yaml(path, clockmaster_ip=clockmaster_ip)
        self.setting = new_setting
        # self._skew_adjust.target_ports = copy(new_setting.target_port)
        # self._skew_adjust.pull()
        # self.config_rfswitches()
        # drop old measurement cache to avoid confusion after parameter change
        self._measured_waveform.clear()
        self._estimated.clear()

    def prepare(self) -> None:
        self.config_rfswitches()

    @classmethod
    def from_yaml(
        cls,
        skew_yaml: str,
        *,
        box_yaml: str | None = None,
        sysdb: SystemConfigDatabase | None = None,
        system: Quel1System | None = None,
        boxes: list[str] | None = None,
        ignore_boxes: list[str] | None = None,
        clockmaster_ip: str | None = None,
    ) -> Skew:
        setting = SkewSetting.from_yaml(skew_yaml, clockmaster_ip=clockmaster_ip)

        if sysdb is None and box_yaml is None:
            raise ValueError("Either sysdb or box_yaml must be provided")
        if sysdb is not None and box_yaml is not None:
            raise ValueError("Only one of sysdb or box_yaml must be provided")
        if sysdb is None:
            sysdb = SystemConfigDatabase()
            sysdb.define_clockmaster(setting.clockmaster_ip, reset=False)
            sysdb.load_box_yaml(cast(str, box_yaml))
            sysdb.load_skew_yaml(skew_yaml)
        else:
            sysdb.load_skew_yaml(skew_yaml)

        if system is not None and boxes != []:
            raise ValueError("Only one of system or boxes must be provided")

        available_boxes = set(
            list(setting.target_box_names) + [setting.monitor_box_name]
        )

        boxes = [] if boxes is None else boxes
        ignore_boxes = [] if ignore_boxes is None else ignore_boxes
        if boxes:
            if not all([box in available_boxes for box in boxes]):
                raise ValueError("Some boxes in boxes are not available")
        else:
            boxes = list(available_boxes)
        if not all([box in boxes for box in ignore_boxes]):
            raise ValueError("Some boxes in ignore_boxes are not available")
        boxes = [box for box in boxes if box not in ignore_boxes]

        if system is None:
            system = sysdb.create_quel1system(*boxes)
        else:
            system = sysdb.refresh_quel1system(system)

        self = Skew(system=system, sysdb=sysdb, skew_yaml_path=skew_yaml)
        self.setting = setting
        return self

    @property
    def sysdb(self) -> SystemConfigDatabase:
        return self._sysdb

    @property
    def system(self) -> Quel1System:
        return self._system

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
    def acquire_freq_setting(
        cls,
        target_freq: float,  # Hz
        *,
        minimum_cnco_freq: float = 2e9,  # Hz
        lo_step_size: float = 500e6,  # Hz
        cnco_step_size: float = 125e6,  # Hz
    ) -> dict[str, float | str]:
        """
        Compute LO/CNCO/FNCO assuming lower sideband (LSB) only.
        """
        sideband = "L"  # LSB only
        lo_freq = (target_freq + minimum_cnco_freq) // lo_step_size * lo_step_size
        cnco_freq = (lo_freq - target_freq) // cnco_step_size * cnco_step_size
        fnco_freq = 0
        logger.debug(
            f"acquire_freq_setting(): target_freq={target_freq}; lo_freq={lo_freq}, cnco_freq={cnco_freq}, fnco_freq={fnco_freq}, sideband={sideband}"
        )
        return {
            "lo_freq": lo_freq,
            "cnco_freq": cnco_freq,
            "fnco_freq": fnco_freq,
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

    def _acquire_target(
        self,
        port: PORT,
        channel: int | None = None,
        sysdb: SystemConfigDatabase | None = None,
    ) -> str:
        sysdb = self._sysdb if sysdb is None else sysdb
        return self.get_target_by_channel(sysdb, port, channel)

    def target_from_box(self, box_names: list[str]) -> set[PORT]:
        return {
            (box_name, nport)
            for box_name, nport in self._target_port
            if box_name in box_names
        }

    def sync_lo_nco(
        self,
        *,
        src_port: PORT,
        dest_port: PORT,
    ) -> dict[str, str | float | dict[str, float] | dict[int, dict[str, float]]]:
        """src_port の周波数を dest_port に合わせる"""
        return self._sync_lo_nco(
            src_port=src_port,
            dest_port=dest_port,
            system=self._system,
        )

    @classmethod
    def _parse_freqs_from_dump_port(
        cls,
        *,
        dump_port: dict[str, dict[str, float]],
    ) -> tuple[
        dict[str, str | float | dict[str, float] | dict[int, dict[str, float]]],
        dict[int, dict[str, float]],
    ]:
        # target_freq 計算して ch_freqs に追加する
        kws = ["lo_freq", "cnco_freq", "runits", "channels", "sideband"]
        freqs: dict[
            str, str | float | dict[str, float] | dict[int, dict[str, float]]
        ] = {k: v for k, v in dump_port.items() if k in kws}
        ch_freqs = (
            cast(dict[int, dict[str, float]], freqs["channels"])
            if "channels" in freqs
            else cast(dict[int, dict[str, float]], freqs["runits"])
        )

        lo_freq, cnco_freq = (
            cast(float, freqs["lo_freq"]) if "lo_freq" in freqs else 0,
            cast(float, freqs["cnco_freq"]),
        )
        # USB/LSB の target_freq を計算する
        kw = "fnco_freq"
        usb_freqs = {i: lo_freq + (cnco_freq + v[kw]) for i, v in ch_freqs.items()}
        lsb_freqs = {i: lo_freq - (cnco_freq + v[kw]) for i, v in ch_freqs.items()}
        # sideband　が設定されているならそれを使う
        if lo_freq == 0:
            # lo_freq == 0 の場合は sideband を空にする
            freqs["sideband"] = ""
        if "sideband" in freqs:
            sideband = freqs["sideband"]
        else:
            # 9.5GHz を境に USB/LSB を決定する
            sideband = "L" if lo_freq + cnco_freq < 9500e6 else "U"
            freqs["sideband"] = sideband
        if sideband == "L":
            for i, v in lsb_freqs.items():
                ch_freqs[i]["target_freq"] = v
        else:
            for i, v in usb_freqs.items():
                ch_freqs[i]["target_freq"] = v
        return freqs, ch_freqs

    @classmethod
    def _sync_lo_nco(
        cls,
        *,
        src_port: PORT,
        dest_port: PORT,
        system: Quel1System,
    ) -> dict[str, str | float | dict[str, float] | dict[int, dict[str, float]]]:
        dump_port = deepcopy(system.dump_port(*src_port))
        freqs, ch_freqs = cls._parse_freqs_from_dump_port(
            dump_port=dump_port,
        )
        logger.debug(
            f"_sync_lo_nco(): SRC_PORT{src_port}; lo_freq={freqs['lo_freq'] if 'lo_freq' in freqs else 0}, cnco_freq={freqs['cnco_freq']}, channels={freqs['channels']}, sideband={freqs['sideband']}"
        )
        lo_freq, cnco_freq = (
            cast(float, freqs["lo_freq"]) if "lo_freq" in freqs else 0,
            cast(float, freqs["cnco_freq"]) + ch_freqs[0]["fnco_freq"],
        )
        if lo_freq == 0:
            # Case of Destination port is Direct synthesys port
            # Direct synthesys ポートは cnco_freq は 2000MHz - 6000MHz 程度まで OK だが
            # Monitor ポートは 3000MHz まで
            target_freq = cnco_freq
            lo_freq = (target_freq + 1500e6) // 500e6 * 500e6
            cnco_freq = lo_freq - target_freq
        else:
            target_freq = ch_freqs[0]["target_freq"]

        boxname, nport = dest_port
        box = system.box[boxname]
        dump_port = deepcopy(system.dump_port(*dest_port))
        if "channels" in dump_port:
            dump_port["lo_freq"] = lo_freq
            dump_port["cnco_freq"] = cnco_freq
            for k, v in dump_port["channels"].items():
                v["fnco_freq"] = 0

            # ignore rfswitch setting
            del dump_port["rfswitch"]

            config_box: dict[int | tuple[int, int], dict[str, Any]] = {nport: dump_port}
            box.config_box(config_box)
        elif "runits" in dump_port:
            if lo_freq == 0:
                raise ValueError("LO frequency is required for receiver ports")
            freq_setting = cls.acquire_freq_setting(target_freq)

            lo_freq = float(freq_setting["lo_freq"])
            dump_port["lo_freq"] = lo_freq
            system.config_cache[boxname]["ports"][nport]["lo_freq"] = lo_freq

            cnco_freq = float(freq_setting["cnco_freq"])
            dump_port["cnco_freq"] = cnco_freq
            system.config_cache[boxname]["ports"][nport]["cnco_freq"] = cnco_freq

            fnco_freq = float(freq_setting["fnco_freq"])
            for k, v in dump_port["runits"].items():
                v["fnco_freq"] = fnco_freq
                system.config_cache[boxname]["ports"][nport]["runits"][k][
                    "fnco_freq"
                ] = fnco_freq

            # ignore rfswitch setting
            del dump_port["rfswitch"]
            config_box = {nport: dump_port}
            box.config_box(config_box)
        channels_or_runits = freqs.get("channels", freqs.get("runits"))
        logger.debug(
            f"-> DEST_PORT{dest_port}; lo_freq={lo_freq}, cnco_freq={cnco_freq}, "
            f"units={channels_or_runits}, sideband={freqs.get('sideband', '')}"
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
        ports = system.dump_box(box_name)["ports"]
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
                ndelay_or_nwait=tuple(len(channels) * [7 if io == "IN" else 0]),
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
        if not system.is_output_port(name, nport):
            raise ValueError(f"{target_port} is not output port")
        freqs = cls._sync_lo_nco(
            src_port=target_port,
            dest_port=monitor_port,
            system=system,
        )
        ch_freqs = cast(dict[int, dict[str, float]], freqs["channels"])
        DEFAULT_CHANNEL = 0
        target_freq = ch_freqs[DEFAULT_CHANNEL]["target_freq"] * 1e-9
        target = cls.get_target_by_channel(sysdb, target_port)
        sysdb._target_settings[target] = dict(frequency=target_freq)
        monitor = cls.get_target_by_channel(sysdb, monitor_port)
        sysdb._target_settings[monitor] = dict(frequency=target_freq)
        lo_freq = system.get_lo_freq(*monitor_port)
        cnco_freq = system.get_cnco_freq(*monitor_port)
        sideband = system.get_sideband(*monitor_port)
        fnco_freq = system.get_fnco_freq(*monitor_port, channel=DEFAULT_CHANNEL)
        logger.debug(
            f"_setup_monitor_port(): Target {target}:{target_port}; lo_freq={lo_freq}, cnco_freq={cnco_freq}, fnco_freq={fnco_freq}, target_freq={target_freq}, sideband={sideband}"
        )

    @classmethod
    def _setup_trigger_port(
        cls,
        *,
        trigger_port: PORT,
        system: Quel1System,
        sysdb: SystemConfigDatabase,
    ) -> None:
        """nco 設定に合わせて target 周波数を設定する"""
        dump_port = system.dump_port(*trigger_port)
        freqs, ch_freqs = cls._parse_freqs_from_dump_port(dump_port=dump_port)
        lo_freq, cnco_freq, fnco_freq, target_freq, sideband = (
            cast(float, freqs["lo_freq"] if "lo_freq" in freqs else 0),
            cast(float, freqs["cnco_freq"]),
            ch_freqs[cls.DEFAULT_CHANNEL]["fnco_freq"],
            ch_freqs[cls.DEFAULT_CHANNEL]["target_freq"] * 1e-9,
            cast(str, freqs["sideband"]),
        )
        trigger = cls.get_target_by_channel(sysdb, trigger_port)
        sysdb._target_settings[trigger] = dict(frequency=target_freq)
        logger.debug(
            f"_setup_trigger_port(): Trigger {trigger_port}, lo_freq={lo_freq * 1e-9}, cnco_freq={cnco_freq * 1e-9}, fnco_freq={fnco_freq * 1e-9}, target_freq={target_freq}, sideband={sideband}"
        )

    def config_rfswitches(self) -> None:
        setting = cast(SkewSetting, self.setting)
        for box_name, box in self.system.boxes.items():
            box_setting = setting.rf_switches.get(box_name)
            if not box_setting:
                continue
            box.config_rfswitches(box_setting)

    def measure(
        self,
        *,
        show_reference: bool | None = None,
        extra_capture_range: int | None = None,  # multiple of 128 ns
        repeats: int = DEFAULT_REPEATS,
    ) -> None:
        target_ports = self.target_from_box(list(self._system.boxes))
        self._define_targets(
            target_ports=target_ports,
            reference_port=self._reference_port,
            monitor_port=self._monitor_port,
            trigger_nport=self._trigger_nport,
            sysdb=self._sysdb,
        )
        self._measure_targets(
            target_ports,
            show_reference=show_reference,
            extra_capture_range=extra_capture_range,
            repeats=repeats,
        )

    def _measure_targets(
        self,
        target_ports: set[PORT],
        *,
        show_reference: bool | None = None,
        extra_capture_range: int | None = None,
        repeats: int | None = None,
    ) -> None:
        target_ports = self.target_from_box(list(self._system.boxes))
        repeats = self.DEFAULT_REPEATS if repeats is None else repeats
        with tqdm(target_ports) as t:
            for target_port in t:
                t.postfix = f"Target: {target_port}"
                t.update()
                setting = cast(SkewSetting, self.setting)
                if target_port in setting.repeats:
                    repeats = setting.repeats[target_port]
                self._measure(
                    target_port,
                    show_reference=show_reference,
                    extra_capture_range=extra_capture_range,
                    repeats=repeats,
                )
                logger.debug(
                    f"-------- _measure_targets(): Target{target_port} finished --------"
                )

    def _measure(
        self,
        target_port: PORT,
        *,
        reference_port: PORT | None = None,
        monitor_port: PORT | None = None,
        trigger_nport: int | None = None,
        repeats: int | None = None,
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
        extra_capture_range = (
            extra_capture_range
            if extra_capture_range is not None
            else EXTRA_CAPTURE_RANGE
        )
        seq = self._create_check_sequence(
            target_port, capture_range=extra_capture_range
        )
        repeats = self.DEFAULT_REPEATS if repeats is None else repeats
        iqs = self._execute(seq, repeats=repeats)
        self._store(target_port, iqs)

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
        capture_range = (
            DEFAULT_CAPTURE_RANGE if capture_range is None else capture_range
        )
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

    def _execute(
        self, sequence: Sequence, *, repeats: int | None = None
    ) -> npt.NDArray:
        """Executes the measurement, assuming that the sequence contains only a single capture."""
        self._executor.add_sequence(sequence, driver=self._system)
        rst: npt.NDArray | None = None
        repeats = self.DEFAULT_REPEATS if repeats is None else repeats
        for _, data, _ in self._executor.step_execute(
            repeats=repeats,
            interval=REPETITION_PERIOD,
            integral_mode="single",
            dsp_demodulation=False,
            software_demodulation=True,
        ):
            for _, iqs in data.items():
                rst = (
                    iqs[0].sum(axis=0).squeeze()
                )  # iqs format changed shape: (512, 100)　→ (100, 512)
            if rst is None:
                raise RuntimeError("No data acquired")
        return cast(npt.NDArray, rst)

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
        # trigger の target 周波数を設定する
        self._setup_trigger_port(
            trigger_port=trigger_port,
            system=system,
            sysdb=sysdb,
        )

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
            # Define a temporary channel name for provisional use
            if channel not in channel_names_by_channel:
                bname, nport, ch = channel
                port_name = f"{bname}.PORT{nport}"
                channel_name = f"{port_name}.CH{ch}"
                sysdb.add_port_setting(
                    port_name=port_name,
                    box_name=bname,
                    port=nport,
                    ndelay_or_nwait=tuple([0]),
                )
                sysdb._relation_channel_port.append(
                    (
                        channel_name,
                        {
                            "port_name": port_name,
                            "channel_number": ch,
                        },
                    ),
                )
            else:
                channel_name = channel_names_by_channel[channel]
            # Define a temporary target for provisional use
            if channel_name not in defined_channel:
                target_name = channel_name
                sysdb._relation_channel_target.append((target_name, channel_name))
                sysdb._target_settings[target_name] = dict(frequency=0)

    def define_targets(
        self,
        *,
        target_ports: set[PORT],
        reference_port: PORT | None = None,
        monitor_port: PORT | None = None,
        trigger_nport: int | None = None,
    ) -> None:
        """Deprecated public wrapper retained for backward compatibility."""
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
            measured_waveform=OrderedDict(sorted(self._measured_waveform.items())),
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

    def estimated_indices(self) -> dict[PORT, int]:
        """
        Return the current estimated skew index for each PORT.
        """
        return {k: v.idx for k, v in self._estimated.items()}

    def update_skew(self, target_value: int, backup: bool = False) -> bool:
        """
        Update the skew configuration so that each PORT matches the target value.
        
        The skew wait value for each port is adjusted based on the difference
        between the target value and the last estimated skew index.
        The resulting wait value is clamped to a minimum of 0.        

        Parameters
        ----------
        target_value : int
            The desired skew index.
        backup : bool, optional
            If True, create a timestamped backup of the current configuration file
            before applying changes.

        Returns
        -------
        bool
            True if the configuration was modified, False otherwise.
        """
        # backup current skew config data when required
        if backup:
            skewfile = self._skew_yaml_path
            dt = datetime.datetime.now()
            bak = skewfile + '.bak.' + dt.strftime('%Y%m%d_%H%M%S')
            shutil.copy(skewfile, bak)

        # load current skew config data
        with open(self._skew_yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        updated = False
        for k,v in self._estimated.items():
            t,c = k
            diff = (target_value - v.idx)
            cur_wait = config['box_setting'][t]['port_wait'][c]
            new_wait = cur_wait + diff
            new_wait = max(0, new_wait) # clamp to minimum 0
            if cur_wait != new_wait:
                config['box_setting'][t]['port_wait'][c] = new_wait
                updated = True

        # save updated skew config data
        with open(self._skew_yaml_path, 'w') as f:
            yaml.safe_dump(config, f)

        return updated

    # def load(self, filename: str) -> None:
    #     with open(Path(os.getcwd()) / Path(filename), "r") as file:
    #         config = yaml.safe_load(file)
    #     self.setting = SkewSetting.from_yaml_dict(config)
    #     sysdb = self._sysdb
    #     target = self._target_port
    #     self._skew_adjust = SkewAdjust.from_yaml_dict(
    #         config,
    #         sysdb=sysdb,
    #         target_ports=target,
    #     )
    #     self._skew_adjust.push()

    # def load_setting(self, filename: str) -> None:
    #     with open(Path(os.getcwd()) / Path(filename), "r") as file:
    #         config = yaml.safe_load(file)
    #     self.setting = SkewSetting.from_yaml_dict(config)
    #     setting = cast(SkewSetting, self.setting)
    #     self._skew_adjust.target_ports = copy(setting.target_port)
    #     self._skew_adjust.pull()

    # def save(self, filename: str) -> None:
    #     sysdb = self._sysdb
    #     config = {
    #         "time_to_start": sysdb.time_to_start,
    #         "box_setting": {
    #             box_name: {"slot": v // 16, "wait": sysdb.skew[box_name]}
    #             for box_name, v in sysdb.timing_shift.items()
    #         },
    #         "reference_port": port2str(self._reference_port),
    #         "monitor_port": port2str(self._monitor_port),
    #         "trigger_nport": self._trigger_nport,
    #         "target_port": {port2str(v) for v in self._target_port},
    #         "scale": {port2str(p): v for p, v in self._scale.items()},
    #     }
    #     with open(Path(os.getcwd()) / Path(filename), "w") as file:
    #         yaml.safe_dump(config, file)
