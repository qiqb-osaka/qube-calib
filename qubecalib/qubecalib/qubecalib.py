from __future__ import annotations

import logging
import os
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from ipaddress import IPv4Address, IPv6Address, ip_address
from pathlib import Path
from typing import Any, Dict, Final, MutableSequence, Optional, Sequence, Tuple

import json5
from e7awgsw import CaptureParam, WaveSequence
from quel_ic_config import CaptureReturnCode, Quel1BoxType, Quel1ConfigOption

from . import neopulse
from .e7utils import CaptureParamTools
from .neopulse import CapSampledSequence, GenSampledSequence, SampledSequenceBase

logger = logging.getLogger(__name__)


class Direction(Enum):
    FROM_TARGET = "from_target"
    TO_TARGET = "to_target"


class Sideband(Enum):
    UpperSideBand: str = "U"
    LowerSideBand: str = "L"


class QubeCalib:
    def __init__(
        self,
        path_to_database_file: Optional[str | os.PathLike] = None,
    ) -> None:
        self._system_config_database: Final[SystemConfigDatabase] = (
            SystemConfigDatabase()
        )
        self._executor: Final[Executor] = Executor()

        if path_to_database_file is not None:
            self.system_config_database.load(path_to_database_file)

    @property
    def system_config_database(self) -> SystemConfigDatabase:
        return self._system_config_database

    def exec(self) -> Tuple:
        """queue に登録されている command を実行する"""
        return "", "", ""

    def exec_iter(self) -> Executor:
        """queue に登録されている command を実行する iterator を返す"""
        return self._executor

    def add_sequence(
        self,
        sequence: neopulse.Sequence,
        repeats: int = 1,
        interval: Optional[float] = None,
        singleshot: bool = False,
        dsp_demodulation: bool = True,
        software_demodulation: bool = False,
    ) -> None:
        if dsp_demodulation and software_demodulation:
            raise ValueError(
                "dsp_demodulation and softoware_demodulation options cannot be True at the same time"
            )
        sampled_sequence = sequence.convert_to_sampled_sequence()
        resource_map = self._create_target_resource_map([_ for _ in sampled_sequence])
        devseq = Converter.convert_to_device_specific_sequence(
            sampled_sequence,
            resource_map,
            repeats,
            interval,
            singleshot,
            dsp_demodulation,
            software_demodulation,
        )
        print(devseq)
        self._executor.add_command(Sequencer(devseq))

    def add_config(self) -> None:
        self._executor.add_command(Configurator())

    def define_target(
        self,
        target_name: str,
        target_frequency: float,
        channel_name: str,
    ) -> None:
        pass

    def _create_target_resource_map(
        self, target_names: Sequence[str]
    ) -> Dict[str, Dict[str, int | BoxSetting | PortSetting]]:
        # {target_name: sampled_sequence} の形式から
        # TODO {target_name: {box, group, line | rline, channel | runit)} へ変換　？？
        # {target_name: {box, port, channel_number)}} へ変換
        db = self.system_config_database
        return {
            _: {
                "box": db._box_settings[db.get_box_by_target(_)],
                "port": db._port_settings[db.get_port_by_target(_)],
                "channel_number": db.get_channel_number_by_target(_),
            }
            for _ in target_names
        }


class Converter:
    @classmethod
    def convert_to_device_specific_sequence(
        cls,
        sampled_sequence: Dict[str, SampledSequenceBase],
        resource_map: Dict[str, Dict[str, BoxSetting | PortSetting | int]],
        repeats: int,
        interval: Optional[float],
        singleshot: bool,
        dsp_demodulation: bool,
        software_demodulation: bool,
    ) -> Dict[Tuple[str, str, int], WaveSequence | CaptureParam]:
        # sampled_sequence と resource_map から e7 データを生成する
        capseq = cls.convert_to_cap_device_specific_sequence(
            {
                target_name: sseq
                for target_name, sseq in sampled_sequence.items()
                if isinstance(sseq, CapSampledSequence)
            },
            resource_map,
        )
        genseq = cls.convert_to_gen_device_specific_sequence(
            {
                target_name: sseq
                for target_name, sseq in sampled_sequence.items()
                if isinstance(sseq, GenSampledSequence)
            },
            resource_map,
        )
        # gen と cap を分離する
        return genseq | capseq

    @classmethod
    def convert_to_cap_device_specific_sequence(
        cls,
        sampled_sequence: Dict[str, CapSampledSequence],
        resource_map: Dict[str, Dict[str, BoxSetting | PortSetting | int]],
    ) -> Dict[Tuple[str, str, int], CaptureParam]:
        # CaptureParam の生成

        # target_name と (box_name, port_number, channel_number) のマップを作成する
        targets_ids = {
            target_name: (_["box"].box_name, _["port"].port, _["channel_number"])
            for target_name, _ in resource_map.items()
        }
        # ハードウェア復調の場合 channel (unit) に対して単一の target を仮定する
        # ソフトウェア復調の場合は channel 毎にデータを束ねる必要がある TODO 後日実装
        # cap channel と target が 1:1 で対応しているか確認
        if not all(
            [
                _ == 1
                for _ in Counter([targets_ids[_] for _ in sampled_sequence]).values()
            ]
        ):
            raise ValueError("multiple access for single runit will be supported")
        # 戻り値は {(box_name, port_number, channel_number): CaptureParam} の Dict
        return {
            targets_ids[_.target_name]: CaptureParamTools.create(_)
            for _ in sampled_sequence.values()
        }

    @classmethod
    def convert_to_gen_device_specific_sequence(
        cls,
        sampled_sequence: Dict[str, GenSampledSequence],
        resource_map: Dict[str, Dict[str, BoxSetting | PortSetting | int]],
    ) -> Dict[Tuple[str, str, int], WaveSequence]:
        # WaveSequence の生成
        # channel 毎 (awg 毎) にデータを束ねる
        # 各 target 毎に信号を変調する
        # channel 毎に WaveSequence を生成する
        # 戻り値は {(box_name, port_number, channel_number): WaveSequence} の Dict

        # print(sampled_sequence)
        return {}

    @classmethod
    def calc_modulation_frequency(
        cls,
        target_hz: float,
        lo_hz: float,
        cnco_hz: float,
        sideband: str,
        fnco_hz: Optional[float] = None,
    ) -> float:
        if_hz = cnco_hz + fnco_hz if fnco_hz is not None else cnco_hz
        if sideband == Sideband.UpperSideBand.value:
            diff_hz = target_hz - lo_hz - if_hz
        elif sideband == Sideband.LowerSideBand.value:
            diff_hz = -(target_hz - lo_hz) - if_hz
        else:
            raise ValueError("invalid ssb mode")

        return diff_hz


class Command:
    def execute(self) -> Any:
        pass


class Sequencer(Command):
    def __init__(self, e7_settings: Dict) -> None:
        self._e7_setting = e7_settings

    def execute(self) -> Tuple[CaptureReturnCode, Dict, Dict]:
        return CaptureReturnCode.SUCCESS, {}, {}


class Configurator(Command):
    def execute(self) -> None:
        print(f"{self.__class__.__name__} executed")


class Executor:
    def __init__(self) -> None:
        self._work_queue: Final[deque] = deque()

    def __iter__(self) -> Executor:
        if not self._work_queue:
            return self
        last_command = self._work_queue[-1]
        if not isinstance(last_command, Sequencer):
            raise ValueError("_work_queue should end with a Sequencer command")
        return self

    def __next__(self) -> Tuple[Any, Dict, Dict]:
        # ワークキューが空になったら実行を止める
        if not self._work_queue:
            raise StopIteration()
        # 波形送受信以外の処理は一括実行する
        while True:
            if not self._work_queue:
                raise ValueError("_work_queue should end with a Sequencer command")
            next = self._work_queue.pop()
            if isinstance(next, Sequencer):
                break
            next.execute()
        return next.execute()

    def add_command(self, command: Command) -> None:
        self._work_queue.appendleft(command)


@dataclass
class ClockmasterSetting:
    ipaddr: str | IPv4Address | IPv6Address
    reset: bool


@dataclass
class BoxSetting:
    box_name: str
    ipaddr_wss: str | IPv4Address | IPv6Address
    boxtype: Quel1BoxType
    ipaddr_sss: Optional[str | IPv4Address | IPv6Address] = None
    ipaddr_css: Optional[str | IPv4Address | IPv6Address] = None
    config_root: Optional[str | os.PathLike] = None
    config_options: MutableSequence[Quel1ConfigOption] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.ipaddr_wss, str):
            self.ipaddr_wss = ip_address(self.ipaddr_wss)
        elif not isinstance(self.ipaddr_wss, (IPv4Address, IPv6Address)):
            raise ValueError("ipaddr_wss should be instance of IPvxAddress")

        if self.ipaddr_sss is None:
            self.ipaddr_sss = self.ipaddr_wss + (1 << 16)
        elif isinstance(self.ipaddr_sss, str):
            self.ipaddr_sss = ip_address(self.ipaddr_sss)
        elif not isinstance(self.ipaddr_sss, (IPv4Address, IPv6Address)):
            raise ValueError("ipaddr_sss should be instance of IPvxAddress")

        if self.ipaddr_css is None:
            self.ipaddr_css = self.ipaddr_wss + (4 << 16)
        elif isinstance(self.ipaddr_css, str):
            self.ipaddr_css = ip_address(self.ipaddr_css)
        elif not isinstance(self.ipaddr_css, (IPv4Address, IPv6Address)):
            raise ValueError("ipaddr_css should be instance of IPvxAddress")

        if (
            self.boxtype
            in (
                Quel1BoxType.QuBE_OU_TypeA,
                Quel1BoxType.QuBE_RIKEN_TypeA,
                Quel1BoxType.QuEL1_TypeA,
            )
            and not self.config_options
        ):
            self.config_options = [
                Quel1ConfigOption.USE_READ_IN_MXFE0,
                Quel1ConfigOption.USE_READ_IN_MXFE1,
            ]

    def asdict(self) -> dict[str, Any]:
        return {
            "ipaddr_wss": str(self.ipaddr_wss),
            "ipaddr_sss": str(self.ipaddr_sss),
            "ipaddr_css": str(self.ipaddr_css),
            "boxtype": self.boxtype,
            "config_root": str(self.config_root)
            if self.config_root is not None
            else None,
            "config_options": self.config_options,
        }


@dataclass
class PortSetting:
    port_name: str
    box_name: str
    port: int
    lo_freq: Optional[float] = None
    cnco_freq: Optional[float] = None
    sideband: str = "U"
    vatt: int = 0x800
    fnco_freq: Optional[Tuple[float] | Tuple[float, float, float]] = None


class SystemConfigDatabase:
    def __init__(self) -> None:
        self._clockmaster_setting: Optional[ClockmasterSetting] = None
        self._box_settings: Final[Dict[str, BoxSetting]] = {}
        self._box_aliases: Final[Dict[str, str]] = {}
        self._port_settings: Final[Dict[str, PortSetting]] = {}
        self._relation_channel_target: Final[MutableSequence[Tuple[str, str]]] = []
        self._target_settings: Final[Dict[str, Dict[str, Any]]] = {}
        self._relation_channel_port: Final[
            MutableSequence[Tuple[str, Dict[str, str | int]]]
        ] = []

    def set(
        self,
        clockmaster_setting: Optional[Dict] = None,
        box_settings: Optional[Dict] = None,
        box_aliases: Optional[Dict[str, str]] = None,
        port_settings: Optional[Dict[str, Dict[str, Any]]] = None,
        relation_channel_target: Optional[MutableSequence[Tuple[str, str]]] = None,
        target_settings: Optional[Dict[str, Dict[str, Any]]] = None,
        relation_channel_port: Optional[MutableSequence[Tuple[str, str]]] = None,
    ) -> None:
        if clockmaster_setting is not None:
            self._clockmaster_setting = ClockmasterSetting(
                ipaddr=clockmaster_setting["ipaddr"],
                reset=clockmaster_setting["reset"],
            )
        if box_settings is not None:
            for box_name, setting in box_settings.items():
                self.add_box_setting(**({"box_name": box_name} | setting))
        if box_aliases is not None:
            for alias, name in box_aliases.items():
                self._box_aliases[alias] = name
        if port_settings is not None:
            for port_name, setting in port_settings.items():
                setting["box_name"] = self._box_aliases[setting["box_name_or_alias"]]
                setting["fnco_freq"] = setting["band"]
                del setting["box_name_or_alias"]
                del setting["band"]
                self.add_port_setting(**{"port_name": port_name} | setting)
        if relation_channel_target is not None:
            for _ in relation_channel_target:
                self._relation_channel_target.append(_)
        if target_settings is not None:
            for target_name, setting in target_settings.items():
                self._target_settings[target_name] = setting
        if relation_channel_port is not None:
            for _ in relation_channel_port:
                self._relation_channel_port.append(_)

    def load(self, path_to_database_file: str | os.PathLike) -> None:
        with open(Path(os.getcwd()) / Path(path_to_database_file), "r") as file:
            json = json5.load(file)
        # TODO workaround
        settings = {
            k: v
            for k, v in json.items()
            if k
            in [
                "clockmaster_setting",
                "box_settings",
                "box_aliases",
                "target_settings",
                "port_settings",
            ]
        }
        relation_channel_target = [
            [channel[:-2] + "CH" + channel[-1], target]
            for channel, target in json["relation_band_target"]
        ]
        settings["relation_channel_target"] = relation_channel_target
        channels = {channel for channel, target in relation_channel_target}
        relation_channel_port = [
            [_, {"port_name": _[:-3], "channel_number": int(_[-1:])}] for _ in channels
        ]
        settings["relation_channel_port"] = relation_channel_port
        # TODO ----------
        self.set(**settings)

    def add_box_setting(
        self,
        box_name: str,
        ipaddr_wss: str | IPv4Address | IPv6Address,
        boxtype: Quel1BoxType,
        ipaddr_sss: Optional[str | IPv4Address | IPv6Address] = None,
        ipaddr_css: Optional[str | IPv4Address | IPv6Address] = None,
        config_root: Optional[str | os.PathLike] = None,
        config_options: MutableSequence[Quel1ConfigOption] = [],
    ) -> None:
        self._box_settings[box_name] = BoxSetting(
            box_name=box_name,
            ipaddr_wss=ipaddr_wss,
            boxtype=boxtype,
            ipaddr_sss=ipaddr_sss,
            ipaddr_css=ipaddr_css,
            config_root=config_root,
            config_options=config_options,
        )

    def add_port_setting(
        self,
        port_name: str,
        box_name: str,
        port: int,
        lo_freq: float,
        cnco_freq: float,
        sideband: str,
        vatt: int,
        fnco_freq: Tuple[float] | Tuple[float, float, float],
    ) -> None:
        self._port_settings[port_name] = PortSetting(
            port_name=port_name,
            box_name=box_name,
            port=port,
            lo_freq=lo_freq,
            cnco_freq=cnco_freq,
            sideband=sideband,
            vatt=vatt,
            fnco_freq=fnco_freq,
        )

    def get_channel_by_target(self, target_name: str) -> str:
        return {target: channel for channel, target in self._relation_channel_target}[
            target_name
        ]

    def get_channel_number_by_target(self, target_name: str) -> int:
        channel = self.get_channel_by_target(target_name)
        return int(
            {channel: port for channel, port in self._relation_channel_port}[channel][
                "channel_number"
            ]
        )

    def get_port_by_target(self, target_name: str) -> str:
        channel = self.get_channel_by_target(target_name)
        return str(
            {channel: port for channel, port in self._relation_channel_port}[channel][
                "port_name"
            ]
        )

    def get_port_number_by_target(self, target_name: str) -> int:
        return self._port_settings[self.get_port_by_target(target_name)].port

    def get_box_by_target(self, target_name: str) -> str:
        port = self.get_port_by_target(target_name)
        return self._port_settings[port].box_name
