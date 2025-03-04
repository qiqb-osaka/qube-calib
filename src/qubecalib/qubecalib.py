from __future__ import annotations

import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    MutableSequence,
    Optional,
)

from quel_clock_master import QuBEMasterClient, SequencerClient
from quel_ic_config import (
    QUEL1_BOXTYPE_ALIAS,
    Quel1BoxWithRawWss,
    Quel1ConfigOption,
)

from . import neopulse
from .executor import Executor
from .instrument.quel.quel1 import driver as direct
from .instrument.quel.quel1.command import ConfigPort, RfSwitch
from .instrument.quel.quel1.system import BoxPool
from .instrument.quel.quel1.tool import SkewMonitor
from .sysconfdb import SystemConfigDatabase

logger = logging.getLogger(__name__)
from .base import QubeCalibBase


class Direction(Enum):
    FROM_TARGET = "from_target"
    TO_TARGET = "to_target"


class Sideband(Enum):
    UpperSideBand = "U"
    LowerSideBand = "L"


DEFAULT_SIDEBAND = "U"


class QubeCalib(QubeCalibBase):
    def __init__(
        self,
        path_to_database_file: Optional[str | os.PathLike] = None,
    ) -> None:
        super().__init__()
        self._box_configs: dict[str, dict[str, Any]] = {}

        if path_to_database_file is not None:
            self.system_config_database.load(path_to_database_file)

    @property
    def system_config_database(self) -> SystemConfigDatabase:
        return self._system_config_database

    @property
    def sysdb(self) -> SystemConfigDatabase:
        return self._system_config_database

    def create_quel1system(self, box_names: list[str]) -> direct.Quel1System:
        if self.sysdb._clockmaster_setting is None:
            raise ValueError("clock master is not found")
        system = direct.Quel1System.create(
            clockmaster=QuBEMasterClient(self.sysdb._clockmaster_setting.ipaddr),
            boxes=[self.create_named_box(b) for b in box_names],
        )
        return system

    def create_skew_monitor(
        self,
        box_names: list[str],
        *,
        monitor_port: tuple[str, int] = ("", 0),
        trigger_nport: int = 0,
        reference_port: tuple[str, int] = ("", 0),
    ) -> SkewMonitor:
        system = self.create_quel1system(box_names)
        return SkewMonitor(
            system,
            sysdb=self.sysdb,
            executor=self._executor,
            monitor_port=monitor_port,
            trigger_nport=trigger_nport,
            reference_port=reference_port,
        )

    def execute(self) -> tuple:
        return self._executor.execute()

    def step_execute(
        self,
        repeats: int = 1,
        interval: float = 10240,
        integral_mode: str = "integral",  # "single"
        dsp_demodulation: bool = True,
        software_demodulation: bool = False,
    ) -> Executor:
        return self._executor.step_execute(
            repeats=repeats,
            interval=interval,
            integral_mode=integral_mode,
            dsp_demodulation=dsp_demodulation,
            software_demodulation=software_demodulation,
        )

    def modify_target_frequency(self, target_name: str, frequency: float) -> None:
        self.system_config_database._target_settings[target_name]["frequency"] = (
            frequency
        )

    def add_rfswitch(self, box_name: str, port: int, rfswitch: str) -> None:
        """(block / pass), (loop / open)"""
        self._executor.add_command(RfSwitch(box_name, port, rfswitch))

    def add_sequence(
        self,
        sequence: neopulse.Sequence,
        *,
        interval: Optional[float] = None,
        time_offset: dict[str, int] = {},  # {box_name: time_offset}
        time_to_start: dict[str, int] = {},  # {box_name: time_to_start}
    ) -> None:
        self._executor.add_sequence(
            sequence,
            interval=interval,
            time_offset=time_offset,
            time_to_start=time_to_start,
        )
        # # TODO ここは仕様変更が必要
        # # Readout send に位相合わせ機構を導入するため SebSequence にまとめてしまわず Slot 毎に分割しないといけない
        # # 情報を失わせ過ぎた
        # # capture に関連する gen_sequence を取り出して 変調 slice を作成する
        # gen_sampled_sequence, cap_sampled_sequence = (
        #     sequence.convert_to_sampled_sequence()
        # )
        # # settings = self.system_config_database._target_settings
        # # for target_name, gss in gen_sampled_sequence.items():
        # #     if target_name not in settings:
        # #         raise ValueError(f"target({target_name}) is not defined")
        # #     box_names = self.system_config_database.get_boxes_by_target(target_name)
        # #     if not box_names:
        # #         raise ValueError(f"target({target_name}) is not assigned to any box")
        # #     if len(box_names) > 1:
        # #         raise ValueError(f"target({target_name}) is assigned to multiple boxes")
        # #     # tgtset = settings[target_name]
        # #     # skew = tgtset["skew"] if "skew" in tgtset else 0
        # #     box_name = list(box_names)[0]
        # #     skew = self.sysdb.skew[box_name] if box_name in self.sysdb.skew else 0
        # #     gss.padding += skew

        # items_by_target = sequence._get_group_items_by_target()

        # targets = set(
        #     [gtarget for gtarget in gen_sampled_sequence]
        #     + [ctarget for ctarget in cap_sampled_sequence]
        # )
        # resource_map = self._create_target_resource_map(targets)

        # self._executor.add_command(
        #     Sequencer(
        #         gen_sampled_sequence=gen_sampled_sequence,
        #         cap_sampled_sequence=cap_sampled_sequence,
        #         resource_map=resource_map,
        #         group_items_by_target=items_by_target,
        #         time_offset=time_offset,
        #         time_to_start=time_to_start,
        #         interval=interval,
        #         sysdb=self.system_config_database,
        #     )
        # )

    def add_config_port(
        self,
        box_name: str,
        port: int,
        *,
        subport: int = 0,
        lo_freq: Optional[float] = None,
        cnco_freq: Optional[float] = None,
        cnco_locked_with: Optional[int | tuple[int, int]] = None,
        vatt: Optional[int] = None,
        sideband: Optional[str] = None,
        fullscale_current: Optional[int] = None,
        rfswitch: Optional[str] = None,
    ) -> None:
        p = {
            _.port: _
            for _ in self.system_config_database._port_settings.values()
            if _.box_name == box_name
        }
        p["lo_freq"] = lo_freq
        p["cnco_freq"] = cnco_freq
        p["vatt"] = vatt
        p["sideband"] = sideband
        self._executor.add_command(
            ConfigPort(
                box_name,
                port,
                subport=subport,
                lo_freq=lo_freq,
                cnco_freq=cnco_freq,
                cnco_locked_with=cnco_locked_with,
                vatt=vatt,
                sideband=sideband,
                fullscale_current=fullscale_current,
                rfswitch=rfswitch,
            )
        )

    def define_target(
        self,
        target_name: str,
        channel_name: str,
        target_frequency: Optional[float] = None,
    ) -> None:
        db = self.system_config_database
        db._relation_channel_target.append((channel_name, target_name))
        if target_frequency is None and target_name not in db._target_settings:
            raise ValueError(f"frequency of target({target_name}) is not defined")
        if target_frequency is not None:
            db._target_settings[target_name] = {"frequency": target_frequency}

    def define_clockmaster(
        self,
        ipaddr: str,
        reset: bool,
    ) -> None:
        return self.system_config_database.define_clockmaster(
            ipaddr,
            reset,
        )

    def define_box(
        self,
        box_name: str,
        ipaddr_wss: str,
        boxtype: str,
        ipaddr_sss: Optional[str] = None,
        ipaddr_css: Optional[str] = None,
        config_root: Optional[str] = None,
        config_options: MutableSequence[Quel1ConfigOption] = [],
    ) -> dict[str, Any]:
        return self.system_config_database.define_box(
            box_name=box_name,
            ipaddr_wss=ipaddr_wss,
            boxtype=boxtype,
            config_options=config_options,
            ipaddr_sss=ipaddr_sss,
            ipaddr_css=ipaddr_css,
            config_root=config_root,
        )

    def define_channel(
        self,
        channel_name: str,
        port_name: str,
        channel_number: int,
        ndelay_or_nwait: int = 0,
    ) -> None:
        self.system_config_database.define_channel(
            channel_name=channel_name,
            port_name=port_name,
            channel_number=channel_number,
            ndelay_or_nwait=ndelay_or_nwait,
        )

    def define_port(
        self,
        port_name: str,
        box_name: str,
        port_number: int,
        lo_freq: Optional[float] = None,
        cnco_freq: Optional[float] = None,
        sideband: str = DEFAULT_SIDEBAND,
        vatt: int = 0x800,
        fnco_freq: Optional[
            tuple[float] | tuple[float, float] | tuple[float, float, float]
        ] = None,
    ) -> None:
        self.system_config_database.define_port(
            port_name=port_name,
            box_name=box_name,
            port_number=port_number,
            lo_freq=lo_freq,
            cnco_freq=cnco_freq,
            sideband=sideband,
            vatt=vatt,
            fnco_freq=fnco_freq,
        )

    def get_target_info(self, target_name: str) -> dict:
        return {
            "box_name": self.system_config_database.get_boxes_by_target(
                target_name=target_name
            ),
            "port": self.system_config_database.get_ports_by_target(
                target_name=target_name
            ),
            "channel": self.system_config_database.get_channel_numbers_by_target(
                target_name=target_name
            ),
            "target_frequency": self.system_config_database._target_settings[
                target_name
            ]["frequency"],
        }

    def get_box_names_by_targets(self, *target_names: str) -> set[str]:
        return set(
            sum(
                [
                    list(self.get_target_info(target_name)["box_name"])
                    for target_name in target_names
                ],
                [],
            )
        )

    def get_box_name_by_alias(self, alias: str) -> str:
        return self.system_config_database._box_aliases[alias]

    def create_box(
        self,
        box_name: str,
        reconnect: bool = True,
    ) -> Quel1BoxWithRawWss:
        return self.system_config_database.create_box(
            box_name=box_name,
            reconnect=reconnect,
        )

    def create_named_box(
        self, box_name: str, *, reconnect: bool = True
    ) -> direct.NamedBox:
        return direct.NamedBox(
            name=box_name,
            box=self.create_box(
                box_name,
                reconnect=reconnect,
            ),
        )

    def read_clock(self, *box_names: str) -> MutableSequence[tuple[bool, int, int]]:
        return [
            SequencerClient(
                target_ipaddr=str(
                    self.system_config_database._box_settings[_].ipaddr_sss
                )
            ).read_clock()
            for _ in box_names
        ]

    def resync(
        self, *box_names: str
    ) -> MutableSequence[tuple[bool, int, int] | tuple[bool, int]]:
        db = self.system_config_database
        if db._clockmaster_setting is None:
            raise ValueError("clock master is not found")
        master = QuBEMasterClient(master_ipaddr=str(db._clockmaster_setting.ipaddr))
        master.kick_clock_synch(
            [str(db._box_settings[_].ipaddr_sss) for _ in box_names]
        )
        return [self.read_clock(_) for _ in box_names] + [master.read_clock()]

    def show_available_boxtype(self) -> MutableSequence[str]:
        return [_ for _ in QUEL1_BOXTYPE_ALIAS]

    @classmethod
    def quantize_sequence_duration(
        cls,
        sequence_duration: float,
        constrain: float = 10_240,
    ) -> float:
        return sequence_duration // constrain * constrain

    def get_all_box_configs(self) -> dict[str, dict[str, Any]]:
        return {
            box_name: self.system_config_database.create_box(box_name).dump_box()
            for box_name in self.system_config_database._box_settings
        }

    def store_all_box_configs(self, path_to_config_file: str | os.PathLike) -> None:
        with open(Path(os.getcwd()) / Path(path_to_config_file), "w") as fp:
            json.dump(
                self.get_all_box_configs(),
                fp,
                indent=4,
            )

    def load_all_box_configs(self, path_to_config_file: str | os.PathLike) -> None:
        with open(Path(os.getcwd()) / Path(path_to_config_file), "r") as fp:
            configs = json.load(fp)
        for box_name, _ in configs.items():
            ports: dict[int | tuple[int, int], dict[str, Any]] = {
                int(k): v for k, v in _["ports"].items()
            }
            _["ports"] = ports
            for port, port_config in ports.items():
                if "channels" in port_config:
                    port_config["channels"] = {
                        int(k): v for k, v in port_config["channels"].items()
                    }
                if "runits" in port_config:
                    port_config["runits"] = {
                        int(k): v for k, v in port_config["runits"].items()
                    }
        self._box_configs = configs

    def apply_all_box_configs(self) -> None:
        for box_name in self._box_configs:
            self._apply_box_config(box_name)

    def _apply_box_config(self, box_name: str) -> None:
        box = self.create_box(box_name)
        box.config_box(self._box_configs[box_name]["ports"])

    def apply_box_config(self, *target_names: str) -> set[str]:
        box_names = self.get_box_names_by_targets(*target_names)
        for box_name in box_names:
            self._apply_box_config(box_name)
        return box_names

    def clear_command_queue(self) -> None:
        self._executor._work_queue.clear()

    def show_command_queue(self) -> MutableSequence:
        return self._executor._work_queue

    def create_boxpool(self, *box_names: str) -> BoxPool:
        boxpool = BoxPool()
        if self.system_config_database._clockmaster_setting is not None:
            boxpool.create_clock_master(
                ipaddr=str(self.system_config_database._clockmaster_setting.ipaddr),
            )
        for box_name in box_names:
            if box_name not in self.system_config_database._box_settings:
                raise ValueError(f"box({box_name}) is not defined")
            setting = self.system_config_database._box_settings[box_name]
            box = boxpool.create(
                box_name,
                ipaddr_wss=str(setting.ipaddr_wss),
                ipaddr_sss=str(setting.ipaddr_sss),
                ipaddr_css=str(setting.ipaddr_css),
                boxtype=setting.boxtype,
                config_root=Path(setting.config_root)
                if setting.config_root is not None
                else None,
                config_options=setting.config_options,
            )
            box.reconnect()
        return boxpool
