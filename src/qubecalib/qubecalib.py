from __future__ import annotations

import copy
import datetime
import json
import logging
import os
import pathlib
import pickle
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from ipaddress import IPv4Address, IPv6Address, ip_address
from pathlib import Path
from typing import (
    Any,
    Final,
    Iterable,
    MutableMapping,
    MutableSequence,
    Optional,
    TypedDict,
)

import numpy as np
from e7awgsw import CaptureModule, CaptureParam, DspUnit, WaveSequence
from quel_clock_master import QuBEMasterClient, SequencerClient
from quel_ic_config import (
    QUEL1_BOXTYPE_ALIAS,
    CaptureReturnCode,
    Quel1Box,
    Quel1BoxType,
    Quel1ConfigOption,
)

from . import __version__, neopulse
from .e7utils import (
    CaptureParamTools,
    WaveSequenceTools,
    _convert_gen_sampled_sequence_to_blanks_and_waves_chain,
)
from .general_looptest_common_mod import BoxPool, PulseCap, PulseGen, PulseGen_
from .neopulse import (
    CapSampledSequence,
    GenSampledSequence,
    GenSampledSubSequence,
    SampledSequenceBase,
)

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
        self._box_configs: dict[str, dict[str, Any]] = {}

        if path_to_database_file is not None:
            self.system_config_database.load(path_to_database_file)

    @property
    def system_config_database(self) -> SystemConfigDatabase:
        return self._system_config_database

    def execute(self) -> tuple:
        """queue に登録されている command を実行する（未実装）"""
        return "", "", ""

    def step_execute(
        self,
        repeats: int = 1,
        interval: float = 10240,
        integral_mode: str = "integral",  # "single"
        dsp_demodulation: bool = True,
        software_demodulation: bool = False,
    ) -> Executor:
        """queue に登録されている command を実行する iterator を返す"""
        # work queue を舐めて必要な box を生成する
        boxes = self._executor.collect_boxes()
        # もし box が複数で clockmaster_setting が設定されていれば QuBEMasterClient を生成する
        if (
            len(boxes) > 1
            and self.system_config_database._clockmaster_setting is not None
        ):
            self._executor._boxpool.create_clock_master(
                ipaddr=str(self.system_config_database._clockmaster_setting.ipaddr)
            )
        # boxpool を生成する
        for box_name in boxes:
            setting = self._system_config_database._box_settings[box_name]
            box = self._executor._boxpool.create(
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
            status = box.reconnect()
            for mxfe_idx, s in status.items():
                if not s:
                    logger.error(
                        f"be aware that mxfe-#{mxfe_idx} is not linked-up properly"
                    )
        # sequencer に measurement_option を設定する
        for _ in self._executor.collect_sequencers():
            _.set_measurement_option(
                repeats=repeats,
                interval=interval,
                integral_mode=integral_mode,
                dsp_demodulation=dsp_demodulation,
                software_demodulation=software_demodulation,
            )

        return self._executor

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
    ) -> None:
        gen_sampled_sequence, cap_sampled_sequence = (
            sequence.convert_to_sampled_sequence()
        )
        targets = set(
            [_ for _ in gen_sampled_sequence] + [_ for _ in cap_sampled_sequence]
        )
        resource_map = self._create_target_resource_map(targets)
        self._executor.add_command(
            Sequencer(
                gen_sampled_sequence=gen_sampled_sequence,
                cap_sampled_sequence=cap_sampled_sequence,
                resource_map=resource_map,
            )
        )

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
        sideband: str = "U",
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

    def _create_target_resource_map(
        self,
        target_names: Iterable[str],
    ) -> dict[
        str, Iterable[dict[str, BoxSetting | PortSetting | int | dict[str, Any]]]
    ]:
        # {target_name: sampled_sequence} の形式から
        # TODO {target_name: {box, group, line | rline, channel | runit)} へ変換　？？
        # {target_name: {box, port, channel_number)}} へ変換
        db = self.system_config_database
        targets_channels: MutableSequence[tuple[str, set[str]]] = [
            (target_name, db.get_channels_by_target(target_name))
            for target_name in target_names
        ]
        bpc_targets = {
            target_name: [db.get_channel(_) for _ in channels]
            for target_name, channels in targets_channels
        }
        return {
            target_name: [
                {
                    "box": db._box_settings[box_name],
                    "port": db._port_settings[port_name],
                    "channel_number": channel_number,
                    "target": db._target_settings[target_name],
                }
                for box_name, port_name, channel_number in _
            ]
            for target_name, _ in bpc_targets.items()
        }

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
    ) -> Quel1Box:
        return self.system_config_database.create_box(
            box_name=box_name,
            reconnect=reconnect,
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
                # ignore_crc_error_of_mxfe=args.ignore_crc_error_of_mxfe,
                # ignore_access_failure_of_adrf6780=args.ignore_access_failure_of_adrf6780,
            )
            box.reconnect()
        return boxpool

    def execute_raw_e7(
        self,
        boxpool: BoxPool,
        e7_settings: dict[tuple[str, int, int], WaveSequence | CaptureParam],
        disabled: list[DspUnit] = [],
    ) -> tuple:
        for (box_name, _, _), e7 in e7_settings.items():
            if box_name not in boxpool._boxes:
                raise ValueError(f"box({box_name}) is not defined")
            if not isinstance(e7, WaveSequence) and not isinstance(e7, CaptureParam):
                raise ValueError(f"e7({e7}) is not supported")
        box_names = {box_name for box_name, _, _ in e7_settings}
        pg, pc = PulseGen(boxpool), PulseCap(boxpool)
        gen_e7_settings = {
            (box_name, port, channel): e7
            for (box_name, port, channel), e7 in e7_settings.items()
            if isinstance(e7, WaveSequence)
        }
        for (box_name, port, channel), e7 in gen_e7_settings.items():
            pg.create(
                box_name=box_name,
                port=port,
                channel=channel,
                waveseq=e7,
            )
        cap_e7_settings = {
            (box_name, port, channel): e7
            for (box_name, port, channel), e7 in e7_settings.items()
            if isinstance(e7, CaptureParam)
        }
        for (box_name, port, channel), e7 in cap_e7_settings.items():
            enabled = e7.dsp_units_enabled
            for dsp_unit in disabled:
                if dsp_unit in enabled:
                    enabled.remove(dsp_unit)
            e7_copied = copy.deepcopy(e7)
            e7_copied.sel_dsp_units_to_enable(*enabled)
            pc.create(
                box_name=box_name,
                port=port,
                channel=channel,
                capprm=e7_copied,
            )
        for o in list(pg.pulsegens) + list(pc.pulsecaps):
            o.init()

        box_names = {box_name for (box_name, _, _), _ in e7_settings.items()}
        box_configs = {
            box_name: boxpool.get_box(box_name)[0].dump_box() for box_name in box_names
        }
        if pc.pulsecaps:
            capmod_by_port_channel = {
                (p.box_name, p.port, p.channel): (p.box_name, p.capmod)
                for p in pc.pulsecaps
            }
        if not pg.pulsegens and pc.pulsecaps:  # case 2.
            # 機体数に関わらず caps のみ設定されているなら非同期キャプチャする
            status, iqs = pc.capture_now()
            return (
                status,
                iqs,
                {
                    "cap_e7_settings": cap_e7_settings,
                    "capmod_by_port_channel": capmod_by_port_channel,
                    "box_configs": box_configs,
                    "debug": "case 2: capture_now",
                },
            )
        elif len(box_names) == 1 and pg.pulsegens and pc.pulsecaps:  # case 1.
            # caps 及び gens が共に設定されていて機体数が 1 台のみなら clock 系をバイパス
            # trigger を設定
            triggering_pgs = Sequencer.create_triggering_pgs(pg, pc)
            futures = pc.capture_at_trigger_of(triggering_pgs)
            pg.emit_now()
            status, iqs = pc.wait_until_capture_finishes(futures)
            return (
                status,
                iqs,
                {
                    "cap_e7_settings": cap_e7_settings,
                    "gen_e7_settings": gen_e7_settings,
                    "capmod_by_port_channel": capmod_by_port_channel,
                    "box_configs": box_configs,
                    "debug": "case 1: capture_at_trigger_of, emit_now",
                },
            )
        elif len(box_names) == 1 and pg.pulsegens and not pc.pulsecaps:  # case 3.
            # gens のみ設定されていて機体数が 1 台のみなら clock 系をバイパスして同期出力する
            pg.emit_now()
            return (
                {},
                {},
                {
                    "gen_e7_settings": gen_e7_settings,
                    "box_configs": box_configs,
                    "debug": "case 3: emit_now",
                },
            )
        elif len(box_names) != 1 and pg.pulsegens and not pc.pulsecaps:  # case 5.
            # gens のみ設定されていて機体数が複数，clockmaster があるなら clock を経由して同期出力する
            if boxpool._clock_master is None:
                pg.emit_now()
                return (
                    {},
                    {},
                    {
                        "gen_e7_settings": gen_e7_settings,
                        "box_configs": box_configs,
                        "debug": "case 5: emit_now",
                    },
                )
            else:
                pg.emit_at()
                return (
                    {},
                    {},
                    {
                        "gen_e7_settings": gen_e7_settings,
                        "box_configs": box_configs,
                        "debug": "case 5: emit_at",
                    },
                )
        elif len(box_names) != 1 and pg.pulsegens and pc.pulsecaps:  # case 4.
            # caps 及び gens が共に設定されていて機体数が複数，clockmaster があるなら clock を経由して同期出力する
            if boxpool._clock_master is None:
                triggering_pgs = Sequencer.create_triggering_pgs(pg, pc)
                futures = pc.capture_at_trigger_of(triggering_pgs)
                pg.emit_now()
                status, iqs = pc.wait_until_capture_finishes(futures)
                return (
                    status,
                    iqs,
                    {
                        "cap_e7_settings": cap_e7_settings,
                        "gen_e7_settings": gen_e7_settings,
                        "capmod_by_port_channel": capmod_by_port_channel,
                        "box_configs": box_configs,
                        "debug": "case 4: capture_at, emit_now",
                    },
                )
            else:
                triggering_pgs = Sequencer.create_triggering_pgs(pg, pc)
                futures = pc.capture_at_trigger_of(triggering_pgs)
                pg.emit_at()
                status, iqs = pc.wait_until_capture_finishes(futures)
                return (
                    status,
                    iqs,
                    {
                        "cap_e7_settings": cap_e7_settings,
                        "gen_e7_settings": gen_e7_settings,
                        "capmod_by_port_channel": capmod_by_port_channel,
                        "box_configs": box_configs,
                        "debug": "case 4: capture_at, emit_at",
                    },
                )
        else:
            raise ValueError("this setting is not supported yet")

    def pickle_log(self, path_to_log_file: str | os.PathLike) -> None:
        mypath = pathlib.Path(path_to_log_file)
        with open(mypath, "wb") as fp:
            fp.write(pickle.dumps(self._executor.get_log()))

    def unpickle_log(self, path_to_log_file: str | os.PathLike) -> list:
        mypath = pathlib.Path(path_to_log_file)
        with open(mypath, "rb") as fp:
            log = pickle.load(fp)
        return log


class Converter:
    @classmethod
    def convert_to_device_specific_sequence(
        cls,
        sampled_sequence: dict[str, SampledSequenceBase],
        resource_map: dict[str, dict[str, BoxSetting | PortSetting | int]],
        port_config: dict[str, PortConfigAcquirer],
        repeats: int,
        interval: float,
        integral_mode: str,
        dsp_demodulation: bool,
        software_demodulation: bool,
    ) -> dict[tuple[str, int, int], WaveSequence | CaptureParam]:
        # sampled_sequence と resource_map から e7 データを生成する
        # gen と cap を分離する
        capseq = cls.convert_to_cap_device_specific_sequence(
            sampled_sequence={
                target_name: sseq
                for target_name, sseq in sampled_sequence.items()
                if isinstance(sseq, CapSampledSequence)
            },
            resource_map={
                target_name: _
                for target_name, _ in resource_map.items()
                if isinstance(sampled_sequence[target_name], CapSampledSequence)
            },
            port_config={
                target_name: _
                for target_name, _ in port_config.items()
                if isinstance(sampled_sequence[target_name], CapSampledSequence)
            },
            repeats=repeats,
            interval=interval,
            integral_mode=integral_mode,
            dsp_demodulation=dsp_demodulation,
            software_demodulation=software_demodulation,
        )
        genseq = cls.convert_to_gen_device_specific_sequence(
            sampled_sequence={
                target_name: sseq
                for target_name, sseq in sampled_sequence.items()
                if isinstance(sseq, GenSampledSequence)
            },
            resource_map={
                target_name: _
                for target_name, _ in resource_map.items()
                if isinstance(sampled_sequence[target_name], GenSampledSequence)
            },
            port_config={
                target_name: _
                for target_name, _ in port_config.items()
                if isinstance(sampled_sequence[target_name], GenSampledSequence)
            },
            repeats=repeats,
            interval=interval,
        )
        return genseq | capseq

    @classmethod
    def convert_to_cap_device_specific_sequence(
        cls,
        sampled_sequence: dict[str, CapSampledSequence],
        resource_map: dict[str, dict[str, BoxSetting | PortSetting | int]],
        port_config: dict[str, PortConfigAcquirer],
        repeats: int,
        interval: float,
        integral_mode: str,
        dsp_demodulation: bool,
        software_demodulation: bool,
    ) -> dict[tuple[str, int, int], CaptureParam]:
        ndelay_or_nwait_by_target = {
            target_name: _["port"].ndelay_or_nwait[_["channel_number"]]
            if _["port"].ndelay_or_nwait is not None
            else 0
            for target_name, _ in resource_map.items()
        }
        # CaptureParam の生成
        # target 毎の変調周波数の計算
        targets_freqs: MutableMapping[str, float] = {
            target_name: cls.calc_modulation_frequency(
                f_target=_["target"]["frequency"],
                port_config=port_config[target_name],
            )
            for target_name, _ in resource_map.items()
        }
        # target_name と (box_name, port_number, channel_number) のマップを作成する
        # 1:1 の対応を仮定
        targets_ids = {
            target_name: (_["box"].box_name, _["port"].port, _["channel_number"])
            for target_name, _ in resource_map.items()
        }
        ids_targets = {id: target_name for target_name, id in targets_ids.items()}
        if len(targets_ids) != len(ids_targets):
            raise ValueError("multiple targets are assigned.")

        # ハードウェア復調の場合 channel (unit) に対して単一の target を仮定する
        # ソフトウェア復調の場合は channel 毎にデータを束ねる必要がある TODO 後日実装
        # cap channel と target が 1:1 で対応しているか確認
        if not all(
            [
                _ == 1
                for _ in Counter([targets_ids[_] for _ in sampled_sequence]).values()
            ]
        ):
            raise ValueError(
                "multiple access for single runit will be supported, not now"
            )
        # 戻り値は {(box_name, port_number, channel_number): CaptureParam} の dict
        ids_e7 = {
            targets_ids[_.target_name]: CaptureParamTools.create(
                sequence=_,
                capture_delay_words=ndelay_or_nwait_by_target[_.target_name] * 16,
                repeats=repeats,
                interval_samples=int(interval / _.sampling_period),  # samples
            )
            for _ in sampled_sequence.values()
        }
        if integral_mode == "integral":
            ids_e7 = {
                id: CaptureParamTools.enable_integration(capprm=e7)
                for id, e7 in ids_e7.items()
            }
        if dsp_demodulation:
            ids_e7 = {
                id: CaptureParamTools.enable_demodulation(
                    capprm=e7,
                    f_GHz=targets_freqs[ids_targets[id]],
                )
                for id, e7 in ids_e7.items()
            }

        return ids_e7

    @classmethod
    def convert_to_gen_device_specific_sequence(
        cls,
        sampled_sequence: dict[str, GenSampledSequence],
        resource_map: dict[str, dict[str, BoxSetting | PortSetting | int]],
        port_config: dict[str, PortConfigAcquirer],
        repeats: int,
        interval: float,
    ) -> dict[tuple[str, int, int], WaveSequence]:
        # WaveSequence の生成
        # target 毎の変調周波数の計算
        targets_freqs = {
            target_name: cls.calc_modulation_frequency(
                f_target=_["target"]["frequency"],
                port_config=port_config[target_name],
            )
            for target_name, _ in resource_map.items()
        }
        # channel 毎 (awg 毎) にデータを束ねる
        # target_name と (box_name, port_number, channel_number) のマップを作成する
        targets_ids = {
            target_name: (_["box"].box_name, _["port"].port, _["channel_number"])
            for target_name, _ in resource_map.items()
        }
        ndelay_or_nwait_by_id = {
            id: next(
                iter(
                    {
                        _["port"].ndelay_or_nwait[_["channel_number"]]
                        for _ in resource_map.values()
                        if _["box"].box_name == id[0]
                        and _["port"].port == id[1]
                        and _["channel_number"] == id[2]
                    }
                )
            )
            for id in targets_ids.values()
        }
        # (box_name, port_number, channel_number) と {target_name: sampled_sequence} とのマップを作成する
        ids_sampled_sequences = {
            id: {
                _: sampled_sequence[_]
                for _, _id in targets_ids.items()
                if _id == id and _ in sampled_sequence
            }
            for id in targets_ids.values()
        }
        # (box_name, port_number, channel_number) と {target_name: modfreq} とのマップを作成する
        ids_modfreqs = {
            id: {_: targets_freqs[_] for _, _id in targets_ids.items() if _id == id}
            for id in targets_ids.values()
        }

        # channel 毎に WaveSequence を生成する
        # 戻り値は {(box_name, port_number, channel_number): WaveSequence} の dict
        # 周波数多重した sampled_sequence を作成する
        ids_muxed_sequences = {
            id: cls.multiplex(
                sequences=ids_sampled_sequences[id],
                modfreqs=ids_modfreqs[id],
            )
            for id in ids_sampled_sequences
        }
        return {
            id: WaveSequenceTools.create(
                sequence=ids_muxed_sequences[id],
                # wait_words=wait_words,
                wait_words=ndelay_or_nwait_by_id[id],
                repeats=repeats,
                interval_samples=int(
                    interval / ids_muxed_sequences[id].sampling_period
                ),
            )
            for id in ids_sampled_sequences
        }

    @classmethod
    def calc_modulation_frequency(
        cls,
        f_target: float,
        port_config: PortConfigAcquirer,
    ) -> float:
        """
        Calculate modulation frequency from target frequency and port configuration.

        Parameters
        ----------
        f_target : float
            Target frequency in GHz.
        port_config : PortConfigAcquirer
            Port configuration.

        Returns
        -------
        float
            Modulation frequency in GHz.
        """
        # Note that port_config has frequencies in Hz.
        f_lo = port_config.lo_freq * 1e-9  # Hz -> GHz
        f_cnco = port_config.cnco_freq * 1e-9  # Hz -> GHz
        f_fnco = port_config.fnco_freq * 1e-9  # Hz -> GHz
        sideband = port_config.sideband
        if sideband == Sideband.UpperSideBand.value:
            f_diff = f_target - f_lo - (f_cnco + f_fnco)
        elif sideband == Sideband.LowerSideBand.value:
            f_diff = -(f_target - f_lo) - (f_cnco + f_fnco)
        else:
            raise ValueError("invalid ssb mode")

        return f_diff  # GHz

    @classmethod
    def multiplex(
        cls,
        sequences: MutableMapping[str, GenSampledSequence],
        modfreqs: MutableMapping[str, float],
    ) -> GenSampledSequence:
        cls.validate_geometry_identity(sequences)
        if not cls.validate_geometry_identity(sequences):
            raise ValueError(
                "All geometry of sub sequences belonging to the same awg must be equal"
            )
        sequence = sequences[next(iter(sequences))]
        chain = {
            target_name: _convert_gen_sampled_sequence_to_blanks_and_waves_chain(_)
            for target_name, _ in sequences.items()
        }
        begins = {
            target_name: [
                sum(chain[target_name][: i + 1])
                for i, _ in enumerate(chain[target_name][1:])
            ]
            for target_name, _ in sequences.items()
        }
        times = {
            target_name: [
                (begin + np.arange(subseq.real.shape[0])) * sequence.sampling_period
                for begin, subseq in zip(
                    begins[target_name][::2], sequence.sub_sequences
                )
            ]
            for target_name, sequence in sequences.items()
        }
        return GenSampledSequence(
            target_name="",
            prev_blank=sequence.prev_blank,
            post_blank=sequence.post_blank,
            repeats=sequence.repeats,
            sampling_period=sequence.sampling_period,
            sub_sequences=[
                GenSampledSubSequence(
                    real=np.array(
                        [
                            np.real(
                                (
                                    sequences[_].sub_sequences[i].real
                                    + 1j * sequences[_].sub_sequences[i].imag
                                )
                                * np.exp(1j * 2 * np.pi * (modfreqs[_] * times[_][i]))
                            )
                            for _ in sequences
                        ]
                    ).sum(axis=0),
                    imag=np.array(
                        [
                            np.imag(
                                (
                                    sequences[_].sub_sequences[i].real
                                    + 1j * sequences[_].sub_sequences[i].imag
                                )
                                * np.exp(1j * 2 * np.pi * (modfreqs[_] * times[_][i]))
                            )
                            for _ in sequences
                        ]
                    ).sum(axis=0),
                    post_blank=subseq.post_blank,
                    repeats=subseq.repeats,
                )
                for i, subseq in enumerate(sequence.sub_sequences)
            ],
        )

    @classmethod
    def validate_geometry_identity(
        cls,
        sequences: MutableMapping[str, GenSampledSequence],
    ) -> bool:
        # 同一 awg から出すすべての信号の GenSampledSequence の波形構造が同一であることを確認する
        _ = {
            target_name: [
                (_.real.shape, _.imag.shape, _.post_blank, _.repeats)
                for _ in sequence.sub_sequences
            ]
            for target_name, sequence in sequences.items()
        }
        if not all(
            sum(
                [[geometry[0] == __ for __ in geometry] for _, geometry in _.items()],
                [],
            )
        ):
            return False
        else:
            return True


class Command:
    def execute(
        self,
        boxpool: BoxPool,
    ) -> Any:
        pass


class TargetBPC(TypedDict):
    box: Quel1Box
    port: int | tuple[int, int]
    channel: int


class PortConfigAcquirer:
    def __init__(
        self,
        box: Quel1Box,
        port: int | tuple[int, int],
        channel: int,
    ):
        self.dump_config = dp = box.dump_port(port)
        fnco_freq = 0
        if "channels" in dp:
            fnco_freq = dp["channels"][channel]["fnco_freq"]
        if "runits" in dp:
            fnco_freq = dp["runits"][channel]["fnco_freq"]
        sideband = dp["sideband"] if "sideband" in dp else "U"
        self.lo_freq: float = dp["lo_freq"]
        self.cnco_freq: float = dp["cnco_freq"]
        self.fnco_freq: float = fnco_freq
        self.sideband: str = sideband

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(lo_freq={self.lo_freq}, cnco_freq={self.cnco_freq}, fnco_freq={self.fnco_freq}, sideband={self.sideband})"


class RfSwitch(Command):
    def __init__(self, box_name: str, port: int, rfswitch: str):
        self._box_name = box_name
        self._port = port
        self._rfswitch = rfswitch

    def execute(
        self,
        boxpool: BoxPool,
    ) -> None:
        box = boxpool.get_box(self._box_name)[0]
        box.config_rfswitch(self._port, rfswitch=self._rfswitch)


class Sequencer(Command):
    def __init__(
        self,
        gen_sampled_sequence: dict[str, GenSampledSequence],
        cap_sampled_sequence: dict[str, CapSampledSequence],
        resource_map: dict[
            str, Iterable[dict[str, BoxSetting | PortSetting | int | dict[str, Any]]]
        ],
    ):
        self.gen_sampled_sequence = gen_sampled_sequence
        self.cap_sampled_sequence = cap_sampled_sequence
        self.resource_map = resource_map
        # resource_map は以下の形式
        # {
        #   "box": db._box_settings[box_name],
        #   "port": db._port_settings[port_name],
        #   "channel_number": channel_number,
        #   "target": db._target_settings[target_name],
        # }

    def set_measurement_option(
        self,
        repeats: int,
        interval: float,
        integral_mode: str,
        dsp_demodulation: bool,
        software_demodulation: bool,
    ) -> None:
        self.repeats = repeats
        self.interval = interval
        self.integral_mode = integral_mode
        self.dsp_demodulation = dsp_demodulation
        self.software_demodulation = software_demodulation

    def execute(
        self,
        boxpool: BoxPool,
    ) -> tuple[dict[str, CaptureReturnCode], dict[str, list], dict]:
        # cap 用の cap_e7_setting と gen 用の gen_e7setting を作る

        _cap_resource_map: dict[str, MutableSequence[dict[str, Any]]] = {}
        for target_name, ms in self.resource_map.items():
            for m in ms:
                if isinstance(m["box"], BoxSetting):
                    box_name = m["box"].box_name
                else:
                    raise ValueError("box_name is not defined")
                if isinstance(m["port"], PortSetting):
                    port = m["port"].port
                else:
                    raise ValueError("port is not defined")
                if (
                    boxpool.get_box(box_name)[0].dump_port(port)["direction"] == "in"
                    and target_name in self.cap_sampled_sequence
                ):
                    if target_name in _cap_resource_map:
                        _cap_resource_map[target_name].append(m)
                    else:
                        _cap_resource_map[target_name] = [m]
        cap_resource_map: dict[str, Any] = {
            target_name: next(iter(maps))
            for target_name, maps in _cap_resource_map.items()
            if maps
        }
        _gen_resource_map: dict[str, MutableSequence[dict[str, Any]]] = {}
        for target_name, ms in self.resource_map.items():
            for m in ms:
                if isinstance(m["box"], BoxSetting):
                    box_name = m["box"].box_name
                else:
                    raise ValueError("box_name is not defined")
                if isinstance(m["port"], PortSetting):
                    port = m["port"].port
                else:
                    raise ValueError("port is not defined")
                if (
                    boxpool.get_box(box_name)[0].dump_port(port)["direction"] == "out"
                    and target_name in self.gen_sampled_sequence
                ):
                    if target_name in _gen_resource_map:
                        _gen_resource_map[target_name].append(m)
                    else:
                        _gen_resource_map[target_name] = [m]
        gen_resource_map: dict[str, Any] = {
            target_name: next(iter(maps))
            for target_name, maps in _gen_resource_map.items()
            if maps
        }

        # TODO ここで caps や gens が二つ以上だとエラーを出すこと
        # e7 の生成に必要な lo_hz などをまとめた辞書を作る
        cap_target_bpc: dict[str, TargetBPC] = {
            target_name: TargetBPC(
                box=boxpool.get_box(m["box"].box_name)[0],
                port=m["port"].port if isinstance(m["port"], PortSetting) else 0,
                channel=m["channel_number"],
            )
            for target_name, m in cap_resource_map.items()
        }
        gen_target_bpc: dict[str, TargetBPC] = {
            target_name: TargetBPC(
                box=boxpool.get_box(m["box"].box_name)[0],
                port=m["port"].port if isinstance(m["port"], PortSetting) else 0,
                channel=m["channel_number"],
            )
            for target_name, m in gen_resource_map.items()
        }
        cap_target_portconf = {
            target_name: PortConfigAcquirer(
                box=m["box"], port=m["port"], channel=m["channel"]
            )
            for target_name, m in cap_target_bpc.items()
        }
        cap_e7_settings: dict[tuple[str, int, int], CaptureParam] = (
            Converter.convert_to_cap_device_specific_sequence(
                sampled_sequence=self.cap_sampled_sequence,
                resource_map=cap_resource_map,
                # target_freq=target_freq,
                port_config=cap_target_portconf,
                repeats=self.repeats,
                interval=self.interval,
                integral_mode=self.integral_mode,
                dsp_demodulation=self.dsp_demodulation,
                software_demodulation=self.software_demodulation,
            )
        )

        gen_target_portconf = {
            target_name: PortConfigAcquirer(
                box=m["box"], port=m["port"], channel=m["channel"]
            )
            for target_name, m in gen_target_bpc.items()
        }
        gen_e7_settings: dict[tuple[str, int, int], WaveSequence] = (
            Converter.convert_to_gen_device_specific_sequence(
                sampled_sequence=self.gen_sampled_sequence,
                resource_map=gen_resource_map,
                port_config=gen_target_portconf,
                repeats=self.repeats,
                interval=self.interval,
            )
        )

        pc = PulseCap(boxpool)
        pg = PulseGen(boxpool)
        for (box_name, port, channel), e7 in cap_e7_settings.items():
            pc.create(
                box_name=box_name,
                port=port,
                channel=channel,
                capprm=e7,
            )
        for (box_name, port, channel), e7 in gen_e7_settings.items():
            pg.create(
                box_name=box_name,
                port=port,
                channel=channel,
                waveseq=e7,
            )
        for o in list(pg.pulsegens) + list(pc.pulsecaps):
            o.init()
        bpc_capmod = {(_.box_name, _.port, _.channel): _.capmod for _ in pc.pulsecaps}
        bmc_target: dict[tuple[Optional[str], CaptureModule, Optional[int]], str] = {
            (
                m["box"].box_name if isinstance(m["box"], BoxSetting) else None,
                bpc_capmod[
                    (
                        m["box"].box_name if isinstance(m["box"], BoxSetting) else None,
                        m["port"].port if isinstance(m["port"], PortSetting) else None,
                        m["channel_number"],
                    )
                ],
                m["channel_number"] if isinstance(m["channel_number"], int) else None,
            ): target_name
            for target_name, m in cap_resource_map.items()
        }

        box_names = {box_name for (box_name, _, _), _ in cap_e7_settings.items()}
        box_names |= {box_name for (box_name, _, _), _ in gen_e7_settings.items()}
        box_configs = {
            box_name: boxpool.get_box(box_name)[0].dump_box() for box_name in box_names
        }
        # TODO CW 出力については box の機能を使うのが良さげ
        # 制御方式の自動選択
        # TODO caps 及び gens が共に設定されていて機体数が複数台なら clock 系を使用
        # TODO ~~single and caps and gens: capture_at(), emit_now() 1.~~
        # TODO ~~single and caps and not gens: capture_now() 2.~~
        # TODO ~~single and not caps and gens: emit_now() 3.~~
        # TODO ~~single and not caps and not gens X~~
        # TODO not single and caps and gens: capture_at(), emit_at() 4.
        # TODO ~~not single and caps and not gens: capture_now() 2.~~
        # TODO not single and not caps and gens: emit_at() 5.
        # TODO ~~not single and not caps and not gens X~~
        if not pg.pulsegens and pc.pulsecaps:  # case 2.
            # 機体数に関わらず caps のみ設定されているなら非同期キャプチャする
            _status, _iqs = pc.capture_now()
            status, iqs = self.convert_key_from_bmu_to_target(bmc_target, _status, _iqs)
            return (status, iqs) + (
                {
                    "cap_e7_settings": cap_e7_settings,
                    "box_configs": box_configs,
                    "cap_sampled_sequence": self.cap_sampled_sequence,
                    "debug": "case 2: catpure_now",
                },
            )
        elif len(box_names) == 1 and pg.pulsegens and pc.pulsecaps:  # case 1.
            # caps 及び gens が共に設定されていて機体数が 1 台のみなら clock 系をバイパス
            # trigger を設定
            triggering_pgs = self.create_triggering_pgs(pg, pc)
            futures = pc.capture_at_trigger_of(triggering_pgs)
            pg.emit_now()
            _status, _iqs = pc.wait_until_capture_finishes(futures)
            status, iqs = self.convert_key_from_bmu_to_target(bmc_target, _status, _iqs)
            return (status, iqs) + (
                {
                    "cap_e7_settings": cap_e7_settings,
                    "cap_sampled_sequence": self.cap_sampled_sequence,
                    "gen_e7_settings": gen_e7_settings,
                    "gen_sampled_sequence": self.gen_sampled_sequence,
                    "box_configs": box_configs,
                    "debug": "case 1: capture_at_trigger_of, emit_now",
                },
            )
        elif len(box_names) == 1 and pg.pulsegens and not pc.pulsecaps:  # case 3.
            # gens のみ設定されていて機体数が 1 台のみなら clock 系をバイパスして同期出力する
            pg.emit_now()
            return (
                {},
                {},
                {
                    "gen_e7_settings": gen_e7_settings,
                    "gen_sampled_sequence": self.gen_sampled_sequence,
                    "box_configs": box_configs,
                    "debug": "case 3: emit_now",
                },
            )
        elif len(box_names) != 1 and pg.pulsegens and not pc.pulsecaps:  # case 5.
            # gens のみ設定されていて機体数が複数，clockmaster があるなら clock を経由して同期出力する
            if boxpool._clock_master is None:
                pg.emit_now()
                return (
                    {},
                    {},
                    {
                        "gen_e7_settings": gen_e7_settings,
                        "gen_sampled_sequence": self.gen_sampled_sequence,
                        "box_configs": box_configs,
                        "debug": "case 5: emit_now",
                    },
                )
            else:
                pg.emit_at()
                return (
                    {},
                    {},
                    {
                        "gen_e7_settings": gen_e7_settings,
                        "gen_sampled_sequence": self.gen_sampled_sequence,
                        "box_configs": box_configs,
                        "debug": "case 5: emit_at",
                    },
                )
        elif len(box_names) != 1 and pg.pulsegens and pc.pulsecaps:  # case 4.
            # caps 及び gens が共に設定されていて機体数が複数，clockmaster があるなら clock を経由して同期出力する
            if boxpool._clock_master is None:
                triggering_pgs = self.create_triggering_pgs(pg, pc)
                futures = pc.capture_at_trigger_of(triggering_pgs)
                pg.emit_now()
                _status, _iqs = pc.wait_until_capture_finishes(futures)
                status, iqs = self.convert_key_from_bmu_to_target(
                    bmc_target, _status, _iqs
                )
                return (status, iqs) + (
                    {
                        "gen_e7_settings": gen_e7_settings,
                        "gen_sampled_sequence": self.gen_sampled_sequence,
                        "cap_e7_settings": cap_e7_settings,
                        "cap_sampled_sequence": self.cap_sampled_sequence,
                        "box_configs": box_configs,
                        "debug": "case 4: capture_at_trigger_of, emit_now",
                    },
                )
            else:
                triggering_pgs = self.create_triggering_pgs(pg, pc)
                futures = pc.capture_at_trigger_of(triggering_pgs)
                pg.emit_at()
                _status, _iqs = pc.wait_until_capture_finishes(futures)
                status, iqs = self.convert_key_from_bmu_to_target(
                    bmc_target, _status, _iqs
                )
                return (status, iqs) + (
                    {
                        "gen_e7_settings": gen_e7_settings,
                        "gen_sampled_sequence": self.gen_sampled_sequence,
                        "cap_e7_settings": cap_e7_settings,
                        "cap_sampled_sequence": self.cap_sampled_sequence,
                        "box_configs": box_configs,
                        "debug": "case 4: capture_at_trigger_of, emit_at",
                    },
                )
        else:
            raise ValueError("this setting is not supported yet")

    @classmethod
    def convert_key_from_bmu_to_target(
        cls,
        bmc_target: dict[tuple[Optional[str], CaptureModule, Optional[int]], str],
        status: dict[tuple[str, CaptureModule], CaptureReturnCode],
        iqs: dict[tuple[str, CaptureModule], dict[int, list]],
    ) -> tuple[dict[str, CaptureReturnCode], dict[str, list]]:
        _iqs = {
            bmc_target[(box_name, capm, capu)]: __iqs
            for (box_name, capm), _ in iqs.items()
            for capu, __iqs in _.items()
        }
        _status = {
            bmc_target[(box_name, capm, capu)]: status[(box_name, capm)]
            for (box_name, capm), _ in iqs.items()
            for capu in _
        }

        # sort keys of iqs by target name
        sorted_iqs = {key: _iqs[key] for key in sorted(_iqs)}

        return _status, sorted_iqs

    @classmethod
    def create_triggering_pgs(
        cls,
        pg: PulseGen,
        pc: PulseCap,
    ) -> dict[tuple[Any, Any], PulseGen_]:  # (box_name, capmod): PulseGen_
        # (box_names, capmod) 毎に同一グループの trigger の何れかをを割り当てる
        # 同一グループの awg が無ければエラーを返す
        cap = {(_.box_name, _.group, _.capmod) for _ in pc.pulsecaps}
        gen = {(_.box_name, _.group, _) for _ in pg.pulsegens}
        triggering_pgs_ = {
            (cap_[0], cap_[2]): {gen_[2] for gen_ in gen if cap_[:2] == gen_[:2]}
            for cap_ in cap
        }
        if not all([bool(pgs) for pgs in triggering_pgs_.values()]):
            # TODO 別の group の trigger を割り当てるようにチャレンジする
            raise ValueError("invalid trigger")
        triggering_pgs = {
            (box_name, capmod): next(iter(gens))
            for (box_name, capmod), gens in triggering_pgs_.items()
        }
        return triggering_pgs


class Configurator(Command):
    def execute(
        self,
        boxpool: BoxPool,
    ) -> None:
        print(f"{self.__class__.__name__} executed")


class ConfigPort(Command):
    def __init__(
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
        self.box_name = box_name
        self.port = port
        self.subport = subport
        self.lo_freq = lo_freq
        self.cnco_freq = cnco_freq
        self.cnco_locked_with = cnco_locked_with
        self.vatt = vatt
        self.sideband = sideband
        self.fullscale_current = fullscale_current
        self.rfswitch = rfswitch

    def execute(
        self,
        boxpool: BoxPool,
    ) -> None:
        print(f"{self.__class__.__name__} executed")
        # box = boxpool(self.box_name)
        # box.config_port()


class ConfigChannel(Command):
    def __init__(
        self,
        box_name: str,
        port: int,
        channel: int,
        *,
        subport: int = 0,
        fnco_freq: Optional[float] = None,
    ):
        self.box_name = box_name
        self.port = port
        self.channel = channel
        self.subport = subport
        self.fnco_freq = fnco_freq

    def execute(
        self,
        boxpool: BoxPool,
    ) -> None:
        print(f"{self.__class__.__name__} executed")
        # box = boxpool(self.box_name)
        # box.config_port()


class Executor:
    def __init__(self) -> None:
        self._work_queue: Final[deque] = deque()
        self._boxpool: BoxPool = BoxPool()
        self._config_buffer: Final[deque] = deque()

    def reset(self) -> None:
        self._work_queue.clear()
        self._boxpool = BoxPool()

    def collect_boxes(self) -> set[Any]:
        return set(
            sum(
                [
                    [
                        __["box"].box_name
                        for _ in command.resource_map.values()
                        for __ in _
                    ]
                    for command in self._work_queue
                    if isinstance(command, Sequencer)
                ],
                [],
            )
        )

    def collect_sequencers(self) -> set[Sequencer]:
        return {_ for _ in self._work_queue if isinstance(_, Sequencer)}

    def __iter__(self) -> Executor:
        # if not self._work_queue:
        #     return self
        # last_command = self._work_queue[-1]
        # if not isinstance(last_command, Sequencer):
        #     raise ValueError("_work_queue should end with a Sequencer command")
        self.clear_log()  # clear config for last execution
        return self

    def __next__(self) -> tuple[Any, dict, dict]:
        # ワークキューが空になったら実行を止める
        if not self._work_queue:
            raise StopIteration()
        # Sequencer が見つかるまでコマンドを逐次実行
        while True:
            # もしワークキューが空になったらエラーを出す
            if not self._work_queue:
                raise ValueError(
                    "command que should include at least one Sequencer command."
                )
            next = self._work_queue.pop()
            # 次に実行するコマンドが Sequencer ならばループを抜ける
            if isinstance(next, Sequencer):
                break
            # Sequencer 以外のコマンドを逐次実行
            next.execute(self._boxpool)
        for command in self._work_queue:
            # もしコマンドキューに Sequencer が残っていれば次の Sequencer を実行する
            if isinstance(command, Sequencer):
                status, iqs, config = next.execute(self._boxpool)
                user_name = os.getlogin()
                current_pyfile = os.path.abspath(__file__)
                date_time = datetime.datetime.now()
                clock_ns = time.clock_gettime_ns(time.CLOCK_REALTIME)
                self._config_buffer.append(
                    (
                        config,
                        user_name,
                        current_pyfile,
                        __version__,
                        date_time,
                        clock_ns,
                    )
                )
                return status, iqs, config
        # これ以上 Sequencer がなければ残りのコマンドを実行する
        status, iqs, config = next.execute(self._boxpool)
        user_name = os.getlogin()
        current_pyfile = os.path.abspath(__file__)
        date_time = datetime.datetime.now()
        clock_ns = time.clock_gettime_ns(time.CLOCK_REALTIME)
        self._config_buffer.append(
            (
                config,
                user_name,
                current_pyfile,
                __version__,
                date_time,
                clock_ns,
            )
        )
        for command in self._work_queue:
            command.execute(self._boxpool)
        return status, iqs, config

    def add_command(self, command: Command) -> None:
        self._work_queue.appendleft(command)

    def get_log(self) -> list:
        return list(self._config_buffer)

    def clear_log(self) -> None:
        self._config_buffer.clear()


@dataclass
class ClockmasterSetting:
    ipaddr: str | IPv4Address | IPv6Address
    reset: bool

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


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

        self.config_options = []

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

    def asjsonable(self) -> dict[str, Any]:
        dct = self.asdict()
        dct["boxtype"] = {v: k for k, v in QUEL1_BOXTYPE_ALIAS.items()}[dct["boxtype"]]
        return dct


@dataclass
class PortSetting:
    port_name: str
    box_name: str
    port: int
    lo_freq: Optional[float] = None  # will be obsolete
    cnco_freq: Optional[float] = None  # will be obsolete
    sideband: str = "U"  # will be obsolete
    vatt: int = 0x800  # will be obsolete
    fnco_freq: Optional[tuple[float, ...]] = None  # will be obsolete
    ndelay_or_nwait: tuple[int, ...] = ()

    def asdict(self) -> dict[str, Any]:
        return {
            "port_name": self.port_name,
            "box_name": self.box_name,
            "port": self.port,
            "ndelay_or_nwait": self.ndelay_or_nwait,
        }


class SystemConfigDatabase:
    def __init__(self) -> None:
        self._clockmaster_setting: Optional[ClockmasterSetting] = None
        self._box_settings: Final[dict[str, BoxSetting]] = {}
        self._box_aliases: Final[dict[str, str]] = {}
        self._port_settings: Final[dict[str, PortSetting]] = {}
        self._relation_channel_target: Final[MutableSequence[tuple[str, str]]] = []
        self._target_settings: Final[dict[str, dict[str, Any]]] = {}
        self._relation_channel_port: Final[
            MutableSequence[tuple[str, dict[str, str | int]]]
        ] = []

    def define_clockmaster(
        self,
        ipaddr: str,
        reset: bool,
    ) -> None:
        self._clockmaster_setting = ClockmasterSetting(
            ipaddr=ipaddr,
            reset=reset,
        )

    def set(
        self,
        clockmaster_setting: Optional[dict] = None,
        box_settings: Optional[dict] = None,
        box_aliases: Optional[dict[str, str]] = None,
        port_settings: Optional[dict[str, dict[str, Any]]] = None,
        relation_channel_target: Optional[MutableSequence[tuple[str, str]]] = None,
        target_settings: Optional[dict[str, dict[str, Any]]] = None,
        relation_channel_port: Optional[MutableSequence[tuple[str, str]]] = None,
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
                if setting["box_name"] in self._box_aliases:
                    setting["box_name"] = self._box_aliases[setting["box_name"]]
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
            configs = json.load(file)
        # TODO workaround
        settings = {
            k: v
            for k, v in configs.items()
            if k
            in [
                "clockmaster_setting",
                "box_settings",
                "box_aliases",
                "target_settings",
                "port_settings",
            ]
        }
        relation_channel_target = configs["relation_channel_target"]
        settings["relation_channel_target"] = relation_channel_target
        relation_channel_port = configs["relation_channel_port"]
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
        if isinstance(boxtype, str):
            boxtype = QUEL1_BOXTYPE_ALIAS[boxtype]
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
        lo_freq: float = 0,
        cnco_freq: float = 0,
        sideband: str = "",
        vatt: int = 0,
        fnco_freq: tuple[float] | tuple[float, float, float] = (0.0,),
        ndelay_or_nwait: tuple[int, ...] = (),
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
            ndelay_or_nwait=ndelay_or_nwait,
        )

    def get_channels_by_target(
        self,
        target_name: str,
    ) -> __builtins__.set[str]:
        return {
            channel
            for channel, target in self._relation_channel_target
            if target == target_name
        }

    def get_channel_numbers_by_target(
        self,
        target_name: str,
    ) -> __builtins__.set[int]:
        channels = self.get_channels_by_target(target_name)
        return {
            int(
                {_: port for _, port in self._relation_channel_port}[channel][
                    "channel_number"
                ]
            )
            for channel in channels
        }

    def get_channel(
        self,
        channel_name: str,
    ) -> tuple[str, str, int]:
        port_name = self.get_port_by_channel(channel_name)
        box_name = self._port_settings[port_name].box_name
        channel_number = self.get_channel_number_by_channel(channel_name)
        return box_name, port_name, channel_number

    def get_port_by_channel(self, channel_name: str) -> str:
        retval = {channel: port for channel, port in self._relation_channel_port}[
            channel_name
        ]["port_name"]
        return retval if isinstance(retval, str) else ""

    def get_channel_number_by_channel(self, channel_name: str) -> int:
        retval = {channel: port for channel, port in self._relation_channel_port}[
            channel_name
        ]["channel_number"]
        return retval if isinstance(retval, int) else 0

    def get_ports_by_target(
        self,
        target_name: str,
    ) -> __builtins__.set[str]:
        channels = self.get_channels_by_target(target_name)
        return {
            str(
                {channel: port for channel, port in self._relation_channel_port}[_][
                    "port_name"
                ]
            )
            for _ in channels
        }

    def get_port_numbers_by_target(
        self,
        target_name: str,
    ) -> __builtins__.set[int]:
        return {
            self._port_settings[_].port for _ in self.get_ports_by_target(target_name)
        }

    def get_boxes_by_target(
        self,
        target_name: str,
    ) -> __builtins__.set[str]:
        ports = self.get_ports_by_target(target_name)
        return {self._port_settings[port].box_name for port in ports}

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
        box_setting = BoxSetting(
            box_name=box_name,
            ipaddr_wss=ipaddr_wss,
            boxtype=QUEL1_BOXTYPE_ALIAS[boxtype],
            config_options=config_options,
            ipaddr_sss=ipaddr_sss,
            ipaddr_css=ipaddr_css,
            config_root=config_root,
        )
        self._box_settings[box_name] = box_setting
        return box_setting.asdict()

    def define_channel(
        self,
        channel_name: str,
        port_name: str,
        channel_number: int,
        ndelay_or_nwait: int = 0,
    ) -> None:
        self._relation_channel_port.append(
            (
                channel_name,
                {
                    "port_name": port_name,
                    "channel_number": channel_number,
                },
            ),
        )
        _ndelay_or_nwait = list(self._port_settings[port_name].ndelay_or_nwait)
        if channel_number < len(_ndelay_or_nwait):
            _ndelay_or_nwait[channel_number] = ndelay_or_nwait
        else:
            _ = [0 for _ in range(channel_number + 1)]
            for i, v in enumerate(_ndelay_or_nwait):
                _[i] = v
            _[channel_number] = ndelay_or_nwait
            _ndelay_or_nwait = _
        self._port_settings[port_name].ndelay_or_nwait = tuple(_ndelay_or_nwait)

    def define_port(
        self,
        port_name: str,
        box_name: str,
        port_number: int,
        lo_freq: Optional[float] = None,
        cnco_freq: Optional[float] = None,
        sideband: str = "U",
        vatt: int = 0x800,
        fnco_freq: Optional[
            tuple[float] | tuple[float, float] | tuple[float, float, float]
        ] = None,
        # ndelay_or_nwait: tuple[int, ...] = [],
    ) -> None:
        if port_name in self._port_settings:
            ndelay_or_nwait = self._port_settings[port_name].ndelay_or_nwait
        else:
            ndelay_or_nwait = ()
        self._port_settings[port_name] = PortSetting(
            port_name=port_name,
            box_name=box_name,
            port=port_number,
            lo_freq=lo_freq,
            cnco_freq=cnco_freq,
            sideband=sideband,
            vatt=vatt,
            fnco_freq=fnco_freq,
            ndelay_or_nwait=ndelay_or_nwait,
        )

    def create_box(
        self,
        box_name: str,
        reconnect: bool = True,
    ) -> Quel1Box:
        s = self._box_settings[box_name]
        box = Quel1Box.create(
            ipaddr_wss=str(s.ipaddr_wss),
            ipaddr_sss=str(s.ipaddr_sss),
            ipaddr_css=str(s.ipaddr_css),
            boxtype=s.boxtype,
            config_root=Path(s.config_root) if s.config_root is not None else None,
            config_options=s.config_options if s.config_options else None,
        )
        if reconnect:
            if not all([_ for _ in box.link_status().values()]):
                box.relinkup(use_204b=False)
            status = box.reconnect()
            for mxfe_idx, _ in status.items():
                if not _:
                    logger.error(
                        f"be aware that mxfe-#{mxfe_idx} is not linked-up properly"
                    )
        return box

    def asdict(self) -> dict[str, Any]:
        return {
            "clockmaster_setting": self._clockmaster_setting.asdict()
            if self._clockmaster_setting is not None
            else None,
            "box_settings": {
                box_name: _.asdict() for box_name, _ in self._box_settings.items()
            },
            "box_aliases": self._box_aliases,
            "port_settings": {
                port_name: _.asdict() for port_name, _ in self._port_settings.items()
            },
            "target_settings": self._target_settings,
            "relation_channel_target": self._relation_channel_target,
            "relation_channel_port": self._relation_channel_port,
        }

    def asjson(self) -> str:
        box_settings = {
            box_name: _.asdict() for box_name, _ in self._box_settings.items()
        }
        for dct in box_settings.values():
            dct["boxtype"] = {v: k for k, v in QUEL1_BOXTYPE_ALIAS.items()}[
                dct["boxtype"]
            ]
        return json.dumps(
            {
                "clockmaster_setting": self._clockmaster_setting.asdict()
                if self._clockmaster_setting is not None
                else None,
                "box_settings": box_settings,
                "box_aliases": self._box_aliases,
                "port_settings": {
                    port_name: _.asdict()
                    for port_name, _ in self._port_settings.items()
                },
                "target_settings": self._target_settings,
                "relation_channel_target": self._relation_channel_target,
                "relation_channel_port": self._relation_channel_port,
            },
            indent=4,
        )
