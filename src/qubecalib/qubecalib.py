from __future__ import annotations

import datetime
import getpass
import json
import logging
import math
import os
import time
import warnings
from collections import Counter, deque
from enum import Enum
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
import numpy.typing as npt
from e7awgsw import CaptureModule, CaptureParam, DspUnit, WaveSequence
from quel_clock_master import QuBEMasterClient, SequencerClient
from quel_ic_config import (
    QUEL1_BOXTYPE_ALIAS,
    CaptureReturnCode,
    Quel1BoxType,
    Quel1BoxWithRawWss,
)
from typing_extensions import deprecated

from . import __version__, neopulse
from .e7utils import (
    CaptureParamTools,
    WaveSequenceTools,
    _convert_gen_sampled_sequence_to_blanks_and_waves_chain,
)
from .instrument.quel.quel1 import driver as direct
from .neopulse import (
    CapSampledSequence,
    Capture,
    GenSampledSequence,
    GenSampledSubSequence,
    Slot,
    Waveform,
)
from .sysconfdb import BoxSetting, PortSetting, SystemConfigDatabase

logger = logging.getLogger(__name__)


class Direction(Enum):
    FROM_TARGET = "from_target"
    TO_TARGET = "to_target"


class Sideband(Enum):
    UpperSideBand = "U"
    LowerSideBand = "L"


DEFAULT_SIDEBAND = "U"


class QubeCalib:
    def __init__(
        self,
        path_to_database_file: Optional[str | os.PathLike] = None,
    ) -> None:
        self._system_config_database: Final[SystemConfigDatabase] = (
            SystemConfigDatabase()
        )
        self._executor: Final[Executor] = Executor(self._system_config_database)
        self._box_configs: dict[str, dict[str, Any]] = {}

        if path_to_database_file is not None:
            self.system_config_database.load(path_to_database_file)

    @property
    def system_config_database(self) -> SystemConfigDatabase:
        return self._system_config_database

    @property
    def sysdb(self) -> SystemConfigDatabase:
        return self._system_config_database

    @property
    def Quel1BoxType(self) -> type[Quel1BoxType]:
        return Quel1BoxType

    @deprecated("use sysdb.create_quel1system() instead")
    def create_quel1system(self, box_names: list[str]) -> direct.Quel1System:
        return self.quel1_create_quel1system(*box_names)

    @deprecated("use sysdb.create_quel1system() instead")
    def quel1_create_quel1system(self, *box_names: str) -> direct.Quel1System:
        if self.sysdb._clockmaster_setting is None:
            raise ValueError("clock master is not found")
            # TODO : ここは例外を投げるのではなく、 None を設定するようにし，　single box モードを設ける
        system = direct.Quel1System.create(
            clockmaster=QuBEMasterClient(self.sysdb._clockmaster_setting.ipaddr),
            boxes=[self.create_named_box(b) for b in box_names],
        )
        return system

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
        # TODO ここは仕様変更が必要
        # Readout send に位相合わせ機構を導入するため SebSequence にまとめてしまわず Slot 毎に分割しないといけない
        # 情報を失わせ過ぎた
        # capture に関連する gen_sequence を取り出して 変調 slice を作成する
        gen_sampled_sequence, cap_sampled_sequence = (
            sequence.convert_to_sampled_sequence()
        )
        # settings = self.system_config_database._target_settings
        # for target_name, gss in gen_sampled_sequence.items():
        #     if target_name not in settings:
        #         raise ValueError(f"target({target_name}) is not defined")
        #     box_names = self.system_config_database.get_boxes_by_target(target_name)
        #     if not box_names:
        #         raise ValueError(f"target({target_name}) is not assigned to any box")
        #     if len(box_names) > 1:
        #         raise ValueError(f"target({target_name}) is assigned to multiple boxes")
        #     # tgtset = settings[target_name]
        #     # skew = tgtset["skew"] if "skew" in tgtset else 0
        #     box_name = list(box_names)[0]
        #     skew = self.sysdb.skew[box_name] if box_name in self.sysdb.skew else 0
        #     gss.padding += skew

        items_by_target = sequence._get_group_items_by_target()

        targets = set(
            [gtarget for gtarget in gen_sampled_sequence]
            + [ctarget for ctarget in cap_sampled_sequence]
        )
        resource_map = self._create_target_resource_map(targets)

        self._executor.add_command(
            Sequencer(
                gen_sampled_sequence=gen_sampled_sequence,
                cap_sampled_sequence=cap_sampled_sequence,
                resource_map=resource_map,
                group_items_by_target=items_by_target,
                time_offset=time_offset,
                time_to_start=time_to_start,
                interval=interval,
                sysdb=self.system_config_database,
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
        # config_root: Optional[str] = None,
        # config_options: MutableSequence[Quel1ConfigOption] = [],
    ) -> dict[str, Any]:
        return self.system_config_database.define_box(
            box_name=box_name,
            ipaddr_wss=ipaddr_wss,
            boxtype=boxtype,
            # config_options=config_options,
            ipaddr_sss=ipaddr_sss,
            ipaddr_css=ipaddr_css,
            # config_root=config_root,
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

    def _create_target_resource_map(
        self,
        target_names: Iterable[str],
    ) -> dict[
        str, Iterable[dict[str, BoxSetting | PortSetting | int | dict[str, float]]]
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
    ) -> Quel1BoxWithRawWss:
        return self.system_config_database.create_box(
            box_name=box_name,
            reconnect=reconnect,
        )

    @deprecated("use sysdb.create_named_box() instead")
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
                # config_root=Path(setting.config_root)
                # if setting.config_root is not None
                # else None,
                # config_options=setting.config_options,
            )
            box.reconnect()
        return boxpool


class Converter:
    @classmethod
    def convert_to_device_specific_sequence(
        cls,
        gen_sampled_sequence: dict[str, GenSampledSequence],
        cap_sampled_sequence: dict[str, CapSampledSequence],
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
            gen_sampled_sequence=gen_sampled_sequence,
            cap_sampled_sequence=cap_sampled_sequence,
            resource_map={
                target_name: _
                for target_name, _ in resource_map.items()
                if target_name in cap_sampled_sequence
            },
            port_config={
                target_name: _
                for target_name, _ in port_config.items()
                if target_name in cap_sampled_sequence
            },
            repeats=repeats,
            interval=interval,
            integral_mode=integral_mode,
            dsp_demodulation=dsp_demodulation,
            software_demodulation=software_demodulation,
        )
        genseq = cls.convert_to_gen_device_specific_sequence(
            gen_sampled_sequence=gen_sampled_sequence,
            cap_sampled_sequence=cap_sampled_sequence,
            resource_map={
                target_name: _
                for target_name, _ in resource_map.items()
                if target_name in gen_sampled_sequence
            },
            port_config={
                target_name: _
                for target_name, _ in port_config.items()
                if target_name in gen_sampled_sequence
            },
            repeats=repeats,
            interval=interval,
        )
        return genseq | capseq

    @classmethod
    def convert_to_cap_device_specific_sequence(
        cls,
        gen_sampled_sequence: dict[str, GenSampledSequence],
        cap_sampled_sequence: dict[str, CapSampledSequence],
        resource_map: dict[str, dict[str, BoxSetting | PortSetting | int]],
        port_config: dict[str, PortConfigAcquirer],
        repeats: int,
        interval: float,
        integral_mode: str,
        dsp_demodulation: bool,
        software_demodulation: bool,
    ) -> dict[tuple[str, int, int], CaptureParam]:
        # 線路に起因する遅延
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
        for target_name, freq in targets_freqs.items():
            cap_sampled_sequence[target_name].modulation_frequency = freq
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
                for _ in Counter(
                    [targets_ids[_] for _ in cap_sampled_sequence]
                ).values()
            ]
        ):
            raise ValueError(
                "multiple access for single runit will be supported, not now"
            )
        # 戻り値は {(box_name, port_number, channel_number): CaptureParam} の dict
        sseqs = [seq for seq in cap_sampled_sequence.values()]
        # fps = [padding] + len(sseqs[1:]) * [0]
        # lbs = len(sseqs[:-1]) * [0] + [padding]
        # padding は WaveSequence の長さと合わせるために設けた
        # 原則 WaveSequence は wait = 0 とする
        # TODO Skew の調整はこれから実装する。　wait の単位を確認すること。1Saで調整できるように。
        ids_e7 = {
            targets_ids[sseq.target_name]: CaptureParamTools.create(
                sequence=sseq,
                capture_delay_words=ndelay_or_nwait_by_target[sseq.target_name]
                * 16,  # ndelay は 16 words = 1 block の単位
                repeats=repeats,
                interval_samples=int(interval / sseq.sampling_period),  # samples
            )
            for sseq in sseqs
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
        gen_sampled_sequence: dict[str, GenSampledSequence],
        cap_sampled_sequence: dict[str, CapSampledSequence],
        resource_map: dict[
            str, dict[str, BoxSetting | PortSetting | int | dict[str, float]]
        ],
        port_config: dict[str, PortConfigAcquirer],
        repeats: int,
        interval: float,
    ) -> dict[tuple[str, int, int], WaveSequence]:
        # for target_name, gss in gen_sampled_sequence.items():
        #     print(target_name, gss.padding, end=" ")
        #     # for sub in gss.sub_sequences:
        #     #     print(sub.padding, end=" ")
        # print()
        # WaveSequence の生成
        SAMPLING_PERIOD = 2

        # target 毎の変調周波数の計算と channel 毎にデータを束ねる
        targets_freqs: MutableMapping[str, float] = {}
        targets_ids: MutableMapping[str, tuple[str, int, int]] = {}
        for target_name, rmap in resource_map.items():
            # target 毎の変調周波数の計算
            rmap_target = rmap["target"]
            if not isinstance(rmap_target, dict):
                raise ValueError("target is not defined")
            targets_freqs[target_name] = cls.calc_modulation_frequency(
                f_target=rmap_target["frequency"],
                port_config=port_config[target_name],
            )

            # targets_freqs = {
            #     target_name: cls.calc_modulation_frequency(
            #         f_target=rmap["target"]["frequency"],
            #         port_config=port_config[target_name],
            #     )
            #     for target_name, rmap in resource_map.items()
            # }

            # channel 毎 (awg 毎) にデータを束ねる
            # target_name と (box_name, port_number, channel_number) のマップを作成する
            rmap_box = rmap["box"]
            if not isinstance(rmap_box, BoxSetting):
                raise ValueError("box is not defined")
            rmap_port = rmap["port"]
            if not isinstance(rmap_port, PortSetting):
                raise ValueError("port is not defined")
            rmap_channel_number = rmap["channel_number"]
            if not isinstance(rmap_channel_number, int):
                raise ValueError("channel_number is not defined")
            targets_ids[target_name] = (
                rmap_box.box_name,
                rmap_port.port,
                rmap_channel_number,
            )

        # readout のタイミングを考慮して位相補償を行う
        for target_name, freq in targets_freqs.items():
            gen_sampled_sequence[target_name].modulation_frequency = freq
        for target, seq in gen_sampled_sequence.items():
            # readout target でない場合はスキップ
            if target not in cap_sampled_sequence:
                continue
            modulation_angular_frequency = 2 * np.pi * targets_freqs[target]
            timing_list = seq.readout_timings
            offset_list = cap_sampled_sequence[target].readin_offsets
            # もし readout_timings と readin_offsets が None なら後方互換でスキップ
            if timing_list is None and offset_list is None:
                continue
            if timing_list is None:
                raise ValueError("readout_timings is not defined")
            if offset_list is None:
                raise ValueError("readin_offsets is not defined")
            for subseq, timings, offsets in zip(
                seq.sub_sequences, timing_list, offset_list
            ):
                wave = subseq.real + 1j * subseq.imag
                for (begin, end), (offsetb, _) in zip(timings, offsets):
                    offset_phase = modulation_angular_frequency * offsetb
                    b = math.floor(begin / SAMPLING_PERIOD)
                    e = math.floor(end / SAMPLING_PERIOD)
                    wave[b:e] = wave[b:e] * np.exp(-1j * offset_phase)
                subseq.real = np.real(wave)
                subseq.imag = np.imag(wave)

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
        ids_sampled_sequences: dict[
            tuple[str, int, int], dict[str, GenSampledSequence]
        ] = {}
        for target, box_port_channel in targets_ids.items():
            if target in gen_sampled_sequence:
                if box_port_channel not in ids_sampled_sequences:
                    ids_sampled_sequences[box_port_channel] = {}
                ids_sampled_sequences[box_port_channel][target] = gen_sampled_sequence[
                    target
                ]
        # ids_sampled_sequences = {
        #     id: {
        #         _: sampled_sequence[_]
        #         for _, _id in targets_ids.items()
        #         if _id == id and _ in sampled_sequence
        #     }
        #     for id in targets_ids.values()
        # }
        # (box_name, port_number, channel_number) と {target_name: modfreq} とのマップを作成する
        ids_modfreqs = {
            id: {_: targets_freqs[_] for _, _id in targets_ids.items() if _id == id}
            for id in targets_ids.values()
        }
        # 最初の subsequence の先頭に padding 分の 0 を追加する
        # TODO sampling_period を見るようにする
        for target, seq in gen_sampled_sequence.items():
            padding = seq.padding
            subseq = seq.sub_sequences[0]
            subseq.real = np.concatenate([np.zeros(padding), subseq.real])
            subseq.imag = np.concatenate([np.zeros(padding), subseq.imag])
        # ここから全部の subseq で位相回転 (omega(t-t0))しないといけない
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

        if port_config.dump_config["direction"] == "out":
            f_diff = cls._calc_modulation_frequency(
                target_freq=f_target,
                lo_freq=f_lo,
                cnco_freq=f_cnco,
                fnco_freq=f_fnco,
                sideband=sideband,
            )

            if 0.5 < abs(f_diff):
                p = port_config
                warnings.warn(
                    f"Modulation frequency abs({f_diff}) of {p._box_name}:{p._port}:{p._channel} is too high. f_target={f_target} GHz, f_lo={f_lo} GHz, f_cnco={f_cnco} GHz, f_fnco={f_fnco} GHz, sideband={sideband}"
                )
        elif port_config.dump_config["direction"] == "in":
            opposite = "L" if sideband == "U" else "U"
            f_diff = cls._calc_modulation_frequency(
                target_freq=f_target,
                lo_freq=f_lo,
                cnco_freq=f_cnco,
                fnco_freq=f_fnco,
                sideband=sideband,
            )
            o_f_diff = cls._calc_modulation_frequency(
                target_freq=f_target,
                lo_freq=f_lo,
                cnco_freq=f_cnco,
                fnco_freq=f_fnco,
                sideband=opposite,
            )
            # TODO in port の sideband の取り扱いが曖昧
            # Dsp の設定を見るべきかな
            # とりあえず abs が小さい方で判定するが sideband の設定に応じた diff を復調周波数として採用
            mindiff = f_diff if abs(f_diff) < abs(o_f_diff) else o_f_diff
            if 0.25 < abs(mindiff):
                p = port_config
                warnings.warn(
                    f"Modulation frequency abs({mindiff}) of {p._box_name}:{p._port}:{p._channel} is too high. f_target={f_target} GHz, f_lo={f_lo} GHz, f_cnco={f_cnco} GHz, f_fnco={f_fnco} GHz, sideband={sideband}"
                )
        else:
            raise ValueError(f"{port_config} invalid direction")

        return f_diff  # GHz

    @classmethod
    def _calc_modulation_frequency(
        cls,
        target_freq: float,
        lo_freq: float,
        cnco_freq: float,
        fnco_freq: float,
        sideband: str,
    ) -> float:
        """Calculate modulation frequency from target frequency and port configuration."""
        # Note that port_config has frequencies in Hz.
        if sideband == Sideband.UpperSideBand.value:
            f_diff = target_freq - lo_freq - (cnco_freq + fnco_freq)
        elif sideband == Sideband.LowerSideBand.value:
            f_diff = -(target_freq - lo_freq) - (cnco_freq + fnco_freq)
        else:
            raise ValueError("invalid ssb mode")

        return f_diff  # GHz

    @classmethod
    def multiplex(
        cls,
        sequences: MutableMapping[str, GenSampledSequence],
        modfreqs: MutableMapping[str, float],
    ) -> GenSampledSequence:
        # sequences は {target_name: GenSampledSequence} の dict だが
        # 基本データを取得するために代表する先頭の sequence を取得する
        # 基本データは全て同じであることを前提とする
        cls.validate_geometry_identity(sequences)
        if not cls.validate_geometry_identity(sequences):
            raise ValueError(
                "All geometry of sub sequences belonging to the same awg must be equal"
            )
        sequence = sequences[next(iter(sequences))]
        padding = sequence.padding

        chain = {
            target_name: _convert_gen_sampled_sequence_to_blanks_and_waves_chain(subseq)
            for target_name, subseq in sequences.items()
        }
        begins = {
            target_name: [
                sum(chain[target_name][: i + 1])
                for i, _ in enumerate(chain[target_name][1:])
            ]
            for target_name in sequences
        }
        # 変調する時間軸を生成する
        # padding 分基準時間をずらす t - t0 処理が加わっている
        SAMPLING_PERIOD = sequence.sampling_period
        times = {
            target_name: [
                (begin + np.arange(subseq.real.shape[0]) - padding) * SAMPLING_PERIOD
                for begin, subseq in zip(
                    begins[target_name][::2],
                    sequence.sub_sequences,
                )
            ]
            for target_name, sequence in sequences.items()
        }
        # 変調および多重化した複素信号
        waves = [
            np.array(
                [
                    (
                        sequences[target].sub_sequences[i].real
                        + 1j * sequences[target].sub_sequences[i].imag
                    )
                    * np.exp(1j * 2 * np.pi * (modfreqs[target] * times[target][i]))
                    for target in sequences
                ]
            ).sum(axis=0)
            for i, _ in enumerate(sequence.sub_sequences)
        ]

        return GenSampledSequence(
            target_name="",
            prev_blank=sequence.prev_blank,
            post_blank=sequence.post_blank,
            repeats=sequence.repeats,
            sampling_period=sequence.sampling_period,
            sub_sequences=[
                GenSampledSubSequence(
                    real=np.real(waves[i]),
                    imag=np.imag(waves[i]),
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
    box: Quel1BoxWithRawWss
    port: int | tuple[int, int]
    channel: int
    box_name: str


class PortConfigAcquirer:
    def __init__(
        self,
        boxpool: BoxPool,
        box_name: str,
        box: Quel1BoxWithRawWss,
        port: int | tuple[int, int],
        channel: int,
    ):
        # boxpool にキャッシュされている box の設定を取得する
        if box_name not in boxpool._box_config_cache:
            boxpool._box_config_cache[box_name] = box.dump_box()
        dump_box = boxpool._box_config_cache[box_name]["ports"]
        self.dump_config = dp = dump_box[port]
        sideband = dp["sideband"] if "sideband" in dp else DEFAULT_SIDEBAND
        fnco_freq = 0
        if port in box.get_output_ports():
            fnco_freq = dp["channels"][channel]["fnco_freq"]
        if port in box.get_input_ports():
            fnco_freq = dp["runits"][channel]["fnco_freq"]
            if port in box.get_read_input_ports():
                lpbackps = box.get_loopbacks_of_port(port)
                if lpbackps:
                    lpbackp = next(iter(lpbackps))
                    dumped_port = dump_box[lpbackp]
                    sideband = (
                        dumped_port["sideband"]
                        if "sideband" in dumped_port
                        else DEFAULT_SIDEBAND
                    )
            elif port in box.get_monitor_input_ports():
                lpbackps = box.get_loopbacks_of_port(port)
                if lpbackps:
                    lpbackp = next(iter(lpbackps))
                    dumped_port = dump_box[lpbackp]
                    sideband = (
                        dumped_port["sideband"]
                        if "sideband" in dumped_port
                        else DEFAULT_SIDEBAND
                    )
        self.lo_freq: float = dp["lo_freq"]
        self.cnco_freq: float = dp["cnco_freq"]
        self.fnco_freq: float = fnco_freq
        self.sideband: str = sideband
        self._box_name = box_name
        self._port = port
        self._channel = channel

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
        *,
        sysdb: SystemConfigDatabase,
        time_offset: dict[str, int] = {},
        time_to_start: dict[str, int] = {},
        group_items_by_target: dict[str, dict[int, MutableSequence[Slot]]] = {},
        interval: Optional[float] = None,
    ):
        self.gen_sampled_sequence = gen_sampled_sequence
        self.cap_sampled_sequence = cap_sampled_sequence
        self.group_items_by_terget = group_items_by_target  # TODO ここは begin, end の境界だけわかれば良いので過剰
        # むしろオブジェクトは不要（シリアライズして送るのに面倒）
        self.resource_map = resource_map
        self.syncoffset_by_boxname = time_offset  # taps
        self.timetostart_by_boxname = time_to_start  # sysref
        self.interval = interval

        settings = sysdb._target_settings
        for target_name, gss in gen_sampled_sequence.items():
            if target_name not in settings:
                raise ValueError(f"target({target_name}) is not defined")
            box_names = sysdb.get_boxes_by_target(target_name)
            if not box_names:
                raise ValueError(f"target({target_name}) is not assigned to any box")
            if len(box_names) > 1:
                raise ValueError(f"target({target_name}) is assigned to multiple boxes")
            # tgtset = settings[target_name]
            # skew = tgtset["skew"] if "skew" in tgtset else 0
            box_name = list(box_names)[0]
            skew = sysdb.skew[box_name] if box_name in sysdb.skew else 0
            gss.padding += skew

        # resource_map は以下の形式
        # {
        #   "box": db._box_settings[box_name],
        #   "port": db._port_settings[port_name],
        #   "channel_number": channel_number,
        #   "target": db._target_settings[target_name],
        # }
        self.sysdb = sysdb
        self._sideload_settings: list[
            direct.AwgSetting | direct.RunitSetting | direct.TriggerSetting
        ] = []  # サイドロード用の設定

        # readout の target set を作る
        readout_targets = {
            target
            for target, subseq in group_items_by_target.items()
            for items in subseq.values()
            for item in items
            if isinstance(item, Waveform)
        }
        # readout のタイミング (begin, end) 辞書を作る
        readout_timings: dict[str, MutableSequence[list[tuple[float, float]]]] = {
            target: [
                [
                    (begin, begin + duration)
                    for item in items
                    if isinstance(item, Waveform)
                    if (begin := item.begin) is not None
                    and (duration := item.duration) is not None
                ]
                for items in group_items_by_target[target].values()
            ]
            for target in readout_targets
        }
        # remove empty items
        readout_timings = {
            target: [item for item in items if item]
            for target, items in readout_timings.items()
        }
        # remove empty subseqs
        readout_timings = {
            target: items for target, items in readout_timings.items() if items
        }
        # readout_timings が 空なら後方互換
        if readout_timings:
            for target_name, gseq in gen_sampled_sequence.items():
                # readout_timings に target_name は含まれているはず
                gseq.readout_timings = readout_timings[target_name]

        readin_targets = {
            target
            for target, subseq in group_items_by_target.items()
            for items in subseq.values()
            for item in items
            if isinstance(item, Capture)
        }
        readin_offsets: dict[str, MutableSequence[list[tuple[float, float]]]] = {
            target: [
                [
                    (begin, begin + duration)
                    for item in items
                    if isinstance(item, Capture)
                    if (begin := item.begin) is not None
                    and (duration := item.duration) is not None
                ]
                for nodeid, items in group_items_by_target[target].items()
            ]
            for target in readin_targets
        }
        # remove empty items
        readin_offsets = {
            target: [item for item in items if item]
            for target, items in readin_offsets.items()
        }
        # remove empty subseqs
        readin_offsets = {
            target: items for target, items in readin_offsets.items() if items
        }
        if readin_offsets:
            for target_name, cseq in cap_sampled_sequence.items():
                cseq.readin_offsets = readin_offsets[target_name]

    def set_measurement_option(
        self,
        repeats: int,
        interval: float,
        integral_mode: str,
        dsp_demodulation: bool = True,
        software_demodulation: bool = False,
        phase_compensation: bool = True,  # TODO not work
    ) -> None:
        self.repeats = repeats
        self.interval = interval
        self.integral_mode = integral_mode
        self.dsp_demodulation = dsp_demodulation
        self.software_demodulation = software_demodulation
        self.phase_compensation = phase_compensation

    def generate_cap_resource_map(self, boxpool: BoxPool) -> dict[str, Any]:
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
                    boxpool.get_port_direction(box_name, port) == "in"
                    and target_name in self.cap_sampled_sequence
                ):
                    if target_name in _cap_resource_map:
                        _cap_resource_map[target_name].append(m)
                    else:
                        _cap_resource_map[target_name] = [m]
        return {
            target_name: next(iter(maps))
            for target_name, maps in _cap_resource_map.items()
            if maps
        }

    def calc_first_padding(self) -> int:
        csseq = self.cap_sampled_sequence
        first_blank = min(
            [seq.prev_blank for sseq in csseq.values() for seq in sseq.sub_sequences]
        )
        return ((first_blank - 1) // 64 + 1) * 64 - first_blank  # Sa

    def generate_e7_settings(
        self,
        boxpool: BoxPool,
    ) -> tuple[
        dict[tuple[str, int, int], CaptureParam],
        dict[tuple[str, int, int], WaveSequence],
        dict[str, Any],
    ]:
        # cap 用の cap_e7_setting と gen 用の gen_e7setting を作る
        cap_resource_map = self.generate_cap_resource_map(boxpool)
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
                    boxpool.get_port_direction(box_name, port) == "out"
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
                box_name=m["box"].box_name,
            )
            for target_name, m in cap_resource_map.items()
        }
        gen_target_bpc: dict[str, TargetBPC] = {
            target_name: TargetBPC(
                box=boxpool.get_box(m["box"].box_name)[0],
                port=m["port"].port if isinstance(m["port"], PortSetting) else 0,
                channel=m["channel_number"],
                box_name=m["box"].box_name,
            )
            for target_name, m in gen_resource_map.items()
        }
        cap_target_portconf = {
            target_name: PortConfigAcquirer(
                boxpool=boxpool,
                box_name=m["box_name"],
                box=m["box"],
                port=m["port"],
                channel=m["channel"],
            )
            for target_name, m in cap_target_bpc.items()
        }

        # first_blank = min(
        #     [seq.prev_blank for sseq in csseq.values() for seq in sseq.sub_sequences]
        # )
        # first_padding = ((first_blank - 1) // 64 + 1) * 64 - first_blank  # Sa
        # ref_sequence = next(iter(csseq.values()))
        first_padding = self.calc_first_padding()

        for target_name, cseq in self.cap_sampled_sequence.items():
            cseq.padding += first_padding
        for target_name, gseq in self.gen_sampled_sequence.items():
            gseq.padding += first_padding

        interval = self.interval if self.interval is not None else 10240
        cap_e7_settings: dict[tuple[str, int, int], CaptureParam] = (
            Converter.convert_to_cap_device_specific_sequence(
                gen_sampled_sequence=self.gen_sampled_sequence,
                cap_sampled_sequence=self.cap_sampled_sequence,
                resource_map=cap_resource_map,
                # target_freq=target_freq,
                port_config=cap_target_portconf,
                repeats=self.repeats,
                interval=interval,
                integral_mode=self.integral_mode,
                dsp_demodulation=self.dsp_demodulation,
                software_demodulation=self.software_demodulation,
            )
        )
        # phase_offset_list_by_target = {
        #     target: [-2 * np.pi * cap_fmod[target] * t for t in reference_time_list]
        #     for target, reference_time_list in reference_time_list_by_target.items()
        # }

        gen_target_portconf = {
            target_name: PortConfigAcquirer(
                boxpool=boxpool,
                box_name=m["box_name"],
                box=m["box"],
                port=m["port"],
                channel=m["channel"],
            )
            for target_name, m in gen_target_bpc.items()
        }
        gen_e7_settings: dict[tuple[str, int, int], WaveSequence] = (
            Converter.convert_to_gen_device_specific_sequence(
                gen_sampled_sequence=self.gen_sampled_sequence,
                cap_sampled_sequence=self.cap_sampled_sequence,
                resource_map=gen_resource_map,
                port_config=gen_target_portconf,
                repeats=self.repeats,
                interval=interval,
            )
        )
        return cap_e7_settings, gen_e7_settings, cap_resource_map

    def execute(
        self,
        boxpool: BoxPool,
    ) -> tuple[dict[str, CaptureReturnCode], dict[str, list], dict]:
        quel1system = self.create_quel1system(boxpool)
        c, g, m = self.generate_e7_settings(boxpool)

        settings: list[
            direct.RunitSetting | direct.AwgSetting | direct.TriggerSetting
        ] = []
        for (name, port, runit), cprm in c.items():
            settings.append(
                direct.RunitSetting(
                    runit=direct.RunitId(
                        box=name,
                        port=port,
                        runit=runit,
                    ),
                    cprm=cprm,
                )
            )
        for (name, port, channel), wseq in g.items():
            settings.append(
                direct.AwgSetting(
                    awg=direct.AwgId(
                        box=name,
                        port=port,
                        channel=channel,
                    ),
                    wseq=wseq,
                )
            )
        settings += self.select_trigger(quel1system, settings)
        if len(settings) == 0:
            raise ValueError("no settings")

        if self._sideload_settings:
            action = direct.Action.build(
                system=quel1system, settings=self._sideload_settings
            )
        else:
            action = direct.Action.build(system=quel1system, settings=settings)
        status, results = action.action()
        return self.parse_capture_results(status, results, action, m)

    def parse_capture_results(
        self,
        status: dict[tuple[str, int], CaptureReturnCode],
        results: dict[tuple[str, int, int], npt.NDArray[np.complex64]],
        action: direct.Action,
        crmap: dict[str, Any],
    ) -> tuple[dict[str, CaptureReturnCode], dict[str, list], dict]:
        bpc2target = {}
        for target, m in crmap.items():
            box, port, channel = m["box"].box_name, m["port"].port, m["channel_number"]
            bpc2target[(box, port, channel)] = target
        # status = {}
        # for (box, port), code in status.items():
        #     status[(box, port)] = code
        data = {}
        for (box, port, runit), datum in results.items():
            data[(box, port, runit)] = datum
        cprms = {}
        if isinstance(action._action, direct.multi.Action):
            for box, act in action._action._actions.items():
                for runit_id, cprm in act._cprms.items():
                    cprms[(box, runit_id.port, runit_id.runit)] = cprm
        elif isinstance(action._action, tuple):
            box, act = action._action
            for runit_id, cprm in act._cprms.items():
                cprms[(box, runit_id.port, runit_id.runit)] = cprm
        rstatus, rresults = {}, {}
        for (box, port, runit), target in bpc2target.items():
            s, r = self.parse_capture_result(
                status[(box, port)],
                data[(box, port, runit)],
                cprms[(box, port, runit)],
            )
            target = bpc2target[(box, port, runit)]
            rstatus[target] = s
            rresults[target] = r
        return rstatus, rresults, {}

    def parse_capture_result(
        self,
        status: CaptureReturnCode,
        data: npt.NDArray[np.complex64],
        cprm: CaptureParam,
    ) -> tuple[CaptureReturnCode, list[npt.NDArray[np.complex64]]]:
        # num_expected_words = cprm.calc_capture_samples()
        if DspUnit.INTEGRATION in cprm.dsp_units_enabled:
            data = data.reshape(1, -1)
        else:
            data = data.reshape(cprm.num_integ_sections, -1)
        if DspUnit.SUM in cprm.dsp_units_enabled:
            width = list(range(len(cprm.sum_section_list))[1:])
            result = np.hsplit(data, width)
        else:
            b = DspUnit.DECIMATION not in cprm.dsp_units_enabled
            ssl = cprm.sum_section_list
            ws = [w if b else int(w // 4) for w, _ in ssl[:-1]]
            word = cprm.NUM_SAMPLES_IN_ADC_WORD
            width = np.cumsum(np.array(ws))
            c = np.hsplit(data, width * word)
            result = [di.transpose() for di in c]
        return status, result

    def create_quel1system(self, boxpool: BoxPool) -> direct.Quel1System:
        quel1system = direct.Quel1System.create(
            clockmaster=boxpool._clock_master,
            boxes=[
                direct.NamedBox(
                    name,
                    box,
                )
                for name, (box, _) in boxpool._boxes.items()
            ],
        )
        quel1system.trigger = self.sysdb.trigger
        for box_name, timing_shift in self.sysdb.timing_shift.items():
            quel1system.timing_shift[box_name] = timing_shift
        quel1system.displacement = self.sysdb.time_to_start
        return quel1system

    def convert(
        self,
        cap_e7_settings: dict[tuple[str, int, int], CaptureParam],
        gen_e7_settings: dict[tuple[str, int, int], WaveSequence],
    ) -> list[direct.AwgSetting | direct.RunitSetting | direct.TriggerSetting]:
        settings: list[
            direct.AwgSetting | direct.RunitSetting | direct.TriggerSetting
        ] = []
        for (box_name, port, runit), e7 in cap_e7_settings.items():
            settings.append(
                direct.RunitSetting(
                    runit=direct.RunitId(box=box_name, port=port, runit=runit),
                    cprm=e7,
                )
            )
        for (box_name, port, channel), e7 in gen_e7_settings.items():
            settings.append(
                direct.AwgSetting(
                    awg=direct.AwgId(box=box_name, port=port, channel=channel),
                    wseq=e7,
                )
            )
        return settings

    @staticmethod
    def is_empty_trigger(
        settings: list[direct.AwgSetting | direct.RunitSetting | direct.TriggerSetting],
    ) -> bool:
        for s in settings:
            if isinstance(s, direct.TriggerSetting):
                return False
        return True

    def select_trigger(
        self,
        quel1system: direct.Quel1System,
        settings: list[direct.AwgSetting | direct.RunitSetting | direct.TriggerSetting],
    ) -> list[direct.TriggerSetting]:
        if not self.is_empty_trigger(settings):
            raise ValueError("trigger is already set")

        # トリガを自動で設定する
        result: list[direct.TriggerSetting] = []
        caps: list[tuple[int, direct.RunitId]] = []
        gens: list[tuple[int, direct.AwgId]] = []
        for setting in settings:
            if isinstance(setting, direct.RunitSetting):
                # 右肺か左肺かの情報を付加して runit の設定を抽出する
                box = quel1system.box[setting.runit.box]
                port, subport = box._decode_port(setting.runit.port)
                group, rline = box._convert_any_port(port)
                # capmod = box.rmap.get_capture_module_of_rline(group, rline)
                caps.append((group, setting.runit))
            elif isinstance(setting, direct.AwgSetting):
                # 右肺か左肺かの情報を付加して awg の設定を抽出する
                box = quel1system.box[setting.awg.box]
                port, subport = box._decode_port(setting.awg.port)
                group, rline = box._convert_any_port(port)
                gens.append((group, setting.awg))
        # もし quel1system に明示的に trigger が設定されているならそれを使う
        defined_awgs = [s.awg for s in settings if isinstance(s, direct.AwgSetting)]
        for runit_group, runit_id in caps:
            if (runit_id.box, runit_id.port) in quel1system.trigger:
                trig_name, trig_nport, trig_nchannel = quel1system.trigger[
                    (runit_id.box, runit_id.port)
                ]
                awg = direct.AwgId(
                    box=trig_name, port=trig_nport, channel=trig_nchannel
                )
                if runit_id.box != trig_name:
                    raise ValueError(
                        f"invalid trigger {runit_id.box, runit_id.port} for {trig_name, trig_nport, trig_nchannel}"
                    )
                if awg not in defined_awgs:
                    raise ValueError(
                        f"trigger {trig_name, trig_nport, trig_nchannel} not found in settings"
                    )
                result.append(
                    direct.TriggerSetting(
                        triggerd_port=runit_id.port,
                        trigger_awg=awg,
                    )
                )
        # もし capture のみあるいは awgs のみなら tigger は設定しない
        if all([bool(caps), not bool(gens)]) or all([not bool(caps), bool(gens)]):
            return result
        pre_defined_triggers = {(s.trigger_awg.box, s.triggerd_port) for s in result}
        for runit_group, runit_id in caps:
            if (runit_id.box, runit_id.port) in pre_defined_triggers:
                continue  # もしすでに trigger が設定されているならスキップ
            for awg_group, awg_id in gens:
                if runit_id.box == awg_id.box and runit_group == awg_group:
                    result.append(
                        direct.TriggerSetting(
                            triggerd_port=runit_id.port,
                            trigger_awg=awg_id,
                        )
                    )
                    break
            # 別の group の trigger を割り当てるようにチャレンジ
            for awg_group, awg_id in gens:
                if runit_id.box == awg_id.box:
                    result.append(
                        direct.TriggerSetting(
                            triggerd_port=runit_id.port,
                            trigger_awg=awg_id,
                        )
                    )
                    break
            else:
                raise ValueError("invalid trigger")
        return result

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
    def __init__(self, sysdb: SystemConfigDatabase) -> None:
        self._work_queue: Final[deque] = deque()
        self._boxpool: BoxPool = BoxPool()
        self._config_buffer: Final[deque] = deque()
        self.sysdb = sysdb

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
            self.check_config()
            self._boxpool._box_config_cache.clear()
            self._boxpool = BoxPool()
            self.clear_log()
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
                # for box, _ in self._boxpool._boxes.values():
                #     box.initialize_all_awgs()
                break
            # Sequencer 以外のコマンドを逐次実行
            next.execute(self._boxpool)
        for command in self._work_queue:
            # もしコマンドキューに Sequencer が残っていれば次の Sequencer を実行する
            if isinstance(command, Sequencer):
                # if self._quel1system is None:
                #     raise ValueError("Quel1System is not defined")
                # status, iqs, config = next.execute(self._boxpool, self._quel1system)
                results = next.execute(self._boxpool)
                user_name = getpass.getuser()
                current_pyfile = os.path.abspath(__file__)
                date_time = datetime.datetime.now()
                clock_ns = time.clock_gettime_ns(time.CLOCK_REALTIME)
                self._config_buffer.append(
                    (
                        # config,
                        user_name,
                        current_pyfile,
                        __version__,
                        date_time,
                        clock_ns,
                    )
                )
                if not self._work_queue:
                    self.check_config()
                    self._boxpool._box_config_cache.clear()
                    self._boxpool = BoxPool()
                    self.clear_log()
                # return status, iqs, config
                return results
        # これ以上 Sequencer がなければ残りのコマンドを実行する
        # if self._quel1system is None:
        #     raise ValueError("Quel1System is not defined")
        # status, iqs, config = next.execute(self._boxpool, self._quel1system)
        results = next.execute(self._boxpool)
        # status, iqs, config = next.execute(self._boxpool)
        user_name = getpass.getuser()
        current_pyfile = os.path.abspath(__file__)
        date_time = datetime.datetime.now()
        clock_ns = time.clock_gettime_ns(time.CLOCK_REALTIME)
        self._config_buffer.append(
            (
                # config,
                user_name,
                current_pyfile,
                __version__,
                date_time,
                clock_ns,
            )
        )
        for command in self._work_queue:
            command.execute(self._boxpool)
        if not self._work_queue:
            self.check_config()
            self._boxpool._box_config_cache.clear()
            self._boxpool = BoxPool()
            self.clear_log()
        # return status, iqs, config
        return results

    def check_config(self) -> None:
        box_configs = {
            box_name: self._boxpool.get_box(box_name)[0].dump_box()
            for box_name in self._boxpool._box_config_cache
        }
        for box_name, initial in self._boxpool._box_config_cache.items():
            if box_name not in box_configs:
                raise ValueError(f"The BoxPool is inconsistent with {box_name}")
            final = box_configs[box_name]
            if initial != final:
                logger.warning(
                    f"The box {box_name} configuration has changed since the start of the process: {initial} -> {final}"
                )

    def add_command(self, command: Command) -> None:
        self._work_queue.appendleft(command)

    def get_log(self) -> list:
        return list(self._config_buffer)

    def clear_log(self) -> None:
        self._config_buffer.clear()

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
        boxes = self.collect_boxes()
        # もし box が複数で clockmaster_setting が設定されていれば QuBEMasterClient を生成する
        if len(boxes) > 1 and self.sysdb._clockmaster_setting is not None:
            self._boxpool.create_clock_master(
                ipaddr=str(self.sysdb._clockmaster_setting.ipaddr)
            )
        # boxpool を生成する
        for box_name in boxes:
            setting = self.sysdb._box_settings[box_name]
            box = self._boxpool.create(
                box_name,
                ipaddr_wss=str(setting.ipaddr_wss),
                ipaddr_sss=str(setting.ipaddr_sss),
                ipaddr_css=str(setting.ipaddr_css),
                boxtype=setting.boxtype,
                # config_root=Path(setting.config_root),
                # if setting.config_root is not None
                # else None,
                # config_options=setting.config_options,
            )
            status = box.reconnect()
            for mxfe_idx, s in status.items():
                if not s:
                    logger.error(
                        f"be aware that mxfe-#{mxfe_idx} is not linked-up properly"
                    )

        # sequencer に measurement_option を設定する
        for sequencer in self.collect_sequencers():
            if sequencer.interval is None:
                new_interval = interval
            else:
                new_interval = sequencer.interval
            sequencer.set_measurement_option(
                repeats=repeats,
                interval=new_interval,
                integral_mode=integral_mode,
                dsp_demodulation=dsp_demodulation,
                software_demodulation=software_demodulation,
            )

        return self

    def _create_target_resource_map(
        self,
        target_names: Iterable[str],
    ) -> dict[
        str, Iterable[dict[str, BoxSetting | PortSetting | int | dict[str, float]]]
    ]:
        # {target_name: sampled_sequence} の形式から
        # TODO {target_name: {box, group, line | rline, channel | runit)} へ変換　？？
        # {target_name: {box, port, channel_number)}} へ変換
        db = self.sysdb
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

    def add_sequence(
        self,
        sequence: neopulse.Sequence,
        *,
        interval: Optional[float] = None,
        time_offset: dict[str, int] = {},  # {box_name: time_offset}
        time_to_start: dict[str, int] = {},  # {box_name: time_to_start}
    ) -> None:
        # TODO ここは仕様変更が必要
        # Readout send に位相合わせ機構を導入するため SebSequence にまとめてしまわず Slot 毎に分割しないといけない
        # 情報を失わせ過ぎた
        # capture に関連する gen_sequence を取り出して 変調 slice を作成する
        gen_sampled_sequence, cap_sampled_sequence = (
            sequence.convert_to_sampled_sequence()
        )

        items_by_target = sequence._get_group_items_by_target()

        targets = set(
            [gtarget for gtarget in gen_sampled_sequence]
            + [ctarget for ctarget in cap_sampled_sequence]
        )
        resource_map = self._create_target_resource_map(targets)

        self.add_command(
            Sequencer(
                gen_sampled_sequence=gen_sampled_sequence,
                cap_sampled_sequence=cap_sampled_sequence,
                resource_map=resource_map,
                group_items_by_target=items_by_target,
                time_offset=time_offset,
                time_to_start=time_to_start,
                interval=interval,
                sysdb=self.sysdb,
            )
        )


class BoxPool:
    SYSREF_PERIOD: int = 2_000
    DEFAULT_NUM_SYSREF_MEASUREMENTS: Final[int] = 100

    def __init__(self) -> None:
        self._clock_master = (
            None  # QuBEMasterClient(settings["CLOCK_MASTER"]["ipaddr"])
        )
        self._boxes: dict[str, tuple[Quel1BoxWithRawWss, SequencerClient]] = {}
        self._linkstatus: dict[str, bool] = {}
        self._estimated_timediff: dict[str, int] = {}
        self._cap_sysref_time_offset: int = 0
        self._port_direction: dict[tuple[str, int], str] = {}
        self._box_config_cache: dict[str, dict] = {}

    def create_clock_master(
        self,
        ipaddr: str,
    ) -> None:
        self._clock_master = QuBEMasterClient(master_ipaddr=ipaddr)

    def measure_timediff(
        self, num_iters: int = DEFAULT_NUM_SYSREF_MEASUREMENTS
    ) -> tuple[str, int]:
        sqcs = {name: sqc for name, (_, sqc) in self._boxes.items()}
        counter_at_sysref_clk = {name: 0 for name in self._boxes}
        for _ in range(num_iters):
            for name, sqc in sqcs.items():
                m = sqc.read_clock()
                if len(m) < 2:
                    raise RuntimeError("firmware doesn't support this measurement")
                counter_at_sysref_clk[name] += m[2] % self.SYSREF_PERIOD
        avg: dict[str, int] = {
            name: round(cntr / num_iters)
            for name, cntr in counter_at_sysref_clk.items()
        }
        refname = list(self._boxes.keys())[0]
        adj = avg[refname]
        self._estimated_timediff = {name: cntr - adj for ipaddr, cntr in avg.items()}
        self._cap_sysref_time_offset = avg[refname]
        return refname, avg[refname]

    def create(
        self,
        box_name: str,
        *,
        ipaddr_wss: str,
        ipaddr_sss: str,
        ipaddr_css: str,
        boxtype: Quel1BoxType,
        # config_root: Optional[Path],
        # config_options: Optional[Collection[Quel1ConfigOption]] = None,
    ) -> Quel1BoxWithRawWss:
        box = Quel1BoxWithRawWss.create(
            ipaddr_wss=ipaddr_wss,
            ipaddr_sss=ipaddr_sss,
            ipaddr_css=ipaddr_css,
            boxtype=boxtype,
            # config_root=config_root,
            # config_options=config_options,
        )
        sqc = SequencerClient(ipaddr_sss)
        self._boxes[box_name] = (box, sqc)
        self._linkstatus[box_name] = False
        return box

    def init(self, reconnect: bool = True, resync: bool = True) -> None:
        self.scan_link_status(reconnect=reconnect)
        self.reset_awg()
        if self._clock_master is None:
            return

        # if resync:
        #     self.resync()
        # if not self.check_clock():
        #     raise RuntimeError("failed to acquire time count from some clocks")

    def scan_link_status(
        self,
        reconnect: bool = False,
    ) -> None:
        for name, (box, sqc) in self._boxes.items():
            link_status: bool = True
            if reconnect:
                if not all(box.reconnect().values()):
                    if all(
                        box.reconnect(
                            ignore_crc_error_of_mxfe=box.css.get_all_groups()
                        ).values()
                    ):
                        logger.warning(
                            f"crc error has been detected on MxFEs of {name}"
                        )
                    else:
                        logger.error(
                            f"datalink between MxFE and FPGA of {name} is not working"
                        )
                        link_status = False
            else:
                if not all(box.link_status().values()):
                    if all(
                        box.link_status(
                            ignore_crc_error_of_mxfe=box.css.get_all_groups()
                        ).values()
                    ):
                        logger.warning(
                            f"crc error has been detected on MxFEs of {name}"
                        )
                    else:
                        logger.error(
                            f"datalink between MxFE and FPGA of {name} is not working"
                        )
                        link_status = False
            self._linkstatus[name] = link_status

    def reset_awg(self) -> None:
        for name, (box, _) in self._boxes.items():
            box.easy_stop_all(control_port_rfswitch=True)
            box.initialize_all_awgs()

    def get_box(
        self,
        name: str,
    ) -> tuple[Quel1BoxWithRawWss, SequencerClient]:
        if name in self._boxes:
            box, sqc = self._boxes[name]
            return box, sqc
        else:
            raise ValueError(f"invalid name of box: '{name}'")

    def get_port_direction(self, box_name: str, port: int) -> str:
        if (box_name, port) not in self._port_direction:
            box = self.get_box(box_name)[0]
            self._port_direction[(box_name, port)] = box.dump_port(port)["direction"]
        return self._port_direction[(box_name, port)]
