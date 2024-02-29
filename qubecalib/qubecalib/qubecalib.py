from __future__ import annotations

import logging
import os
import re
import warnings
from collections import defaultdict, deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from enum import EnumMeta
from ipaddress import IPv4Address, IPv6Address, ip_address
from pathlib import Path
from typing import (
    Any,
    Collection,
    Dict,
    Final,
    List,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import json5
import numpy as np
import numpy.typing as npt
from e7awgsw import CaptureParam, DspUnit, WaveSequence
from quel_clock_master import SequencerClient
from quel_ic_config import QUEL1_BOXTYPE_ALIAS, Quel1BoxType, Quel1ConfigOption
from quel_ic_config_utils import (
    CaptureReturnCode,
    SimpleBoxIntrinsic,
    create_box_objects,
)
from quel_ic_config_utils.linkupper import LinkupFpgaMxfe

from . import backendqube as backend
from . import neopulse as pulse
from .backendqube import Sections, Sideband
from .temp.general_looptest_common import BoxPool, PulseCap, PulseGen
from .temp.general_looptest_common_mod import (
    BoxPoolMod,
    PulseCapSinglebox,
    PulseGenSinglebox,
)
from .temp.quel1_wave_subsystem_mod import Quel1WaveSubsystemMod

PulseGen, PulseCap, CaptureParam

logger = logging.getLogger(__name__)


def _value_of(enum: EnumMeta, target: str) -> EnumMeta:
    if target not in [_.value for _ in enum]:
        raise ValueError(f"{target} is invalid")
    return [_ for _ in enum if _.value == target][0]


class BoxSettings:
    def __init__(
        self,
        dct: dict[str, dict[str, Any]],
    ):
        self._aliases: MutableMapping[str, str] = dict()
        self._contents: dict[str, BoxSetting] = dict()
        for k, v in dct.items():
            self.add_box_setting_dict(k, v)

    def add_box_setting_dict(
        self,
        boxname: str,
        setting: dict[str, Any],
    ) -> None:
        setting["boxtype"] = QUEL1_BOXTYPE_ALIAS[setting["boxtype"]]
        if "config_options" in setting:
            setting["config_options"] = [
                self._value_of(Quel1ConfigOption, _) for _ in setting["config_options"]
            ]
        self._contents[boxname] = BoxSetting(**setting)

    def _value_of(self, enum: EnumMeta, target: str) -> EnumMeta:
        if target not in [_.value for _ in enum]:
            raise ValueError(f"{target} is invalid")
        return [_ for _ in enum if _.value == target][0]

    @property
    def aliases(self) -> MutableMapping[str, str]:
        return self._aliases

    @aliases.setter
    def aliases(self, aliases: MutableMapping[str, str]) -> None:
        self._aliases = aliases

    def get_boxname(self, boxname_or_alias: str) -> str:
        if boxname_or_alias in self.aliases:
            return self.aliases[boxname_or_alias]
        elif boxname_or_alias in self._contents:
            return boxname_or_alias
        else:
            raise ValueError(f"boxname_or_alias: {boxname_or_alias} is not found")

    def __getitem__(self, key: str) -> BoxSetting:
        _key = self.get_boxname(key)
        return self._contents[_key]

    def __setitem__(self, key: str, value: BoxSetting) -> None:
        try:
            _key = self.get_boxname(key)
        except ValueError:
            _key = key
        self._contents[_key] = value


class QcJson(dict):
    def __init__(
        self,
        path_for_setting_file_of_boxes: Optional[str | os.PathLike] = None,
    ):
        if path_for_setting_file_of_boxes is None:
            raise ValueError("no filename")
        with open(Path(os.getcwd()) / Path(path_for_setting_file_of_boxes), "r") as f:
            dct = json5.load(f)
        super().__init__(**dct)


class QubeCalib:
    @classmethod
    def _exec(cls, command_queue: MutableSequence[dict]) -> None:
        pass

    def __init__(
        self,
        path_for_setting_file_of_boxes: Optional[str | os.PathLike] = None,
        settings: dict[str, Any] = {},
    ):
        self._boxpool: Optional[BoxPool] = None
        self._command_queue: MutableSequence = deque()
        self._clockmaster_setting: Optional[ClockMasterSetting] = None
        self._box_settings: BoxSettings = BoxSettings({})
        self._port_settings: dict[str, dict[str, str | int]] = {}
        self._target_settings: Final[dict[str, dict[str, str | Optional[float]]]] = {}
        self._default_exclude_boxnames: tuple = ()
        self._sequencer_setting = None
        self._band_by_target: Final[MutableMapping[str, MutableSequence[str]]] = {}
        self._captparam_settings = CaptureParamSettings()
        self._captparam_settings._band_by_targets = self._band_by_target
        if path_for_setting_file_of_boxes:
            self._load(path_for_setting_file_of_boxes)
        elif settings is not None:
            if "clockmaster_setting" in settings:
                self._clockmaster_setting = ClockMasterSetting(
                    **settings["clockmaster_setting"]
                )
            if "box_settings" in settings:
                for k, _ in settings["box_settings"].items():
                    self._box_settings.add_box_setting_dict(k, _)
            if "box_aliases" in settings:
                for k, _ in settings["box_aliases"].items():
                    self._box_settings._aliases[k] = _
            if "port_settings" in settings:
                for k, _ in settings["port_settings"].items():
                    self._port_settings[k] = _
            if "band_by_target" in settings:
                for k, _ in settings["band_by_target"].items():
                    self._band_by_target[k] = _

    def _load(
        self,
        path_for_setting_file_of_boxes: Optional[str | os.PathLike],
    ) -> None:
        if path_for_setting_file_of_boxes is None:
            raise ValueError("no filename")
        with open(Path(os.getcwd()) / Path(path_for_setting_file_of_boxes), "r") as f:
            jsn = json5.load(f)
            if "clockmaster_setting" in jsn:
                self._clockmaster_setting = ClockMasterSetting(
                    **jsn["clockmaster_setting"]
                )
            if "box_settings" in jsn:
                self._box_settings = BoxSettings(jsn["box_settings"])
            if "box_aliases" in jsn:
                self._box_settings.aliases = jsn["box_aliases"]
            if "port_settings" in jsn:
                self._port_settings = jsn["port_settings"]
            if "band_by_target" in jsn:
                for k, _ in jsn["band_by_target"].items():
                    self._band_by_target[k] = _

    @property
    def captparam_setting(self) -> CaptureParamSettings:
        return self._captparam_settings

    @property
    def boxname_by_ipaddrwss(self) -> dict[str, str]:
        return {
            str(self._box_settings[_].ipaddr_wss): _
            for _ in self._box_settings._contents
        }

    def get_defined_targets(self) -> dict[str, str]:
        return {_: _ for _ in self._band_by_target}

    def get_box_type(self, box_name_or_alias: str) -> Quel1BoxType:
        box_setting = self._box_settings[box_name_or_alias]
        return box_setting.boxtype

    def get_num_channels_of_port(self, box_name_or_alias: str, port: int) -> int:
        boxname = self._box_settings.get_boxname(box_name_or_alias)
        _, _, box, _ = self._create_box(boxname)
        group, line = self._convert_all_port(boxname, port)
        if isinstance(line, int):
            return box.css.get_num_channels_of_line(group, line)
        elif isinstance(line, str):
            return box.css.get_num_rchannels_of_rline(group, line)
        else:
            raise ValueError("invalid line value")

    def get_port_type(self, box_name_or_alias: str, port: int) -> str:
        boxname = self._box_settings.get_boxname(box_name_or_alias)
        _, _, box, _ = self._create_box(boxname)
        group, line = self._convert_all_port(boxname, port)
        if isinstance(line, int):
            return "gen"
        elif isinstance(line, str):
            return "cap"

    def _convert_all_port(self, box_name_or_alias: str, port: int) -> tuple[int, int]:
        settings = self._box_settings[box_name_or_alias].asdict()
        _, _, _, _, box = create_box_objects(refer_by_port=True, **settings)
        group, line = box._convert_all_port(port)
        return group, line

    # def connect(self):
    #     pass

    # def disconnect(self):
    #     pass

    def config_sequencer(self, duration: float, repeats: int = 1) -> None:
        self._sequencer_setting = SequencerSetting(duration, repeats)

    def convert_sequence(
        self, sequence: backend.Sequence
    ) -> list[tuple[str, WaveSequence | CaptureParam]]:
        if self._sequencer_setting is None:
            raise ValueError("exec QubeCalib.config_sequencer() is required")
        cmap = self.create_channelmap()
        section = backend.acquire_section(sequence, cmap)
        period = backend.quantize_sequence_duration(self._sequencer_setting.duration)
        sequence_setting = self.convert(
            sequence.flatten(),
            section,
            cmap,
            period,
            self._sequencer_setting.repeats,
        )
        return sequence_setting

    def parse_bandname(self, band_name: str) -> tuple[str, int]:
        r = re.match("^(MUX|Q)(\d+)(GEN|CAP)?B(\d)$", band_name)
        if r is None:
            raise ValueError(f"invalid bandname:{band_name}")
        port_name, channel = (
            r.group(1) + r.group(2) + (r.group(3) if r.group(3) is not None else ""),
            int(r.group(4)),
        )
        return port_name, channel

    def create_pulsegen_setting(self, band_name: str, waveseq: WaveSequence) -> dict:
        port_name, channel = self.parse_bandname(band_name)
        s = self._port_settings[port_name]
        port = int(s["port"])
        box_name_or_alias = str(s["box_name_or_alias"])
        group, line = self._convert_all_port(box_name_or_alias, port)

        setting = {
            "name": band_name,
            "port": port,
            "group": group,
            "line": line,
            "channel": channel,
            "waveseq": waveseq,
            # TODO ポート設定については後で考える
            # "lo_freq": s["lo_freq"],
            # "cnco_freq": s["cnco_freq"],
            # "sideband": s["sideband"],
            # "vatt": s["vatt"],
            # "fnco_freq": s["band"][channel],
        }
        return setting

    def create_pulsecap_setting(self, band_name: str, cptprm: CaptureParam) -> dict:
        port_name, channel = self.parse_bandname(band_name)
        s = self._port_settings[port_name]
        port = int(s["port"])
        box_name_or_alias = str(s["box_name_or_alias"])
        group, rline = self._convert_all_port(box_name_or_alias, port)
        setting = {
            "name": band_name,
            "port": port,
            "group": group,
            "rline": rline,
            "channel": channel,
            "capprm": cptprm,
            # TODO ポート設定については後で考える
            # "lo_freq": s["lo_freq"],
            # "cnco_freq": s["cnco_freq"],
            # "sideband": s["sideband"],
            # "fnco_freq": s["band"][0],
        }
        return setting

    def invoke_sequencer(self, sequence: backend.Sequence) -> Tuple[Any, Any]:
        settings: dict[str, Any] = {}
        sequence_setting = self.convert_sequence(sequence)
        box_settings = {
            "BOX" + self._box_settings.get_boxname(_): self._box_settings[_].asdict()
            for _ in {
                self._port_settings[port_name]["box_name_or_alias"]
                for port_name in {band_name[:-2] for band_name, _ in sequence_setting}
            }
        }
        pulsegen_settings = {
            band_name: self.create_pulsegen_setting(band_name, _)
            for band_name, _ in sequence_setting
            if isinstance(_, WaveSequence)
        }
        pulsecap_settings = {
            band_name: self.create_pulsecap_setting(band_name, _)
            for band_name, _ in sequence_setting
            if isinstance(_, CaptureParam)
        }
        captparam_settings = self._captparam_settings
        if len(box_settings) == 1:
            settings = {
                "box_settings": box_settings,
                "pulsegen_settings": pulsegen_settings,
                "pulsecap_settings": pulsecap_settings,
                "captparam_settings": captparam_settings,
            }
            return InvokeSequencerSinglebox(settings).execute()
        # print(len(box_settings), box_settings)
        if self._clockmaster_setting is None:
            raise ValueError("clockmaster setting is not defined")
        settings = {
            "clockmaster_setting": self._clockmaster_setting.asdict(),
            "box_settings": box_settings,
        }
        return InvokeSequencer(settings).execute()

    def debug_invoke_sequencer(self, sequence: backend.Sequence) -> Tuple[Any, Any]:
        settings: dict[str, Any] = {}
        sequence_setting = self.convert_sequence(sequence)
        box_settings = {
            "BOX" + self._box_settings.get_boxname(_): self._box_settings[_].asdict()
            for _ in {
                self._port_settings[port_name]["box_name_or_alias"]
                for port_name in {band_name[:-2] for band_name, _ in sequence_setting}
            }
        }
        pulsegen_settings = {
            band_name: self.create_pulsegen_setting(band_name, _)
            for band_name, _ in sequence_setting
            if isinstance(_, WaveSequence)
        }
        pulsecap_settings = {
            band_name: self.create_pulsecap_setting(band_name, _)
            for band_name, _ in sequence_setting
            if isinstance(_, CaptureParam)
        }
        captparam_settings = self._captparam_settings
        if len(box_settings) == 1:
            settings = {
                "box_settings": box_settings,
                "pulsegen_settings": pulsegen_settings,
                "pulsecap_settings": pulsecap_settings,
                "captparam_settings": captparam_settings,
            }
            return settings
        #     return InvokeSequencerSinglebox(settings).execute()
        # # print(len(box_settings), box_settings)
        # if self._clockmaster_setting is None:
        #     raise ValueError("clockmaster setting is not defined")
        # settings = {
        #     "clockmaster_setting": self._clockmaster_setting.asdict(),
        #     "box_settings": box_settings,
        # }
        # return InvokeSequencer(settings).execute()

    # def invoke_sequencer_singlebox(
    #     self,
    #     sequence_setting: list[tuple[str, WaveSequence | CaptureParam]],
    #     box_settings: dict[str, Any],
    # ) -> None:
    #     settings = {
    #         "box_settings": box_settings,
    #     }
    #     c = InvokeSequencerSinglebox(settings)
    #     c.execute()

    def dump_config(
        self,
        box_name_or_alias: str,
        port: Optional[int] = None,
    ) -> (
        dict[str, dict[int, dict[int | str, dict[str, Any]]]]
        | dict[int, dict[int | str, dict[str, Any]]]
    ):
        settings = self._box_settings[box_name_or_alias].asdict()
        css, wss, rmap, linkupper, box = create_box_objects(
            refer_by_port=True, **settings
        )
        box_dump_config = box.dump_config()
        if port is None:
            return box_dump_config
        else:
            return box_dump_config[f"port-#{port:02}"]

    def connect(
        self,
        *targets: str,
    ) -> None:
        if not targets:
            boxpool = self._create_all_boxes()
            for box, _ in boxpool._boxes.values():
                box.init()
        else:
            ports = {
                _[:-2] for _ in sum([self._band_by_target[_] for _ in targets], [])
            }
            box_names_or_aliases = {
                self._port_settings[_]["box_name_or_alias"] for _ in ports
            }
            boxnames = {self._box_settings.get_boxname(_) for _ in box_names_or_aliases}
            self._boxpool, b_and_s = self._create_boxes(*boxnames)
            for box, _ in b_and_s.values():
                box.init()

    def disconnect(self) -> None:
        self._boxpool = None

    # def _convert_all_port(
    #     self,
    #     box_name_or_alias: str,
    #     port: int,
    # ) -> tuple[int, int | str]:
    #     # TODO: Fix abuse API
    #     boxtype = self.get_box_type(box_name_or_alias)
    #     if port not in SimpleBox._PORT2LINE[boxtype]:
    #         raise ValueError(f"invalid port: {port}")
    #     return SimpleBox._PORT2LINE[boxtype][port]

    def set_clockmaster_setting(self, ipaddr: str | IPv4Address | IPv6Address) -> None:
        if isinstance(ipaddr, str):
            ipaddr = ip_address(ipaddr)
        self._clockmaster_setting = ClockMasterSetting(ipaddr=ipaddr, reset=True)

    def add_box_setting(
        self,
        boxname: str,
        ipaddr_wss: str | IPv4Address | IPv6Address,
        boxtype: Quel1BoxType,
        ipaddr_sss: Optional[str | IPv4Address | IPv6Address] = None,
        ipaddr_css: Optional[str | IPv4Address | IPv6Address] = None,
        config_root: Any = None,
        config_options: list[Quel1ConfigOption] = [],
    ) -> None:
        self._box_settings[boxname] = BoxSetting(
            ipaddr_wss=ipaddr_wss,
            ipaddr_sss=ipaddr_sss,
            ipaddr_css=ipaddr_css,
            boxtype=boxtype,
            config_root=config_root,
            config_options=config_options,
        )

    @property
    def exclude_boxnames(self) -> tuple:
        return self._default_exclude_boxnames

    @property
    def exclude_boxnames_or_aliases(self) -> tuple:
        raise ValueError("exclude_boxnames_or_aliases property can not access to read")

    @exclude_boxnames_or_aliases.setter
    def exclude_boxnames_or_aliases(self, boxnames_or_aliases: tuple[str]) -> None:
        default = self._default_exclude_boxnames
        self._default_exclude_boxnames = tuple(
            [self._box_settings.get_boxname(_) for _ in default + boxnames_or_aliases]
        )

    # def convert_sequence_to_setting(
    #     self,
    #     sequence: pulse.Sequence,
    #     channel_map: backend.ChannelMap,
    #     sequence_duration: int,
    #     repeats: int,
    # ) -> Any:
    #     section = backend.acquire_section(sequence, channel_map)
    #     period = backend.quantize_sequence_duration(sequence_duration)
    #     setup = backend.convert(
    #         sequence.flatten(), section, channel_map, period, repeats
    #     )
    #     return ()

    # def exec(self, pulse: pulse.Sequence) -> tuple[dict[str, Any]]:
    #     return ({"": None},)

    @classmethod
    def _create_channel(
        cls,
        boxpool: BoxPool,
    ) -> None:
        pass

    def _get_boxname_all(
        self, exclude_boxnames_or_aliases: tuple = tuple()
    ) -> list[str]:
        exclude_boxnames: list[str] = [
            self._box_settings.get_boxname(_) for _ in exclude_boxnames_or_aliases
        ]
        excludes = exclude_boxnames + list(self.exclude_boxnames)
        boxnames = [_ for _ in self._box_settings._contents if _ not in excludes]
        return boxnames

    def _create_clockmaster_setting(self) -> dict[str, dict[str, str | bool]]:
        if isinstance(self._clockmaster_setting, ClockMasterSetting):
            _clockmaster_setting = self._clockmaster_setting.asdict()
        return {
            "CLOCK_MASTER": _clockmaster_setting,
        }

    def _create_box(
        self,
        box_name_or_alias: str,
    ) -> tuple[BoxPool, str, SimpleBoxIntrinsic, SequencerClient]:
        _settings = self._create_clockmaster_setting()
        boxname = self._box_settings.get_boxname(box_name_or_alias)

        __settings = dict()
        __settings["BOX" + boxname] = self._box_settings[boxname].asdict()

        boxpool = BoxPool(_settings | __settings)
        box, sc = BoxPoolMod.get_box(boxpool, boxname)
        # box.init()
        return boxpool, boxname, box, sc

    def _create_boxes(
        self,
        *box_names_or_aliases: str,
    ) -> tuple[BoxPool, dict[str, tuple[SimpleBoxIntrinsic, SequencerClient]]]:
        _settings = self._create_clockmaster_setting()
        box_names = [self._box_settings.get_boxname(_) for _ in box_names_or_aliases]
        __settings = {"BOX" + _: self._box_settings[_].asdict() for _ in box_names}

        boxpool = BoxPool(_settings | __settings)
        boxes_and_seqctrls = {_: BoxPoolMod.get_box(boxpool, _) for _ in box_names}
        # box.init()
        return boxpool, boxes_and_seqctrls

    def _create_all_boxes(
        self,
        exclude_boxnames_or_aliases: tuple = tuple(),
    ) -> BoxPool:
        settings = self._create_clockmaster_setting()
        for _ in self._get_boxname_all(exclude_boxnames_or_aliases):
            settings["BOX" + _] = self._box_settings[_].asdict()
        boxpool = BoxPool(settings)
        # for k, (box, _) in boxpool._boxes.items():
        #     box.init()
        return boxpool

    def _show_clock(
        self,
        boxpool: BoxPool,
        boxname_or_alias: str,
    ) -> tuple[bool, int, int]:
        """同期用FPGA内部クロックの現在値を表示する

        Returns
        -------
        tuple[bool, int, int]
            _description_
        """
        boxname = self._box_settings.get_boxname(boxname_or_alias)
        return BoxPoolMod.get_box(boxpool, boxname)[1].read_clock()

    def show_clock_all(
        self,
        exclude_boxnames_or_aliases: tuple = tuple(),
    ) -> dict[str, tuple[bool, int, int] | tuple[bool, int]]:
        """登録されているすべての筐体の同期用FPGA内部クロックの現在値を表示する

        Returns
        -------
        tuple[tuple[bool, int], dict[str, tuple[bool, int, int]]]
            _description_
        """
        boxpool = self._create_all_boxes(exclude_boxnames_or_aliases)

        boxnames = list(BoxPoolMod.get_boxnames(boxpool))
        client_results = {_: self._show_clock(boxpool, _) for _ in boxnames}
        results: dict[str, tuple[bool, int, int] | tuple[bool, int]] = {
            k: v for k, v in client_results.items()
        }
        results["master"] = boxpool._clock_master.read_clock()
        return results

    def resync_box(
        self,
        boxname_or_alias: str,
    ) -> None:
        """resync したい機体だけを個別に resync する"""
        # boxpool = self._create_boxpool()
        # name, box, sc = self._create_box(
        #     boxpool=boxpool, boxname_or_alias=boxname_or_alias
        # )
        boxpool, _, _, sc = self._create_box(boxname_or_alias)
        boxpool._clock_master.kick_clock_synch((sc.ipaddress,))

    def linkup(
        self,
        boxname_or_alias: str,
        skip_init: bool = False,
        save_dirpath: Optional[Path] = None,
        background_noise_threshold: Optional[float] = 512,
    ) -> dict[int, bool]:
        # boxpool = self._create_boxpool()
        # _, box, _ = self._create_box(boxpool, boxname_or_alias)
        _, _, box, _ = self._create_box(boxname_or_alias)
        linkupper = LinkupFpgaMxfe(box.css, box.wss, box.rmap)

        mxfe_list: Sequence[int] = (0, 1)
        hard_reset: bool = False
        use_204b: bool = False
        ignore_crc_error_of_mxfe: Optional[Collection[int]] = None
        ignore_access_failure_of_adrf6780: Optional[Collection[int]] = None
        ignore_lock_failure_of_lmx2594: Optional[Collection[int]] = None
        ignore_extraordinal_converter_select_of_mxfe: Optional[Collection[int]] = None

        if ignore_crc_error_of_mxfe is None:
            ignore_crc_error_of_mxfe = {}

        if ignore_access_failure_of_adrf6780 is None:
            ignore_access_failure_of_adrf6780 = {}

        if ignore_lock_failure_of_lmx2594 is None:
            ignore_lock_failure_of_lmx2594 = {}

        if ignore_extraordinal_converter_select_of_mxfe is None:
            ignore_extraordinal_converter_select_of_mxfe = {}

        if not skip_init:
            linkupper._css.configure_peripherals(
                ignore_access_failure_of_adrf6780=ignore_access_failure_of_adrf6780,
                ignore_lock_failure_of_lmx2594=ignore_lock_failure_of_lmx2594,
            )
            linkupper._css.configure_all_mxfe_clocks(
                ignore_lock_failure_of_lmx2594=ignore_lock_failure_of_lmx2594,
            )
        linkup_ok: dict[int, bool] = {}
        for mxfe in mxfe_list:
            linkup_ok[mxfe] = linkupper.linkup_and_check(
                mxfe,
                hard_reset=hard_reset,
                use_204b=use_204b,
                background_noise_threshold=background_noise_threshold,
                ignore_crc_error=mxfe in ignore_crc_error_of_mxfe,
                ignore_extraordinal_converter_select=mxfe
                in ignore_extraordinal_converter_select_of_mxfe,
                save_dirpath=save_dirpath,
            )

        return linkup_ok

    # def get_port_property(
    #     self,
    #     box_name_or_alias: str,
    #     port: Optional[int],
    # ) -> None:
    #     s = self.dump_config(box_name_or_alias, port)

    def create_single_box(
        self, setting: Dict[str, Dict[str, Any]]
    ) -> Tuple[Any, SimpleBoxIntrinsic]:
        box_setting = setting
        box_name = {_ for _ in box_setting}.pop()

        _, _, _, _, box = create_box_objects(
            refer_by_port=False,
            **box_setting[box_name],
        )
        if not isinstance(box, SimpleBoxIntrinsic):
            raise ValueError(f"unsupported boxtype: {box_setting[box_name]['boxtype']}")

        link_status: bool = True
        if not box.init().values():
            # print(box.init(ignore_crc_error_of_mxfe=box.css.get_all_groups()).values())
            if box.init(ignore_crc_error_of_mxfe=box.css.get_all_groups()).values():
                logger.warning(f"crc error has been detected on MxFEs of {box_name}")
            else:
                logger.error(
                    f"datalink between MxFE and FPGA of {box_name} is not working"
                )
                link_status = False
        return link_status, box

    def create_target(
        self,
        target_name: str,
        frequency: Optional[float] = None,
    ) -> None:
        if target_name in self._target_settings:
            raise ValueError(f"{target_name} has already been created.")
        self._target_settings[target_name] = {"frequency": frequency}

    def edit_target(
        self,
        target_name: str,
        frequency: float,
    ) -> None:
        if target_name not in self._target_settings:
            raise ValueError(f"{target_name} is not found.")
        self._target_settings[target_name]["frequency"] = frequency

    @property
    def all_defined_targets(self) -> set[str]:
        return {_ for _ in self._band_by_target}

    def create_channelmap(self) -> backend.ChannelMap:
        # create inverse dict
        targets_by_band: dict[str, set[str]] = defaultdict(set)
        for target, bands in self._band_by_target.items():
            for _ in bands:
                targets_by_band[_].add(target)

        m = backend.ChannelMap()
        for band, targets in targets_by_band.items():
            m[band] = [
                backend.Target(
                    name=_,
                    frequency=self._target_settings[_]["frequency"],
                )
                for _ in targets
            ]
        return m

    def descide_direction_from_band_name(
        self,
        band_name: str,
    ) -> backend.Direction:  # TODO 外に出してもいい
        r = re.match("(Q\d+)B(\d)", band_name)
        if r is not None:
            return backend.Direction.TO_TARGET
        r = re.match("(MUX\d+GEN)B(\d)", band_name)
        if r is not None:
            return backend.Direction.TO_TARGET
        r = re.match("(MUX\d+CAP)B(\d)", band_name)
        if r is not None:
            return backend.Direction.FROM_TARGET
        raise ValueError("format is invalid")

    def convert(
        self,
        sequence: pulse.Sequence,
        section: Sections,
        channel_map: backend.ChannelMap,  # ここは周波数設定を内部に保持しないといけない
        period: float,
        repeats: int = 1,
        warn: bool = False,
    ) -> list[tuple[str, WaveSequence | CaptureParam]]:
        # channel_map = self.get_cannelmap()
        sequence = sequence.flatten()
        # channel2slot = r = organize_slots(sequence) # target -> slot
        channel2slot = sequence.slots
        a, c = backend.split_gen_cap(channel_map)  # target -> gen_port, cap_port
        # print("channel2slot", channel2slot)
        # print("gen", a)
        # print("cap", c)

        # Tx チャネルはデータ変調
        for target in a:
            if target.name not in channel2slot:
                with warnings.catch_warnings():
                    if not warn:
                        warnings.simplefilter("ignore")
                    warnings.warn(f"Channel {target} is Iqnored.")
                continue
            for slot in channel2slot[target.name]:
                if isinstance(backend.body(slot), pulse.SlotWithIQ):
                    # print(f"awg: {k.frequency}")
                    if target.frequency is None:
                        raise ValueError("frequency is None")
                    # print(a[target], self._port_settings)

                    if a[target].startswith("MUX"):
                        r = re.match("^(MUX\d+GEN)B(\d)$", a[target])
                        if r is None:
                            raise ValueError("Invalid port label")
                        port_name, band = r.group(1), int(r.group(2))
                    else:
                        r = re.match("^(Q\d+)B(\d)$", a[target])
                        if r is None:
                            raise ValueError("invalid port name")
                        port_name, band = r.group(1), int(r.group(2))

                    mp = {
                        "lo_freq": "lo",
                        "cnco_freq": "cnco",
                        "sideband": "sideband",
                    }
                    _ = {
                        mp[k]: v
                        for k, v in self._port_settings[port_name].items()
                        if k in ["lo_freq", "cnco_freq", "sideband"]
                    }
                    _["fnco"] = self._port_settings[port_name]["band"][band]
                    _["rf"] = target.frequency
                    m = calc_modulation_frequency(**_)  # TODO

                    # target.frequency が nan で設定される場合がある
                    if np.isnan(m):
                        # print("slot.iq", slot.iq)
                        slot.miq = slot.iq
                        break

                    t = slot.sampling_points
                    # slot.miq = slot.iq
                    slot.miq = slot.iq * np.exp(1j * 2 * np.pi * (m * t))
                    # print("slot.miq", slot.miq)

        # 各AWG/UNITの各セクション毎に属する Slot のリストの対応表を作る
        section2slots: dict[Sections, MutableSequence[pulse.Slot]] = {}
        for (
            band_name,
            sections,
        ) in section.items():  # k:band_name, v: List[TxSection] | List[RxSection]
            for vv in sections:  # vv: TxSection | RxSection
                if vv not in section2slots:
                    section2slots[vv] = deque([])
                if band_name not in channel_map:
                    with warnings.catch_warnings():
                        if not warn:
                            warnings.simplefilter("ignore")
                        warnings.warn(f"PhyChannel {band_name} is Iqnored.")
                    continue
                for target in channel_map[band_name]:  # c:Channel
                    if (
                        target.name not in channel2slot
                    ):  # sequence の定義内で使っていない論理チャネルがあり得る
                        with warnings.catch_warnings():
                            if not warn:
                                warnings.simplefilter("ignore")
                            warnings.warn(f"Logical Channel {c} is Iqnored.")
                        continue
                    for s in channel2slot[target.name]:
                        if vv.repeats == 1:
                            bawg = (
                                self.descide_direction_from_band_name(band_name)
                                == backend.Direction.TO_TARGET
                            ) and isinstance(backend.body(s), pulse.SlotWithIQ)
                        else:
                            bawg = (
                                self.descide_direction_from_band_name(band_name)
                                == backend.Direction.TO_TARGET
                            ) and isinstance(s, pulse.SlotWithIQ)
                        bunit = (
                            self.descide_direction_from_band_name(band_name)
                            == backend.Direction.FROM_TARGET
                        ) and isinstance(backend.body(s), pulse.Range)
                        if bawg:
                            if vv.begin <= s.begin and s.begin <= vv.end:
                                section2slots[vv].append(s)
                        elif bunit:
                            for i in range(vv.repeats):
                                if (
                                    vv.begin + i * vv.total <= s.begin
                                    and s.begin + s.duration <= vv.end + i * vv.total
                                ):
                                    section2slots[vv].append(s)

        # 各セクション毎に Chunk を合成する
        awgs = [
            band_name
            for band_name in channel_map
            if self.descide_direction_from_band_name(band_name)
            == backend.Direction.TO_TARGET
        ]
        for i, k in enumerate(awgs):
            if k in section:
                for s in section[k]:
                    t = s.sampling_points
                    s.iq[:] = 0
                    ss = section2slots[s]
                    for v in ss:
                        rng = (v.begin <= t) * (t < v.end)
                        s.iq[rng] += v.miq  # / len(ss)
                    if (
                        max(abs(np.real(s.iq))) > 32767
                        or max(abs(np.imag(s.iq))) > 32767
                    ):
                        raise ValueError("Exceeds the maximum allowable output.")
        # 束ねるチャネルを WaveSequence に変換
        awgs = [
            bname
            for bname in channel_map
            if self.descide_direction_from_band_name(bname)
            == backend.Direction.TO_TARGET
        ]  # channel_map から AWG 関連だけ抜き出す
        wseqs = [
            (k, backend.chunks2wseq(section[k], period, repeats))
            for k in awgs
            if k in section
        ]  # chunk obj を wseq へ変換する
        # return [(k, section[k]) for k in awgs if k in section]

        units = [
            bname
            for bname in channel_map
            if self.descide_direction_from_band_name(bname)
            == backend.Direction.FROM_TARGET
        ]
        capts = [
            (k, backend.sect2capt(section[k], period, repeats))
            for k in units
            if k in section
        ]

        # print(wseqs, capts)
        return wseqs + capts

        # sender = {
        #     f"SENDER{i}": QcPulseGenSetting(
        #         boxname=c.boxname, port=c.port, channel=c.channel, wave=w
        #     )
        #     for i, (c, w) in enumerate([(c, w) for c, w in wseqs])
        # }
        # capturer = {
        #     f"CAPTURER{i}": QcPulseCapSetting(
        #         boxname=c.boxname, port=c.port, channel=c.channel, captparam=p
        #     )
        #     for i, (c, p) in enumerate([(c, p) for c, p in capts])
        # }

        # return QcSetup((sender | capturer))


def calc_modulation_frequency(
    lo: int,
    cnco: float,
    rf: float,
    sideband: str,
    fnco: Optional[float] = None,
) -> float:
    if fnco is None:
        return calc_cap_modulation_frequency(lo, cnco, rf, sideband)
    else:
        return calc_gen_modulation_frequency(lo, cnco, rf, sideband, fnco)


def calc_cap_modulation_frequency(
    lo: int,
    cnco: float,
    rf: float,
    sideband: str,
) -> float:
    # frequency is normalized as GHz
    # if isinstance(self.line, int):
    #     raise ValueError("invalid line type")
    # retval = self.dump_config()

    # if not isinstance(retval["lo_hz"], (float, int)):
    #     raise ValueError()
    # lo_hz = float(retval["lo_hz"])
    lo_hz = lo

    # if not isinstance(retval["cnco_hz"], (float, int)):
    #     raise ValueError()
    # if_hz = retval["cnco_hz"]
    # rf_hz = rf_freq * 1e9
    if_hz = cnco
    rf_hz = rf  # * 1e9

    if sideband == Sideband.UpperSideBand.value:
        diff_hz = rf_hz - lo_hz - if_hz
    elif sideband == Sideband.LowerSideBand.value:
        diff_hz = lo_hz - if_hz - rf_hz
    else:
        raise ValueError("invalid ssb mode.")

    return diff_hz  # * 1e-9  # return value is normalized as GHz


def calc_gen_modulation_frequency(
    lo: int,
    cnco: float,
    rf: float,
    sideband: str,
    fnco: float,
) -> float:
    # frequency is normalized as GHz
    # if isinstance(self.line, str):
    #     raise ValueError("invalid line type")

    # retval = self.dump_config()
    # if not isinstance(retval["lo_hz"], (float, int)):
    #     raise ValueError()
    # if not isinstance(retval["cnco_hz"], (float, int)):
    #     raise ValueError()
    # if not isinstance(retval["channels"], list):
    #     raise ValueError()

    # lo_hz = retval["lo_hz"]
    # fnco_hz = retval["channels"][self.channel]["fnco_hz"]
    # if_hz = retval["cnco_hz"] + fnco_hz
    # rf_hz = rf_freq * 1e9
    lo_hz = lo
    if_hz = cnco + fnco
    rf_hz = rf  # * 1e9

    if sideband == Sideband.UpperSideBand.value:
        diff_hz = rf_hz - lo_hz - if_hz
    elif sideband == Sideband.LowerSideBand.value:
        diff_hz = lo_hz - if_hz - rf_hz
    else:
        raise ValueError("invalid ssb mode.")

    return diff_hz  # * 1e-9  # return value is normalized as GHz


@dataclass
class PulseGenSetting:
    boxname: str
    port: int
    channel: int
    wave: WaveSequence = None
    lo_freq: Optional[float] = None
    cnco_freq: Optional[float] = None


@dataclass
class PulseCapSetting:
    boxname: str
    port: int
    channel: int
    captparam: CaptureParam = None


@dataclass
class PortSetting:
    alias: str
    box_name_or_alias: str
    port: int
    lo_freq: float = -1
    cnco_freq: float = -1
    sideband: str = ""
    vatt: int = -1
    band: tuple = tuple()
    # boxpool: Optional[BoxPool] = None

    # def __post_init__(self) -> None:
    #     if self.boxpool is None:
    #         raise ValueError("boxpool is needed")


@dataclass
class ClockMasterSetting:
    ipaddr: str | IPv4Address | IPv6Address
    reset: bool = True

    def asdict(self) -> dict[str, str | bool]:
        return {
            "ipaddr": str(self.ipaddr),
            "reset": self.reset,
        }


@dataclass
class BoxSetting:
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
            self.ipaddr_css = self.ipaddr_wss + (1 << 16)
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

    def asjson(self) -> dict[str, Any]:
        pass


@dataclass
class Channel:
    boxname: str
    port: int
    lane: int
    freq: float
    lo_freq: float = -1
    cnco_freq: float = -1
    fnco_freq: float = -1
    sideband: str = ""
    vatt: int = -1


@dataclass
class TxChannel(Channel):
    wave: Optional[WaveSequence] = None
    pulsegen: PulseGen = field(init=False)


@dataclass
class RxChannel(Channel):
    cprm: Optional[CaptureParam] = None
    pulsecap: PulseCap = field(init=False)


@dataclass
class SequencerSetting:
    duration: float
    repeats: int


class CommandBase:
    def execute(self) -> Tuple[Any, Any]:
        return None, None


class InvokeSequencerSinglebox(CommandBase):
    def __init__(self, settings: dict[str, Any]):
        self._settings = settings

    def execute(
        self,
    ) -> (
        Tuple[
            Dict[Tuple[str, int, int], CaptureReturnCode],
            Dict[Tuple[str, int, int], npt.NDArray[np.complex64]],
        ]
        | Tuple[None, None]
    ):
        box_setting = self._settings["box_settings"]
        box_name = {_ for _ in box_setting}.pop()
        # TODO dump box するためだけにこれしてた
        # _, _, _, _, box = create_box_objects(
        #     refer_by_port=True,
        #     **box_setting[box_name],
        # )
        # self._box_name = box_name
        # self._box = box
        # self._linkstatus = False
        # self.init()
        # dump_box = box.dump_config()

        _, _, _, _, box = create_box_objects(
            refer_by_port=False,
            **box_setting[box_name],
        )
        if not isinstance(box, SimpleBoxIntrinsic):
            raise ValueError(f"unsupported boxtype: {box_setting[box_name]['boxtype']}")
        self._box_name = box_name
        self._box = box
        self._linkstatus = False

        self.init()

        # pgs = self.create_pulsegens(self._settings["pulsegen_settings"], dump_box)
        pgs = self.create_pulsegens(self._settings["pulsegen_settings"])
        pcs = self.create_pulsecaps(self._settings["pulsecap_settings"])

        # print([_.capprm.capture_delay for _ in pcs])
        self._settings["captparam_settings"].apply({_.name: _.capprm for _ in pcs})
        # print([_.capprm.capture_delay for _ in pcs])

        # 品質検査で重要だが，後で検討する．とりあえず無効化
        # cp.check_noise(show_graph=False)

        # boxname に属する pulsecap, pulsegen を集めた _pcs, _pgs を作る（単体の場合不要）
        # 各 box に対してループする（単体の場合不要）
        #     _pcs あり _pgs あり の場合 len(_pcs) and len(_pgs)
        #         self.capture_at_trigger_of() を実行し thread の queue に溜める
        #     _pcs あり _pgs なし の場合 len(_pcs) and not len(_pgs)
        #         self.capture_now() を実行する
        # box が単体の場合 self.emit_now() を実行する
        # box が複数の場合 self.emit_at() を実行する（単体の場合不要）
        # future に格納されたデータをデコードして status, iqs を返す

        status, iqs = {}, {}
        future = {}

        if not len(pgs) and not len(pcs):
            raise ValueError("no pulse setting")
        if len(pcs) and len(pgs):
            # print("self.capture_at_trigger()")
            triggering_pg = next(iter(pgs))
            future.update(self.capture_at_trigger_of(pcs, triggering_pg))
        else:
            # print("self.capture_now()")
            results = self.capture_now(pcs)
            for (_boxname, _port), (_status, _iqs) in results.items():
                # _status, _iqs = _future.result()
                __status = {(_boxname, _port, _channel): _status for _channel in _iqs}
                __iqs = {
                    (_boxname, _port, _channel): _iqs[_channel] for _channel in _iqs
                }
                status.update(__status)
                iqs.update(__iqs)

        self.emit_now(pgs)
        # print("self.emit_now()")

        if len(future):
            for (_boxname, _port), _future in future.items():
                _status, _iqs = _future.result()
                __status = {(_boxname, _port, _channel): _status for _channel in _iqs}
                __iqs = {
                    (_boxname, _port, _channel): _iqs[_channel] for _channel in _iqs
                }
                status.update(__status)
                iqs.update(__iqs)

        return status, iqs

    def init(self) -> None:
        box, name = self._box, self._box_name
        link_status: bool = True
        if not box.init().values():
            # print(box.init(ignore_crc_error_of_mxfe=box.css.get_all_groups()).values())
            if box.init(ignore_crc_error_of_mxfe=box.css.get_all_groups()).values():
                logger.warning(f"crc error has been detected on MxFEs of {name}")
            else:
                logger.error(f"datalink between MxFE and FPGA of {name} is not working")
                link_status = False
        self._linkstatus = link_status

        self.reset_awg()

    def reset_awg(self) -> None:
        if self._linkstatus:
            self._box.easy_stop_all(control_port_rfswitch=True)
            self._box.wss.initialize_all_awgs()

    def create_pulsegens(
        self,
        pulsegen_settings: dict[str, Any],
    ) -> Set[PulseGenSinglebox]:
        pgs: Set[PulseGenSinglebox] = set()
        for _, pulsegen_setting in pulsegen_settings.items():
            del pulsegen_setting["port"]
            pg = PulseGenSinglebox(
                box_status=self._linkstatus,
                box=self._box,
                **pulsegen_setting,
            )
            pg.init()
            pgs.add(pg)
        return pgs

    def create_pulsecaps(
        self,
        pulsecap_settings: dict[str, Any],
    ) -> Set[PulseCapSinglebox]:
        pcs: Set[PulseCapSinglebox] = set({})
        for pulsecap_name, pulsecap_setting in pulsecap_settings.items():
            del pulsecap_setting["port"]
            pc = PulseCapSinglebox(
                box_status=self._linkstatus,
                box=self._box,
                **pulsecap_setting,
            )
            pc.init()
            pcs.add(pc)
        return pcs

    def _capture_now(
        self,
        pcs: Set[PulseCapSinglebox],
    ) -> Tuple[CaptureReturnCode, Dict[int, npt.NDArray[np.complex64]]]:
        capmods = {_.capmod for _ in pcs}
        if len(capmods) != 1:
            raise ValueError("single capmod is estimated")
        wss = next(iter({_.box.wss for _ in pcs}))
        capmod = next(iter(capmods))
        capu_capprm = {_.channel: _.capprm for _ in pcs}
        capunits = tuple(capu_capprm.keys())
        num_expected_words = {capu: 0 for capu in capu_capprm.keys()}
        thunk = Quel1WaveSubsystemMod.simple_capture_start(
            wss,
            capmod,
            capunits,
            capu_capprm,
            num_expected_words=num_expected_words,
            triggering_awg=None,
        )
        status, iqs = thunk.result()
        return status, iqs

    def capture_now(
        self,
        pcs: Set[PulseCapSinglebox],
    ) -> Dict[
        Tuple[str, str], Tuple[CaptureReturnCode, Dict[int, npt.NDArray[np.complex64]]]
    ]:
        boxnames = {_.boxname for _ in pcs}
        if len(boxnames) != 1:
            raise ValueError("single box is estimated")
        boxname = next(iter(boxnames))

        result = {}
        for capmod in {pc.capmod for pc in pcs}:
            name = next(iter({pc.name for pc in pcs if pc.capmod == capmod}))
            _pcs = {pc for pc in pcs if pc.capmod == capmod}
            _status, _iqs = self._capture_now(_pcs)
            id = (boxname, name)
            result[id] = (_status, _iqs)

        return result

    def _capture_at_trigger_of(
        self,
        pcs: Set[PulseCapSinglebox],
        pg: PulseGenSinglebox,
    ) -> Future[Tuple[CaptureReturnCode, Dict[int, npt.NDArray[np.complex64]]]]:
        capmods = {pc.capmod for pc in pcs}
        if len(capmods) != 1:
            raise ValueError("single capmod is estimated")
        wss = next(iter({pc.box.wss for pc in pcs}))
        if pg.box.wss != wss:
            raise ValueError("can not be triggered by an awg of the other box")
        capmod = next(iter(capmods))
        capu_capprm = {pc.channel: pc.capprm for pc in pcs}
        capunits = tuple(capu_capprm.keys())
        num_expected_words = {capu: 0 for capu in capu_capprm.keys()}
        future = Quel1WaveSubsystemMod.simple_capture_start(
            wss,
            capmod,
            capunits,
            capu_capprm,
            num_expected_words=num_expected_words,
            triggering_awg=pg.awg,
        )
        return future

    def capture_at_trigger_of(
        self,
        pcs: Set[PulseCapSinglebox],
        _pg: PulseGenSinglebox,
    ) -> Dict[
        Tuple[str, Tuple[int, str]],
        Future[Tuple[CaptureReturnCode, Dict[int, npt.NDArray[np.complex64]]]],
    ]:
        box = {_.box for _ in pcs}
        if len(box) != 1:
            raise ValueError("single box is estimated")
        box = next(iter(box))
        # print(box)

        result = {}
        for capmod in {_.capmod for _ in pcs}:
            port = next(iter({(_.group, _.rline) for _ in pcs if _.capmod == capmod}))
            __pcs = {_ for _ in pcs if _.capmod == capmod}
            future = self._capture_at_trigger_of(__pcs, _pg)
            id = ("", port)
            result[id] = future

        return result

    def emit_now(self, pgs: Set[PulseGenSinglebox]) -> None:
        """単体のボックスで複数のポートから PulseGen に従ったパルスを出射する．パルスは同期して出射される．"""
        if len(pgs) == 0:
            logger.warn("no pulse generator to activate")

        self._box.wss.start_emission([_.awg for _ in pgs])

    def stop_now(self, pgs: Set[PulseGenSinglebox]) -> None:
        """PulseGen で指定された awg のバルスを停止する．"""
        if len(pgs) == 0:
            logger.warn("no pulse generator to activate")

        self._box.wss.stop_emission([_.awg for _ in pgs])


class InvokeSequencer(CommandBase):
    pass
    # def __init__(self, settings: dict[str, Any]):
    #     self._settings = settings

    # def execute(self) -> None:
    #     boxpool_setting = {
    #         "CLOCK_MASTER": self._settings["clockmaster_setting"],
    #     } | {
    #         box_name: box_setting
    #         for box_name, box_setting in self._settings["box_settings"].items()
    #     }
    #     # print(boxpool_setting)
    #     boxpool = BoxPool(boxpool_setting)
    #     # print([v for k, v in boxpool._boxes.items()])

    # # def _create_pulsegens(self, settings) -> None:
    # #     pgs = {PulseGen(name, boxpool)}

    # # def _create_pulsecaps(self, settings) -> None:
    # #     pass


class RetrieveCaptureResults(CommandBase):
    pass


class CaptureParamSettings:
    def __init__(self) -> None:
        self._settings: MutableMapping[str, CaptureParamSetting] = defaultdict(
            CaptureParamSetting
        )
        self._band_by_targets: MutableMapping[str, MutableSequence[str]] = {}

    def add_capture_delay(self, target_name: str, capture_delay: int) -> None:
        self._settings[target_name].additional_capture_delay = capture_delay

    def enable_integration(self, target_name: str) -> None:
        self._settings[target_name].dsp_units.append(CaptureParamDspIntegraton())

    def apply(self, captparam_by_bandname: MutableMapping[str, CaptureParam]) -> None:
        for band_name, captparam in captparam_by_bandname.items():
            self._settings[self.get_target_name(band_name)].apply(captparam)

    def set_band_by_target(
        self, band_by_target: MutableMapping[str, MutableSequence[str]]
    ) -> None:
        self._band_by_targets = band_by_target  # should sync with QubeCalib

    def get_target_name(self, band_name: str) -> str:
        target_by_band: MutableMapping[str, str] = defaultdict()
        for band, targets in self._band_by_targets.items():
            for target in targets:
                target_by_band[target] = band
        return target_by_band[band_name]


@dataclass
class CaptureParamSetting:
    additional_capture_delay: int = 0
    dsp_units: List[CaptureParamDsp] = field(default_factory=list)

    def apply(self, captparam: CaptureParam) -> None:
        captparam.capture_delay += self.additional_capture_delay
        dspunits = [_ for dsp in self.dsp_units for _ in dsp.dspunits]
        captparam.sel_dsp_units_to_enable(*dspunits)
        for dsp in self.dsp_units:
            dsp.apply(captparam)


@dataclass
class CaptureParamDsp:
    dspunits: Sequence[DspUnit]

    def apply(self, captparam: CaptureParam) -> None:
        pass


# @dataclass
# class CaptureParamDspDemodulation(CaptureParamDsp):
#     target_frequency: float
#     baseband_frequency: float
#     SAMPLING_PERIOD: float = 2 * nS
#     dspunits: Sequence[DspUnit] = (
#         DspUnit.COMPLEX_FIR,  # DSPのBPFを有効化
#         DspUnit.DECIMATION,  # DSPの間引1/4を有効化
#         DspUnit.COMPLEX_WINDOW,  # 複素窓関数を有効化
#     )

#     def apply(self, captparam: CaptureParam) -> None:
#         t = 4 * np.arange(p.NUM_COMPLEXW_WINDOW_COEFS) * SAMPLING_PERIOD
#         m = u.calc_modulation_frequency(rf_freq=o.frequency)
#         captparam.complex_window_coefs = list(
#             np.round((2**31 - 1) * np.exp(-1j * 2 * np.pi * (m * t)))
#         )
#         captparam.complex_fir_coefs = acquisition_fir_coefficient(
#             -m / MHz
#         )  # BPFの係数を設定

#     # from QubeServer.py by Tabuchi
#     # DSPのバンドパスフィルターを構成するFIRの係数を生成.
#     def acquisition_fir_coefficient(self, bb_frequency: float) -> List:
#         ADCBB_SAMPLE_R = 500
#         ACQ_MAX_FCOEF = (
#             16  # The maximum number of the FIR filter taps prior to decimation process.
#         )
#         ACQ_FCBIT_POW_HALF = 2**15  # equivalent to 2^(ACQ_FCOEF_BITS-1).

#         sigma = 100.0  # nanoseconds
#         freq_in_mhz = bb_frequency  # MHz
#         n_of_band = (
#             16  # The maximum number of the FIR filter taps prior to decimation process.
#         )
#         band_step = 500 / n_of_band
#         band_idx = int(freq_in_mhz / band_step + 0.5 + n_of_band) - n_of_band
#         band_center = band_step * band_idx
#         x = np.arange(ACQ_MAX_FCOEF) - (ACQ_MAX_FCOEF - 1) / 2
#         gaussian = np.exp(-0.5 * x**2 / (sigma**2))
#         phase_factor = (
#             2 * np.pi * (band_center / ADCBB_SAMPLE_R) * np.arange(ACQ_MAX_FCOEF)
#         )
#         coeffs = gaussian * np.exp(1j * phase_factor) * (1 - 1e-3)
#         return list(
#             (np.real(coeffs) * ACQ_FCBIT_POW_HALF).astype(int)
#             + 1j * (np.imag(coeffs) * ACQ_FCBIT_POW_HALF).astype(int)
#         )


@dataclass
class CaptureParamDspIntegraton(CaptureParamDsp):
    dspunits: Sequence[DspUnit] = (DspUnit.INTEGRATION,)


@dataclass
class CaptureParamDspSum(CaptureParamDsp):
    dspunits: Sequence[DspUnit] = (DspUnit.SUM,)


@dataclass
class CaptureParamDspClassification(CaptureParamDsp):
    decision_func_params: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
        (0, 0, 0),
        (0, 0, 0),
    )
    dspunits: Sequence[DspUnit] = (DspUnit.CLASSIFICATION,)
