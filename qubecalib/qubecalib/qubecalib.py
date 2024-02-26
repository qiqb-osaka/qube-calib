from __future__ import annotations

import logging
import os
from collections import deque
from dataclasses import dataclass, field
from enum import EnumMeta
from ipaddress import IPv4Address, IPv6Address, ip_address
from pathlib import Path
from typing import (
    Any,
    Collection,
    MutableMapping,
    MutableSequence,
    Optional,
    Sequence,
)

import json5
from e7awgsw import CaptureParam, WaveSequence
from quel_ic_config import QUEL1_BOXTYPE_ALIAS, Quel1BoxType, Quel1ConfigOption
from quel_ic_config_utils import SimpleBox
from quel_ic_config_utils.linkupper import LinkupFpgaMxfe

from . import backendqube as backend
from . import neopulse as pulse
from .temp.general_looptest_common import BoxPool, PulseCap, PulseGen
from .temp.general_looptest_common_mod import BoxPoolMod

PulseGen, PulseCap, CaptureParam

logger = logging.getLogger(__name__)


def _value_of(enum: EnumMeta, target: str) -> EnumMeta:
    if target not in [_.value for _ in enum]:
        raise ValueError(f"{target} is invalid")
    return [_ for _ in enum if _.value == target][0]


# class SettingsForBox:
#     def __init__(self, dct: dict[str, Any]):
#         self._aliases: Mapping[str, str] = dict()
#         self._contents = dct
#         for v in self._contents.values():
#             v["boxtype"] = QUEL1_BOXTYPE_ALIAS[v["boxtype"]]
#             v["config_options"] = [
#                 self._value_of(Quel1ConfigOption, _) for _ in v["config_options"]
#             ]

#     def _value_of(self, enum: EnumMeta, target: str) -> EnumMeta:
#         if target not in [_.value for _ in enum]:
#             raise ValueError(f"{target} is invalid")
#         return [_ for _ in enum if _.value == target][0]

#     @property
#     def aliases(self) -> Mapping[str, str]:
#         return self._aliases

#     @aliases.setter
#     def aliases(self, aliases: Mapping[str, str]) -> None:
#         self._aliases = aliases

#     def get_boxname(self, boxname_or_alias: str) -> str:
#         if boxname_or_alias in self.aliases:
#             return self.aliases[boxname_or_alias]
#         elif boxname_or_alias in self._contents:
#             return boxname_or_alias
#         else:
#             raise ValueError(f"boxname_or_alias: {boxname_or_alias} is not found")

#     def __getitem__(self, key: Any) -> Any:
#         _key = self.get_boxname(key)
#         return self._contents[_key]


class BoxSettings:
    def __init__(
        self,
        dct: dict[str, dict[str, Any]],
    ):
        self._aliases: MutableMapping[str, str] = dict()
        self._contents: dict[str, BoxSetting] = dict()
        for k, v in dct.items():
            self.add_box_setting_dict(k, v)
            # v["boxtype"] = QUEL1_BOXTYPE_ALIAS[v["boxtype"]]
            # v["config_options"] = [
            #     self._value_of(Quel1ConfigOption, _) for _ in v["config_options"]
            # ]
            # self._contents[k] = BoxSetting(**v)

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
        self._command_queue: MutableSequence = deque()
        self._clockmaster_setting: Optional[ClockMasterSetting] = None
        self._box_settings: BoxSettings = BoxSettings({})
        self._port_settings: dict[str, dict[str, str | int]] = {}
        self._default_exclude_boxnames: tuple = ()
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
        # else:
        #     self._clockmaster_setting: Optional[ClockMasterSetting] = None
        #     self._box_settings: BoxSettings = BoxSettings({})
        #     # self._settings_for_box = SettingsForBox({})
        #     self._default_exclude_boxnames: tuple = ()
        #     self._port_settings: dict[str, dict[str, str | int]] = {}
        #     # self._setting_for_clockmaster = {"ipaddr": "10.3.0.255", "reset": True}

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
            # if "settings_for_box" in jsn:
            #     self._settings_for_box = SettingsForBox(jsn["settings_for_box"])
            # if "aliases_of_boxname" in jsn:
            #     self._settings_for_box.aliases = jsn["aliases_of_boxname"]
            if "port_settings" in jsn:
                self._port_settings = jsn["port_settings"]

    def get_box_type(self, box_name_or_alias: str) -> Quel1BoxType:
        box_setting = self._box_settings[box_name_or_alias]
        return box_setting.boxtype

    def _convert_all_port(
        self, box_name_or_alias: str, port: int
    ) -> tuple[int, int | str]:
        # TODO: Fix abuse API
        boxtype = self.get_box_type(box_name_or_alias)
        if port not in SimpleBox._PORT2LINE[boxtype]:
            raise ValueError(f"invalid port: {port}")
        return SimpleBox._PORT2LINE[boxtype][port]

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

    def convert_sequence_to_setting(
        self,
        sequence: pulse.Sequence,
        channel_map: backend.ChannelMap,
        sequence_duration: int,
        repeats: int,
    ) -> Any:
        section = backend.acquire_section(sequence, channel_map)
        period = backend.quantize_sequence_duration(sequence_duration)
        setup = backend.convert(
            sequence.flatten(), section, channel_map, period, repeats
        )
        return ()

    def exec(self, pulse: pulse.Sequence) -> tuple[dict[str, Any]]:
        return ({"": None},)

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

    # def _create_boxpool(self, exclude_boxnames_or_aliases: tuple = tuple()) -> BoxPool:
    #     if isinstance(self._clockmaster_setting, ClockMasterSetting):
    #         _clockmaster_setting = self._clockmaster_setting.asdict()
    #     settings = {
    #         "clockmaster_setting": _clockmaster_setting,
    #     }
    #     for _ in self._get_boxname_all():
    #         settings["BOX" + _] = self._box_settings

    #     # return BoxPoolMod.create_boxpool(settings=setting.asdict())
    #     return BoxPoolMod(settings)

    def _create_box(
        self,
        # boxpool: BoxPool,
        boxname_or_alias: str,
    ) -> tuple[BoxPool, str]:
        _settings = self._create_clockmaster_setting()
        boxname = self._box_settings.get_boxname(boxname_or_alias)

        # if "BOX" + boxname in boxpool._boxes:
        #     raise ValueError(f"box: {boxname} is alreaddy created")
        __settings = dict()
        __settings["BOX" + boxname] = self._box_settings[boxname].asdict()

        boxpool = BoxPool(_settings | __settings)
        #     BoxPoolMod.add_box(boxpool, settings=settings)
        return boxpool, boxname

    def _create_all_boxes(
        self,
        exclude_boxnames_or_aliases: tuple = tuple(),
    ) -> BoxPool:
        # exclude_boxnames: list[str] = [
        #     self._settings_for_box.get_boxname(_) for _ in exclude_boxnames_or_aliases
        # ]
        # excludes = exclude_boxnames + list(self.exclude_boxnames)
        # boxnames = [_ for _ in self._settings_for_box._contents if _ not in excludes]
        settings = self._create_clockmaster_setting()
        for _ in self._get_boxname_all(exclude_boxnames_or_aliases):
            settings["BOX" + _] = self._box_settings[_].asdict()

        # boxpool = self._create_boxpool()
        # for _ in boxnames:
        #     self._create_box(boxpool, _)
        # settings = {boxname: self._settings_for_box[boxname] for boxname in boxnames}
        # BoxPoolMod._parse_settings(boxpool, settings=settings)

        return BoxPool(settings)

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
        # if exclude_boxnames_or_aliases is tuple():
        #     exclude_boxnames_or_aliases = self.exclude_boxnames_or_aliases
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
        boxpool, boxname = self._create_box(boxname_or_alias)
        box, sc = BoxPoolMod.get_box(boxpool, boxname)
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
        boxpool, boxname = self._create_box(boxname_or_alias)
        box, _ = BoxPoolMod.get_box(boxpool, boxname)
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

    def port_config(self) -> None:
        pass


@dataclass
class PortConfig:
    name: str
    boxname: str
    port: int
    lo_freq: float = -1
    cnco_freq: float = -1
    sideband: str = ""
    vatt: int = -1
    fnco_freq: tuple = tuple()
    boxpool: Optional[BoxPool] = None

    def __post_init__(self) -> None:
        if self.boxpool is None:
            raise ValueError("boxpool is needed")


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
