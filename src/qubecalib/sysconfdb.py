from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from ipaddress import IPv4Address, IPv6Address, ip_address
from pathlib import Path
from typing import Any, Final, MutableSequence, Optional, Set

import yaml
from quel_clock_master import QuBEMasterClient
from quel_ic_config import (
    QUEL1_BOXTYPE_ALIAS,
    Quel1BoxType,
    Quel1BoxWithRawWss,
    Quel1ConfigOption,
)

from .instrument.quel.quel1 import driver as direct
from .instrument.quel.quel1.driver import Quel1PortType

DEFAULT_SIDEBAND = "U"

logger = logging.getLogger(__name__)


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
        self.timing_shift: Final[dict[str, int]] = {}
        self.skew: Final[dict[str, int]] = {}
        self.trigger: dict[tuple[str, int], tuple[str, Quel1PortType, int]] = {}
        self.time_to_start: int = 0

    @property
    def box_settings(self) -> dict[str, BoxSetting]:
        return self._box_settings

    def copy(self) -> SystemConfigDatabase:
        """Return a copy of the current instance."""
        return deepcopy(self)

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
        port_settings: Optional[dict[str, dict[str, object]]] = None,
        relation_channel_target: Optional[MutableSequence[tuple[str, str]]] = None,
        target_settings: Optional[dict[str, dict[str, object]]] = None,
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

    def load_box_yaml(self, filename: str) -> None:
        with open(Path(os.getcwd()) / Path(filename), "r") as file:
            yaml_dict = yaml.safe_load(file)
        self._load_box_yaml(yaml_dict)

    def _load_box_yaml(self, yaml_dict: dict) -> None:
        for name, setting in yaml_dict.items():
            self.add_box_setting(
                box_name=name,
                ipaddr_wss=setting["address"],
                boxtype=setting["type"],
                adapter=setting["adapter"],
            )

    def load_skew_yaml(self, filename: str) -> None:
        with open(Path(os.getcwd()) / Path(filename), "r") as file:
            yaml_dict = yaml.safe_load(file)
        self._load_skew_yaml(yaml_dict)

    def _load_skew_yaml(self, yaml_dict: dict) -> None:
        for name, setting in yaml_dict["box_setting"].items():
            self.timing_shift[name] = setting["slot"] * 16
            self.skew[name] = setting["wait"]
        self.time_to_start = yaml_dict["time_to_start"]

    def add_box_setting(
        self,
        box_name: str,
        ipaddr_wss: str | IPv4Address | IPv6Address,
        boxtype: Quel1BoxType,
        ipaddr_sss: Optional[str | IPv4Address | IPv6Address] = None,
        ipaddr_css: Optional[str | IPv4Address | IPv6Address] = None,
        config_root: Optional[str | os.PathLike] = None,
        config_options: MutableSequence[Quel1ConfigOption] = [],
        adapter: str | None = None,
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
            adapter=adapter,
        )

    def add_port_setting(
        self,
        port_name: str,
        box_name: str,
        port: Quel1PortType,
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

    def assign_target_to_channel(self, *, target: str, channel: str) -> None:
        self._relation_channel_target.append((channel, target))

    def define_target(
        self,
        target: str,
        *,
        frequency: float = 0,
        channels: list[str] | None = None,
    ) -> None:
        if target in self._target_settings:
            raise ValueError(f"target {target} is already defined")
        self._target_settings[target] = {"frequency": frequency}
        if channels is not None:
            for channel in channels:
                if (channel, target) not in self._relation_channel_target:
                    self.assign_target_to_channel(channel=channel, target=target)

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
    ) -> __builtins__.set[Quel1PortType]:
        return {
            self._port_settings[ports].port
            for ports in self.get_ports_by_target(target_name)
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
        adapter: str | None = None,
    ) -> dict[str, object]:
        box_setting = BoxSetting(
            box_name=box_name,
            ipaddr_wss=ipaddr_wss,
            boxtype=QUEL1_BOXTYPE_ALIAS[boxtype],
            config_options=config_options,
            ipaddr_sss=ipaddr_sss,
            ipaddr_css=ipaddr_css,
            config_root=config_root,
            adapter=adapter,
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
        port_number: Quel1PortType,
        lo_freq: Optional[float] = None,
        cnco_freq: Optional[float] = None,
        sideband: str = DEFAULT_SIDEBAND,
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
    ) -> Quel1BoxWithRawWss:
        s = self._box_settings[box_name]
        box = Quel1BoxWithRawWss.create(
            ipaddr_wss=str(s.ipaddr_wss),
            ipaddr_sss=str(s.ipaddr_sss),
            ipaddr_css=str(s.ipaddr_css),
            boxtype=s.boxtype,
        )
        if reconnect:
            if not all([_ for _ in box.link_status().values()]):
                box.relinkup(use_204b=False, background_noise_threshold=350)
            status = box.reconnect()
            for mxfe_idx, _ in status.items():
                if not _:
                    logger.error(
                        f"be aware that mxfe-#{mxfe_idx} is not linked-up properly"
                    )
        return box

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

    def create_quel1system(self, *box_names: str) -> direct.Quel1System:
        if self._clockmaster_setting is None:
            raise ValueError("clock master is not found")
            # TODO : ここは例外を投げるのではなく、 None を設定するようにし，　single box モードを設ける?
        system = direct.Quel1System.create(
            clockmaster=QuBEMasterClient(self._clockmaster_setting.ipaddr),
            boxes=[self.create_named_box(b) for b in box_names],
        )
        return system

    def asdict(self) -> dict[str, object]:
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

    def get_target_name(
        self,
        *,
        box_name: str,
    ) -> Set[tuple[str, str]]:
        ps = self._port_settings
        port_names = {n for n, s in ps.items() if s.box_name == box_name}
        rcp = self._relation_channel_port
        channel_names = {n for n, c in rcp if c["port_name"] in port_names}
        rct = self._relation_channel_target
        relation_target_channel = {(t, c) for c, t in rct if c in channel_names}
        return relation_target_channel

    def get_targets_by_box(
        self,
        box_name: str,
    ) -> Set[tuple[str, str]]:
        return self.get_target_name(box_name=box_name)

    def get_target_by_port(
        self,
        *,
        box_name: str,
        port: int,
    ) -> Set[tuple[str, str]]:
        return self.get_targets_by_port(
            box_name=box_name,
            port=port,
        )

    def get_targets_by_port(
        self,
        *,
        box_name: str,
        port: int,
    ) -> Set[tuple[str, str]]:
        ps = self._port_settings
        port_names = {
            n for n, s in ps.items() if s.box_name == box_name and s.port == port
        }
        rcp = self._relation_channel_port
        channel_names = {n for n, c in rcp if c["port_name"] in port_names}
        rct = self._relation_channel_target
        relation_target_channel = {(t, c) for c, t in rct if c in channel_names}
        return relation_target_channel

    def get_targets_by_channel(
        self,
        box_name: str,
        port: int,
        channel: int,
    ) -> Set[str]:
        relation_target_channel = self.get_targets_by_port(box_name=box_name, port=port)
        channels = {c for _, c in relation_target_channel}
        if not channels:
            raise ValueError(
                f"no target is assigned to the channel {box_name, port, channel}"
            )
        try:
            channel_id = {
                p["channel_number"]: c
                for c, p in self._relation_channel_port
                if c in channels
            }[channel]
        except KeyError:
            raise ValueError(f"invalid channel number {box_name, port, channel}")
        targets = {t for t, c in relation_target_channel if c == channel_id}
        if not targets:
            raise ValueError(
                f"no target is assigned to the channel {box_name, port, channel}"
            )
        return targets

    def get_target_by_channel(
        self,
        box_name: str,
        port: int,
        channel: int,
    ) -> str:
        relation_target_channel = self.get_target_by_port(box_name=box_name, port=port)
        channels = {c for t, c in relation_target_channel}
        if not channels:
            raise ValueError(
                f"no target is assigned to the channel {box_name, port, channel}"
            )
        try:
            channel_id = {
                p["channel_number"]: c
                for c, p in self._relation_channel_port
                if c in channels
            }[channel]
        except KeyError:
            raise ValueError(f"invalid channel number {box_name, port, channel}")
        targets = {t for t, c in relation_target_channel if c == channel_id}
        if not targets:
            raise ValueError(
                f"no target is assigned to the channel {box_name, port, channel}"
            )
        return next(iter(targets))


@dataclass
class ClockmasterSetting:
    ipaddr: str | IPv4Address | IPv6Address
    reset: bool

    def asdict(self) -> dict[str, object]:
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
    adapter: str | None = None

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
            "adapter": self.adapter,
        }

    def asjsonable(self) -> dict[str, Any]:
        dct = self.asdict()
        dct["boxtype"] = {v: k for k, v in QUEL1_BOXTYPE_ALIAS.items()}[dct["boxtype"]]
        return dct


@dataclass
class PortSetting:
    port_name: str
    box_name: str
    port: Quel1PortType
    lo_freq: Optional[float] = None  # will be obsolete
    cnco_freq: Optional[float] = None  # will be obsolete
    sideband: str = DEFAULT_SIDEBAND  # will be obsolete
    vatt: int = 0x800  # will be obsolete
    fnco_freq: Optional[tuple[float, ...]] = None  # will be obsolete
    ndelay_or_nwait: tuple[int, ...] = ()

    def asdict(self) -> dict[str, object]:
        return {
            "port_name": self.port_name,
            "box_name": self.box_name,
            "port": self.port,
            "ndelay_or_nwait": self.ndelay_or_nwait,
        }
