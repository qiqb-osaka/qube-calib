from __future__ import annotations

import datetime
from logging import getLogger
from typing import Any, Final, NamedTuple, Optional, cast

from quel_ic_config import Quel1Box, Quel1PortType
from quel_ic_config_utils import configuration

from .....drivers.quel1.quel1driver import Quel1Driver
from .....e7utils import QuBEMasterClient
from . import single

logger = getLogger(__name__)


class NamedBox(NamedTuple):
    name: str
    box: Quel1Box


class BoxSetting(NamedTuple):
    name: str
    settings: list[single.AwgSetting | single.RunitSetting | single.TriggerSetting]


class Quel1System:
    def __init__(
        self,
        clockmaster: QuBEMasterClient,
        boxes: dict[str, Quel1Box],
        *,
        quel1driver: Quel1Driver | None = None,
    ) -> None:
        self._clockmaster: Final[QuBEMasterClient] = clockmaster
        self._boxes: Final[dict[str, Quel1Box]] = boxes
        self.displacement: int = 0
        self.timing_shift: Final[dict[str, int]] = {
            b: 0 for b in boxes
        }  # this parameter must be a multiple of 16
        self.config_cache: Final[dict[str, dict[str, Any]]] = {}
        self.monitor_input_ports: Final[dict[str, set[int | tuple[int, int]]]] = {}
        self.config_fetched_at: Optional[datetime.datetime] = None
        self.trigger: dict[
            tuple[str, Quel1PortType], tuple[str, Quel1PortType, int]
        ] = {}
        self._quel1driver: Final[Quel1Driver] = quel1driver or Quel1Driver()

    @classmethod
    def create_with_driver(
        cls,
        *,
        box_names: tuple[str, ...],
        clockmaster: QuBEMasterClient | None = None,
        update_config_cache: bool = True,
    ) -> Quel1System:
        conf = configuration.load_default_configuration()
        if clockmaster is None:
            clockmaster = QuBEMasterClient("0.0.0.0")
        if box_names is None:
            box_names = [box.name for box in conf.boxes]
        drv = Quel1Driver.from_configuration(list(box_names))
        boxes = cls.get_boxes(drv)
        self = cls(
            clockmaster=clockmaster,
            boxes=boxes,
            quel1driver=drv,
        )
        if update_config_cache:
            self.update_config_cache()
        return self

    @classmethod
    def create(
        cls,
        *,
        clockmaster: QuBEMasterClient,
        boxes: list[Quel1Box | NamedBox],
        update_copnfig_cache: bool = True,
    ) -> Quel1System:
        boxes_dict = {}
        for box in boxes:
            if isinstance(box, NamedBox):
                boxes_dict[box.name] = box.box
            else:
                boxes_dict[box._dev.wss._ipaddr_wss] = box
        self = cls(clockmaster, boxes_dict)
        if update_copnfig_cache:
            self.update_config_cache()
        return self

    @staticmethod
    def get_boxes(drv: Quel1Driver) -> dict[str, Quel1Box]:
        return {name: drv.get_box(name) for name in drv.list_active()}

    @property
    def boxes(self) -> dict[str, Quel1Box]:
        drv = self._quel1driver
        return self.get_boxes(drv)

    @property
    def box(self) -> dict[str, Quel1Box]:
        return self.boxes

    def release(self) -> None:
        boxes = list(self._boxes.keys())
        for name in boxes:
            del self._boxes[name]
        self._quel1driver.release()

    def initialize(self, *box_names: str) -> None:
        if not box_names:
            box_names = tuple(self.boxes.keys())
        for b in box_names:
            self.box[b].initialize_all_awgunits()
            self.box[b].initialize_all_capunits()

    def update_config_cache(self, *box_names: str) -> None:
        if not box_names:
            box_names = tuple(self.boxes.keys())
        self.config_cache.clear()
        self.monitor_input_ports.clear()
        for b in box_names:
            self.config_cache[b] = self.box[b].dump_box()
            self.monitor_input_ports[b] = self.box[b].get_monitor_input_ports()
        self.config_fetched_at = datetime.datetime.now()

    def dump_box(self, box_name: str) -> dict[str, Any]:
        if self.config_fetched_at is None:
            raise ValueError("config cache is empty")
        if box_name not in self.boxes:
            raise ValueError(f"box {box_name} not found in system")
        return self.config_cache[box_name]

    def dump_port(self, box_name: str, port: Quel1PortType) -> dict[str, Any]:
        if self.config_fetched_at is None:
            raise ValueError("config cache is empty")
        if box_name not in self.boxes:
            raise ValueError(f"box {box_name} not found in system")
        return self.config_cache[box_name]["ports"][port]

    def is_output_port(self, box_name: str, port: Quel1PortType) -> bool:
        if self.config_fetched_at is None:
            raise ValueError("config cache is empty")
        if box_name not in self.boxes:
            raise ValueError(f"box {box_name} not found in system")
        return self.config_cache[box_name]["ports"][port]["direction"] == "out"

    def is_input_port(self, box_name: str, port: Quel1PortType) -> bool:
        if self.config_fetched_at is None:
            raise ValueError("config cache is empty")
        if box_name not in self.boxes:
            raise ValueError(f"box {box_name} not found in system")
        return self.config_cache[box_name]["ports"][port]["direction"] == "in"

    def get_monitor_input_ports(self, box_name: str) -> set[int | tuple[int, int]]:
        if self.config_fetched_at is None:
            raise ValueError("config cache is empty")
        if box_name not in self.boxes:
            raise ValueError(f"box {box_name} not found in system")
        return self.monitor_input_ports[box_name]

    def get_lo_freq(self, box_name: str, port: Quel1PortType) -> float | None:
        port_cfg = self.dump_port(box_name, port)
        return cast(float, port_cfg["lo_freq"]) if "lo_freq" in port_cfg else None

    def get_cnco_freq(self, box_name: str, port: Quel1PortType) -> float:
        port_cfg = self.dump_port(box_name, port)
        return cast(float, port_cfg["cnco_freq"])

    def get_fnco_freq(self, box_name: str, port: Quel1PortType, channel: int) -> float:
        port_cfg = self.dump_port(box_name, port)
        if "channels" in port_cfg:
            key = "channels"
        elif "runits" in port_cfg:
            key = "runits"
        else:
            raise ValueError(
                f"no channel information found in port-{port} of {box_name}"
            )
        ch_cfgs = cast(dict[int, dict[str, float]], port_cfg[key])
        return ch_cfgs[channel]["fnco_freq"]

    def get_sideband(self, box_name: str, port: Quel1PortType) -> str | None:
        port_cfg = self.dump_port(box_name, port)
        return cast(str, port_cfg["sideband"]) if "sideband" in port_cfg else None
