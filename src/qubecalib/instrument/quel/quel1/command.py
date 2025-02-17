from __future__ import annotations

from typing import Optional

from ....command import Command
from .system import BoxPool


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
