"""後方互換性確保のためのモジュール群
"""

from __future__ import annotations

import os
import weakref
from enum import Enum
from ipaddress import IPv4Address, IPv6Address, ip_address
from typing import Any, Dict

from ..qcbox import QcBox, QcBoxFactory

# from tqdm import tqdm


class SSB(Enum):
    USB = "U"
    LSB = "L"


class QubeBase:
    pass


class ClassicQube(QubeBase):
    def __init__(self, qcbox: QcBox):
        _PORT_QuBE_OU_TypeA: Dict[int, type[Port]] = {
            0: TxPortC1,
            1: RxPort,
            2: TxPortC1,
            3: DummyPort,
            4: DummyPort,
            5: TxPortC3,
            6: TxPortC3,
            7: TxPortC3,
            8: TxPortC3,
            9: DummyPort,
            10: DummyPort,
            11: TxPortC1,
            12: RxPort,
            13: TxPortC1,
        }
        self._qcbox = qcbox
        self.port0 = None
        self.port1 = None
        self.port2 = None
        self.port3 = None
        self.port4 = None
        self.port5 = None
        self.port6 = None
        self.port7 = None
        self.port8 = None
        self.port9 = None
        self.port10 = None
        self.port11 = None
        self.port12 = None
        self.port13 = None
        for i in range(14):
            # p = getattr(self, f"port{i}")
            setattr(self, f"port{i}", _PORT_QuBE_OU_TypeA[i](self, i))
        self.gpio = GPIO(self)

    @property
    def ipmulti(self) -> IPv4Address | IPv6Address:
        return ip_address(self._qcbox._ipaddr_sss)

    @property
    def qcbox(self) -> QcBox:
        return self._qcbox

    def dump_config(self) -> dict[str, dict[str, Any]]:
        return self.qcbox.dump_config()


class Peripheral:
    def __init__(self, parent: Any):
        self._parent = weakref.ref(parent)

    @property
    def parent(self) -> Any:
        return self._parent()


class GPIO(Peripheral):
    def write_value(self, value: int) -> None:
        pass


class PeripheralWithID(Peripheral):
    def __init__(self, parent: Any, id: int):
        super().__init__(parent)
        self._id = id

    @property
    def id(self) -> int:
        return self._id


class Port(PeripheralWithID):
    def __init__(self, parent: ClassicQube, id: int):
        super().__init__(parent, id)
        self.nco = Nco(self, "cnco_hz")
        self.mix = Mix(self)

    @property
    def qube(self) -> ClassicQube:
        o = self._parent()
        if o is None:
            raise ValueError("qube object is missing.")
        return o

    def dump_config(self) -> dict[str, Any]:
        return self.qube.qcbox.dump_port_config(self.id)


class TxPort(Port):
    def __init__(self, parent: ClassicQube, id: int):
        super().__init__(parent, id)
        self.lo = Lo(self)
        self.nco = Nco(self, "cnco_hz")


class TxPortC1(TxPort):
    def __init__(self, parent: ClassicQube, id: int):
        super().__init__(parent, id)
        self.awg0 = AWG(self, 0)


class TxPortC3(TxPort):
    def __init__(self, parent: ClassicQube, id: int):
        super().__init__(parent, id)
        self.awg0 = AWG(self, 0)
        self.awg1 = AWG(self, 1)
        self.awg2 = AWG(self, 2)


class AWG(PeripheralWithID):
    def __init__(self, parent: TxPort, id: int):
        super().__init__(parent, id)
        self.nco = Nco(self, "fnco_hz")

    @property
    def port(self) -> TxPort:
        o = self._parent()
        if o is None:
            raise ValueError("port object is missing.")
        return o

    def calc_modulation_frequency(self, mhz: float) -> float:
        port = self.port
        lo_mhz = port.lo.mhz
        if_mhz = port.nco.mhz + self.nco.mhz
        rf_mhz = mhz
        if self.port.mix.ssb == SSB.USB:
            diff_mhz = rf_mhz - lo_mhz - if_mhz
        elif self.port.mix.ssb == SSB.LSB:
            diff_mhz = lo_mhz - if_mhz - rf_mhz
        else:
            raise ValueError("invalid ssb mode.")
        return diff_mhz

    def dump_config(self) -> dict[str, Any]:
        port_conf = self.port.dump_config()
        return port_conf["channels"][self.id]

    modulation_frequency = calc_modulation_frequency


class RxPort(Port):
    def __init__(self, parent: ClassicQube, id: int):
        super().__init__(parent, id)
        self.capt = CPT(self, 0)


class CPT(PeripheralWithID):
    def __init__(self, parent: RxPort, id: int):
        super().__init__(parent, id)
        self.unit0 = UNIT(self, 0)
        self.unit1 = UNIT(self, 1)
        self.unit2 = UNIT(self, 2)
        self.unit3 = UNIT(self, 3)

    @property
    def port(self) -> RxPort:
        return self.parent


class UNIT(PeripheralWithID):
    def __init__(self, parent: CPT, id: int):
        super().__init__(parent, id)

    @property
    def cpt(self) -> CPT:
        return self.parent


class DummyPort(Port):
    pass


class Lo(Peripheral):
    def __init__(self, parent: Port):
        super().__init__(parent)

    @property
    def port(self) -> Port:
        if self.parent is None:
            raise ValueError("port object is missing.")
        return self.parent

    @property
    def mhz(self) -> float:
        port_conf = self.port.dump_config()
        return port_conf["lo_hz"] * 1e-6

    @mhz.setter
    def mhz(self, mhz: float) -> None:
        box = self.port.qube.qcbox.box
        box.config_port(self.port.id, lo_freq=mhz * 1_000_000)


class Mix(Peripheral):
    def __init__(self, parent: Port):
        super().__init__(parent)

    @property
    def port(self) -> Port:
        if self.parent is None:
            raise ValueError("port object is missing.")
        return self.parent

    @property
    def ssb(self) -> SSB:
        port_conf = self.port.dump_config()
        if port_conf["sideband"] == "L":
            return SSB.LSB
        elif port_conf["sideband"] == "U":
            return SSB.USB
        else:
            raise ValueError("sideband report from equipment is invalid")

    @property
    def vatt(self) -> int:
        port_conf = self.port.dump_config()
        if "vatt" not in port_conf:
            raise ValueError("vatt is referenced before assigned")
        return port_conf["vatt"]

    @vatt.setter
    def vatt(self, value: int) -> None:
        box = self.port.qube.qcbox.box
        box.config_port(self.port.id, vatt=value)


class Nco(Peripheral):
    def __init__(self, parent: Port | AWG, key: str):
        super().__init__(parent)
        self._key = key

    @property
    def parent(self) -> Port | AWG:
        o = self._parent()
        if o is None:
            raise ValueError("parent object is missing.")
        return o

    @property
    def mhz(self) -> float:
        parent_conf = self.parent.dump_config()
        return parent_conf[self._key] * 1e-6

    @mhz.setter
    def mhz(self, mhz: float) -> None:
        if isinstance(self.parent, Port):
            box = self.parent.qube.qcbox.box
            port_id = self.parent.id
            box.config_port(port_id, cnco_freq=mhz * 1_000_000)
        else:
            box = self.parent.port.qube.qcbox.box
            port_id = self.parent.port.id
            channel_id = self.parent.id
            box.config_channel(port_id, channel_id, fnco_freq=mhz * 1_000_000)


class Qube:
    @classmethod
    def create(cls, config_path: str | os.PathLike) -> ClassicQube:
        return ClassicQube(QcBoxFactory.produce(config_path))
