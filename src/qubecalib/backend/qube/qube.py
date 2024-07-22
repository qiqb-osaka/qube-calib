from __future__ import annotations

from ipaddress import ip_address
from typing import Final

from quel_ic_config import Quel1Box, Quel1BoxType

from ... import neopulse as pls
from ..baseclasses import Backend
# from ..e7awg.driver import Driver as BoxDriver

DEFAULT_NUM_WORKERS: Final[int] = 4
DEFAULT_CAPTURE_TIMEOUT: Final[float] = 60.0


# @dataclass
# class E7Setting:
#     ipaddr: IPv4Address


# @dataclass
# class AwgSetting(E7Setting):
#     awg: AWG
#     wseq: WaveSequence


# @dataclass
# class CapuSetting(E7Setting):
#     capu: CaptureUnit
#     capprm: CaptureParam

#     @property
#     def capm(self) -> CaptureModule:
#         return CaptureUnit.get_module(self.capu)


class Qube(Backend):
    # def __init__(self) -> None:
    #     self._driver: Optional[Driver] = None

    # def create_e7driver(self, ipaddr: str) -> BoxDriver:
    #     return BoxDriver(ip_address(ipaddr))

    def create_box(self, ipaddr: str, boxtype: Quel1BoxType) -> Quel1Box:
        return Quel1Box.create(ipaddr_wss=ipaddr, boxtype=boxtype)

    def append_sequence(
        self,
        sequence: pls.Sequence,
        time_offset: dict[str, int] = {},
        time_to_start: dict[str, int] = {},
    ) -> None:
        print(sequence)
        print(time_offset)
        print(time_to_start)


# class Driver:
#     def __init__(self, clock_master_ipaddr: Optional[IPv4Address] = None) -> None:
#         self._executor = ThreadPoolExecutor(max_workers=DEFAULT_NUM_WORKERS)
#         self._awgctrl_lock = threading.Lock()
#         self._capctrl_lock = threading.Lock()
#         self._clock_master = (
#             QuBEMasterClient(str(clock_master_ipaddr)) if clock_master_ipaddr else None
#         )

#     def load(self, *settings: list[E7Setting]) -> None:
#         pass

#     def start(self, disabled: list[DspUnit]) -> None:
#         pass

#     def emit_at(self, *settings: list[AwgSetting]) -> None:
#         pass
