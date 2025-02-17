from __future__ import annotations

import logging
from pathlib import Path
from typing import Collection, Final, Optional

from quel_clock_master import QuBEMasterClient, SequencerClient
from quel_ic_config import (
    Quel1BoxType,
    Quel1BoxWithRawWss,
    Quel1ConfigOption,
)

logger = logging.getLogger(__name__)


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
        config_root: Optional[Path],
        config_options: Optional[Collection[Quel1ConfigOption]] = None,
    ) -> Quel1BoxWithRawWss:
        box = Quel1BoxWithRawWss.create(
            ipaddr_wss=ipaddr_wss,
            ipaddr_sss=ipaddr_sss,
            ipaddr_css=ipaddr_css,
            boxtype=boxtype,
            config_root=config_root,
            config_options=config_options,
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
