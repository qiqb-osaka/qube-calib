from __future__ import annotations

import logging
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from ipaddress import IPv4Address, IPv6Address, ip_address
from typing import Final, Optional

import numpy as np
import numpy.typing as npt
from quel_clock_master import QuBEMasterClient, SequencerClient
from quel_ic_config import Quel1BoxWithRawWss

from . import single

logger = logging.getLogger(__name__)


def complete_ipaddrs(
    ipaddr_wss: IPv4Address | IPv6Address,
) -> tuple[IPv4Address | IPv6Address, IPv4Address | IPv6Address]:
    ipaddr_sss = ipaddr_wss + (1 << 16)
    ipaddr_css = ipaddr_wss + (4 << 16)
    return ipaddr_sss, ipaddr_css


@dataclass
class E7Setting:
    pass


@dataclass
class TimingSetting(E7Setting):
    offset: int = 0
    time_to_start: int = 0


class BoxPool:
    SYSREF_PERIOD: Final[int] = 2000
    # TODO: tried to find the best value, but the best value changes link-up by link-up. so, calibration is required.
    TIMING_OFFSET: Final[int] = 0
    DEFAULT_NUM_SYSREF_MEASUREMENTS: Final[int] = 100

    def __init__(
        self,
        master_ipaddr: IPv4Address | IPv6Address,
        *,
        boxes: dict[str, Quel1BoxWithRawWss],
    ):
        """create a pool of boxes

        Parameters
        ----------
        ipaddr : IPv4Address | IPv6Address
            ip address of the clock master
        boxes : dict[str, Quel1Box]
            boxes to be controlled
        """
        self._clock_master = QuBEMasterClient(str(master_ipaddr))
        self._boxes = boxes
        self._sqcs = self._create_sqcs(self._boxes)
        self._linkstatus = {name: False for name in self._boxes}
        self._estimated_timediff = {name: 0 for name in self._boxes}
        self._cap_sysref_time_offset: int = 0

    def _create_sqcs(
        self, boxes: dict[str, Quel1BoxWithRawWss]
    ) -> dict[str, SequencerClient]:
        result = {}
        for name, box in boxes.items():
            ipaddr_sss, _ = complete_ipaddrs(ip_address(box._dev.wss._wss_addr))
            result[name] = SequencerClient(str(ipaddr_sss))
        return result

    def reset_awg(self) -> None:
        for box in self._boxes.values():
            box.easy_stop_all()
            box.initialize_all_awgs()
            box.initialize_all_capunits()

    def resync(self) -> None:
        self._clock_master.reset()  # TODO: confirm whether it is harmless or not.
        self._clock_master.kick_clock_synch(
            [sqc.ipaddress for sqc in self._sqcs.values()]
        )

    def check_clock(self) -> bool:
        valid_m, cntr_m = self._clock_master.read_clock()
        t = {}
        for name, sqc in self._sqcs.items():
            t[name] = sqc.read_clock()

        flag = True
        if valid_m:
            logger.info(f"master: {cntr_m:d}")
        else:
            flag = False
            logger.info("master: not found")

        for name, (valid, cntr, cntr_last_sysref) in t.items():
            if valid:
                logger.info(f"{name:s}: {cntr:d} {cntr_last_sysref:d}")
            else:
                flag = False
                logger.info(f"{name:s}: not found")
        return flag

    def get_box(self, name: str) -> Quel1BoxWithRawWss:
        if name in self._boxes:
            box = self._boxes[name]
            return box
        else:
            raise ValueError(f"invalid name of box: '{name}'")

    def get_sqc(self, name: str) -> SequencerClient:
        if name in self._boxes:
            sqc = self._sqcs[name]
            return sqc
        else:
            raise ValueError(f"invalid name of box: '{name}'")

    def get_linkstatus(self, name: str) -> bool:
        if name in self._boxes:
            return self._linkstatus[name]
        else:
            raise ValueError(f"invalid name of box: '{name}'")

    def measure_timediff(
        self, boxname: str, num_iters: int = DEFAULT_NUM_SYSREF_MEASUREMENTS
    ) -> None:
        counter_at_sysref_clk: dict[str, int] = {boxname: 0 for boxname in self._boxes}

        for _ in range(num_iters):
            for name, sqc in self._sqcs.items():
                m = sqc.read_clock()
                if len(m) < 2:
                    raise RuntimeError(
                        f"firmware of {name} doesn't support this measurement"
                    )
                counter_at_sysref_clk[name] += m[2] % self.SYSREF_PERIOD

        avg: dict[str, int] = {
            boxname: round(cntr / num_iters)
            for boxname, cntr in counter_at_sysref_clk.items()
        }
        adj = avg[boxname]
        self._estimated_timediff = {
            boxname: cntr - adj for boxname, cntr in avg.items()
        }
        logger.info(f"estimated time difference: {self._estimated_timediff}")

        self._cap_sysref_time_offset = avg[boxname]


class PulseGen:
    def __init__(self, box: Quel1BoxWithRawWss, port: int, channel: int) -> None:
        self.group, self.line = box._convert_any_port(port)
        if isinstance(self.line, str):
            raise ValueError("the port is not for AWG")
        self.awg = box.rmap.get_awg_of_channel(self.group, self.line, channel)
        self.awg_spec = (port, channel)


class Driver:
    SYSREF_PERIOD: int = 2_000

    def __init__(
        self,
        boxpool: BoxPool,
        settings: dict[str, list[single.E7Setting | E7Setting]],
    ) -> None:
        self._boxpool = boxpool
        single_settings = {
            name: [s for s in setting if isinstance(s, single.E7Setting)]
            for name, setting in settings.items()
        }
        self._drivers = {
            name: single.Driver(self._boxpool.get_box(name), setting)
            for name, setting in single_settings.items()
        }
        self._pulsegens = {
            name: {
                PulseGen(self._boxpool.get_box(name), port, channel)
                for port, channel in driver._channels
            }
            for name, driver in self._drivers.items()
        }
        self._timings: dict[str, TimingSetting] = {}
        for name, setting in settings.items():
            for s in setting:
                if isinstance(s, TimingSetting):
                    self._timings[name] = s

    def start(self, timeout: Optional[float] = None) -> Results:
        thunks = {
            name: driver.capture_start(timeout)
            for name, driver in self._drivers.items()
        }
        self.emit_at()
        return Results(self._drivers, thunks)
        # return {
        #     name: driver.get_results(thunks[name])
        #     for name, driver in self._drivers.items()
        # }

    def emit_at(self) -> None:
        if not len(self._pulsegens):
            logger.warning("no pulse generator to activate")

        MIN_TIME_OFFSET = 12_500_000

        bitmaps_by_box: dict[str, int] = {name: 0 for name in self._boxpool._boxes}
        for name, pgs in self._pulsegens.items():
            b = self._boxpool._boxes[name]
            for pg in pgs:
                awg_idx = b.rmap.get_awg_of_channel(
                    *(b._convert_output_channel(pg.awg_spec))
                )
                bitmaps_by_box[name] |= 1 << awg_idx

        # initialize awgs
        for name, pgs in self._pulsegens.items():
            b = self._boxpool._boxes[name]
            sqc = self._boxpool._sqcs[name]
            b.prepare_for_emission({pg.awg_spec for pg in pgs})
            # Notes: the following is not required actually, just for debug purpose.
            valid_read, current_time, last_sysref_time = sqc.read_clock()
            if valid_read:
                logger.info(
                    f"boxname: {name}, current time: {current_time}, "
                    f"sysref offset: {last_sysref_time % self.SYSREF_PERIOD}"
                )
            else:
                raise RuntimeError("failed to read current clock")

        for name in self._boxpool._boxes:
            self._boxpool.measure_timediff(name)

        fstsqc = list(self._boxpool._sqcs.values())[0]
        valid_read, current_time, last_sysref_time = fstsqc.read_clock()
        sysref_offset = self._boxpool._cap_sysref_time_offset
        logger.info(
            f"sysref offset: average: {sysref_offset},  latest: {last_sysref_time % self.SYSREF_PERIOD}"
        )
        if abs(last_sysref_time % self.SYSREF_PERIOD - sysref_offset) > 4:
            logger.warning("large fluctuation of sysref is detected on the FPGA")
        base_time = current_time + MIN_TIME_OFFSET
        tamate_offset = (16 - (base_time - sysref_offset) % 16) % 16
        base_time += tamate_offset

        sqcs = self._boxpool._sqcs
        timediff = self._boxpool._estimated_timediff
        timings = self._timings
        tts = {
            name: timings[name].time_to_start if name in timings else 0 for name in sqcs
        }
        offset = {name: timings[name].offset if name in timings else 0 for name in sqcs}
        bitmap = bitmaps_by_box

        for name, sqc in sqcs.items():
            t = base_time - timediff[name] + tts[name] + 16 * offset[name]
            valid_sched = sqc.add_sequencer(t, awg_bitmap=bitmap[name])
            if not valid_sched:
                raise RuntimeError("failed to schedule AWG start")
            logger.info(f"scheduled at {t}")


class Results:
    def __init__(
        self,
        drivers: dict[str, single.Driver],
        thunks: dict[str, dict[int, Future]],
    ) -> None:
        self._results = {
            name: driver.get_results(thunks[name]) for name, driver in drivers.items()
        }
        self._queue: Final[deque] = deque()

    @property
    def box(self) -> dict[str, single.Results]:
        return self._results

    def __iter__(self) -> Results:
        self._queue.clear()
        for name, box_result in self._results.items():
            for port, port_result in box_result.port.items():
                for runit, data in port_result.runit.items():
                    for sum_section, _ in enumerate(data.sum_section):
                        self._queue.appendleft((name, port, runit, sum_section))
        return self

    def __next__(self) -> tuple[tuple[str, int, int, int], npt.NDArray[np.complex64]]:
        if not self._queue:
            raise StopIteration
        name, port, runit, sum_section = self._queue.pop()
        return (name, port, runit, sum_section), self._results[name].port[port].runit[
            runit
        ].sum_section[sum_section]
