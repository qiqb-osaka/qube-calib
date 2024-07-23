from __future__ import annotations

import logging
from collections import UserList, defaultdict
from dataclasses import dataclass
from ipaddress import IPv4Address, IPv6Address, ip_address
from typing import Final, Optional

from e7awgsw import AWG, CaptureModule, CaptureParam, CaptureUnit, WaveSequence
from quel_clock_master import QuBEMasterClient, SequencerClient

from . import single
from .single import Result

LOGGER = logging.getLogger(__name__)
MIN_TIME_OFFSET: int = 12_500_000
SYSREF_PERIOD: int = 2_000
DEFAULT_NUM_SYSREF_MEASUREMENTS: Final[int] = 100


def get_logger() -> logging.Logger:
    return LOGGER


@dataclass
class Results:
    results: dict[tuple[str, CaptureUnit], Result]


@dataclass
class E7Setting:
    ipaddr: IPv4Address | IPv6Address


@dataclass
class AwgSetting(E7Setting):
    awg: AWG
    wseq: WaveSequence


@dataclass
class CapuSetting(E7Setting):
    capu: CaptureUnit
    cprm: CaptureParam


@dataclass
class TriggerSetting(E7Setting):
    capm: CaptureModule
    awg: AWG


@dataclass
class TimingSetting(E7Setting):
    displacement: int
    time_to_start: int


@dataclass
class ClockmasterSetting(E7Setting):
    pass


class Setting(UserList):
    def append_awg(
        self, ipaddr: IPv4Address | IPv6Address, awg: AWG, wseq: WaveSequence
    ) -> Setting:
        self.append(AwgSetting(ipaddr, awg, wseq))
        return self

    def append_capu(
        self, ipaddr: IPv4Address | IPv6Address, capu: CaptureUnit, cprm: CaptureParam
    ) -> Setting:
        self.append(CapuSetting(ipaddr, capu, cprm))
        return self

    def append_trig(
        self, ipaddr: IPv4Address | IPv6Address, capm: CaptureModule, awg: AWG
    ) -> Setting:
        self.append(TriggerSetting(ipaddr, capm, awg))
        return self

    def append_master(self, ipaddr: IPv4Address | IPv6Address) -> Setting:
        self.append(ClockmasterSetting(ipaddr))
        return self

    def append_timing(
        self, ipaddr: IPv4Address | IPv6Address, displacement: int, time_to_start: int
    ) -> Setting:
        self.append(TimingSetting(ipaddr, displacement, time_to_start))
        return self

    @classmethod
    def get_module(cls, capu: CaptureUnit) -> CaptureModule:
        return CaptureUnit.get_module(capu)


class SeqClient:
    def __init__(
        self, ipaddr: IPv4Address | IPv6Address, timing: Optional[TimingSetting] = None
    ) -> None:
        self._ipaddr = ipaddr
        self._core = SequencerClient(str(ipaddr + (1 << 16)))
        self._cap_sysref_time_offset = 0
        self.estimated_timediff = 0
        if timing is not None:
            self.displacement = timing.displacement
            self.time_to_start = timing.time_to_start
        else:
            self.displacement = 0
            self.time_to_start = 0
        self._scheduled_time = 0

    def check_flucturation(self) -> int:
        _, current_time, last_sysref_time = self._core.read_clock()
        LOGGER.debug(
            f"sysref offset: average: {self._cap_sysref_time_offset}, latest: {last_sysref_time % SYSREF_PERIOD}"
        )
        if abs(last_sysref_time % SYSREF_PERIOD - self._cap_sysref_time_offset) > 4:
            LOGGER.warning(
                f"large fluctuation of sysref is detected on FPGA {self._ipaddr}"
            )
        return current_time

    def read_clock(self) -> tuple[bool, int, int]:
        return self._core.read_clock()

    def add_sequencer(
        self,
        base_time: int,
        *,
        awg_bitmap: int,
    ) -> bool:
        self._scheduled_time = base_time
        return self._core.add_sequencer(base_time, awg_bitmap=awg_bitmap)


class Driver:
    def __init__(
        self,
        # ipaddr: Optional[IPv4Address | IPv6Address],
        settings: list[E7Setting],
    ) -> None:
        self._cap_sysref_time_offset = 0

        self._validate_settings(settings)

        clock_master_setting = self._pop_clock_master_setting(settings)
        self._clock_master = self._create_clock_master(clock_master_setting)

        driver_settings_by_ipaddrs: dict[
            IPv4Address | IPv6Address,
            list[single.E7Setting],
        ] = defaultdict(list)
        capu_settings = [s for s in settings if isinstance(s, CapuSetting)]
        for cs in capu_settings:
            driver_settings_by_ipaddrs[ip_address(cs.ipaddr)].append(
                single.CapuSetting(capu=cs.capu, cprm=cs.cprm)
            )
            settings.remove(cs)
        other_settings = [
            _
            for _ in settings
            if isinstance(_, AwgSetting) or isinstance(_, TriggerSetting)
        ]
        for os in other_settings:
            obj = (
                single.AwgSetting(awg=os.awg, wseq=os.wseq)
                if isinstance(os, AwgSetting)
                else single.TriggerSetting(capm=os.capm, awg=os.awg)
                if isinstance(os, TriggerSetting)
                else None
            )
            if obj is None:
                raise ValueError(f"Unsupported stting type: {os}")
            driver_settings_by_ipaddrs[ip_address(os.ipaddr)].append(obj)
            settings.remove(os)

        self._drivers_by_ipaddrs = {
            str(ip): single.Driver(ip, s)
            for ip, s in driver_settings_by_ipaddrs.items()
        }

        sqc_settings_by_ipaddrs: dict[
            IPv4Address | IPv6Address, Optional[TimingSetting]
        ] = {ip: None for ip in driver_settings_by_ipaddrs}
        sqc_settings = [s for s in settings if isinstance(s, TimingSetting)]
        for ss in sqc_settings:
            sqc_settings_by_ipaddrs[ip_address(ss.ipaddr)] = ss
            settings.remove(ss)

        self._seqclients_by_ipaddrs = {
            str(ip): SeqClient(ip, s) for ip, s in sqc_settings_by_ipaddrs.items()
        }

        if len(settings) != 0:
            raise ValueError(f"Unsupported stting type: {settings}")

        # self.offset = {ip: 0 for ip in self._drivers_by_ipaddrs}
        # self.tts = {ip: 0 for ip in self._drivers_by_ipaddrs}
        # self.estimated_timediff = {ip: 0 for ip in self._drivers_by_ipaddrs}

    def _validate_settings(self, settings: list[E7Setting]) -> None:
        acceptable_types = [
            AwgSetting,
            CapuSetting,
            TriggerSetting,
            TimingSetting,
            ClockmasterSetting,
        ]
        for s in settings:
            if not any([isinstance(s, c) for c in acceptable_types]):
                raise ValueError(f"Unsupported setting type: {s}")

    def _pop_clock_master_setting(
        self,
        settings: list[E7Setting],
    ) -> Optional[ClockmasterSetting]:
        clock_master_settings = [
            s for s in settings if isinstance(s, ClockmasterSetting)
        ]
        for s in clock_master_settings:
            settings.remove(s)
        if clock_master_settings:
            return clock_master_settings[-1]
        else:
            return None

    def _create_clock_master(
        self, setting: Optional[ClockmasterSetting]
    ) -> Optional[QuBEMasterClient]:
        return (
            QuBEMasterClient(master_ipaddr=str(setting.ipaddr))
            if setting is not None
            else None
        )

    def start(self, timeout: float = single.DEFAULT_CAPTURE_TIMEOUT) -> Results:
        self._estimate_timediff()
        for drv in self._drivers_by_ipaddrs.values():
            drv.capctrl_lock.acquire()
            drv.capture_at()
        for drv in self._drivers_by_ipaddrs.values():
            drv.awgctrl_lock.acquire()
        self.emit_at()
        futures = {
            ip: drv.wait_until_capture_finishes(timeout)
            for ip, drv in self._drivers_by_ipaddrs.items()
        }
        multidrv_results = {ip: f.result() for ip, f in futures.items()}
        results = {
            (ip, capu): single_result
            for ip, single_results in multidrv_results.items()
            for capu, single_result in single_results.results.items()
        }
        for drv in self._drivers_by_ipaddrs.values():
            drv.awgctrl_lock.release()
            drv.capctrl_lock.release()
        return Results(results=results)

    # from quelware
    def emit_at(self) -> None:
        primary_ipaddr = list(self._seqclients_by_ipaddrs)[0]
        primary_sqc = self._seqclients_by_ipaddrs[primary_ipaddr]
        awg_bitmap: dict[str, int] = defaultdict(int)
        for ip, drv in self._drivers_by_ipaddrs.items():
            for awg in drv.awgs:
                awg_bitmap[ip] |= 1 << awg

        for drv in self._drivers_by_ipaddrs.values():
            drv.clear_before_starting_emission()

        valid_read, current_time, last_sysref_time = primary_sqc.read_clock()
        if valid_read:
            LOGGER.info(
                f"current time: {current_time}, last sysref time: {last_sysref_time}"
            )
        else:
            raise RuntimeError("Failed to read clock")
        if abs(last_sysref_time % SYSREF_PERIOD - self._cap_sysref_time_offset) > 4:
            LOGGER.warning(
                f"large fluctuation of sysref is detected on FPGA {primary_ipaddr}"
            )
        base_time = current_time + MIN_TIME_OFFSET
        tamate_offset = (16 - (base_time - self._cap_sysref_time_offset) % 16) % 16
        base_time += tamate_offset

        for ip, drv in self._drivers_by_ipaddrs.items():
            sqc = self._seqclients_by_ipaddrs[ip]
            schedule = (
                base_time
                - sqc.estimated_timediff
                + sqc.time_to_start
                + 16 * sqc.displacement
            )
            valid_sched = sqc.add_sequencer(
                base_time=schedule,
                awg_bitmap=awg_bitmap[ip],
            )
            if not valid_sched:
                raise RuntimeError("Failed to schedule AWG start")
        LOGGER.info("Scheduling completed")

    # from quelware
    def _estimate_timediff(
        self, num_iters: int = DEFAULT_NUM_SYSREF_MEASUREMENTS
    ) -> tuple[str, int]:
        counter_at_sysref_clk = {ip: 0 for ip in self._seqclients_by_ipaddrs}
        for _ in range(num_iters):
            for ip, sqc in self._seqclients_by_ipaddrs.items():
                m = sqc.read_clock()
                if len(m) < 2:
                    raise RuntimeError("Firmware doesn't support sysref measurement")
                counter_at_sysref_clk[ip] += m[2] % SYSREF_PERIOD
        avg: dict[str, int] = {
            ip: round(cntr / num_iters) for ip, cntr in counter_at_sysref_clk.items()
        }
        refip = list(self._seqclients_by_ipaddrs)[0]
        adj = avg[refip]
        for ip, cntr in avg.items():
            self._seqclients_by_ipaddrs[ip].estimated_timediff = cntr - adj
        self._cap_sysref_time_offset = avg[refip]
        return refip, avg[refip]
