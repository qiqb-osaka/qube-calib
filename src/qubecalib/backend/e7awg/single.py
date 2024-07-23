from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from ipaddress import IPv4Address, IPv6Address
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from e7awgsw import (
    AWG,
    AwgCtrl,
    CaptureCtrl,
    CaptureModule,
    CaptureParam,
    CaptureUnit,
    DspUnit,
    WaveSequence,
)
from e7awgsw.memorymap import CaptureCtrlRegs, CaptureParamRegs

# Readout や Monitor などの構成に関する知識はライブラリには含まれない

LOGGER = logging.getLogger(__name__)

DEFAULT_NUM_WORKERS: Final[int] = 4
DEFAULT_CAPTURE_TIMEOUT: Final[float] = 60.0


def get_logger() -> logging.Logger:
    return LOGGER


# from quelware
class CaptureReturnCode(Enum):
    CAPTURE_TIMEOUT = 1
    CAPTURE_ERROR = 2
    BROKEN_DATA = 3
    SUCCESS = 4


@dataclass
class Result:
    status: CaptureReturnCode
    data: list[npt.NDArray[np.complex64]] | list[npt.NDArray[np.int16]]


@dataclass
class Results:
    status: CaptureReturnCode
    results: Final[dict[CaptureUnit, Result]] = field(default_factory=dict)


@dataclass
class E7Setting:
    pass


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


class Driver:
    def __init__(
        self,
        ipaddr: IPv4Address | IPv6Address,
        settings: list[E7Setting],
    ) -> None:
        self._ipaddr = str(ipaddr)
        self._awgctrl = AwgCtrl(self._ipaddr)
        self._capctrl = CaptureCtrl(self._ipaddr)
        self._awgctrl_lock = threading.Lock()
        self._capctrl_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=DEFAULT_NUM_WORKERS)
        self._settings = settings

        self._capus: Final[list[CaptureUnit]] = [
            s.capu for s in settings if isinstance(s, CapuSetting)
        ]
        self._awgs: Final[list[AWG]] = [
            s.awg for s in settings if isinstance(s, AwgSetting)
        ]
        self._capus_with_triggers: Final[list[CaptureUnit]] = []
        self._cprms_by_capus: Final[dict[CaptureUnit, CaptureParam]] = {
            s.capu: s.cprm for s in settings if isinstance(s, CapuSetting)
        }
        self._initialize()
        self._load(settings)

    @property
    def awgctrl(self) -> AwgCtrl:
        return self._awgctrl

    @property
    def capctrl(self) -> CaptureCtrl:
        return self._capctrl

    @property
    def awgctrl_lock(self) -> threading.Lock:
        return self._awgctrl_lock

    @property
    def capctrl_lock(self) -> threading.Lock:
        return self._capctrl_lock

    @property
    def awgs(self) -> list[AWG]:
        return self._awgs

    def _initialize(self) -> None:
        with self._capctrl_lock:
            self._capctrl.initialize(*self._capus)
        with self._awgctrl_lock:
            self._awgctrl.initialize(*self._awgs)
            self._awgctrl.terminate_awgs(*self._awgs)
            self._awgctrl.clear_awg_stop_flags(*self._awgs)

    def _load(self, settings: list[E7Setting]) -> None:
        for setting in settings:
            if isinstance(setting, AwgSetting):
                with self._awgctrl_lock:
                    self._awgctrl.set_wave_sequence(setting.awg, setting.wseq)
            elif isinstance(setting, CapuSetting):
                with self._capctrl_lock:
                    self._capctrl.set_capture_params(setting.capu, setting.cprm)
            elif isinstance(setting, TriggerSetting):
                pass
            else:
                raise ValueError(f"Unknown setting type: {setting}")
        for setting in self._settings:
            if isinstance(setting, TriggerSetting):
                with self._capctrl_lock:
                    self._capctrl.select_trigger_awg(setting.capm, setting.awg)
                for u in CaptureModule.get_units(setting.capm):
                    if u in self._capus:
                        self._capus_with_triggers.append(u)

    def start(self, timeout: float = DEFAULT_CAPTURE_TIMEOUT) -> Results:
        with self._capctrl_lock:
            self.capture_at()  # or self.capture_now() for individual capture
            future = self.wait_until_capture_finishes(timeout)
            self.emit_now()
            result = future.result()
        return result

    def abort(self) -> None:  # for capture
        # TODO implement
        pass

    def stop(self, awgs: list[AWG] = []) -> None:  # for awg
        if not awgs:
            awgs = self._awgs
        with self._awgctrl_lock:
            self._awgctrl.terminate_awgs(*awgs)
            self._awgctrl.clear_awg_stop_flags(*awgs)

    def capture_at(self) -> None:
        """Caputre data when trigger is received. Locked capctrl is assumed."""
        # Capture data when trigger is received
        self._capctrl.enable_start_trigger(
            *[u for u in self._capus if u in self._capus_with_triggers]
        )
        # Capture data immediately for units without trigger
        capus = [u for u in self._capus if u not in self._capus_with_triggers]
        self._capctrl.start_capture_units(*capus)
        LOGGER.debug(
            f"Capture units {', '.join([str(u) for u in capus])} started immediately"
        )

    def capture_now(self) -> None:
        """Capture data immediately without waiting for the trigger. Unlocked capctrl is required."""
        with self._capctrl_lock:
            self._capctrl.start_capture_units(self._capus)

    def wait_until_capture_finishes(
        self, timeout: float = DEFAULT_CAPTURE_TIMEOUT
    ) -> Future[Results]:
        return self._executor.submit(
            self._capture_thread,
            timeout,
        )

    def emit_now(self) -> None:
        with self._awgctrl_lock:
            self._awgctrl.start_awgs(*self._awgs)

    # from quelware
    def _check_capture_units_done(
        self,
    ) -> bool:
        for capu in self._capus:
            val = self._capctrl._CaptureCtrl__reg_access.read_bits(
                CaptureCtrlRegs.Addr.capture(capu),
                CaptureCtrlRegs.Offset.STATUS,
                CaptureCtrlRegs.Bit.STATUS_DONE,
                1,
            )
            if val == 0:
                return False
        else:
            return True

    # from quelware
    def _clear_capture_unit_done(
        self,
    ) -> None:
        for capu in self._capus:
            for v in (0, 1):
                self._capctrl._CaptureCtrl__reg_access.write_bits(
                    CaptureCtrlRegs.Addr.capture(capu),
                    CaptureCtrlRegs.Offset.CTRL,
                    CaptureCtrlRegs.Bit.CTRL_DONE_CLR,
                    1,
                    v,
                )

    # from quelware
    def _wait_for_capture_data(
        self,
        timeout: float = DEFAULT_CAPTURE_TIMEOUT,
    ) -> bool:
        # waiting for the completion of captureing
        polling_interval = min(max(0.01, timeout / 1000.0), 0.1)
        t0: float = time.perf_counter()
        completed: bool = False
        while time.perf_counter() - t0 < timeout:
            time.sleep(polling_interval)
            if self._check_capture_units_done():
                completed = True
                break
        if completed:
            self._clear_capture_unit_done()
        else:
            # TODO: investigate the reason this happens even when `timeout` is large enough
            LOGGER.warning(
                f"timeout happens at capture units {', '.join([str(x) for x in self._capus])}, capture aborted"
            )
        return completed

    # from quelware
    def _check_capture_error(
        self,
    ) -> bool:
        errdict: dict[int, list[Any]] = self._capctrl.check_err(*self._capus)
        errflag = False
        for capu, errlist in errdict.items():
            for err in errlist:
                LOGGER.warning(f"capture unit {self._capus[capu]}: {err}")
                errflag = True
        return errflag

    # from quelware
    def _capture_thread(
        self,
        timeout: float = DEFAULT_CAPTURE_TIMEOUT,
    ) -> Results:
        ready: bool = self._wait_for_capture_data(timeout)
        if not ready:
            return Results(status=CaptureReturnCode.CAPTURE_TIMEOUT)
        if self._check_capture_error():
            return Results(status=CaptureReturnCode.CAPTURE_ERROR)
        results = Results(status=CaptureReturnCode.SUCCESS)
        for capu in self._capus:
            results.results[capu] = self._retrieve_capture_data(capu)
        return results

    def _retrieve_capture_data(
        self,
        capu: CaptureUnit,
    ) -> Result:
        status: CaptureReturnCode = CaptureReturnCode.SUCCESS
        cprm = self._cprms_by_capus[capu]
        n_sample_expected = cprm.calc_capture_samples()
        n_sample_captured = self._capctrl.num_captured_samples(capu)
        if n_sample_captured == n_sample_expected:
            LOGGER.debug(
                f"the capture unit {self._ipaddr}:{capu} captured {n_sample_captured} samples"
            )
        else:
            # TODO: investigate the reason this happens
            LOGGER.warning(
                "the capture unit {self._ipaddr}:{capu} captured {n_sample_captured} samples, "
                "should be {n_sample_expected} samples"
            )
            status = CaptureReturnCode.BROKEN_DATA

        if DspUnit.CLASSIFICATION in cprm.dsp_units_enabled:
            d = np.array(
                list(
                    self._capctrl.get_classification_results(
                        capu,
                        n_sample_captured,
                    )
                ),
                dtype=np.int16,
            )
        else:
            c = np.array(
                self._capctrl.get_capture_data(
                    capu,
                    n_sample_captured,
                ),
                dtype=np.float32,
            )
        d = c[:, 0] + c[:, 1] * 1j

        if DspUnit.INTEGRATION in cprm.dsp_units_enabled:
            d = d.reshape(1, -1)
        else:
            d = d.reshape(cprm.num_integ_sections, -1)

        if DspUnit.SUM in cprm.dsp_units_enabled:
            e = np.hsplit(d, list(range(len(cprm.sum_section_list))[1:]))
        else:
            f = np.hsplit(
                d,
                np.cumsum(
                    np.array(
                        [
                            w
                            if DspUnit.DECIMATION not in cprm.dsp_units_enabled
                            else int(w / 4)
                            for w, _ in cprm.sum_section_list[:-1]
                        ]
                    )
                )
                * cprm.NUM_SAMPLES_IN_ADC_WORD,
            )
            e = [o.transpose() for o in f]
        return Result(
            status=status,
            data=e,
        )

    # TODO: direct access to the register will be implemented in the future
    def _get_dsp_units_enabled(self, capu: CaptureUnit) -> list[DspUnit]:
        """Locked capctrl is assumed."""
        base_addr = CaptureParamRegs.Addr.capture(capu)
        result = self._capctrl._CaptureCtrl__reg_access.read(
            base_addr, CaptureParamRegs.Offset.DSP_MODULE_ENABLE
        )
        return result

    def clear_before_starting_emission(self) -> None:
        """Locked awgctrl is assumed."""
        self._awgctrl.clear_awg_stop_flags(*self._awgs)
