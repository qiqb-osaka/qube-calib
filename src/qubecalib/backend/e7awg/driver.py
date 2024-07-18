from __future__ import annotations

import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from ipaddress import IPv4Address, IPv6Address
from typing import Any, Final, Iterable, Optional

import numpy as np
import numpy.typing as npt
from e7awgsw import (
    AWG,
    AwgCtrl,
    CaptureCtrl,
    CaptureParam,
    CaptureUnit,
    WaveSequence,
)
from e7awgsw.memorymap import CaptureCtrlRegs
from quel_ic_config import CaptureReturnCode

# 一部 quel_ic_config の機能を使っているため、quel_ic_config がインストールされている必要がある
# ほとんどの機能は e7awgsw のために作られている

DEFAULT_NUM_WORKERS: Final[int] = 4
DEFAULT_CAPTURE_TIMEOUT: Final[float] = 60.0


@dataclass
class Result:
    status: CaptureReturnCode
    data: npt.NDArray[np.complex64]


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
    capprm: CaptureParam
    trigger: Optional[AWG] = None


class Driver:
    def __init__(self, ipaddr: IPv4Address | IPv6Address) -> None:
        self._executor = ThreadPoolExecutor(max_workers=DEFAULT_NUM_WORKERS)
        self._awgctrl_lock = threading.Lock()
        self._capctrl_lock = threading.Lock()
        self._ipaddr = str(ipaddr)
        self._awgctrl: Optional[AwgCtrl] = None
        self._capctrl: Optional[CaptureCtrl] = None
        self._capus: Final[list[CaptureUnit]] = []
        self._awgs: Final[list[AWG]] = []
        # self._capu_with_trigger: Iterable[CapuSetting] = {}
        # self._capu_without_trigger: Iterable[CapuSetting] = {}
        self._capu_with_trigger: Final[list[CaptureUnit]] = []

    def load(self, *settings: list[E7Setting]) -> None:
        self._capctrl = self._create_capctrl(
            {s.capu for s in settings if isinstance(s, CapuSetting)}
        )
        self._awgctrl = self._create_awgctrl(
            {s.awg for s in settings if isinstance(s, AwgSetting)}
        )
        self._load_e7settings(*settings)
        self._set_trigger(*settings)

    def kick(self) -> Results:
        if self._capctrl is None or self._awgctrl is None:
            raise ValueError("Driver not loaded")
        with self._capctrl_lock:
            self._capctrl.enable_start_trigger(
                *[u for u in self._capus if u in self._capu_with_trigger]
            )
            self._capctrl.start_capture_units(
                *[u for u in self._capus if u not in self._capu_with_trigger]
            )
        future = self._executor.submit(
            self._capture_thread,
            capctrl=self._capctrl,
            capus=self._capus,
        )
        self._emit_now(self._awgctrl, self._awgs)
        return future.result()

    def _emit_now(self, awgctrl: AwgCtrl, awgs: list[AWG]) -> None:
        with self._awgctrl_lock:
            awgctrl.start_awgs(*awgs)

    # from quelware
    def _check_capture_units_done(
        self,
        capctrl: CaptureCtrl,
        capus: list[CaptureUnit],
    ) -> bool:
        with self._capctrl_lock:
            for capu in capus:
                val = capctrl._CaptureCtrl__reg_access.read_bits(
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
        capctrl: CaptureCtrl,
        capus: list[CaptureUnit],
    ) -> None:
        with self._capctrl_lock:
            for capu in capus:
                for v in (0, 1):
                    capctrl._CaptureCtrl__reg_access.write_bits(
                        CaptureCtrlRegs.Addr.capture(capu),
                        CaptureCtrlRegs.Offset.CTRL,
                        CaptureCtrlRegs.Bit.CTRL_DONE_CLR,
                        1,
                        v,
                    )

    # from quelware
    def _wait_for_capture_data(
        self,
        capctrl: CaptureCtrl,
        capus: list[CaptureUnit],
        timeout: float,
    ) -> bool:
        # waiting for the completion of captureing
        polling_interval = min(max(0.01, timeout / 1000.0), 0.1)
        t0: float = time.perf_counter()
        completed: bool = False
        while time.perf_counter() - t0 < timeout:
            time.sleep(polling_interval)
            if self._check_capture_units_done(capctrl, capus):
                completed = True
                break
        if completed:
            self._clear_capture_unit_done(capctrl, capus)
        else:
            # TODO: investigate the reason this happens even when `timeout` is large enough
            print(
                f"timeout happens at capture units {', '.join([str(x) for x in capus])}, capture aborted"
            )
        return completed

    # from quelware
    def _check_capture_error(
        self,
        capctrl: CaptureCtrl,
        capus: list[CaptureUnit],
    ) -> bool:
        with self._capctrl_lock:
            errdict: dict[int, list[Any]] = capctrl.check_err(*capus)
            errflag = False
            for capu, errlist in errdict.items():
                for err in errlist:
                    print(f"capture unit {capus[capu]}: {err}")
                    errflag = True
            return errflag

    # from quelware
    def _capture_thread(
        self,
        capctrl: CaptureCtrl,
        capus: list[CaptureUnit],
        timeout: float = DEFAULT_CAPTURE_TIMEOUT,
    ) -> Results:
        ready: bool = self._wait_for_capture_data(capctrl, capus, timeout)
        if not ready:
            return Results(status=CaptureReturnCode.CAPTURE_TIMEOUT)
        if self._check_capture_error(capctrl, capus):
            return Results(status=CaptureReturnCode.CAPTURE_ERROR)
        results = Results(status=CaptureReturnCode.SUCCESS)
        for capu in capus:
            results.results[capu] = self._retrieve_capture_data(capctrl, capu)
        return results

    def _retrieve_capture_data(
        self,
        capctrl: CaptureCtrl,
        capu: CaptureUnit,
    ) -> Result:
        status: CaptureReturnCode = CaptureReturnCode.SUCCESS
        with self._capctrl_lock:
            n_sample_captured = capctrl.num_captured_samples(capu)
            c = np.array(
                capctrl.get_capture_data(capu, n_sample_captured), dtype=np.float32
            )
        return Result(
            status=status,
            data=c[:, 0] + c[:, 1] * 1j,
        )

    def _create_capctrl(self, capus: Iterable[CaptureUnit]) -> CaptureCtrl:
        with self._capctrl_lock:
            capctrl = CaptureCtrl(self._ipaddr)
            capctrl.initialize(*capus)
        return capctrl

    def _create_awgctrl(self, awgs: Iterable[AWG]) -> AwgCtrl:
        with self._awgctrl_lock:
            awgctrl = AwgCtrl(self._ipaddr)
            awgctrl.initialize(*awgs)
            awgctrl.terminate_awgs(*awgs)
            awgctrl.clear_awg_stop_flags(*awgs)
        return awgctrl

    def _load_e7settings(self, *settings: list[E7Setting]) -> None:
        if self._awgctrl is None or self._capctrl is None:
            raise ValueError("Driver not loaded")
        with self._awgctrl_lock, self._capctrl_lock:
            for setting in settings:
                if isinstance(setting, AwgSetting):
                    self._awgctrl.set_wave_sequence(setting.awg, setting.wseq)
                    self._awgs.append(setting.awg)
                elif isinstance(setting, CapuSetting):
                    self._capctrl.set_capture_params(setting.capu, setting.capprm)
                    self._capus.append(setting.capu)
                else:
                    raise ValueError(f"Unknown setting type: {setting}")

    def _set_trigger(self, *settings: list[E7Setting]) -> None:
        capu_settings = [s for s in settings if isinstance(s, CapuSetting)]
        # self._capu_with_trigger: dict[CaptureUnit] = {
        #     s.capu: s.trigger for s in capu_settings if s.trigger
        # }
        # self._capu_without_trigger: Iterable[CaptureUnit] = {
        #     s.capu for s in capu_settings if not s.trigger
        # }
        triggers_by_capm = defaultdict(list)
        for s in capu_settings:
            if s.trigger is not None:
                triggers_by_capm[CaptureUnit.get_module(s.capu)].append(s.trigger)
                self._capu_with_trigger.append(s.capu)
        if self._capctrl is None:
            raise ValueError("CaptureCtrl not loaded")
        with self._capctrl_lock:
            for capm, triggers in triggers_by_capm.items():
                if len(triggers) > 1:
                    raise ValueError(f"Multiple triggers for {capm}")
                trigger = next(iter(triggers))
                self._capctrl.select_trigger_awg(capm, trigger)
