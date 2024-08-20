from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Final, Optional

import numpy as np
import numpy.typing as npt
from e7awgsw import CaptureParam, DspUnit, WaveSequence
from quel_ic_config import CaptureReturnCode, Quel1BoxWithRawWss


@dataclass
class E7Setting:
    port: int


@dataclass
class AwgSetting(E7Setting):
    # port: int
    channel: int
    wseq: WaveSequence


@dataclass
class CapuSetting(E7Setting):
    # port: int
    runit: int
    cprm: CaptureParam


@dataclass
class TriggerSetting(E7Setting):
    # port: int  # port
    trigger: tuple[int, int]  # port, channel


class UnitResult:
    def __init__(self, iq: npt.NDArray[np.complex64], cprm: CaptureParam) -> None:
        self._result = self._parse(iq, cprm)

    @property
    def sum_section(self) -> list[npt.NDArray[np.complex64]]:
        return self._result

    def _parse(
        self,
        iq: npt.NDArray[np.complex64],
        cprm: CaptureParam,
    ) -> list[npt.NDArray[np.complex64]]:
        dsp_units_enabled = cprm.dsp_units_enabled

        if DspUnit.INTEGRATION in dsp_units_enabled:
            a = iq.reshape(1, -1)
        else:
            a = iq.reshape(cprm.num_integ_sections, -1)

        if DspUnit.SUM in dsp_units_enabled:
            b = np.hsplit(a, list(range(len(cprm.sum_section_list))[1:]))
        else:
            cond = DspUnit.DECIMATION not in dsp_units_enabled
            ssl = cprm.sum_section_list[:-1]
            c = np.hsplit(
                a,
                np.cumsum(np.array([w if cond else int(w / 4) for w, _ in ssl]))
                * cprm.NUM_SAMPLES_IN_ADC_WORD,
            )
            b = [o.transpose() for o in c]
        return b


class PortResult:
    def __init__(
        self,
        result: tuple[CaptureReturnCode, dict[int, npt.NDArray[np.complex64]]],
        cprms: dict[int, CaptureParam],
    ) -> None:
        self.return_code = result[0]
        self._result = {
            runit: UnitResult(result[1][runit], cprm) for runit, cprm in cprms.items()
        }

    @property
    def runit(self) -> dict[int, UnitResult]:
        return self._result


class Results:
    def __init__(self, results: dict[int, PortResult]) -> None:
        self._results = results
        self._queue: Final[deque] = deque()

    @property
    def port(self) -> dict[int, PortResult]:
        return self._results

    def get(
        self, *, port: int, runit: int, sum_section: int
    ) -> npt.NDArray[np.complex64]:
        return self._results[port].runit[runit].sum_section[sum_section]

    def __iter__(self) -> Results:
        self._queue.clear()
        for port, port_result in self._results.items():
            for runit, runit_result in port_result.runit.items():
                for sum_section, _ in enumerate(runit_result.sum_section):
                    self._queue.appendleft((port, runit, sum_section))
        return self

    def __next__(self) -> tuple[tuple[int, int, int], npt.NDArray[np.complex64]]:
        if not self._queue:
            raise StopIteration
        port, runit, sum_section = self._queue.pop()
        return (port, runit, sum_section), self.port[port].runit[runit].sum_section[
            sum_section
        ]


class Driver:
    def __init__(self, box: Quel1BoxWithRawWss, settings: list[E7Setting]) -> None:
        self._box = box
        self._runits_by_ports: Final[dict[int, list[int]]] = defaultdict(list)
        self._cprms_by_ports: Final[dict[int, dict[int, CaptureParam]]] = defaultdict(
            dict
        )
        self._channels: Final[list[tuple[int, int]]] = []
        self._triggers: Final[dict[int, tuple[int, int]]] = {}
        self._load(settings)

    def _load(self, settings: list[E7Setting]) -> None:
        for setting in settings:
            if isinstance(setting, AwgSetting):
                self._channels.append((setting.port, setting.channel))
                self._box.config_channel(
                    setting.port,
                    setting.channel,
                    wave_param=setting.wseq,
                )
            elif isinstance(setting, CapuSetting):
                self._runits_by_ports[setting.port].append(setting.runit)
                self._cprms_by_ports[setting.port][setting.runit] = setting.cprm
                self._box.config_runit(
                    setting.port,
                    setting.runit,
                    capture_param=setting.cprm,
                )
            elif isinstance(setting, TriggerSetting):
                self._triggers[setting.port] = setting.trigger
            else:
                raise ValueError(f"unsupported setting: {setting}")

    def capture_start(
        self, timeout: Optional[float] = None
    ) -> dict[
        int, Future[tuple[CaptureReturnCode, dict[int, npt.NDArray[np.complex64]]]]
    ]:
        if timeout is None:
            timeout = self._box._dev.wss.DEFAULT_CAPTURE_TIMEOUT
        return {
            port: self._box.capture_start(
                port,
                runits,
                triggering_channel=self._triggers[port],
                timeout=timeout,
            )
            for port, runits in self._runits_by_ports.items()
        }

    def get_results(self, thunks: dict[int, Future]) -> Results:
        return Results(
            {
                port: PortResult(
                    result=thunk.result(),
                    cprms=self._cprms_by_ports[port],
                )
                for port, thunk in thunks.items()
            }
        )

    def start(self, timeout: Optional[float] = None) -> Results:
        thunks = self.capture_start(timeout)
        self._box.start_emission(self._channels)
        return self.get_results(thunks)
