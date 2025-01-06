from __future__ import annotations

from concurrent.futures import Future
from logging import getLogger
from types import MappingProxyType
from typing import Final, NamedTuple, Optional

import numpy as np
import numpy.typing as npt
from quel_clock_master import QuBEMasterClient
from quel_ic_config import CaptureReturnCode, Quel1BoxWithRawWss

from . import single
from .single import AwgSetting, RunitSetting, TriggerSetting

logger = getLogger(__name__)


class NamedBox(NamedTuple):
    name: str
    box: Quel1BoxWithRawWss


class BoxSetting(NamedTuple):
    name: str
    settings: list[AwgSetting | RunitSetting | TriggerSetting]


class Quel1System:
    def __init__(
        self,
        clockmaster: QuBEMasterClient,
        boxes: MappingProxyType[str, Quel1BoxWithRawWss],
    ) -> None:
        self._clockmaster: Final[QuBEMasterClient] = clockmaster
        self._boxes: Final[MappingProxyType[str, Quel1BoxWithRawWss]] = boxes

    @classmethod
    def create(
        cls,
        *,
        clockmaster: QuBEMasterClient,
        boxes: list[Quel1BoxWithRawWss | NamedBox],
    ) -> Quel1System:
        boxes_dict = {}
        for box in boxes:
            if isinstance(box, NamedBox):
                boxes_dict[box.name] = box.box
            else:
                boxes_dict[box._dev.wss._wss_addr] = box
        return cls(clockmaster, MappingProxyType(boxes_dict))

    @property
    def boxes(self) -> MappingProxyType[str, Quel1BoxWithRawWss]:
        return self._boxes

    @property
    def box(self) -> MappingProxyType[str, Quel1BoxWithRawWss]:
        return self._boxes


class Action:
    SYSREF_PERIOD: Final[int] = 2_000
    TIMING_OFFSET: Final[int] = 0
    MIN_TIME_OFFSET = 12_500_000
    DEFAULT_NUM_SYSREF_MEASUREMENTS: Final[int] = 100

    def __init__(
        self,
        quel1system: Quel1System,
        actions: MappingProxyType[str, single.Action],
        estimated_timediff: MappingProxyType[str, int],
        reference_box_name: str,
        ref_sysref_time_offset: int,
    ) -> None:
        self._quel1system: Final[Quel1System] = quel1system
        self._actions: Final[MappingProxyType[str, single.Action]] = actions
        self._estimated_timediff: Final[MappingProxyType[str, int]] = estimated_timediff
        self._reference_box_name: Final[str] = reference_box_name
        self._ref_sysref_time_offset: Final[int] = ref_sysref_time_offset

    @classmethod
    def build(
        cls,
        *,
        quel1system: Quel1System,
        settings: list[BoxSetting],
    ) -> Action:
        master = quel1system._clockmaster
        logger.warning(f"clock of master: {master.read_clock()}")
        actions: dict[str, single.Action] = {}
        for box_settings in settings:
            name = box_settings.name
            box = quel1system.box[name]
            box.initialize_all_awgs()
            box.initialize_all_capunits()
            awg_ids = [
                s.awg for s in box_settings.settings if isinstance(s, AwgSetting)
            ]
            box.prepare_for_emission(awg_ids)
            current_time, last_sysref_time = box.read_current_and_latched_clock()
            logger.warning(
                f"clock of {name}, current: {current_time}, last sysref: {last_sysref_time}, last sysref offset: {cls._mod_by_sysref(last_sysref_time)}"
            )
            actions[name] = single.Action.build(
                box=box,
                settings=box_settings.settings,
            )

        average_offsets_at_sysref_clock = {
            name: cls._measure_average_offset_at_sysref_clock(action.box)
            for name, action in actions.items()
        }
        reference_box_name = cls._get_reference_box_name(actions)
        ref_sysref_time_offset = average_offsets_at_sysref_clock[reference_box_name]
        estimated_timediff = {}
        for name, avgcntr in average_offsets_at_sysref_clock.items():
            estimated_timediff[name] = avgcntr - ref_sysref_time_offset
            logger.warning(
                f"estimated time difference of {name}: {estimated_timediff[name]}"
            )
        return cls(
            quel1system,
            MappingProxyType(actions),
            MappingProxyType(estimated_timediff),
            reference_box_name,
            ref_sysref_time_offset,
        )

    @classmethod
    def _measure_average_offset_at_sysref_clock(
        cls,
        box: Quel1BoxWithRawWss,
        num_iters: Optional[int] = None,
    ) -> int:
        if num_iters is None:
            num_iters = cls.DEFAULT_NUM_SYSREF_MEASUREMENTS
        offsets = [
            cls._mod_by_sysref(box.read_current_and_latched_clock()[1])
            for _ in range(num_iters)
        ]
        return round(sum(offsets) / num_iters)

    @classmethod
    def _get_reference_box_name(cls, actions: dict[str, single.Action]) -> str:
        for name, action in actions.items():
            if cls.has_capture_setting(action):
                return name
        raise ValueError("no box has capture setting")

    @staticmethod
    def has_capture_setting(action: single.Action) -> bool:
        return True if action._cprms else False

    @classmethod
    def _mod_by_sysref(cls, t: int) -> int:
        h = cls.SYSREF_PERIOD // 2
        return (t + h) % cls.SYSREF_PERIOD - h

    def capture_start(
        self,
    ) -> dict[
        str,
        dict[
            int,
            Future[tuple[CaptureReturnCode, dict[int, npt.NDArray[np.complex64]]]],
        ],
    ]:
        futures = {
            name: action.capture_start()
            for name, action in self._actions.items()
            if self.has_capture_setting(action)
        }
        return futures

    def capture_stop(
        self,
        futures: dict[
            str,
            dict[
                int,
                Future[tuple[CaptureReturnCode, dict[int, npt.NDArray[np.complex64]]]],
            ],
        ],
    ) -> tuple[
        dict[tuple[str, int], CaptureReturnCode],
        dict[tuple[str, int, int], npt.NDArray[np.complex64]],
    ]:
        box_results = {}
        for name, future in futures.items():
            box_results[name] = self._actions[name].capture_stop(future)
        status, data = {}, {}
        for name, (box_status, box_data) in box_results.items():
            for port, capt_return_code in box_status.items():
                status[(name, port)] = capt_return_code
            for (port, runit), runit_data in box_data.items():
                data[(name, port, runit)] = runit_data
        return status, data

    def action(
        self,
    ) -> tuple[
        dict[tuple[str, int], CaptureReturnCode],
        dict[tuple[str, int, int], npt.NDArray[np.complex64]],
    ]:
        futures = self.capture_start()
        self.emit_at()
        results = self.capture_stop(futures)
        return results
        # return {}

    def emit_at(
        self,
        min_time_offset: int = MIN_TIME_OFFSET,
        displacement: int = 0,
    ) -> None:
        for name, action in self._actions.items():
            box = action.box
            current_time, last_sysref_time = box.read_current_and_latched_clock()
            logger.warning(
                f"sysref offset of {name}: latest: {self._mod_by_sysref(last_sysref_time)}"
            )

        box = self._quel1system.box[self._reference_box_name]
        current_time, last_sysref_time = box.read_current_and_latched_clock()
        logger.warning(
            f"sysref offset of reference box {self._reference_box_name}: average: {self._ref_sysref_time_offset},  latest: {self._mod_by_sysref(last_sysref_time)}"
        )

        # Notes: checking the fluctuation of sysref trigger (just for information).
        fluctuation = (
            self._mod_by_sysref(last_sysref_time) - self._ref_sysref_time_offset
        )
        if abs(fluctuation) > 4:
            logger.warning(
                f"large fluctuation (= {fluctuation}) of sysref is detected from the previous timing measurement"
            )

        base_time = current_time + min_time_offset
        tamate_offset = (16 - (base_time - self._ref_sysref_time_offset) % 16) % 16
        # tamate_offset = (base_time - self._ref_sysref_time_offset) % 16
        base_time += tamate_offset
        base_time += displacement  # inducing clock displacement for performance evaluation (must be 0 usually).
        base_time += self.TIMING_OFFSET
        for name, action in self._actions.items():
            t = base_time + self._estimated_timediff[name]
            action.box.reserve_emission(set(action._wseqs.keys()), t)
            logger.warning(f"reserving emission of {name} at {t}")
