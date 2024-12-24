from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Future
from typing import Final, NamedTuple, Optional

import numpy as np
import numpy.typing as npt
from e7awgsw import CaptureParam, WaveSequence
from quel_ic_config import CaptureReturnCode, Quel1BoxWithRawWss, Quel1WaveSubsystem


class Awg(NamedTuple):
    port: int
    channel: int


class AwgSetting(NamedTuple):
    awg: Awg
    wseq: WaveSequence


class Runit(NamedTuple):
    port: int
    runit: int


class RunitSetting(NamedTuple):
    runit: Runit
    cprm: CaptureParam


class TriggerSetting(NamedTuple):
    triggerd_port: int
    trigger_awg: Awg  # port, channel


class Task:
    def __init__(
        self,
        *,
        box: Quel1BoxWithRawWss,
        settings: list[RunitSetting | AwgSetting | TriggerSetting],
    ) -> None:
        self._runits_by_ports: Final[dict[int, list[int]]] = defaultdict(list)
        # Results のパースに必要だが... quelware の実装を見てから決めよう todo_20241213
        # self._cprms_by_ports: Final[dict[int, dict[int, CaptureParam]]] = defaultdict(
        #     dict
        # )
        self._channels: Final[list[Awg]] = []
        self._triggers: Final[dict[int, Awg]] = {}
        self._box: Final[Quel1BoxWithRawWss] = box
        self._load(settings)

    def _load(
        self,
        settings: list[RunitSetting | AwgSetting | TriggerSetting],
    ) -> None:
        if self._runits_by_ports or self._channels or self._triggers:
            raise ValueError("already loaded")
        if not settings:
            raise ValueError("no settings provided")
        for setting in settings:
            if isinstance(setting, AwgSetting):
                self._channels.append(setting.awg)
                self._box.config_channel(
                    port=setting.awg.port,
                    channel=setting.awg.channel,
                    wave_param=setting.wseq,
                )
            elif isinstance(setting, RunitSetting):
                self._runits_by_ports[setting.runit.port].append(setting.runit.runit)
                # todo_20241213 関連
                # self._cprms_by_ports[setting.port][setting.runit] = setting.cprm
                self._box.config_runit(
                    port=setting.runit.port,
                    runit=setting.runit.runit,
                    capture_param=setting.cprm,
                )
            elif isinstance(setting, TriggerSetting):
                self._triggers[setting.triggerd_port] = setting.trigger_awg
            else:
                raise ValueError(f"unsupported setting: {setting}")

    def _capture_start(
        self,
        *,
        timeout: Optional[float] = None,
    ) -> dict[
        int, Future[tuple[CaptureReturnCode, dict[int, npt.NDArray[np.complex64]]]]
    ]:
        if timeout is None:
            timeout = Quel1WaveSubsystem.DEFAULT_CAPTURE_TIMEOUT
        t = self._triggers
        futures = {
            port: self._box.capture_start(
                port,
                runits,
                triggering_channel=(t[port].port, t[port].channel)
                if port in self._triggers
                else None,
                timeout=timeout,
            )
            for port, runits in self._runits_by_ports.items()
        }
        return futures

    def run(
        self,
        *,
        timeout: Optional[float] = None,
    ) -> Optional[
        dict[
            int, Future[tuple[CaptureReturnCode, dict[int, npt.NDArray[np.complex64]]]]
        ]
    ]:
        # _trigger が _channels に含まれていなければ capm が tigger を待ち続けてしまうのでこれを防ぐ
        for port, trigger in self._triggers.items():
            if trigger not in self._channels:
                raise ValueError(
                    f"trigger awg {(trigger.port, trigger.channel)} for triggerd port {port} is not provided"
                )
            if port not in self._runits_by_ports:
                raise ValueError(
                    f"triggerd port {port} is not provided in runit settings"
                )
        if self._runits_by_ports:
            futures = self._capture_start(timeout=timeout)
        else:
            futures = None
        if self._channels:  # _channels が空の場合は AWG は起動しない
            self._box.start_emission(self._channels)
        return futures

    # box を変更されたくないので getter を用意し setter は用意しない
    # 細かな制御は直接 box を操作することで行う
    @property
    def box(self) -> Quel1BoxWithRawWss:
        return self._box
