from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Future
from types import MappingProxyType
from typing import Final, NamedTuple, Optional, Union

import numpy as np
import numpy.typing as npt
from e7awgsw import CaptureParam, WaveSequence
from quel_ic_config import CaptureReturnCode, Quel1BoxWithRawWss, Quel1WaveSubsystem

Quel1PortType = Union[int, tuple[int, int]]


class AwgId(NamedTuple):
    port: Quel1PortType
    channel: int


class AwgSetting(NamedTuple):
    awg: AwgId
    wseq: WaveSequence


class RunitId(NamedTuple):
    port: int
    runit: int


class RunitSetting(NamedTuple):
    runit: RunitId
    cprm: CaptureParam


class TriggerSetting(NamedTuple):
    trigger_awg: AwgId  # port, channel
    triggerd_port: int


class Action:
    def __init__(
        self,
        box: Quel1BoxWithRawWss,
        wseqs: MappingProxyType[AwgId, WaveSequence],
        cprms: MappingProxyType[RunitId, CaptureParam],
        triggers: MappingProxyType[int, AwgId],
    ) -> None:
        self._box: Final[Quel1BoxWithRawWss] = box
        self._wseqs: Final[MappingProxyType[AwgId, WaveSequence]] = wseqs
        self._cprms: Final[MappingProxyType[RunitId, CaptureParam]] = cprms
        self._triggers: Final[MappingProxyType[int, AwgId]] = triggers

    @classmethod
    def build(
        cls,
        *,
        box: Quel1BoxWithRawWss,
        settings: list[RunitSetting | AwgSetting | TriggerSetting],
    ) -> Action:
        wseqs, cprms, triggers = cls.parse_settings(settings)
        self = cls(box, wseqs, cprms, triggers)
        self._load_to_device()
        awgs = set([(s.port, s.channel) for s in self._wseqs])
        box.prepare_for_emission(awgs)
        return self

    @staticmethod
    def parse_settings(
        settings: list[RunitSetting | AwgSetting | TriggerSetting],
    ) -> tuple[
        MappingProxyType[AwgId, WaveSequence],
        MappingProxyType[RunitId, CaptureParam],
        MappingProxyType[int, AwgId],
    ]:
        # ValueError 1
        if not settings:
            raise ValueError("no settings provided")
        wseqs, cprms, triggers = {}, {}, {}
        for setting in settings:
            if isinstance(setting, AwgSetting):
                wseqs[setting.awg] = setting.wseq
            elif isinstance(setting, RunitSetting):
                cprms[setting.runit] = setting.cprm
            elif isinstance(setting, TriggerSetting):
                triggers[setting.triggerd_port] = setting.trigger_awg
            else:
                raise ValueError(f"unsupported setting: {setting}")
        # wseqs, cprms, triggers
        # False, False, False -> ValueError 1
        # True,  False, False
        # False, True,  False
        # True,  True,  False -> ValueError 2
        # False, False, True  -> ValueError 3
        # True,  False, True  -> ValueError 3
        # False, True,  True  -> ValueError 4
        # True,  True,  True
        # ValueError 2
        if all([bool(wseqs), bool(cprms), not bool(triggers)]):
            raise ValueError("both wseqs and cprms are provided without triggers")
        if triggers:
            cap_ports = {runit.port for runit in cprms}
            for port in triggers:
                # ValueError 3
                if port not in cap_ports:
                    raise ValueError(
                        f"triggerd port {port} is not provided in runit settings"
                    )
            awgs = set(wseqs.keys())
            for port, awg in triggers.items():
                # ValueError 4
                if awg not in awgs:
                    raise ValueError(
                        f"trigger {awg} for triggerd port {port} is not provided"
                    )
        return (
            MappingProxyType(wseqs),
            MappingProxyType(cprms),
            MappingProxyType(triggers),
        )

    def _load_to_device(self) -> None:
        for awg, wseq in self._wseqs.items():
            self.box.config_channel(
                port=awg.port,
                channel=awg.channel,
                wave_param=wseq,
            )
        for runit, cprm in self._cprms.items():
            self.box.config_runit(
                port=runit.port,
                runit=runit.runit,
                capture_param=cprm,
            )

    def capture_start(
        self,
        *,
        timeout: Optional[float] = None,
    ) -> dict[
        int, Future[tuple[CaptureReturnCode, dict[int, npt.NDArray[np.complex64]]]]
    ]:
        # _trigger が _channels に含まれていなければ capm が tigger を待ち続けてしまうのでこれを防ぐ
        channels = {awg for awg in self._wseqs}
        runits_by_ports = defaultdict(list)
        for runit in self._cprms:
            runits_by_ports[runit.port].append(runit.runit)
        for port, trigger in self._triggers.items():
            if trigger not in channels:
                raise ValueError(
                    f"trigger {trigger} for triggerd port {port} is not provided"
                )
            if port not in runits_by_ports:
                raise ValueError(
                    f"triggerd port {port} is not provided in runit settings"
                )
        if timeout is None:
            timeout = Quel1WaveSubsystem.DEFAULT_CAPTURE_TIMEOUT
        if runits_by_ports:
            return {
                port: self._box.capture_start(
                    port,
                    runits,
                    triggering_channel=(
                        self._triggers[port].port,
                        self._triggers[port].channel,
                    )
                    if port in self._triggers
                    else None,
                    timeout=timeout,
                )
                for port, runits in runits_by_ports.items()
            }
        else:
            return {}

    def start_emission(self) -> None:
        awg_specs = set([(s.port, s.channel) for s in self._wseqs])
        if awg_specs:  # _channels が空の場合は AWG は起動しない
            self._box.start_emission(awg_specs)

    def capture_stop(
        self, futures: dict[int, Future]
    ) -> tuple[
        dict[int, CaptureReturnCode], dict[tuple[int, int], npt.NDArray[np.complex64]]
    ]:
        status, data = {}, {}
        for port, future in futures.items():
            capt_return_code, runit_data = future.result()
            status[port] = capt_return_code
            for runit, d in runit_data.items():
                data[(port, runit)] = d
        return status, data

    def action(
        self,
    ) -> tuple[
        dict[int, CaptureReturnCode], dict[tuple[int, int], npt.NDArray[np.complex64]]
    ]:
        # wseqs, cprms, triggers
        # True,  False, False -> AWG only
        # False, True,  False -> Capture only
        # True,  True,  True  -> Triggered Capture
        # Triggered Capture
        if all([bool(self._wseqs), bool(self._cprms), bool(self._triggers)]):
            futures = self.capture_start()
            self.start_emission()
            return self.capture_stop(futures)
        # Awg only
        elif all([bool(self._wseqs), not bool(self._cprms), not bool(self._triggers)]):
            self.start_emission()
            return {}, {}
        # Capture only
        elif all([not bool(self._wseqs), bool(self._cprms), not bool(self._triggers)]):
            futures = self.capture_start()
            return self.capture_stop(futures)
        else:
            raise ValueError("unsupported action")  # 基本的には起こらないはず

    # box を変更されたくないので getter を用意し setter は用意しない
    # 細かな制御は直接 box を操作することで行う
    @property
    def box(self) -> Quel1BoxWithRawWss:
        return self._box
