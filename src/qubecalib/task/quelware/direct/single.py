from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Future
from types import MappingProxyType
from typing import Final, NamedTuple, Optional

import numpy as np
import numpy.typing as npt
from e7awgsw import CaptureParam, WaveSequence
from quel_ic_config import CaptureReturnCode, Quel1BoxWithRawWss, Quel1WaveSubsystem


class AwgId(NamedTuple):
    port: int
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
    triggerd_port: int
    trigger_awg: AwgId  # port, channel


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
        box.prepare_for_emission(set(self._wseqs.keys()))
        return self

    @staticmethod
    def parse_settings(
        settings: list[RunitSetting | AwgSetting | TriggerSetting],
    ) -> tuple[
        MappingProxyType[AwgId, WaveSequence],
        MappingProxyType[RunitId, CaptureParam],
        MappingProxyType[int, AwgId],
    ]:
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
        awg_specs = self._wseqs.keys()
        if awg_specs:  # _channels が空の場合は AWG は起動しない
            self._box.start_emission(awg_specs)

    def capture_stop(
        self, futures: dict[int, Future]
    ) -> dict[tuple[int, int], npt.NDArray[np.complex64]]:
        results = {}
        for port, future in futures.items():
            capt_return_code, runit_data = future.result()
            if capt_return_code is not CaptureReturnCode.SUCCESS:
                raise ValueError(
                    f"Capture failed at port (= {port}) . {capt_return_code}"
                )
            for runit, data in runit_data.items():
                results[(port, runit)] = data

        return results

    def action(self) -> dict[tuple[int, int], npt.NDArray[np.complex64]]:
        futures = self.capture_start()
        self.start_emission()
        return self.capture_stop(futures)

    # box を変更されたくないので getter を用意し setter は用意しない
    # 細かな制御は直接 box を操作することで行う
    @property
    def box(self) -> Quel1BoxWithRawWss:
        return self._box
