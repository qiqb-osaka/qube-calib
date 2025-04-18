from __future__ import annotations

from collections import defaultdict
from typing import Final, NamedTuple

import numpy as np
import numpy.typing as npt
from e7awgsw import CaptureParam, WaveSequence
from quel_ic_config import CaptureReturnCode

from . import multi, single
from .single import Quel1PortType


class AwgId(NamedTuple):
    box: str
    port: Quel1PortType
    channel: int


class RunitId(NamedTuple):
    box: str
    port: int
    runit: int


class AwgSetting(NamedTuple):
    awg: AwgId
    wseq: WaveSequence


class RunitSetting(NamedTuple):
    runit: RunitId
    cprm: CaptureParam


class TriggerSetting(NamedTuple):
    trigger_awg: AwgId  # box, port, channel
    triggerd_port: int  # port


def _convert_to_box_setting_dict(
    settings: list[RunitSetting | AwgSetting | TriggerSetting],
) -> dict[str, list[single.AwgSetting | single.RunitSetting | single.TriggerSetting]]:
    rslt: dict[
        str, list[single.AwgSetting | single.RunitSetting | single.TriggerSetting]
    ] = defaultdict(list)
    for s in settings:
        if isinstance(s, RunitSetting):
            rslt[s.runit.box].append(
                single.RunitSetting(
                    single.RunitId(
                        s.runit.port,
                        s.runit.runit,
                    ),
                    s.cprm,
                )
            )
        elif isinstance(s, AwgSetting):
            rslt[s.awg.box].append(
                single.AwgSetting(
                    single.AwgId(
                        s.awg.port,
                        s.awg.channel,
                    ),
                    s.wseq,
                )
            )
        elif isinstance(s, TriggerSetting):
            rslt[s.trigger_awg.box].append(
                single.TriggerSetting(
                    single.AwgId(
                        s.trigger_awg.port,
                        s.trigger_awg.channel,
                    ),
                    s.triggerd_port,
                )
            )
    return rslt


def _convert_to_box_settings(
    settings: list[RunitSetting | AwgSetting | TriggerSetting],
) -> list[multi.BoxSetting]:
    d = _convert_to_box_setting_dict(settings)
    return [multi.BoxSetting(name, s) for name, s in d.items()]


class Action:
    def __init__(self, action: tuple[str, single.Action] | multi.Action) -> None:
        self._action: Final[tuple[str, single.Action] | multi.Action] = action

    @classmethod
    def build(
        cls,
        *,
        system: multi.Quel1System,
        settings: list[RunitSetting | AwgSetting | TriggerSetting],
    ) -> Action:
        if not settings:
            raise ValueError("no settings provided")
        s = _convert_to_box_settings(settings)
        for box in [box for box in s]:
            if box.name not in system.boxes:
                raise ValueError(f"box {box.name} not found in system")
            # There is no problem with having many systems.boxes.
        if len(s) == 1:
            self = cls(
                (
                    s[0].name,
                    single.Action.build(
                        box=system.box[s[0].name],
                        settings=s[0].settings,
                    ),
                )
            )
        else:
            self = cls(multi.Action.build(quel1system=system, settings=s))
        return self

    def action(
        self,
    ) -> tuple[
        dict[tuple[str, int], CaptureReturnCode],
        dict[tuple[str, int, int], npt.NDArray[np.complex64]],
    ]:
        if isinstance(self._action, tuple):
            name = self._action[0]
            status, data = self._action[1].action()
            return {(name, k): v for k, v in status.items()}, {
                (name, k[0], k[1]): v for k, v in data.items()
            }
        elif isinstance(self._action, multi.Action):
            return self._action.action()
