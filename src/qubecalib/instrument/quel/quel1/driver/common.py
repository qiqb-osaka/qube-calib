from __future__ import annotations

from typing import NamedTuple

from e7awgsw import CaptureParam, WaveSequence
from quel_ic_config import Quel1PortType


class AwgId(NamedTuple):
    box: str
    port: Quel1PortType
    channel: int


class RunitId(NamedTuple):
    box: str
    port: Quel1PortType
    runit: int


class AwgSetting(NamedTuple):
    awg: AwgId
    wseq: WaveSequence


class RunitSetting(NamedTuple):
    runit: RunitId
    cprm: CaptureParam


class TriggerSetting(NamedTuple):
    trigger_awg: AwgId  # box, port, channel
    triggerd_port: Quel1PortType  # port


class Action:
    pass
