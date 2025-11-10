from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from quel_ic_config import AwgParam, CapParam

from ...task import ConcreteTaskBase, TaskTemplate
from .bandref import BandRef
from .quel1driver import Quel1Driver


@dataclass
class BoxAdjustParam:
    pass


@dataclass
class BandParam:
    pass


@dataclass
class Quel1SynchronizedPulseCaptureTemplate(TaskTemplate):
    pass


@dataclass
class Quel1Waveforms:
    band: BandRef
    wave_key: str
    iq: NDArray[np.complex64]


@dataclass
class Quel1WaveformLoadTask(ConcreteTaskBase):
    driver_kind: str = "quel1"
    waveforms: list[Quel1Waveforms] = field(default_factory=list)

    def execute(self) -> Any:
        drv = self.driver()
        if drv is None:
            raise RuntimeError("Driver is not available in the current context.")
        if not isinstance(drv, Quel1Driver):
            raise TypeError("Driver must be an instance of Quel1Driver.")
        for waveform in self.waveforms:
            band = waveform.band
            box = drv.box(band.box_key)
            box.register_wavedata(
                port=band.port,
                channel=band.band_key,
                name=waveform.wave_key,
                iq=waveform.iq,
            )


@dataclass
class Quel1PulseCaptureTask(ConcreteTaskBase):
    driver_kind: str = "quel1"
    box_adjust: dict[str, BoxAdjustParam] = field(default_factory=dict)
    commands: dict[BandRef, CapParam | AwgParam] = field(default_factory=dict)

    def execute(self) -> Any:
        drv = self.driver()
        if drv is None:
            raise RuntimeError("Driver is not available in the current context.")
        if not isinstance(drv, Quel1Driver):
            raise TypeError("Driver must be an instance of Quel1Driver.")

        for band, command in self.commands.items():
            if isinstance(command, AwgParam):
                drv.box(band.box_key).config_channel(
                    port=band.port,
                    channel=band.band_key,
                    awg_param=cast(AwgParam, command),
                )
            elif isinstance(command, CapParam):
                drv.box(band.box_key).config_runit(
                    port=band.port,
                    runit=band.band_key,
                    capture_param=cast(CapParam, command),
                )
            else:
                raise TypeError(
                    f"Unsupported command type for band {band}: {type(command)}"
                )

        boxes = {band.box_key for band in self.commands.keys()}

        if len(boxes) == 1:
            box = drv.box(next(iter(boxes)))
            runits = {
                (band.port, band.band_key)
                for band, command in self.commands.items()
                if band.box_key == box.name
                if isinstance(command, CapParam)
            }
            channels = [
                (band.port, band.band_key)
                for band, command in self.commands.items()
                if band.box_key == box.name
                if isinstance(command, AwgParam)
            ]
            cur = box.get_current_timecounter()
            thunk_ri, thunk_ro = box.start_capture_by_awg_trigger(
                runits=runits,
                channels=channels,
                timecounter=cur + 125_000_000 // 10,
            )
            thunk_ro.result()
            iqs_ri_readers = thunk_ri.result()
            wave_dict = {
                BandRef(
                    box_key=box.name, port=port, band_key=band_key
                ): readers.as_wave_dict()
                for (port, band_key), readers in iqs_ri_readers.items()
            }
            return wave_dict
        else:
            pass
