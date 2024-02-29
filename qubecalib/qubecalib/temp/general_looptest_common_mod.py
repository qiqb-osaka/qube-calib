from __future__ import annotations

import logging
import socket
from typing import Any, Final, Mapping, Optional

from e7awgsw import CaptureParam, WaveSequence
from quel_clock_master import SequencerClient
from quel_ic_config_utils import (
    SimpleBoxIntrinsic,
    create_box_objects,
)

from .general_looptest_common import BoxPool, PulseCap, PulseGen
from .quel1_wave_subsystem_mod import Quel1WaveSubsystemMod

socket, Any, Mapping, SequencerClient, SimpleBoxIntrinsic, create_box_objects
BoxPool, PulseGen, PulseCap

logger = logging.getLogger(__name__)


class BoxPoolMod:
    @classmethod
    def get_box(
        self,
        boxpool: BoxPool,
        boxname: str,
    ) -> tuple[SimpleBoxIntrinsic, SequencerClient]:
        return boxpool._boxes["BOX" + boxname]

    @classmethod
    def get_linkstatus(
        self,
        boxpool: BoxPool,
        boxname: str,
    ) -> bool:
        return boxpool._linkstatus["BOX" + boxname]

    @classmethod
    def get_boxnames(
        self,
        boxpool: BoxPool,
    ) -> list[str]:
        return [_.replace("BOX", "", 1) for _ in boxpool._boxes]


# class BoxPoolMod(BoxPool):
#     def _parse_settings(self, settings: Mapping[str, Mapping[str, Any]]) -> None:
#         for k, v in settings.items():
#             if k.startswith("BOX"):
#                 kwd = k.replace("BOX", "", 1)
#                 _, _, _, _, box = create_box_objects(**v, refer_by_port=False)
#                 if not isinstance(box, SimpleBoxIntrinsic):
#                     raise ValueError(f"unsupported boxtype: {v['boxtype']}")
#                 sqc = SequencerClient(v["ipaddr_sss"])
#                 self._boxes[kwd] = (box, sqc)
#                 self._linkstatus[kwd] = False


# class BoxPoolMod:
#     @classmethod
#     def create_boxpool(cls, settings: dict[str, dict[str, Any]]) -> BoxPool:
#         _settings = dict()
#         if "clockmaster_setting" not in settings:
#             raise ValueError("colockmaster_setting field required")
#         _settings["CLOCK_MASTER"] = settings["clockmaster_setting"]
#         boxpool = BoxPool(settings=_settings)
#         if "box_settings" not in settings:
#             return boxpool
#         cls.add_box(boxpool, settings["box_settings"])
#         return boxpool

#     @classmethod
#     def add_box(
#         cls,
#         boxpool: BoxPool,
#         settings: Mapping[str, Any],
#     ) -> None:
#         for k, v in settings.items():
#             _, _, _, _, box = create_box_objects(**v, refer_by_port=False)
#             if not isinstance(box, SimpleBoxIntrinsic):
#                 raise ValueError(f"unsupported boxtype: {v['boxtype']}")
#             sqc = SequencerClient(v["ipaddr_sss"])
#             boxpool._boxes[k] = (box, sqc)
#             boxpool._linkstatus[k] = False


class PulseGenSinglebox:
    NUM_SAMPLES_IN_WAVE_BLOCK: Final[int] = 64

    def __init__(
        self,
        name: str,
        box_status: bool,
        box: SimpleBoxIntrinsic,
        group: int,
        line: int,
        channel: int,
        waveseq: WaveSequence,
        *,
        lo_freq: Optional[float] = None,
        cnco_freq: Optional[float] = None,
        fnco_freq: Optional[float] = None,
        sideband: Optional[str] = None,
        vatt: Optional[int] = None,
    ):
        if not box_status:
            raise RuntimeError(
                f"sender '{name}' is not available due to link problem of '{box}'"
            )

        self.name = name
        self.box = box
        self.group = group
        self.line = line
        self.channel = channel
        self.awg = box.rmap.get_awg_of_channel(group, line, channel)
        self.waveseq = waveseq

        self.lo_freq = lo_freq
        self.cnco_freq = cnco_freq
        self.fnco_freq = fnco_freq
        self.sideband = sideband
        self.vatt = vatt

    def init(self) -> None:
        self.init_config()
        self.init_wave()

    def init_config(self) -> None:
        self.box.config_line(
            group=self.group,
            line=self.line,
            lo_freq=self.lo_freq,
            cnco_freq=self.cnco_freq,
            sideband=self.sideband,
            vatt=self.vatt,
        )
        self.box.config_channel(
            group=self.group,
            line=self.line,
            channel=self.channel,
            fnco_freq=self.fnco_freq,
        )
        self.box.open_rfswitch(self.group, self.line)
        if self.box.is_loopedback_monitor(self.group):
            self.box.open_rfswitch(self.group, "m")

    def init_wave(self) -> None:
        Quel1WaveSubsystemMod.set_wave(
            self.box.wss,
            self.awg,
            self.waveseq,
        )

    def emit_now(self) -> None:
        """単体のチャネルからバルスを出射する．チャネル間のパルスは非同期で出射される．"""
        self.box.wss.start_emission(awgs=(self.awg,))

    def stop_now(self) -> None:
        """単体のチャネルを停止する．"""
        self.box.wss.stop_emission(awgs=(self.awg,))


class PulseCapSinglebox:
    def __init__(
        self,
        name: str,
        box_status: bool,
        box: SimpleBoxIntrinsic,
        group: int,
        rline: str,
        channel: int,
        capprm: CaptureParam,
        *,
        lo_freq: Optional[float] = None,
        cnco_freq: Optional[float] = None,
        fnco_freq: Optional[float] = None,
        sideband: Optional[str] = None,
        background_noise_threshold: float = 256,
    ):
        if not box_status:
            raise RuntimeError(
                f"sender '{name}' is not available due to link problem of '{box}'"
            )

        self.name = name
        self.box = box
        self.group = group
        self.rline = rline
        self.channel = channel
        # self.runit: int = channel
        self.capmod = self.box.rmap.get_capture_module_of_rline(group, rline)
        self.capprm = capprm

        self.lo_freq = lo_freq
        self.cnco_freq = cnco_freq
        self.fnco_freq = fnco_freq
        self.sideband = sideband

        self.background_noise_threshold = background_noise_threshold

    def init(self) -> None:
        self.box.config_rline(
            group=self.group,
            rline=self.rline,
            lo_freq=self.lo_freq,
            cnco_freq=self.cnco_freq,
        )
        # self.box.config_runit(
        #     group=self.group,
        #     rline=self.rline,
        #     runit=self.runit,
        #     fnco_freq=self.fnco_freq,
        # )
        # Notes: receive signal from input_port, not rather than internal loop-back.
        self.box.open_rfswitch(group=self.group, line=self.rline)


class PulseGenMod:
    @classmethod
    def init_config(
        cls,
        pulsegen: PulseGen,
    ) -> None:
        pg = pulsegen

        kwargs: dict[str, int | float | str] = {
            "group": pg.group,
            "line": pg.line,
        }
        if pg.lo_freq != -1:
            kwargs["lo_freq"] = pg.lo_freq
        if pg.cnco_freq != -1:
            kwargs["cnco_freq"] = pg.cnco_freq
        if pg.sideband == "U" or pg.sideband == "L":
            kwargs["sideband"] = pg.sideband
        if pg.vatt != -1:
            kwargs["vatt"] = pg.vatt
        pulsegen.box.config_line(**kwargs)

        kwargs = {
            "group": pg.group,
            "line": pg.line,
            "channel": pg.channel,
        }
        if pg.fnco_freq != -1:
            kwargs["fnco_freq"] = pg.fnco_freq
        pg.box.config_channel(**kwargs)
        pg.box.open_rfswitch(pg.group, pg.line)
        if pg.box.is_loopedback_monitor(pg.group):
            pg.box.open_rfswitch(pg.group, "m")  # for external loop-back lines

    @classmethod
    def init_wave(
        cls,
        wave: WaveSequence,
        pulsegen: PulseGen,
    ) -> None:
        Quel1WaveSubsystemMod.set_wave(
            pulsegen.box.wss,
            pulsegen.awg,
            wave,
        )

    @classmethod
    def init(
        cls,
        wave: WaveSequence,
        pulsegen: PulseGen,
    ) -> None:
        cls.init_config(pulsegen)
        cls.init_wave(wave, pulsegen)
