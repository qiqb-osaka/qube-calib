from __future__ import annotations

import logging
import socket
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Collection, Dict, Mapping, MutableSequence, Optional, Tuple

from e7awgsw import CaptureParam, WaveSequence
from quel_clock_master import QuBEMasterClient, SequencerClient
from quel_ic_config import (
    CaptureReturnCode,
    Quel1Box,
    Quel1BoxType,
    Quel1ConfigOption,
)

# from .general_looptest_common import BoxPool, PulseCap, PulseGen
from .quel1_wave_subsystem_mod import Quel1WaveSubsystemMod

socket, Any, Mapping, SequencerClient
# BoxPool, PulseGen, PulseCap

logger = logging.getLogger(__name__)


# class BoxPoolMod:
#     @classmethod
#     def get_box(
#         self,
#         boxpool: BoxPool,
#         boxname: str,
#     ) -> tuple[Quel1BoxIntrinsic, SequencerClient]:
#         return boxpool._boxes["BOX" + boxname]

#     @classmethod
#     def get_linkstatus(
#         self,
#         boxpool: BoxPool,
#         boxname: str,
#     ) -> bool:
#         return boxpool._linkstatus["BOX" + boxname]

#     @classmethod
#     def get_boxnames(
#         self,
#         boxpool: BoxPool,
#     ) -> list[str]:
#         return [_.replace("BOX", "", 1) for _ in boxpool._boxes]


class BoxPool:
    def __init__(self) -> None:
        self._clock_master = (
            None  # QuBEMasterClient(settings["CLOCK_MASTER"]["ipaddr"])
        )
        self._boxes: Dict[str, Tuple[Quel1Box, SequencerClient]] = {}
        self._linkstatus: Dict[str, bool] = {}
        # self._parse_settings(settings)
        # self._estimated_timediff: Dict[str, int] = {
        #     boxname: 0 for boxname in self._boxes
        # }
        # self._cap_sysref_time_offset: int = 0

    def set_clock_master(
        self,
        ipaddr: str,
    ) -> None:
        self._clockmaster = QuBEMasterClient(master_ipaddr=ipaddr)

    def create(
        self,
        box_name: str,
        *,
        ipaddr_wss: str,
        ipaddr_sss: str,
        ipaddr_css: str,
        boxtype: Quel1BoxType,
        config_root: Optional[Path],
        config_options: Optional[Collection[Quel1ConfigOption]] = None,
    ) -> Quel1Box:
        box = Quel1Box.create(
            ipaddr_wss=ipaddr_wss,
            ipaddr_sss=ipaddr_sss,
            ipaddr_css=ipaddr_css,
            boxtype=boxtype,
            config_root=config_root,
            config_options=config_options,
        )
        sqc = SequencerClient(ipaddr_sss)
        self._boxes[box_name] = (box, sqc)
        self._linkstatus[box_name] = False
        return box

    def init(self, reconnect: bool = True, resync: bool = True) -> None:
        self.scan_link_status(reconnect=reconnect)
        self.reset_awg()
        if self._clock_master is None:
            return

        # if resync:
        #     self.resync()
        # if not self.check_clock():
        #     raise RuntimeError("failed to acquire time count from some clocks")

    def scan_link_status(
        self,
        reconnect: bool = False,
    ) -> None:
        for name, (box, sqc) in self._boxes.items():
            link_status: bool = True
            if reconnect:
                if not all(box.reconnect().values()):
                    if all(
                        box.reconnect(
                            ignore_crc_error_of_mxfe=box.css.get_all_groups()
                        ).values()
                    ):
                        logger.warning(
                            f"crc error has been detected on MxFEs of {name}"
                        )
                    else:
                        logger.error(
                            f"datalink between MxFE and FPGA of {name} is not working"
                        )
                        link_status = False
            else:
                if not all(box.link_status().values()):
                    if all(
                        box.link_status(
                            ignore_crc_error_of_mxfe=box.css.get_all_groups()
                        ).values()
                    ):
                        logger.warning(
                            f"crc error has been detected on MxFEs of {name}"
                        )
                    else:
                        logger.error(
                            f"datalink between MxFE and FPGA of {name} is not working"
                        )
                        link_status = False
            self._linkstatus[name] = link_status

    def reset_awg(self) -> None:
        for name, (box, _) in self._boxes.items():
            box.easy_stop_all(control_port_rfswitch=True)
            box.initialize_all_awgs()

    def get_box(
        self,
        name: str,
    ) -> Tuple[Quel1Box, SequencerClient]:
        if name in self._boxes:
            box, sqc = self._boxes[name]
            return box, sqc
        else:
            raise ValueError(f"invalid name of box: '{name}'")


class PulseGen:
    def __init__(
        self,
        boxpool: BoxPool,
    ) -> None:
        self.boxpool = boxpool
        self.pulsegens: MutableSequence[PulseGen_] = []

    def create(
        self,
        box_name: str,
        port: int,
        channel: int,
        waveseq: WaveSequence,
    ) -> None:
        pg = PulseGen_(
            box_name=box_name,
            port=port,
            channel=channel,
            waveseq=waveseq,
            boxpool=self.boxpool,
        )
        pg.init()  # TODO ここで実行しても良い？
        self.pulsegens.append(pg)

    def emit_now(self) -> None:
        """単体のボックス毎に複数のポートから PulseGen に従ったパルスを出射する．ボックスの単位でパルスは同期して出射されるが，ボックス館では同期されない．"""
        if not self.pulsegens:
            logger.warn("no pulse generator to activate")
            return
        box_names = {_.box_name for _ in self.pulsegens}
        for box_name in box_names:
            box, _ = self.boxpool.get_box(box_name)
            box.wss.start_emission([_.awg for _ in self.pulsegens])


class PulseGen_:
    # NUM_SAMPLES_IN_WAVE_BLOCK: Final[int] = 64

    def __init__(
        self,
        # name: str,
        # box_status: bool,
        box_name: str,
        # box: Quel1Box,
        port: int,
        # group: int,
        # line: int,
        channel: int,
        waveseq: WaveSequence,
        boxpool: BoxPool,
        # *,
        # lo_freq: Optional[float] = None,
        # cnco_freq: Optional[float] = None,
        # fnco_freq: Optional[float] = None,
        # sideband: Optional[str] = None,
        # vatt: Optional[int] = None,
    ):
        # if not box_status:
        #     raise RuntimeError(
        #         f"sender '{(box, port, channel)}' is not available due to link problem of '{box}'"
        #     )

        # self.name = name
        box, sqc = boxpool.get_box(box_name)
        if not all(box.link_status().values()):
            raise RuntimeError(
                f"sender is not available due to link problem of '{box_name}'"
            )

        self.box_name: str = box_name
        self.box: Quel1Box = box
        self.sqc: SequencerClient = sqc

        # self.box = box
        self.port, self.subport = self.box.decode_port(port)
        group, line = self.box._convert_any_port(port)
        if isinstance(line, str):
            raise ValueError("invalid port")
        self.group = group
        self.line = line
        self.channel = channel  # TODO: better to check the validity
        self.awg = box.rmap.get_awg_of_channel(group, line, channel)
        self.awg_spec = (port, channel)
        self.waveseq = waveseq
        self.boxpool = boxpool

        # self.lo_freq = lo_freq
        # self.cnco_freq = cnco_freq
        # self.fnco_freq = fnco_freq
        # self.sideband = sideband
        # self.vatt = vatt

    def init(self) -> None:
        self.init_config()
        self.init_wave()

    def init_config(self) -> None:
        # self.box.config_line(
        #     group=self.group,
        #     line=self.line,
        #     lo_freq=self.lo_freq,
        #     cnco_freq=self.cnco_freq,
        #     sideband=self.sideband,
        #     vatt=self.vatt,
        # )
        # self.box.config_channel(
        #     group=self.group,
        #     line=self.line,
        #     channel=self.channel,
        #     fnco_freq=self.fnco_freq,
        # )
        # self.box.close_rfswitch(self.group, self.line)
        self.box.config_rfswitch(port=self.port, rfswitch="pass")
        # if self.box.is_loopedback_monitor(self.group):
        #     self.box.open_rfswitch(self.group, "m")

    def init_wave(self) -> None:
        Quel1WaveSubsystemMod.set_wave(
            self.box.wss,
            self.awg,
            self.waveseq,
        )

    # def emit_now(self) -> None:
    #     """単体のチャネルからバルスを出射する．チャネル間のパルスは非同期で出射される．"""
    #     self.box.wss.start_emission(awgs=(self.awg,))

    # def stop_now(self) -> None:
    #     """単体のチャネルを停止する．"""
    #     self.box.wss.stop_emission(awgs=(self.awg,))


class PulseCap:
    def __init__(
        self,
        boxpool: BoxPool,
    ) -> None:
        self.boxpool = boxpool
        self.pulsecaps: MutableSequence[PulseCap_] = []

    def create(
        self,
        box_name: str,
        port: int,
        channel: int,
        capprm: CaptureParam,
    ) -> None:
        pc = PulseCap_(
            box_name=box_name,
            port=port,
            channel=channel,
            capprm=capprm,
            boxpool=self.boxpool,
        )
        pc.init()  # TODO ここで実行しても良い？
        self.pulsecaps.append(pc)

    def capture_now(
        self,
    ) -> Tuple[Dict[Tuple[str, int], CaptureReturnCode], Dict[Tuple[str, int], Any]]:
        # box_name, capmod 毎に pulsecaps をまとめる
        box_names_capms = {(_.box_name, _.capmod) for _ in self.pulsecaps}
        box_names_capms__pulsecaps = {
            (box_name, capm): [
                _ for _ in self.pulsecaps if _.box_name == box_name and _.capmod == capm
            ]
            for box_name, capm in box_names_capms
        }
        status_, iqs_ = {}, {}
        for (box_name, capm), pulsecaps in box_names_capms__pulsecaps.items():
            box, cqs = self.boxpool.get_box(box_name)
            capus_capprms = {_.channel: _.capprm for _ in pulsecaps}
            future = Quel1WaveSubsystemMod.simple_capture_start(
                wss=box.wss,
                capmod=capm,
                capunits_capprms=capus_capprms,
                triggering_awg=None,
            )
            status, iqs = future.result()
            status_[(box_name, capm)] = status
            iqs_[(box_name, capm)] = iqs
        return status_, iqs_

    def capture_at_trigger_of(
        self,
        triggering_pgs: Dict[
            Tuple[str, int], PulseGen_
        ],  # (box_name, capmod): PulseGen_
    ) -> Dict[Tuple[str, int], Future]:
        # box_name, capmod 毎に pulsecaps をまとめる
        box_names_capms = {(_.box_name, _.capmod) for _ in self.pulsecaps}
        box_names_capms__pulsecaps = {
            (box_name, capm): [
                _ for _ in self.pulsecaps if _.box_name == box_name and _.capmod == capm
            ]
            for box_name, capm in box_names_capms
        }
        # # (box_name, capmod): (box_name, port, channel) のマップを作る
        # capmods_portchannels = {
        #     (_.box_name, _.capmod): (_.box_name, _.port, _.channel)
        #     for _ in self.pulsecaps
        # }
        # # box_name, capmod 毎に triggering_pg をまとめる
        # capms__triggering_pgs = {
        #     (box_name, capm): triggering_pgs[(box_name, port, channel)]
        #     for (box_name, capm), (_, port, channel) in capmods_portchannels
        # }
        # print(triggering_pgs)

        # status_, iqs_ = {}, {}
        futures = {}
        for (box_name, capm), pulsecaps in box_names_capms__pulsecaps.items():
            box, cqs = self.boxpool.get_box(box_name)
            capus_capprms = {_.channel: _.capprm for _ in pulsecaps}
            future = Quel1WaveSubsystemMod.simple_capture_start(
                wss=box.wss,
                capmod=capm,
                capunits_capprms=capus_capprms,
                triggering_awg=triggering_pgs[(box_name, capm)].awg,
                timeout=10,  # TODO setting from qubecalib layer
            )
            futures[(box_name, capm)] = future
        return futures

    def wait_until_capture_finishes(
        self,
        futures: Dict[Tuple[str, int], Future],
    ) -> Tuple[Dict[Tuple[str, int], CaptureReturnCode], Dict[Tuple[str, int], Dict]]:
        result = {
            (box_name, capm): future.result()
            for (box_name, capm), future in futures.items()
        }
        status = {k: v[0] for k, v in result.items()}
        iqs = {k: v[1] for k, v in result.items()}
        return status, iqs


class PulseCap_:
    def __init__(
        self,
        # name: str,
        # box_status: bool,
        # box: Quel1Box,
        box_name: str,
        port: int,
        # group: int,
        # rline: str,
        channel: int,
        capprm: CaptureParam,
        boxpool: BoxPool,
        # *,
        # lo_freq: Optional[float] = None,
        # cnco_freq: Optional[float] = None,
        # fnco_freq: Optional[float] = None,
        # sideband: Optional[str] = None,
        # background_noise_threshold: float = 256,
    ):
        # if not box_status:
        #     raise RuntimeError(
        #         f"capturer is not available due to link problem of '{box}'"
        #     )

        # self.name = name
        box, sqc = boxpool.get_box(box_name)
        if not all(box.link_status().values()):
            raise RuntimeError(
                f"capture is not available due to link problem of '{box_name}'"
            )

        self.box_name = box_name
        self.box = box
        self.sqc = sqc

        self.port, self.support = self.box.decode_port(port)
        group, rline = self.box._convert_any_port(port)
        if isinstance(rline, int):
            raise ValueError("invalid port")
        self.group = group
        self.rline = rline
        self.channel = channel
        # self.runit: int = channel
        self.capmod = self.box.rmap.get_capture_module_of_rline(group, rline)
        self.capprm = capprm
        self.boxpool = boxpool

        # self.lo_freq = lo_freq
        # self.cnco_freq = cnco_freq
        # self.fnco_freq = fnco_freq
        # self.sideband = sideband

        # self.background_noise_threshold = background_noise_threshold

    def init(self) -> None:
        self.init_config()
        # self.init_capture()

    def init_config(self) -> None:
        # self.box.config_rline(
        #     group=self.group,
        #     rline=self.rline,
        #     lo_freq=self.lo_freq,
        #     cnco_freq=self.cnco_freq,
        # )
        # self.box.config_runit(
        #     group=self.group,
        #     rline=self.rline,
        #     runit=self.channel,
        #     fnco_freq=self.fnco_freq,
        # )
        # Notes: receive signal from input_port, not rather than internal loop-back.
        self.box.config_rfswitch(port=self.port, rfswitch="open")
        # self.box.close_rfswitch(group=self.group, line=self.rline)

    def init_capture(self) -> None:
        # wss._setup_capture_units 内で同様の目的のコードが実行されている
        pass


# class PulseGenMod:
#     @classmethod
#     def init_config(
#         cls,
#         pulsegen: PulseGen,
#     ) -> None:
#         pg = pulsegen

#         kwargs: dict[str, int | float | str] = {
#             "group": pg.group,
#             "line": pg.line,
#         }
#         if pg.lo_freq != -1:
#             kwargs["lo_freq"] = pg.lo_freq
#         if pg.cnco_freq != -1:
#             kwargs["cnco_freq"] = pg.cnco_freq
#         if pg.sideband == "U" or pg.sideband == "L":
#             kwargs["sideband"] = pg.sideband
#         if pg.vatt != -1:
#             kwargs["vatt"] = pg.vatt
#         pulsegen.box.config_line(**kwargs)

#         kwargs = {
#             "group": pg.group,
#             "line": pg.line,
#             "channel": pg.channel,
#         }
#         if pg.fnco_freq != -1:
#             kwargs["fnco_freq"] = pg.fnco_freq
#         pg.box.config_channel(**kwargs)
#         pg.box.close_rfswitch(pg.group, pg.line)
#         # pg.box.open_rfswitch(pg.group, pg.line)
#         # if pg.box.is_loopedback_monitor(pg.group):
#         #     pg.box.open_rfswitch(pg.group, "m")  # for external loop-back lines

#     @classmethod
#     def init_wave(
#         cls,
#         wave: WaveSequence,
#         pulsegen: PulseGen,
#     ) -> None:
#         Quel1WaveSubsystemMod.set_wave(
#             pulsegen.box.wss,
#             pulsegen.awg,
#             wave,
#         )

#     @classmethod
#     def init(
#         cls,
#         wave: WaveSequence,
#         pulsegen: PulseGen,
#     ) -> None:
#         cls.init_config(pulsegen)
#         cls.init_wave(wave, pulsegen)
