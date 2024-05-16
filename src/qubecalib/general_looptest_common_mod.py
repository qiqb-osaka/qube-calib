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

from .quel1_wave_subsystem_mod import Quel1WaveSubsystemMod

socket, Any, Mapping, SequencerClient

logger = logging.getLogger(__name__)


class BoxPool:
    def __init__(self) -> None:
        self._clock_master = (
            None  # QuBEMasterClient(settings["CLOCK_MASTER"]["ipaddr"])
        )
        self._boxes: Dict[str, Tuple[Quel1Box, SequencerClient]] = {}
        self._linkstatus: Dict[str, bool] = {}

    def create_clock_master(
        self,
        ipaddr: str,
    ) -> None:
        self._clock_master = QuBEMasterClient(master_ipaddr=ipaddr)

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

    def emit_at(self) -> None:
        min_time_offset = 125_000_000
        time_counts = [i * 125_000_000 for i in range(1)]

        if not len(self.pulsegens):
            logger.warning("no pulse generator to activate")

        box_names = set([_.box_name for _ in self.pulsegens])
        for box_name in box_names:
            box = next(iter({_.box for _ in self.pulsegens if _.box_name == box_name}))
            sqc = next(iter({_.sqc for _ in self.pulsegens if _.box_name == box_name}))
            awgs = {_.awg for _ in self.pulsegens if _.box_name == box_name}
            box.wss.clear_before_starting_emission(awgs)
            valid_read, current_time, last_sysref_time = sqc.read_clock()
            if valid_read:
                logger.info(
                    f"current time: {current_time},  last sysref time: {last_sysref_time}"
                )
            else:
                raise RuntimeError("failed to read current clock")

            base_time = (
                current_time + min_time_offset
            )  # TODO: implement constraints of the start timing
            for i, time_count in enumerate(time_counts):
                valid_sched = sqc.add_sequencer(base_time + time_count)
                if not valid_sched:
                    raise RuntimeError("failed to schedule AWG start")
            logger.info("scheduling completed")


class PulseGen_:
    # NUM_SAMPLES_IN_WAVE_BLOCK: Final[int] = 64

    def __init__(
        self,
        box_name: str,
        port: int,
        channel: int,
        waveseq: WaveSequence,
        boxpool: BoxPool,
    ):
        box, sqc = boxpool.get_box(box_name)
        if not all(box.link_status().values()):
            raise RuntimeError(
                f"sender is not available due to link problem of '{box_name}'"
            )

        self.box_name: str = box_name
        self.box: Quel1Box = box
        self.sqc: SequencerClient = sqc

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
        # self.box.config_rfswitch(port=self.port, rfswitch="pass")
        # if self.box.is_loopedback_monitor(self.group):
        #     self.box.open_rfswitch(self.group, "m")
        pass

    def init_wave(self) -> None:
        Quel1WaveSubsystemMod.set_wave(
            self.box.wss,
            self.awg,
            self.waveseq,
        )


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
            Tuple[str, int], PulseGen_  # (box_name, capmod): PulseGen_
        ],
    ) -> Dict[Tuple[str, int], Future]:
        # box_name, capmod 毎に pulsecaps をまとめる
        box_names_capms = {(_.box_name, _.capmod) for _ in self.pulsecaps}
        box_names_capms__pulsecaps = {
            (box_name, capm): [
                _ for _ in self.pulsecaps if _.box_name == box_name and _.capmod == capm
            ]
            for box_name, capm in box_names_capms
        }

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
        box_name: str,
        port: int,
        channel: int,
        capprm: CaptureParam,
        boxpool: BoxPool,
    ):
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
        self.capmod = self.box.rmap.get_capture_module_of_rline(group, rline)
        self.capprm = capprm
        self.boxpool = boxpool

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
        # self.box.config_rfswitch(port=self.port, rfswitch="open")
        # self.box.close_rfswitch(group=self.group, line=self.rline)
        pass

    def init_capture(self) -> None:
        # wss._setup_capture_units 内で同様の目的のコードが実行されている
        pass
