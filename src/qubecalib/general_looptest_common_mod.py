from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Final, MutableSequence, Optional

import numpy as np
import numpy.typing as npt
from e7awgsw import CaptureParam, WaveSequence
from quel_clock_master import QuBEMasterClient, SequencerClient
from quel_ic_config import (
    CaptureReturnCode,
    Quel1Box,
    Quel1BoxType,
    Quel1ConfigOption,
)

from .quel1_wave_subsystem_mod import Quel1WaveSubsystemMod

logger = logging.getLogger(__name__)


class BoxPool:
    SYSREF_PERIOD: int = 2_000
    DEFAULT_NUM_SYSREF_MEASUREMENTS: Final[int] = 100

    def __init__(self) -> None:
        self._clock_master = (
            None  # QuBEMasterClient(settings["CLOCK_MASTER"]["ipaddr"])
        )
        self._boxes: dict[str, tuple[Quel1Box, SequencerClient]] = {}
        self._linkstatus: dict[str, bool] = {}
        self._estimated_timediff: dict[str, int] = {}
        self._cap_sysref_time_offset: int = 0
        self._port_direction: dict[tuple[str, int], str] = {}
        self._box_config_cache: dict[str, dict] = {}

    def create_clock_master(
        self,
        ipaddr: str,
    ) -> None:
        self._clock_master = QuBEMasterClient(master_ipaddr=ipaddr)

    def measure_timediff(
        self, num_iters: int = DEFAULT_NUM_SYSREF_MEASUREMENTS
    ) -> tuple[str, int]:
        sqcs = {name: sqc for name, (_, sqc) in self._boxes.items()}
        counter_at_sysref_clk = {name: 0 for name in self._boxes}
        for _ in range(num_iters):
            for name, sqc in sqcs.items():
                m = sqc.read_clock()
                if len(m) < 2:
                    raise RuntimeError("firmware doesn't support this measurement")
                counter_at_sysref_clk[name] += m[2] % self.SYSREF_PERIOD
        avg: dict[str, int] = {
            name: round(cntr / num_iters)
            for name, cntr in counter_at_sysref_clk.items()
        }
        refname = list(self._boxes.keys())[0]
        adj = avg[refname]
        self._estimated_timediff = {name: cntr - adj for ipaddr, cntr in avg.items()}
        self._cap_sysref_time_offset = avg[refname]
        return refname, avg[refname]

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
    ) -> tuple[Quel1Box, SequencerClient]:
        if name in self._boxes:
            box, sqc = self._boxes[name]
            return box, sqc
        else:
            raise ValueError(f"invalid name of box: '{name}'")

    def get_port_direction(self, box_name: str, port: int) -> str:
        if (box_name, port) not in self._port_direction:
            box = self.get_box(box_name)[0]
            self._port_direction[(box_name, port)] = box.dump_port(port)["direction"]
        return self._port_direction[(box_name, port)]


@dataclass
class Resource:
    name: str
    box: Quel1Box
    sqc: SequencerClient
    awgs: set[int]
    awgbitmap: int
    timediff: int
    offset: int  # tap
    tts: int


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
        """単体のボックス毎に複数のポートから PulseGen に従ったパルスを出射する．ボックスの単位でパルスは同期して出射されるが，ボックス間では同期されない．"""
        if not self.pulsegens:
            logger.warn("no pulse generator to activate")
            return
        box_names = {_.box_name for _ in self.pulsegens}
        for box_name in box_names:
            box, _ = self.boxpool.get_box(box_name)
            box.wss.start_emission([_.awg for _ in self.pulsegens])

    def emit_at(
        self,
        offset: dict[str, int] = {},
        tts: dict[str, int] = {},
    ) -> None:
        if not len(self.pulsegens):
            logger.warning("no pulse generator to activate")

        MIN_TIME_OFFSET = 12_500_000

        box_names = set([pg.box_name for pg in self.pulsegens])
        sqc = next(
            iter({g.sqc for g in self.pulsegens if g.box_name == next(iter(box_names))})
        )
        valid_read, current_time, last_sysref_time = sqc.read_clock()
        if valid_read:
            logger.info(
                f"current time: {current_time},  last sysref time: {last_sysref_time}"
            )
        else:
            raise RuntimeError("failed to read current clock")
        base_time = current_time + MIN_TIME_OFFSET
        tamate_offset = (
            16 - (base_time - self.boxpool._cap_sysref_time_offset) % 16
        ) % 16
        base_time += tamate_offset

        awgs_by_boxname = {}
        for boxname in box_names:
            awgs_by_boxname[boxname] = {
                g.awg for g in self.pulsegens if g.box_name == boxname
            }

        awgbitmap_by_boxname: dict[str, int] = defaultdict(int)
        for boxname, awgs in awgs_by_boxname.items():
            for awg in awgs:
                awgbitmap_by_boxname[boxname] |= 1 << awg

        resources_by_boxname: dict[str, Resource] = {}
        for g in self.pulsegens:
            resources_by_boxname[g.box_name] = Resource(
                name=g.box_name,
                box=g.box,
                sqc=g.sqc,
                awgs=awgs_by_boxname[g.box_name],
                awgbitmap=awgbitmap_by_boxname[g.box_name],
                timediff=self.boxpool._estimated_timediff[g.box_name]
                if g.box_name in self.boxpool._estimated_timediff
                else 0,
                offset=offset[g.box_name] if g.box_name in offset else 0,
                tts=tts[g.box_name] if g.box_name in tts else 0,
            )

        for _, r in resources_by_boxname.items():
            r.box.wss.clear_before_starting_emission(r.awgs)

        for _, r in resources_by_boxname.items():
            schedule = base_time - r.timediff + r.tts + 16 * r.offset
            valid_sched = r.sqc.add_sequencer(schedule, awg_bitmap=r.awgbitmap)
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

        self.port, self.subport = self.box._decode_port(port)
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
    ) -> tuple[dict[tuple[str, int], CaptureReturnCode], dict[tuple[str, int], Any]]:
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
        triggering_pgs: dict[
            tuple[str, int], PulseGen_  # (box_name, capmod): PulseGen_
        ],
    ) -> dict[tuple[str, int], Future]:
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
        futures: dict[tuple[str, int], Future],
    ) -> tuple[dict[tuple[str, int], CaptureReturnCode], dict[tuple[str, int], dict]]:
        result = {
            (box_name, capm): future.result()
            for (box_name, capm), future in futures.items()
        }
        status = {k: v[0] for k, v in result.items()}
        iqs = {k: v[1] for k, v in result.items()}
        return status, iqs

    def measure_background_noise(
        self,
    ) -> dict[tuple[str, int, int], tuple[float, float, npt.NDArray[np.complex64]]]:
        cprm = CaptureParam()
        cprm.add_sum_section(
            num_words=4096 // CaptureParam.NUM_SAMPLES_IN_ADC_WORD,
            num_post_blank_words=1,
        )
        # backup capprm
        backup = {(c.box_name, c.port, c.channel): c.capprm for c in self.pulsecaps}
        # load capprm
        for c in self.pulsecaps:
            c.capprm = cprm
        status, iq = self.capture_now()
        # restore capprm
        for c in self.pulsecaps:
            c.capprm = backup[(c.box_name, c.port, c.channel)]

        result = {}
        for c in self.pulsecaps:
            idx = (c.box_name, c.capmod)
            if status[idx] == CaptureReturnCode.SUCCESS:
                data = iq[idx][c.channel][0].squeeze()
                noise_avg, noise_max = np.average(abs(data)), max(abs(data))
                result[(c.box_name, c.port, c.channel)] = (noise_avg, noise_max, data)
            else:
                raise RuntimeError(f"capture failure due to {status}")
        return result


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

        self.port, self.support = self.box._decode_port(port)
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
