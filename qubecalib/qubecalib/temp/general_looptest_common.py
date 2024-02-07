import logging
from concurrent.futures import Future
from typing import Any, Dict, Final, List, Mapping, Sequence, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from quel_clock_master import QuBEMasterClient, SequencerClient
from quel_ic_config_utils import (
    CaptureResults,
    CaptureReturnCode,
    SimpleBoxIntrinsic,
    create_box_objects,
)

logger = logging.getLogger(__name__)


class BoxPool:
    SYSREF_PERIOD: Final[int] = 2000
    # TODO: find the best value by careful experiments. it seems that the best value is 6 or around.
    TIMING_OFFSET: Final[int] = 0
    DEFAULT_NUM_SYSREF_MEASUREMENTS: Final[int] = 100

    def __init__(self, settings: Mapping[str, Mapping[str, Any]]):
        self._clock_master = QuBEMasterClient(settings["CLOCK_MASTER"]["ipaddr"])
        self._boxes: Dict[str, Tuple[SimpleBoxIntrinsic, SequencerClient]] = {}
        self._linkstatus: Dict[str, bool] = {}
        self._parse_settings(settings)
        self._estimated_timediff: Dict[str, int] = {
            boxname: 0 for boxname in self._boxes
        }
        self._cap_sysref_time_offset: int = 0

    def _parse_settings(self, settings: Mapping[str, Mapping[str, Any]]):
        for k, v in settings.items():
            if k.startswith("BOX"):
                _, _, _, _, box = create_box_objects(**v, refer_by_port=False)
                if not isinstance(box, SimpleBoxIntrinsic):
                    raise ValueError(f"unsupported boxtype: {v['boxtype']}")
                sqc = SequencerClient(v["ipaddr_sss"])
                self._boxes[k] = (box, sqc)
                self._linkstatus[k] = False

    def init(self, resync: bool = True):
        for name, (box, sqc) in self._boxes.items():
            link_status: bool = True
            if not all(box.init().values()):
                if all(
                    box.init(ignore_crc_error_of_mxfe=box.css.get_all_groups()).values()
                ):
                    logger.warning(f"crc error has been detected on MxFEs of {name}")
                else:
                    logger.error(
                        f"datalink between MxFE and FPGA of {name} is not working"
                    )
                    link_status = False
            self._linkstatus[name] = link_status

        self.reset_awg()
        if resync:
            self.resync()
        if not self.check_clock():
            raise RuntimeError("failed to acquire time count from some clocks")

    def reset_awg(self):
        for name, (box, _) in self._boxes.items():
            if self._linkstatus[name]:
                box.easy_stop_all(control_port_rfswitch=True)
                box.wss.initialize_all_awgs()

    def resync(self):
        self._clock_master.reset()  # TODO: confirm whether it is harmless or not.
        self._clock_master.kick_clock_synch(
            [sqc.ipaddress for _, (_, sqc) in self._boxes.items()]
        )

    def check_clock(self) -> bool:
        valid_m, cntr_m = self._clock_master.read_clock()
        t = {}
        for name, (_, sqc) in self._boxes.items():
            t[name] = sqc.read_clock()

        flag = True
        if valid_m:
            logger.info(f"master: {cntr_m:d}")
        else:
            flag = False
            logger.info("master: not found")

        for name, (valid, cntr, cntr_last_sysref) in t.items():
            if valid:
                logger.info(f"{name:s}: {cntr:d} {cntr_last_sysref:d}")
            else:
                flag = False
                logger.info(f"{name:s}: not found")
        return flag

    def get_box(self, name: str) -> Tuple[bool, SimpleBoxIntrinsic, SequencerClient]:
        if name in self._boxes:
            box, sqc = self._boxes[name]
            return self._linkstatus[name], box, sqc
        else:
            raise ValueError(f"invalid name of box: '{name}'")

    def emit_at(
        self,
        cp: "PulseCap",
        pgs: Set["PulseGen"],
        min_time_offset: int,
        time_counts=Sequence[int],
        displacement: int = 0,
    ) -> Dict[str, List[int]]:
        if len(pgs) == 0:
            logger.warning("no pulse generator to activate")

        pg_by_box: Dict[str, Set["PulseGen"]] = {box: set() for box in self._boxes}
        for pg in pgs:
            pg_by_box[pg.boxname].add(pg)

        if cp.boxname not in pg_by_box:
            raise RuntimeError("impossible to trigger the capturer")

        # initialize awgs
        targets: Dict[str, SequencerClient] = {}
        for boxname, pgs in pg_by_box.items():
            box, sqc = self._boxes[boxname]
            if len(pgs) == 0:
                continue
            targets[boxname] = sqc
            awgs = {pg.awg for pg in pgs}
            box.wss.clear_before_starting_emission(awgs)
            # Notes: the following is not required actually, just for debug purpose.
            valid_read, current_time, last_sysref_time = sqc.read_clock()
            if valid_read:
                logger.info(
                    f"boxname: {boxname}, current time: {current_time}, "
                    f"sysref offset: {last_sysref_time % self.SYSREF_PERIOD}"
                )
            else:
                raise RuntimeError("failed to read current clock")

        valid_read, current_time, last_sysref_time = targets[cp.boxname].read_clock()
        logger.info(
            f"sysref offset: average: {self._cap_sysref_time_offset},  latest: {last_sysref_time % self.SYSREF_PERIOD}"
        )
        if (
            abs(last_sysref_time % self.SYSREF_PERIOD - self._cap_sysref_time_offset)
            > 4
        ):
            logger.warning("large fluctuation of sysref is detected on the FPGA")
        base_time = current_time + min_time_offset
        offset = (16 - (base_time - self._cap_sysref_time_offset) % 16) % 16
        base_time += offset
        base_time += displacement  # inducing clock displacement for performance evaluation (must be 0 usually).
        base_time += (
            self.TIMING_OFFSET
        )  # Notes: the safest timing to issue trigger, at the middle of two AWG block.
        schedule: Dict[str, List[int]] = {boxname: [] for boxname in targets}
        for i, time_count in enumerate(time_counts):
            for boxname, sqc in targets.items():
                t = base_time + time_count + self._estimated_timediff[boxname]
                valid_sched = sqc.add_sequencer(t)
                if not valid_sched:
                    raise RuntimeError("failed to schedule AWG start")
                schedule[boxname].append(t)
        logger.info("scheduling completed")
        return schedule

    def measure_timediff(
        self, cp: "PulseCap", num_iters: int = DEFAULT_NUM_SYSREF_MEASUREMENTS
    ) -> None:
        counter_at_sysref_clk: Dict[str, int] = {boxname: 0 for boxname in self._boxes}

        for i in range(num_iters):
            for name, (_, sqc) in self._boxes.items():
                m = sqc.read_clock()
                if len(m) < 2:
                    raise RuntimeError(
                        f"firmware of {name} doesn't support this measurement"
                    )
                counter_at_sysref_clk[name] += m[2] % self.SYSREF_PERIOD

        avg: Dict[str, int] = {
            boxname: round(cntr / num_iters)
            for boxname, cntr in counter_at_sysref_clk.items()
        }
        adj = avg[cp.boxname]
        self._estimated_timediff = {
            boxname: cntr - adj for boxname, cntr in avg.items()
        }
        logger.info(f"estimated time difference: {self._estimated_timediff}")

        self._cap_sysref_time_offset = avg[cp.boxname]


class PulseGen:
    NUM_SAMPLES_IN_WAVE_BLOCK: Final[int] = 64  # this should be taken from e7awgsw

    def __init__(
        self,
        name: str,
        box: str,
        group: int,
        line: int,
        lo_freq: float,
        cnco_freq: float,
        fnco_freq: float,
        sideband: str,
        vatt: int,
        amplitude: int,
        wave_samples: int,
        num_repeats: Tuple[int, int],
        num_wait_words: Tuple[int, int],
        boxpool: BoxPool,
    ):
        box_status, box_obj, sqc = boxpool.get_box(box)
        if not box_status:
            raise RuntimeError(
                f"sender '{name}' is not available due to link problem of '{box}'"
            )
        if wave_samples % self.NUM_SAMPLES_IN_WAVE_BLOCK != 0:
            raise ValueError(
                f"wave samples must be multiple of {self.NUM_SAMPLES_IN_WAVE_BLOCK}"
            )

        self.name: str = name
        self.boxname: str = box
        self.box: SimpleBoxIntrinsic = box_obj
        self.sqc: SequencerClient = sqc
        self.group: int = group
        self.line: int = line
        self.channel: int = 0
        self.awg: int = self.box.rmap.get_awg_of_channel(
            self.group, self.line, self.channel
        )

        self.lo_freq = lo_freq
        self.cnco_freq = cnco_freq
        self.fnco_freq = fnco_freq
        self.sideband = sideband
        self.vatt = vatt

        self.amplitude = amplitude
        self.num_wave_blocks = wave_samples // self.NUM_SAMPLES_IN_WAVE_BLOCK
        self.num_repeats = num_repeats
        self.num_wait_words = num_wait_words

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
        self.box.open_rfswitch(self.group, self.line)  # for spectrum analyzer
        if self.box.is_loopedback_monitor(self.group):
            self.box.open_rfswitch(self.group, "m")  # for external loop-back lines

    def init_wave(self) -> None:
        self.box.wss.set_cw(
            self.awg,
            self.amplitude,
            num_wave_blocks=self.num_wave_blocks,
            num_repeats=self.num_repeats,
            num_wait_words=self.num_wait_words,
        )

    def emit_now(self) -> None:
        self.box.wss.start_emission(awgs=(self.awg,))

    def emit_at(self, min_time_offset: int, time_counts: Sequence[int]) -> None:
        self.box.wss.clear_before_starting_emission((self.awg,))
        valid_read, current_time, last_sysref_time = self.sqc.read_clock()
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
            valid_sched = self.sqc.add_sequencer(base_time + time_count)
            if not valid_sched:
                raise RuntimeError("failed to schedule AWG start")
        logger.info("scheduling completed")

    def stop_now(self) -> None:
        self.box.wss.stop_emission(awgs=(self.awg,))


class PulseCap:
    def __init__(
        self,
        name: str,
        box: str,
        group: int,
        rline: Union[None, str],
        lo_freq: float,
        cnco_freq: float,
        fnco_freq: float,
        background_noise_threshold: float,
        boxpool: BoxPool,
    ):
        box_status, box_obj, _ = boxpool.get_box(box)
        if not box_status:
            raise RuntimeError(
                f"sender '{name}' is not available due to link problem of '{box}'"
            )

        self.name: str = name
        self.boxname: str = box
        self.box: SimpleBoxIntrinsic = box_obj
        self.group: int = group
        self.rline: str = self.box.rmap.resolve_rline(self.group, rline)
        self.runit: int = 0
        self.lo_freq = lo_freq
        self.cnco_freq = cnco_freq
        self.fnco_freq = fnco_freq
        self.background_noise_threshold = background_noise_threshold

        self.capmod = self.box.rmap.get_capture_module_of_rline(self.group, self.rline)

    def init(self):
        self.box.config_rline(
            group=self.group,
            rline=self.rline,
            lo_freq=self.lo_freq,
            cnco_freq=self.cnco_freq,
        )
        self.box.config_runit(
            group=self.group,
            rline=self.rline,
            runit=self.runit,
            fnco_freq=self.fnco_freq,
        )
        # Notes: receive signal from input_port, not rather than internal loop-back.
        self.box.open_rfswitch(group=self.group, line=self.rline)

    def capture_now(self, *, num_samples: int, delay: int = 0):
        thunk = self.box.wss.simple_capture_start(
            capmod=self.capmod,
            capunits=(self.runit,),
            num_words=num_samples // 4,
            delay=delay,
            triggering_awg=None,
        )
        status, iqs = thunk.result()
        return status, iqs[self.runit]

    def check_noise(self, show_graph: bool = True):
        status, iq = self.capture_now(num_samples=1024)
        if status == CaptureReturnCode.SUCCESS:
            noise_avg, noise_max = np.average(abs(iq)), max(abs(iq))
            logger.info(
                f"background noise: max = {noise_max:.1f}, avg = {noise_avg:.1f}"
            )
            judge = noise_max < self.background_noise_threshold
            if show_graph:
                plot_iqs({"test": iq})
            if not judge:
                raise RuntimeError(
                    "the capture port is too noisy, check the output ports connected to the capture port"
                )
        else:
            raise RuntimeError(f"capture failure due to {status}")

    def capture_at_single_trigger_of(
        self, *, pg: PulseGen, num_samples: int, delay: int = 0
    ) -> Future:
        if pg.box != self.box:
            raise ValueError("can not be triggered by an awg of the other box")
        return self.box.wss.simple_capture_start(
            capmod=self.capmod,
            capunits=(self.runit,),
            num_words=num_samples // 4,
            delay=delay,
            triggering_awg=pg.awg,
        )

    def capture_at_multiple_triggers_of(
        self, *, pg: PulseGen, num_iters: int, num_samples: int, delay: int = 0
    ) -> CaptureResults:
        if pg.box != self.box:
            raise ValueError("can not be triggered by an awg of the other box")
        return self.box.wss.capture_start(
            num_iters=num_iters,
            capmod=self.capmod,
            capunits=(self.runit,),
            num_words=num_samples // 4,
            delay=delay,
            triggering_awg=pg.awg,
        )


def init_pulsegen(
    settings: Mapping[str, Mapping[str, Any]],
    common_settings: Mapping[str, Any],
    boxpool: BoxPool,
) -> Dict[str, PulseGen]:
    pgs: Dict[str, PulseGen] = {}
    senders = [s for s in settings if s.startswith("SENDER")]
    for sender in senders:
        v: Dict[str, Any] = dict(settings[sender])
        for k in {
            "box",
            "group",
            "line",
            "wave_samples",
            "num_repeats",
            "num_wait_words",
            "amplitude",
            "lo_freq",
            "cnco_freq",
            "fnco_freq",
            "sideband",
            "vatt",
        }:
            if k not in v:
                if k in common_settings:
                    v[k] = common_settings[k]
                else:
                    raise ValueError(f"missing parameter '{k}' for PulseGen")
        pgs[sender] = PulseGen(name=sender, **v, boxpool=boxpool)
        pgs[sender].init()
    return pgs


def init_pulsecap(
    settings: Mapping[str, Mapping[str, Any]],
    common_settings: Mapping[str, Any],
    boxpool: BoxPool,
) -> PulseCap:
    v = dict(settings["CAPTURER"])
    for k in {
        "box",
        "group",
        "rline",
        "background_noise_threshold",
        "lo_freq",
        "cnco_freq",
        "fnco_freq",
    }:
        if k not in v:
            if k in common_settings:
                v[k] = common_settings[k]
            else:
                raise ValueError(f"missing parameter '{k}' for PulseCap")

    pc = PulseCap(name="_", **v, boxpool=boxpool)
    pc.init()
    return pc


def find_chunks(
    iq: npt.NDArray[np.complex64], power_thr=1000.0, space_thr=16, minimal_length=16
) -> Tuple[Tuple[int, int], ...]:
    chunk = (abs(iq) > power_thr).nonzero()[0]
    if len(chunk) == 0:
        logger.info("no pulse!")
        return ()

    gaps = (chunk[1:] - chunk[:-1]) > space_thr
    start_edges = list(chunk[1:][gaps])
    start_edges.insert(0, chunk[0])
    last_edges = list(chunk[:-1][gaps])
    last_edges.append(chunk[-1])
    chunks = tuple(
        [(s, e) for s, e in zip(start_edges, last_edges) if e - s >= minimal_length]
    )

    n_chunks = len(chunks)
    logger.info(f"number_of_chunks: {n_chunks}")
    for i, chunk in enumerate(chunks):
        s, e = chunk
        logger.info(f"  chunk {i}: {e-s} samples, ({s} -- {e})")
    return chunks


def calc_angle(iq) -> Tuple[float, float, float]:
    angle = np.angle(iq)
    min_angle = min(angle)
    max_angle = max(angle)
    if max_angle - min_angle > 6.0:
        angle = (angle + 2 * np.pi) % np.pi

    avg = np.mean(angle) * 180.0 / np.pi
    sd = np.sqrt(np.var(angle)) * 180.0 / np.pi
    delta = (max(angle) - min(angle)) * 180.0 / np.pi
    return avg, sd, delta


def plot_iqs(iq_dict) -> None:
    n_plot = len(iq_dict)
    fig = plt.figure()

    m = 0
    for _, iq in iq_dict.items():
        m = max(m, np.max(abs(np.real(iq))))
        m = max(m, np.max(abs(np.imag(iq))))

    idx = 0
    for title, iq in iq_dict.items():
        ax = fig.add_subplot(n_plot, 1, idx + 1)
        ax.plot(np.real(iq))
        ax.plot(np.imag(iq))
        ax.text(0.05, 0.1, f"{title}", transform=ax.transAxes)
        ax.set_ylim((-m * 1.1, m * 1.1))
        idx += 1
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    import matplotlib
    from quel_ic_config import Quel1BoxType, Quel1ConfigOption

    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} [{levelname:.4}] {name}: {message}",
        style="{",
    )
    matplotlib.use("Qt5agg")

    COMMON_SETTINGS: Mapping[str, Any] = {
        "lo_freq": 11500e6,
        "cnco_freq": 1500.0e6,
        "fnco_freq": 0,
        "sideband": "L",
        "amplitude": 6000.0,
    }

    DEVICE_SETTINGS: Dict[str, Mapping[str, Any]] = {
        "CLOCK_MASTER": {
            "ipaddr": "10.3.0.13",
            "reset": True,
        },
        "BOX0": {
            "ipaddr_wss": "10.1.0.74",
            "ipaddr_sss": "10.2.0.74",
            "ipaddr_css": "10.5.0.74",
            "boxtype": Quel1BoxType.QuEL1_TypeA,
            "config_root": None,
            "config_options": [Quel1ConfigOption.USE_READ_IN_MXFE0],
        },
        "BOX1": {
            "ipaddr_wss": "10.1.0.58",
            "ipaddr_sss": "10.2.0.58",
            "ipaddr_css": "10.5.0.58",
            "boxtype": Quel1BoxType.QuEL1_TypeA,
            "config_root": None,
            "config_options": [],
        },
        "BOX2": {
            "ipaddr_wss": "10.1.0.60",
            "ipaddr_sss": "10.2.0.60",
            "ipaddr_css": "10.5.0.60",
            "boxtype": Quel1BoxType.QuEL1_TypeB,
            "config_root": None,
            "config_options": [],
        },
        "CAPTURER": {
            "box": "BOX0",
            "group": 0,
            "rline": "r",
            "background_noise_threshold": 200.0,
        },
        "SENDER0": {
            "box": "BOX0",
            "group": 1,
            "line": 2,
            "vatt": 0xA00,
            "wave_samples": 64,
            "num_repeats": (1, 1),
            "num_wait_words": (0, 0),
        },
    }

    boxpool = BoxPool(DEVICE_SETTINGS)
    boxpool.init(resync=False)
    pgs = init_pulsegen(DEVICE_SETTINGS, COMMON_SETTINGS, boxpool)
    cp = init_pulsecap(DEVICE_SETTINGS, COMMON_SETTINGS, boxpool)

    cp.check_noise(show_graph=False)
    boxpool.measure_timediff(cp)
