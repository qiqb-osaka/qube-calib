from __future__ import annotations

import logging
from concurrent.futures import Future
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import numpy.typing as npt
from e7awgsw import CaptureParam, WaveSequence
from quel_clock_master import SequencerClient
from quel_ic_config_utils import CaptureReturnCode, SimpleBoxIntrinsic

from qubecalib.temp.general_looptest_common import PulseCap, PulseGen

from .qcbox import QcBox, QcBoxFactory
from .temp.general_looptest_common import BoxPool
from .temp.quel1_wave_subsystem_mod import Quel1WaveSubsystemTools

logger = logging.getLogger(__name__)

# ipaddr の部分は実験環境によって異なる．例えば，阪大，八王子，理研など．config ファイルに入れるのだろうけど，指定方法はどうしよう．
DEVICE_SETTINGS: dict[str, Mapping[str, Any]] = {
    "CLOCK_MASTER": {
        "ipaddr": "10.3.0.255",
        "reset": True,
    },
}


class QcBoxPool(BoxPool):
    def _parse_settings(self, settings: Mapping[str, Mapping[str, Any]]) -> None:
        for k, v in settings.items():
            if k.startswith("BOX"):
                kwd = k[3:]
                qcbox = v["qcbox"]
                self._boxes[kwd] = (
                    qcbox,
                    SequencerClient(str(qcbox.ipaddr_sss)),
                )
                self._linkstatus[kwd] = False

    # 実験環境では複数のユーザがシステムを使うため resync は False をメインとする．
    # box.init() は他の実験環境を壊さないという理解
    def init(self, resync: bool = False) -> None:
        for name, (qcbox, sqc) in self._boxes.items():
            box = qcbox.box._dev
            link_status: bool = True
            if not all(box.init().values()):
                if all(
                    box.init(ignore_crc_error_of_mxfe=box.css.get_all_groups()).values()
                ):
                    logger.warning(
                        f"datalink between MxFE and FPGA of {name} is not working"
                    )
                else:
                    logger.error(
                        f"datalink between MxFE and FPGA of {name} is not working"
                    )
                    link_status = False
            self._linkstatus[name] = link_status

        # 使用する awg だけリセットしたい
        # 出力の直前で実施するためここでは Disable する
        # self.reset_awg()
        # resync したい機体だけを resync できるようにするのは別途したい
        # if resync:
        #     self.resync()
        # 出力の直前で実施するためここでは Disable する
        # if not self.check_clock():
        #     raise RuntimeError("failed to acquire time count from some clocks")


class QcBoxPoolTools:
    @classmethod
    def emit_at(
        cls,
        qcboxpool: QcBoxPool,
        cps: Set[QcPulseCap],
        pgs: Set[QcPulseGen],
        min_time_offset: int,
        time_counts: Sequence[int],
        displacement: int = 0,
    ) -> Dict[str, List[int]]:
        if len(pgs) == 0:
            logger.warning("no pulse generator to activate")

        pg_by_box: Dict[str, Set[QcPulseGen]] = {
            boxname: set() for boxname in qcboxpool._boxes
        }
        for pg in pgs:
            pg_by_box[pg.boxname].add(pg)
        removing_kws = {boxname for boxname, _ in pg_by_box.items() if len(_) == 0}
        for boxname in removing_kws:
            del pg_by_box[boxname]

        for cp in cps:
            if cp.boxname not in pg_by_box:
                raise RuntimeError("impossible to trigger the capturer")

        # initialize awgs
        targets: Dict[str, SequencerClient] = {}
        for boxname, pgs in pg_by_box.items():
            box, sqc = qcboxpool._boxes[boxname]
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
                    f"sysref offset: {last_sysref_time % qcboxpool.SYSREF_PERIOD}"
                )
            else:
                raise RuntimeError("failed to read current clock")

        for cp in cps:
            valid_read, current_time, last_sysref_time = targets[
                cp.boxname
            ].read_clock()
            logger.info(
                f"sysref offset of {cp.boxname}: average: {qcboxpool._cap_sysref_time_offset},  latest: {last_sysref_time % qcboxpool.SYSREF_PERIOD}"
            )
            if (
                abs(
                    last_sysref_time % qcboxpool.SYSREF_PERIOD
                    - qcboxpool._cap_sysref_time_offset
                )
                > 4
            ):
                logger.warning("large fluctuation of sysref is detected on the FPGA")
            base_time = current_time + min_time_offset
            offset = (16 - (base_time - qcboxpool._cap_sysref_time_offset) % 16) % 16
            base_time += offset
            base_time += displacement  # inducing clock displacement for performance evaluation (must be 0 usually).
            base_time += qcboxpool.TIMING_OFFSET  # Notes: the safest timing to issue trigger, at the middle of two AWG block.

        schedule: Dict[str, List[int]] = {boxname: [] for boxname in targets}
        for i, time_count in enumerate(time_counts):
            for boxname, sqc in targets.items():
                t = base_time + time_count + qcboxpool._estimated_timediff[boxname]
                valid_sched = sqc.add_sequencer(t)
                if not valid_sched:
                    raise RuntimeError("failed to schedule AWG start")
                schedule[boxname].append(t)
        logger.info("scheduling completed")
        return schedule


class QcPulseGen(PulseGen):
    # emit_now で 開始
    # emit_at で kick 待ち
    # stop_now で 停止
    # init_wave で 波形設定
    # init_config で ポートなどを設定
    def __init__(
        self,
        name: str,
        boxname: str,
        port: int,
        channel: int,
        wave: WaveSequence,
        lo_freq: Optional[float],
        cnco_freq: Optional[float],
        boxpool: QcBoxPool,
    ):
        box_status, qcbox, sqc = boxpool.get_box(boxname)
        if not box_status:
            raise RuntimeError(
                f"sender '{name}' is not available due to link problem of '{boxname}'"
            )
        self.name = name
        self.boxname = boxname
        self.box: SimpleBoxIntrinsic = qcbox.box._dev
        self.sqc: SequencerClient = sqc
        group, line = qcbox.box._convert_all_port(port)
        if isinstance(line, str):
            raise ValueError(f"{self.__class__.__name__} only accept tx port")
        self.group = group
        self.line = line
        self.channel = channel
        self.wave = wave
        self.awg: int = self.box.rmap.get_awg_of_channel(
            self.group, self.line, self.channel
        )

    def init_config(self) -> None:
        # self.box.config_line(...)
        # self.box.config_channel(...)
        # TODO: 適正なスイッチコンフィグを考える
        # self.box.open_rfswitch(self.group, self.line)
        # if self.box.is_loopedback_monitor(self.group):
        #     self.box.open_rfswitch(self.group, "m")
        pass

    def init_wave(self) -> None:
        Quel1WaveSubsystemTools.set_wave(self.box.wss, self.awg, self.wave)


class QcPulseCap:
    # capture_now で 開始
    # capture_at_single_trigger_of で単体筐体？ wss.simple_capture_start
    # capture_at_multiple_triggers_of で複数筐体？ wss.capture_start
    # capunits の設定に変更が必要っぽい
    # wss.simple_capture_start で triggering_awg を None にすると即時実行
    def __init__(
        self,
        name: str,
        boxname: str,
        port: int,
        channel: int,
        captparam: CaptureParam,
        boxpool: QcBoxPool,
    ):
        box_status, qcbox, _ = boxpool.get_box(boxname)
        if not box_status:
            raise RuntimeError(
                f"sender '{name}' is not available due to link problem of '{boxname}'"
            )
        self.name = name
        self.boxname = boxname
        self.box = qcbox
        self.port = port
        self.channel = channel
        self.capprm = captparam
        self.boxpool = boxpool

        group, rline = qcbox.box._convert_all_port(port)
        if isinstance(rline, int):
            raise ValueError(f"{self.__class__.__name__} only accept rx port")
        self._pulsecap = PulseCap(
            name,
            boxname,
            group,
            rline,
            lo_freq=-1,
            cnco_freq=-1,
            fnco_freq=-1,
            background_noise_threshold=-1,
            boxpool=boxpool,
        )

    @property
    def capmod(self) -> int:
        return self._pulsecap.capmod

    def init(self) -> None:
        # self.box.config_rline(...)
        # self.box.config_runit(...)
        # self.box.open_rfswitch(group=self.group, line=self.rline)
        pass


class QcSystem:
    def __init__(self, *config_paths: str | PathLike):
        self._qcboxes = {
            Path(k).stem: QcBoxFactory.produce(Path(k)) for k in config_paths
        }
        # init 内容を微調整したいので BoxPool.init はここにオーバーライドした
        # SimpleBoxIntrinsic は init() 済みのすでに定義されたものを使いたいので BOX{i} 要素は空で渡して後で追加
        # ここでは path.stem を key とする
        for k, v in self._qcboxes.items():
            DEVICE_SETTINGS[f"BOX{k}"] = {"qcbox": v}
        bp = self._boxpool = QcBoxPool(DEVICE_SETTINGS)
        bp.init(resync=False)

    def resync_box(self, *qcbox_name: str) -> None:
        """resync したい機体だけを個別に resync する"""
        qcboxes = [self._qcboxes[k] for k in qcbox_name if k in self._qcboxes]
        self._boxpool._clock_master.kick_clock_synch(
            [str(qcbox.ipaddr_sss) for qcbox in qcboxes]
        )

    def show_clock_all(
        self,
    ) -> dict[str, tuple[bool, int, int] | tuple[bool, int]]:
        """登録されているすべての筐体の同期用FPGA内部クロックの現在値を表示する

        Returns
        -------
        tuple[tuple[bool, int], dict[str, tuple[bool, int, int]]]
            _description_
        """
        client_results = self.show_clock(*self._qcboxes)
        results: dict[str, tuple[bool, int, int] | tuple[bool, int]] = {
            k: v for k, v in client_results.items()
        }
        results["master"] = self._boxpool._clock_master.read_clock()
        return results

    def show_clock(self, *qcbox_name: str) -> dict[str, tuple[bool, int, int]]:
        """同期用FPGA内部クロックの現在値を表示する

        Returns
        -------
        dict[str, tuple[bool, int, int]]
            _description_
        """
        sqcs = {k: self._boxpool._boxes[k][1] for k in qcbox_name}
        return {k: sqcs[k].read_clock() for k in qcbox_name}

    @property
    def qcboxes(self) -> dict[str, QcBox]:
        return self._qcboxes

    @property
    def qcbox(self) -> dict[str, QcBox]:
        return self.qcboxes

    def init_pulsegen(
        self, setup: Dict[str, QcPulseGenSetting | QcPulseCapSetting]
    ) -> Set[QcPulseGen]:
        pgs = {
            QcPulseGen(name=k, boxpool=self._boxpool, **asdict(v))
            for k, v in setup.items()
            if isinstance(v, QcPulseGenSetting)
        }
        for _ in pgs:
            _.init()
            _.init_config()
            _.init_wave()
        return pgs

    def init_pulsecap(
        self, setup: Dict[str, QcPulseGenSetting | QcPulseCapSetting]
    ) -> Set[QcPulseCap]:
        pcs = {
            QcPulseCap(name=k, boxpool=self._boxpool, **asdict(v))
            for k, v in setup.items()
            if isinstance(v, QcPulseCapSetting)
        }
        for _ in pcs:
            _.init()
        return pcs

    def _capture_now(
        self, pcs: Set[QcPulseCap]
    ) -> Tuple[CaptureReturnCode, Dict[int, npt.NDArray[np.complex64]]]:
        capmods = {pc.capmod for pc in pcs}
        if len(capmods) != 1:
            raise ValueError("single capmod is estimated")
        wss = next(iter({pc.box.wss for pc in pcs}))
        capmod = next(iter(capmods))
        capu_capprm = {pc.channel: pc.capprm for pc in pcs}
        capunits = tuple(capu_capprm.keys())
        num_expected_words = {capu: 0 for capu in capu_capprm.keys()}
        thunk = Quel1WaveSubsystemTools.simple_capture_start(
            wss,
            capmod,
            capunits,
            capu_capprm,
            num_expected_words=num_expected_words,
            triggering_awg=None,
        )
        status, iqs = thunk.result()
        return status, iqs

    def capture_now(
        self, pcs: Set[QcPulseCap]
    ) -> Dict[
        Tuple[str, int], Tuple[CaptureReturnCode, Dict[int, npt.NDArray[np.complex64]]]
    ]:
        boxnames = {pc.boxname for pc in pcs}
        if len(boxnames) != 1:
            raise ValueError("single box is estimated")
        boxname = next(iter(boxnames))

        result = {}
        for capmod in {pc.capmod for pc in pcs}:
            port = next(iter({pc.port for pc in pcs if pc.capmod == capmod}))
            _pcs = {pc for pc in pcs if pc.capmod == capmod}
            _status, _iqs = self._capture_now(_pcs)
            id = (boxname, port)
            result[id] = (_status, _iqs)

        return result

    def _capture_at_trigger_of(
        self, pcs: Set[QcPulseCap], pg: QcPulseGen
    ) -> Future[Tuple[CaptureReturnCode, Dict[int, npt.NDArray[np.complex64]]]]:
        capmods = {pc.capmod for pc in pcs}
        if len(capmods) != 1:
            raise ValueError("single capmod is estimated")
        wss = next(iter({pc.box.wss for pc in pcs}))
        if pg.box.wss != wss:
            raise ValueError("can not be triggered by an awg of the other box")
        capmod = next(iter(capmods))
        capu_capprm = {pc.channel: pc.capprm for pc in pcs}
        capunits = tuple(capu_capprm.keys())
        num_expected_words = {capu: 0 for capu in capu_capprm.keys()}
        future = Quel1WaveSubsystemTools.simple_capture_start(
            wss,
            capmod,
            capunits,
            capu_capprm,
            num_expected_words=num_expected_words,
            triggering_awg=pg.awg,
        )
        return future

    def capture_at_trigger_of(
        self, pcs: Set[QcPulseCap], pg: QcPulseGen
    ) -> Dict[
        Tuple[str, int],
        Future[Tuple[CaptureReturnCode, Dict[int, npt.NDArray[np.complex64]]]],
    ]:
        boxnames = {pc.boxname for pc in pcs}
        if len(boxnames) != 1:
            raise ValueError("single box is estimated")
        boxname = next(iter(boxnames))

        result = {}
        for capmod in {pc.capmod for pc in pcs}:
            port = next(iter({pc.port for pc in pcs if pc.capmod == capmod}))
            _pcs = {pc for pc in pcs if pc.capmod == capmod}
            future = self._capture_at_trigger_of(_pcs, pg)
            id = (boxname, port)
            result[id] = future

        return result

    def _emit_now(self, pgs: Set[QcPulseGen]) -> None:
        """単体のボックスで複数のポートから QcPulseGen に従ったパルスを出射する．パルスは同期して出射される．

        Parameters
        ----------
        pgs : Set[QcPulseGen]
            _description_

        Raises
        ------
        RuntimeError
            _description_
        """
        if len(pgs) == 0:
            logger.warning("no pulse generator to activate")

        boxnames = set([pg.boxname for pg in pgs])
        if len(boxnames) != 1:
            raise RuntimeError(
                f"{self.__class__}._start_synchronized_emission_using_single_box() can use single box only"
            )

        for boxname in boxnames:
            self.qcbox[boxname].wss.start_emission([pg.awg for pg in pgs])

    def _emit_at(self, pgs: Set[QcPulseGen]) -> None:
        min_time_offset = 125_000_000
        time_counts = [i * 125_000_000 for i in range(1)]

        if len(pgs) == 0:
            logger.warning("no pulse generator to activate")

        boxnames = set([pg.boxname for pg in pgs])
        if len(boxnames) != 1:
            raise RuntimeError(
                f"{self.__class__}._start_synchronized_emission_using_single_box() can use single box only"
            )
        qcbox = self.qcbox[next(iter(boxnames))]
        awgs = {pg.awg for pg in pgs}
        sqc = next(iter({pg.sqc for pg in pgs}))
        qcbox.wss.clear_before_starting_emission(awgs)
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

    def _stop_now(self, pgs: Set[QcPulseGen]) -> None:
        """QcPulseGen で指定された awg のパルスを停止する．

        Parameters
        ----------
        pgs : Set[QcPulseGen]
            _description_

        Raises
        ------
        RuntimeError
            _description_
        """
        if len(pgs) == 0:
            logger.warning("no pulse generator to deactivate")
        if not all([isinstance(o, QcPulseGen) for o in pgs]):
            raise ValueError(
                f"{self.__class__.__name__}._stop_emission() only accept QcPulseGen objects."
            )

        for pg in set([_ for _ in pgs]):
            pg.stop_now()

    def execute(
        self, setup: Dict[str, QcPulseGenSetting | QcPulseCapSetting]
    ) -> (
        Tuple[
            Dict[Tuple[str, int, int], CaptureReturnCode],
            Dict[Tuple[str, int, int], npt.NDArray[np.complex64]],
        ]
        | Tuple[None, None]
    ):
        pgs = self.init_pulsegen(setup)
        pcs = self.init_pulsecap(setup)

        if not len(pgs) and not len(pcs):
            raise ValueError("no pulse setting")

        # 品質検査で重要だが，後で検討する．とりあえず無効化
        # cp.check_noise(show_graph=False)
        # boxpool.measure_timediff(cp)

        pgs_by_boxname = self.get_pgs_by_boxname(pgs)
        pcs_by_boxname = self.get_pcs_by_boxname(pcs)

        status, iqs = {}, {}
        future = {}
        for boxname, _pcs in pcs_by_boxname.items():
            _pgs = pgs_by_boxname[boxname] if boxname in pgs_by_boxname else set()
            if len(_pcs):
                if not len(_pgs):
                    results = self.capture_now(_pcs)
                    for (_boxname, _port), (_status, _iqs) in results.items():
                        __status = {
                            (_boxname, _port, _channel): _status for _channel in _iqs
                        }
                        __iqs = {
                            (_boxname, _port, _channel): _iqs[_channel]
                            for _channel in _iqs
                        }
                        status.update(__status)
                        iqs.update(__iqs)
                else:
                    triggering_pg = next(iter(_pgs))
                    _ = self.capture_at_trigger_of(_pcs, triggering_pg)
                    future.update(_)

        if len({boxname for boxname in pgs_by_boxname}) == 1:
            self._emit_now(pgs)
        else:
            for boxname, _pgs in pgs_by_boxname.items():
                self._emit_at(_pgs)

        if len(future):
            for (_boxname, _port), _future in future.items():
                _status, _iqs = _future.result()
                __status = {(_boxname, _port, _channel): _status for _channel in _iqs}
                __iqs = {
                    (_boxname, _port, _channel): _iqs[_channel] for _channel in _iqs
                }
                status.update(__status)
                iqs.update(__iqs)

        return status, iqs

    def get_pgs_by_boxname(self, pgs: Set[QcPulseGen]) -> Dict[str, Set[QcPulseGen]]:
        pgs_by_boxname: Dict[str, Set[QcPulseGen]] = {
            boxname: set() for boxname in self._boxpool._boxes
        }
        for pg in pgs:
            pgs_by_boxname[pg.boxname].add(pg)
        removing_kws = {boxname for boxname, _ in pgs_by_boxname.items() if len(_) == 0}
        for boxname in removing_kws:
            del pgs_by_boxname[boxname]
        return pgs_by_boxname

    def get_pcs_by_boxname(self, pcs: Set[QcPulseCap]) -> Dict[str, Set[QcPulseCap]]:
        pcs_by_boxname: Dict[str, Set[QcPulseCap]] = {
            boxname: set() for boxname in self._boxpool._boxes
        }
        for pc in pcs:
            pcs_by_boxname[pc.boxname].add(pc)
        removing_kws = {boxname for boxname, _ in pcs_by_boxname.items() if len(_) == 0}
        for boxname in removing_kws:
            del pcs_by_boxname[boxname]
        return pcs_by_boxname


@dataclass
class QcPulseGenSetting:
    boxname: str
    port: int
    channel: int
    wave: WaveSequence = None
    lo_freq: Optional[float] = None
    cnco_freq: Optional[float] = None


@dataclass
class QcPulseCapSetting:
    boxname: str
    port: int
    channel: int
    captparam: CaptureParam = None


@dataclass
class QcPulseReadloopSetting:
    pass


@dataclass
class QcPulsePumpSetting:
    pass
