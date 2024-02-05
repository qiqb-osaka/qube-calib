from __future__ import annotations

import logging
from os import PathLike
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
from e7awgsw import CaptureParam, DspUnit, WaveSequence
from quel_ic_config_utils import CaptureReturnCode, Quel1WaveSubsystem
from quel_ic_config_utils.e7resource_mapper import Quel1E7ResourceMapper

from .qcbox import (
    Channel,
    Dict,
    QcBoxFactory,
    QubeOuTypeAQcBox,
    QubeRikenTypeAQcBox,
    RxChannel,
    TxChannel,
)

logger = logging.getLogger(__name__)


class QcSystem:
    def __init__(
        self, *config_paths: str | PathLike
    ):  # TODO:生成時にできるチェックをここでする
        self._boxes = {
            Path(k).stem: QcBoxFactory(Path(k)).produce() for k in config_paths
        }

    @property
    def boxes(self) -> Dict[str, QubeOuTypeAQcBox | QubeRikenTypeAQcBox]:
        return self._boxes

    @property
    def box(self) -> Dict[str, QubeOuTypeAQcBox | QubeRikenTypeAQcBox]:
        return self.boxes


class QcWaveSubsystem:
    @classmethod
    def send_recv(
        cls,
        *setup: Sequence[Channel | WaveSequence | CaptureParam],
        triggering_channel: Sequence[Optional[TxChannel]] = [],
        delay: int = 1,
        timeout: int = 30,
    ) -> Dict | CaptureReturnCode:
        setup_for_send = [(k, v) for k, v in setup if isinstance(k, TxChannel)]
        setup_for_recv = [(k, v) for k, v in setup if isinstance(k, RxChannel)]
        wss_set = set([k[0].wss for k in setup])
        rmap_set = set([k[0].rmap for k in setup])
        if not triggering_channel:
            wss_trigch: dict[Quel1WaveSubsystem, TxChannel | None] = {
                k: None for k in wss_set
            }
        else:
            wss_trigch = {o.wss: o for o in triggering_channel if o is not None}
        if len(wss_set) == 1:
            wss = wss_set.pop()
            rmap = rmap_set.pop()
            return cls.standalone_send_recv(
                wss,
                rmap,
                setup_for_send,
                setup_for_recv,
                wss_trigch[wss],
                delay,
                timeout,
            )
        return {}

    @classmethod
    def _setup_capture_units(
        cls,
        wss: Quel1WaveSubsystem,
        cuhwxs: Dict[int, Tuple[int, int]],
        e7unit_e7param: Dict[int, CaptureParam],
        triggering_awg: int | None = None,
    ) -> None:
        with wss._capctrl_lock:
            wss._capctrl.initialize(*cuhwxs)
            # TODO: ここも unit 毎に別の cap_prms を割り当てられるようにした
            for c, p in e7unit_e7param.items():
                wss._capctrl.set_capture_params(c, p)
            # TODO: it looks better to dump status of capture units for debug.
            if triggering_awg is None:
                wss._capctrl.start_capture_units(*cuhwxs)
            else:
                logger.info(
                    f"capture units {', '.join([str(capmu) for capmu in cuhwxs.values()])} "
                    f"will be triggered by awg {triggering_awg}"
                )
                # TODO: 複数の cuhwxs に対応するのに，ここが単体の capmod 用になっていた
                #       上流で Fix されたら廃止する
                # ====
                for _, v in cuhwxs.items():
                    capmod = v[0]
                    wss._capctrl.select_trigger_awg(capmod, triggering_awg)
                # wss._capctrl.select_trigger_awg(capmod, triggering_awg)
                # ====
                wss._capctrl.enable_start_trigger(*cuhwxs)

    @classmethod
    def _get_capunit_hwidx(
        cls, wss: Quel1WaveSubsystem, capmod: str | Any, capunit: int
    ) -> int:
        muc = wss.get_muc_structure()
        return tuple(muc[capmod].keys())[capunit]

    @classmethod
    def _simple_capture_thread_main(
        cls,
        wss: Quel1WaveSubsystem,
        e7unit_e7param: Dict[int, CaptureParam],
        cuhwxs: Dict[int, Tuple[int, int]],
        timeout: float | None = None,
    ) -> tuple[CaptureReturnCode, dict[int, list[np.ndarray[Any, np.dtype[Any]]]]]:
        timeout = wss.DEFAULT_CAPTURE_TIMEOUT if not timeout else timeout
        ready: bool = wss._wait_for_capture_data(cuhwxs, timeout)
        if not ready:
            return CaptureReturnCode.CAPTURE_TIMEOUT
        if wss._check_capture_error(cuhwxs):
            return CaptureReturnCode.CAPTURE_ERROR
        retcode, data = cls._retrieve_capture_data(wss, e7unit_e7param)
        return retcode, data

    @classmethod
    def standalone_send_recv(
        cls,
        wss: Quel1WaveSubsystem,
        rmap: Quel1E7ResourceMapper,
        setup_for_send: Sequence[Sequence[TxChannel | WaveSequence]],
        setup_for_recv: Sequence[Sequence[RxChannel | CaptureParam]],
        triggering_channel: TxChannel | None = None,
        delay: int = 1,
        timeout: int = 30,
    ) -> tuple[
        CaptureReturnCode, dict[RxChannel, list[np.ndarray[Any, np.dtype[Any]]]]
    ]:
        tx_channels = [c for c, _ in setup_for_send]
        rx_channels = [c for c, _ in setup_for_recv]
        e7unit_e7param = {
            cls._get_capunit_hwidx(wss, c.capmod, c.capunit): p
            for c, p in setup_for_recv
        }
        e7awg_e7wave = {
            rmap.get_awg_of_channel(c.group, c.line, c.channel): p
            for c, p in setup_for_send
        }
        # channel から awg をどうやって持ってこようか
        if triggering_channel is None:
            triggering_awg = None
        else:
            c = triggering_channel
            triggering_awg = rmap.get_awg_of_channel(c.group, c.line, c.channel)
        if triggering_awg is not None:
            wss._validate_awg_hwidxs({triggering_awg})
        # cuhwxs に 複数の capmod がいてもいいように改修
        # {e7unit: (e7module, channel_id)}
        cuhwxs: Dict[int, Tuple[int, int]] = wss._get_capunit_hwidxs(
            [(rc.capmod, rc.capunit) for rc in rx_channels]
        )
        awgs = [
            rmap.get_awg_of_channel(c.group, c.line, c.channel) for c in tx_channels
        ]
        wss.initialize_awgs(awgs)
        with wss._awgctrl_lock:
            wss._awgctrl.terminate_awgs(*awgs)
            for a, w in e7awg_e7wave.items():
                wss._awgctrl.set_wave_sequence(a, w)
        # TODO: 読み出しサンプル数チェックの復活（各モードでの読み出し数の計算を精査）
        cls._setup_capture_units(wss, cuhwxs, e7unit_e7param, triggering_awg)
        wss.start_emission(awgs)
        future = wss._executor.submit(
            cls._simple_capture_thread_main,
            wss,
            e7unit_e7param,
            cuhwxs,
            timeout,
        )
        e7unit_qcchannel = {
            cls._get_capunit_hwidx(wss, c.capmod, c.capunit): c for c in rx_channels
        }
        status, data = future.result()
        data2 = {e7unit_qcchannel[k]: v for k, v in data.items()}
        return status, data2

    @classmethod
    def _retrieve_capture_data(
        cls,
        wss: Quel1WaveSubsystem,
        e7unit_e7param: Dict[int, CaptureParam],
    ) -> tuple[CaptureReturnCode, dict[int, list[np.ndarray[Any, np.dtype[Any]]]]]:
        status: CaptureReturnCode = CaptureReturnCode.SUCCESS
        with wss._capctrl_lock:
            data = {
                u: cls._retrieve_capture_data_single(wss, u, p)
                for u, p in e7unit_e7param.items()
            }

        return status, data

    @classmethod
    def _retrieve_capture_data_single(
        cls,
        wss: Quel1WaveSubsystem,
        e7unit: int,
        e7param: CaptureParam,
    ) -> list[np.ndarray[Any, np.dtype[Any]]]:
        n = wss._capctrl.num_captured_samples(e7unit)
        if DspUnit.CLASSIFICATION in e7param.dsp_units_enabled:  # ４値化
            d = np.array(list(wss._capctrl.get_classification_results(e7unit, n)))
        else:  # ４値化以外
            c = np.array(wss._capctrl.get_capture_data(e7unit, n))
            d = c[:, 0] + 1j * c[:, 1]
        # 取得したデータを整形 Dict[int, ...]
        # INTEGRATION が有効な場合積算区間が１つに潰れる
        if DspUnit.INTEGRATION in e7param.dsp_units_enabled:
            d1 = d.reshape(1, -1)
        else:
            d1 = d.reshape(e7param.num_integ_sections, -1)
        # SUM が有効な場合
        m = len(e7param.sum_section_list)
        if DspUnit.SUM in e7param.dsp_units_enabled:
            d2 = np.hsplit(d1, list(range(m)[1:]))
        else:
            d2x = np.hsplit(
                d1,
                np.cumsum(np.array([w for w, _ in e7param.sum_section_list[:-1]]))
                * e7param.NUM_SAMPLES_IN_ADC_WORD,
            )
            d2 = [di.transpose() for di in d2x]

        return d2
