from __future__ import annotations

import logging
from concurrent.futures import Future
from typing import Dict, MutableSequence, Optional, Tuple

import numpy as np
import numpy.typing as npt
from e7awgsw import CaptureParam, DspUnit, WaveSequence
from quel_ic_config import CaptureReturnCode, Quel1WaveSubsystem

logger = logging.getLogger(__name__)


class Quel1WaveSubsystemMod:
    @classmethod
    def set_wave(cls, wss: Quel1WaveSubsystem, awg: int, wave: WaveSequence) -> None:
        wss.validate_installed_e7awgsw()
        wss._validate_awg_hwidxs({awg})

        with wss._awgctrl_lock:
            wss._awgctrl.terminate_awgs(awg)  # to override current task of unit
            # TODO: should wait for confirminig the termination (?)
            wss._awgctrl.set_wave_sequence(awg, wave)

    @classmethod
    def _retrieve_capture_data(
        cls,
        wss: Quel1WaveSubsystem,
        cuhwxs_capums: Dict[int, Tuple[int, int]],
        cuhwxs_capprms: Dict[int, CaptureParam],
    ) -> Tuple[
        CaptureReturnCode,
        Dict[Tuple[int, int], MutableSequence[npt.NDArray[np.complex64 | np.int16]]],
    ]:
        data: Dict[
            Tuple[int, int], MutableSequence[npt.NDArray[np.complex64 | np.int16]]
        ] = {}
        cuhwx__num_expected_words = {
            cuhwx: capprm.calc_capture_samples()
            / (4 if DspUnit.DECIMATION in capprm.dsp_units_enabled else 1)
            for cuhwx, capprm in cuhwxs_capprms.items()
        }
        status: CaptureReturnCode = CaptureReturnCode.SUCCESS
        with wss._capctrl_lock:
            for cuhwx, capmu in cuhwxs_capums.items():
                n_sample_captured = wss._capctrl.num_captured_samples(cuhwx)
                n_sample_expected = cuhwx__num_expected_words[cuhwx]
                if n_sample_captured == n_sample_expected:
                    logger.info(
                        f"the capture unit {wss._wss_addr}:{capmu} captured {n_sample_captured} samples"
                    )
                else:
                    # TODO: investigate the reason this happens
                    logger.warning(
                        f"the capture unit {wss._wss_addr}:{capmu} captured {n_sample_captured} samples, "
                        f"should be {n_sample_expected} samples"
                    )
                    status = CaptureReturnCode.BROKEN_DATA

                capprm = cuhwxs_capprms[cuhwx]
                if DspUnit.CLASSIFICATION in capprm.dsp_units_enabled:
                    d = np.array(
                        list(
                            wss._capctrl.get_classification_results(
                                cuhwx,
                                n_sample_captured,
                            )
                        ),
                        dtype=np.int16,
                    )
                else:
                    c = np.array(
                        wss._capctrl.get_capture_data(
                            cuhwx,
                            n_sample_captured,
                        ),
                        dtype=np.float32,
                    )
                    d = c[:, 0] + c[:, 1] * 1j

                if DspUnit.INTEGRATION in capprm.dsp_units_enabled:
                    _d = d.reshape(1, -1)
                else:
                    _d = d.reshape(capprm.num_integ_sections, -1)

                if DspUnit.SUM in capprm.dsp_units_enabled:
                    __d = np.hsplit(_d, list(range(len(capprm.sum_section_list))[1:]))
                else:
                    _c = np.hsplit(
                        _d,
                        np.cumsum(
                            np.array([w for w, _ in capprm.sum_section_list[:-1]])
                        )
                        * capprm.NUM_SAMPLES_IN_ADC_WORD,
                    )
                    __d = [di.transpose() for di in _c]

                data[capmu] = __d
        return status, data

    @classmethod
    def _simple_capture_thread_main(
        cls,
        wss: Quel1WaveSubsystem,
        cuhwxs_capmus: Dict[int, Tuple[int, int]],
        cuhwxs_capprms: Dict[int, CaptureParam],
        timeout: float = Quel1WaveSubsystem.DEFAULT_CAPTURE_TIMEOUT,
    ) -> Tuple[
        CaptureReturnCode,
        Dict[int, MutableSequence[npt.NDArray[np.complex64 | np.int16]]],
    ]:
        ready: bool = wss._wait_for_capture_data(cuhwxs_capmus, timeout)
        if not ready:
            return CaptureReturnCode.CAPTURE_TIMEOUT, {}
        if wss._check_capture_error(cuhwxs_capmus):
            return CaptureReturnCode.CAPTURE_ERROR, {}

        retcode, iqs = cls._retrieve_capture_data(
            wss,
            cuhwxs_capmus,
            cuhwxs_capprms,
        )
        return retcode, {capu: iq for (_, capu), iq in iqs.items()}

    @classmethod
    def simple_capture_start(
        cls,
        wss: Quel1WaveSubsystem,
        capmod: int,
        capunits_capprms: Dict[int, CaptureParam],
        *,
        delay: Optional[int] = None,
        triggering_awg: Optional[int] = None,
        timeout: float = Quel1WaveSubsystem.DEFAULT_CAPTURE_TIMEOUT,
    ) -> Future[Tuple[CaptureReturnCode, Dict[int, npt.NDArray[np.complex64]]]]:
        # capmod に所属する capunits 達の測定を開始する
        wss.validate_installed_e7awgsw()

        wss._validate_awg_hwidxs({capmod})
        cuhwxs_capmus = wss._get_capunit_hwidxs(
            [(capmod, capunit) for capunit in capunits_capprms]
        )  # capunit hwidxs -> cuhwxs?
        cuhwxs_capprms = {
            cuhwx: capunits_capprms[capunit]
            for cuhwx, (_, capunit) in cuhwxs_capmus.items()
        }
        if triggering_awg is not None:
            wss._validate_awg_hwidxs({triggering_awg})

        cls._setup_capture_units(
            wss,
            capmod,
            cuhwxs_capprms,
            triggering_awg,
        )
        # num_expected_words は Dsp から自動計算するように変更

        return wss._executor.submit(
            cls._simple_capture_thread_main,
            wss,
            cuhwxs_capmus,
            cuhwxs_capprms,
            timeout,
        )

    @classmethod
    def _setup_capture_units(
        cls,
        wss: Quel1WaveSubsystem,
        capmod: int,
        cuhwxs_capprms: Dict[int, CaptureParam],
        triggering_awg: Optional[int] = None,
    ) -> None:
        # capctrl を初期化し capture_param を cuhwx に設定する
        # triggering_awg が指定されていれば trigger 待機に，いなければ即時キャプチャーを開始する
        # cuhwxs = [_ for _ in cuhwxs_capprms]
        # with wss._capctrl_lock:
        #     wss._capctrl.initialize(*cuhwxs)
        #     # TODO: unit 毎に別の cap_prms を割り当てられるようにした
        #     for cuhwx, capprm in cuhwxs_capprms.items():
        #         wss._capctrl.set_capture_params(cuhwx, capprm)
        cls._setup_capture_units_first_half(wss=wss, cuhwxs_capprms=cuhwxs_capprms)
        # TODO ここで止められる様な手段を設けるべしと三好さんからのアドバイス
        cls._setup_capture_units_second_half(
            wss=wss,
            capmod=capmod,
            cuhwxs_capprms=cuhwxs_capprms,
            triggering_awg=triggering_awg,
        )
        #     # TODO: it looks better to dump status of capture units for debug.
        #     if triggering_awg is None:
        #         wss._capctrl.start_capture_units(*cuhwxs)
        #     else:
        #         logger.info(
        #             f"capture units {', '.join([str(cuhwxs) for cuhwxs in cuhwxs])} "
        #             f"will be triggered by awg {triggering_awg}"
        #         )
        #         wss._capctrl.select_trigger_awg(capmod, triggering_awg)
        #         wss._capctrl.enable_start_trigger(*cuhwxs)

    @classmethod
    def _setup_capture_units_first_half(
        cls,
        wss: Quel1WaveSubsystem,
        cuhwxs_capprms: Dict[int, CaptureParam],
    ) -> None:
        cuhwxs = [_ for _ in cuhwxs_capprms]
        with wss._capctrl_lock:
            wss._capctrl.initialize(*cuhwxs)
            # TODO: unit 毎に別の cap_prms を割り当てられるようにした
            for cuhwx, capprm in cuhwxs_capprms.items():
                wss._capctrl.set_capture_params(cuhwx, capprm)

    @classmethod
    def _setup_capture_units_second_half(
        cls,
        wss: Quel1WaveSubsystem,
        capmod: int,
        cuhwxs_capprms: Dict[int, CaptureParam],
        triggering_awg: Optional[int] = None,
    ) -> None:
        cuhwxs = [_ for _ in cuhwxs_capprms]
        with wss._capctrl_lock:
            # TODO: it looks better to dump status of capture units for debug.
            if triggering_awg is None:
                wss._capctrl.start_capture_units(*cuhwxs)
            else:
                logger.info(
                    f"capture units {', '.join([str(cuhwxs) for cuhwxs in cuhwxs])} "
                    f"will be triggered by awg {triggering_awg}"
                )
                wss._capctrl.select_trigger_awg(capmod, triggering_awg)
                wss._capctrl.enable_start_trigger(*cuhwxs)
