from __future__ import annotations

import logging
from concurrent.futures import Future
from typing import Collection, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from e7awgsw import CaptureParam, WaveSequence
from quel_ic_config_utils import CaptureReturnCode, Quel1WaveSubsystem

logger = logging.getLogger(__name__)


class Quel1WaveSubsystemTools:
    # simple_capture_start は改修
    # capture_start は改修
    @classmethod
    def set_wave(cls, wss: Quel1WaveSubsystem, awg: int, wave: WaveSequence) -> None:
        wss.validate_installed_e7awgsw()
        wss._validate_awg_hwidxs({awg})

        with wss._awgctrl_lock:
            wss._awgctrl.terminate_awgs(awg)  # to override current task of unit
            # TODO: should wait for confirminig the termination (?)
            wss._awgctrl.set_wave_sequence(awg, wave)

    @classmethod
    def wave_gen(
        cls, wss: Quel1WaveSubsystem, wave_by_awg: Dict[int, WaveSequence]
    ) -> None:
        # Note: validation will be done in set_wave()

        for awg, wave in wave_by_awg.items():
            cls.set_wave(wss, awg, wave)
        wss.start_emission(list(wave_by_awg))

    @classmethod
    def _retrieve_capture_data(
        cls,
        wss: Quel1WaveSubsystem,
        cuhwxs: Dict[int, Tuple[int, int]],
        num_expected_words: Dict[int, int],
    ) -> Tuple[CaptureReturnCode, Dict[Tuple[int, int], npt.NDArray[np.complex64]]]:
        data: Dict[Tuple[int, int], npt.NDArray[np.complex64]] = {}
        status: CaptureReturnCode = CaptureReturnCode.SUCCESS
        with wss._capctrl_lock:
            for cuhwx, capunit in cuhwxs.items():
                n_sample_captured = wss._capctrl.num_captured_samples(cuhwx)
                # n_sample_expected = num_expected_words[cuhwx] * 4
                n_sample_expected = n_sample_captured
                if n_sample_captured == n_sample_expected:
                    logger.info(
                        f"the capture unit {wss._wss_addr}:{capunit} captured {n_sample_captured} samples"
                    )
                else:
                    # TODO: investigate the reason this happens
                    logger.warning(
                        f"the capture unit {wss._wss_addr}:{capunit} captured {n_sample_captured} samples, "
                        f"should be {n_sample_expected} samples"
                    )
                    status = CaptureReturnCode.BROKEN_DATA
                data_in_assq: List[Tuple[float, float]] = wss._capctrl.get_capture_data(
                    cuhwx, n_sample_captured
                )
                tmp = np.array(data_in_assq, dtype=np.float32)
                data[capunit] = tmp[:, 0] + tmp[:, 1] * 1j
        return status, data

    @classmethod
    def _simple_capture_thread_main(
        cls,
        wss: Quel1WaveSubsystem,
        cuhwxs: Dict[int, Tuple[int, int]],
        num_expected_words: Dict[int, int],
        timeout: float = Quel1WaveSubsystem.DEFAULT_CAPTURE_TIMEOUT,
    ) -> Tuple[CaptureReturnCode, Dict[int, npt.NDArray[np.complex64]]]:
        ready: bool = wss._wait_for_capture_data(cuhwxs, timeout)
        if not ready:
            return CaptureReturnCode.CAPTURE_TIMEOUT, {}
        if wss._check_capture_error(cuhwxs):
            return CaptureReturnCode.CAPTURE_ERROR, {}

        retcode, iqs = cls._retrieve_capture_data(wss, cuhwxs, num_expected_words)
        return retcode, {runit: iq for (_, runit), iq in iqs.items()}

    @classmethod
    def simple_capture_start(
        cls,
        wss: Quel1WaveSubsystem,
        capmod: int,
        capunits: Collection[int],
        capu_capprm: Dict[int, CaptureParam],
        num_expected_words: Dict[int, int],
        *,
        delay: Optional[int] = None,
        triggering_awg: Optional[int] = None,
        timeout: float = Quel1WaveSubsystem.DEFAULT_CAPTURE_TIMEOUT,
    ) -> Future[Tuple[CaptureReturnCode, Dict[int, npt.NDArray[np.complex64]]]]:
        wss.validate_installed_e7awgsw()

        wss._validate_awg_hwidxs({capmod})
        cuhwxs = wss._get_capunit_hwidxs([(capmod, capunit) for capunit in capunits])
        cuhwx_capprm = {
            cuhwx: capu_capprm[capunit] for cuhwx, (_, capunit) in cuhwxs.items()
        }
        if triggering_awg is not None:
            wss._validate_awg_hwidxs({triggering_awg})

        cls._setup_capture_units(wss, capmod, cuhwxs, cuhwx_capprm, triggering_awg)
        return wss._executor.submit(
            cls._simple_capture_thread_main, wss, cuhwxs, num_expected_words, timeout
        )

    @classmethod
    def _setup_capture_units(
        cls,
        wss: Quel1WaveSubsystem,
        capmod: int,
        cuhwxs: Dict[int, Tuple[int, int]],
        cuhwx_capprm: Dict[int, CaptureParam],
        triggering_awg: Optional[int] = None,
    ) -> None:
        with wss._capctrl_lock:
            wss._capctrl.initialize(*cuhwxs)
            # TODO: unit 毎に別の cap_prms を割り当てられるようにした
            for cuhwx in cuhwxs:
                wss._capctrl.set_capture_params(cuhwx, cuhwx_capprm[cuhwx])
            # TODO: it looks better to dump status of capture units for debug.
            if triggering_awg is None:
                wss._capctrl.start_capture_units(*cuhwxs)
            else:
                logger.info(
                    f"capture units {', '.join([str(capmu) for capmu in cuhwxs.values()])} "
                    f"will be triggered by awg {triggering_awg}"
                )
                wss._capctrl.select_trigger_awg(capmod, triggering_awg)
                wss._capctrl.enable_start_trigger(*cuhwxs)
