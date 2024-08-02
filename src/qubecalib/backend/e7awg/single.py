from __future__ import annotations

import copy
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from ipaddress import IPv4Address, IPv6Address
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from e7awgsw import (
    AWG,
    AwgCtrl,
    CaptureCtrl,
    CaptureModule,
    CaptureParam,
    CaptureUnit,
    DspUnit,
    IqWave,
    WaveSequence,
)
from e7awgsw.memorymap import CaptureCtrlRegs, CaptureParamRegs

# Readout や Monitor などの構成に関する知識はライブラリには含まれない

LOGGER: Final[logging.Logger] = logging.getLogger(__name__)

DEFAULT_NUM_WORKERS: Final[int] = 4
DEFAULT_CAPTURE_TIMEOUT: Final[float] = 60.0  # 60 s
SAMPLING_PERIOD: Final[float] = 2.0  # 2 ns


def get_logger() -> logging.Logger:
    return LOGGER


# from quelware
class CaptureReturnCode(Enum):
    CAPTURE_TIMEOUT = 1
    CAPTURE_ERROR = 2
    BROKEN_DATA = 3
    SUCCESS = 4


@dataclass
class Result:
    status: CaptureReturnCode
    data: list[npt.NDArray[np.complex64]] | list[npt.NDArray[np.int16]]


@dataclass
class Results:
    status: CaptureReturnCode
    results: Final[dict[CaptureUnit, Result]] = field(default_factory=dict)


@dataclass
class E7Setting:
    pass


@dataclass
class AwgSetting(E7Setting):
    awg: AWG
    wseq: WaveSequence


@dataclass
class CapuSetting(E7Setting):
    capu: CaptureUnit
    cprm: CaptureParam


@dataclass
class TriggerSetting(E7Setting):
    capm: CaptureModule
    awg: AWG


class Driver:
    def __init__(
        self,
        ipaddr: IPv4Address | IPv6Address,
        settings: list[E7Setting],
    ) -> None:
        self._ipaddr = str(ipaddr)
        self._awgctrl = AwgCtrl(self._ipaddr)
        self._capctrl = CaptureCtrl(self._ipaddr)
        self._awgctrl_lock = threading.Lock()
        self._capctrl_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=DEFAULT_NUM_WORKERS)
        self._settings = settings

        self._capus: Final[list[CaptureUnit]] = [
            s.capu for s in settings if isinstance(s, CapuSetting)
        ]
        self._awgs: Final[list[AWG]] = [
            s.awg for s in settings if isinstance(s, AwgSetting)
        ]
        self._capus_with_triggers: Final[list[CaptureUnit]] = []
        self._cprms_by_capus: Final[dict[CaptureUnit, CaptureParam]] = {
            s.capu: s.cprm for s in settings if isinstance(s, CapuSetting)
        }
        self._initialize()
        self._load(settings)

    @property
    def awgctrl(self) -> AwgCtrl:
        return self._awgctrl

    @property
    def capctrl(self) -> CaptureCtrl:
        return self._capctrl

    @property
    def awgctrl_lock(self) -> threading.Lock:
        return self._awgctrl_lock

    @property
    def capctrl_lock(self) -> threading.Lock:
        return self._capctrl_lock

    @property
    def awgs(self) -> list[AWG]:
        return self._awgs

    def _initialize(self) -> None:
        with self._capctrl_lock:
            self._capctrl.initialize(*self._capus)
        with self._awgctrl_lock:
            self._awgctrl.initialize(*self._awgs)
            self._awgctrl.terminate_awgs(*self._awgs)
            self._awgctrl.clear_awg_stop_flags(*self._awgs)

    def _load(self, settings: list[E7Setting]) -> None:
        for setting in settings:
            if isinstance(setting, AwgSetting):
                with self._awgctrl_lock:
                    self._awgctrl.set_wave_sequence(setting.awg, setting.wseq)
            elif isinstance(setting, CapuSetting):
                with self._capctrl_lock:
                    self._capctrl.set_capture_params(setting.capu, setting.cprm)
            elif isinstance(setting, TriggerSetting):
                pass
            else:
                raise ValueError(f"Unknown setting type: {setting}")
        for setting in self._settings:
            if isinstance(setting, TriggerSetting):
                with self._capctrl_lock:
                    self._capctrl.select_trigger_awg(setting.capm, setting.awg)
                for u in CaptureModule.get_units(setting.capm):
                    if u in self._capus:
                        self._capus_with_triggers.append(u)

    def start(self, timeout: float = DEFAULT_CAPTURE_TIMEOUT) -> Results:
        with self._capctrl_lock:
            self.capture_at()  # or self.capture_now() for individual capture
            future = self.wait_until_capture_finishes(timeout)
            self.emit_now()
            result = future.result()
        return result

    def abort(self) -> None:  # for capture
        # TODO implement
        pass

    def stop(self, awgs: list[AWG] = []) -> None:  # for awg
        if not awgs:
            awgs = self._awgs
        with self._awgctrl_lock:
            self._awgctrl.terminate_awgs(*awgs)
            self._awgctrl.clear_awg_stop_flags(*awgs)

    def capture_at(self) -> None:
        """Caputre data when trigger is received. Locked capctrl is assumed."""
        # Capture data when trigger is received
        self._capctrl.enable_start_trigger(
            *[u for u in self._capus if u in self._capus_with_triggers]
        )
        # Capture data immediately for units without trigger
        capus = [u for u in self._capus if u not in self._capus_with_triggers]
        self._capctrl.start_capture_units(*capus)
        LOGGER.debug(
            f"Capture units {', '.join([str(u) for u in capus])} started immediately"
        )

    def capture_now(self) -> None:
        """Capture data immediately without waiting for the trigger. Unlocked capctrl is required."""
        with self._capctrl_lock:
            self._capctrl.start_capture_units(self._capus)

    def wait_until_capture_finishes(
        self, timeout: float = DEFAULT_CAPTURE_TIMEOUT
    ) -> Future[Results]:
        return self._executor.submit(
            self._capture_thread,
            timeout,
        )

    def emit_now(self) -> None:
        with self._awgctrl_lock:
            self._awgctrl.start_awgs(*self._awgs)

    # from quelware
    def _check_capture_units_done(
        self,
    ) -> bool:
        for capu in self._capus:
            val = self._capctrl._CaptureCtrl__reg_access.read_bits(
                CaptureCtrlRegs.Addr.capture(capu),
                CaptureCtrlRegs.Offset.STATUS,
                CaptureCtrlRegs.Bit.STATUS_DONE,
                1,
            )
            if val == 0:
                return False
        else:
            return True

    # from quelware
    def _clear_capture_unit_done(
        self,
    ) -> None:
        for capu in self._capus:
            for v in (0, 1):
                self._capctrl._CaptureCtrl__reg_access.write_bits(
                    CaptureCtrlRegs.Addr.capture(capu),
                    CaptureCtrlRegs.Offset.CTRL,
                    CaptureCtrlRegs.Bit.CTRL_DONE_CLR,
                    1,
                    v,
                )

    # from quelware
    def _wait_for_capture_data(
        self,
        timeout: float = DEFAULT_CAPTURE_TIMEOUT,
    ) -> bool:
        # waiting for the completion of captureing
        polling_interval = min(max(0.01, timeout / 1000.0), 0.1)
        t0: float = time.perf_counter()
        completed: bool = False
        while time.perf_counter() - t0 < timeout:
            time.sleep(polling_interval)
            if self._check_capture_units_done():
                completed = True
                break
        if completed:
            self._clear_capture_unit_done()
        else:
            # TODO: investigate the reason this happens even when `timeout` is large enough
            LOGGER.warning(
                f"timeout happens at capture units {', '.join([str(x) for x in self._capus])}, capture aborted"
            )
        return completed

    # from quelware
    def _check_capture_error(
        self,
    ) -> bool:
        errdict: dict[int, list[Any]] = self._capctrl.check_err(*self._capus)
        errflag = False
        for capu, errlist in errdict.items():
            for err in errlist:
                LOGGER.warning(f"capture unit {self._capus[capu]}: {err}")
                errflag = True
        return errflag

    # from quelware
    def _capture_thread(
        self,
        timeout: float = DEFAULT_CAPTURE_TIMEOUT,
    ) -> Results:
        ready: bool = self._wait_for_capture_data(timeout)
        if not ready:
            return Results(status=CaptureReturnCode.CAPTURE_TIMEOUT)
        if self._check_capture_error():
            return Results(status=CaptureReturnCode.CAPTURE_ERROR)
        results = Results(status=CaptureReturnCode.SUCCESS)
        for capu in self._capus:
            results.results[capu] = self._retrieve_capture_data(capu)
        return results

    def _retrieve_capture_data(
        self,
        capu: CaptureUnit,
    ) -> Result:
        status: CaptureReturnCode = CaptureReturnCode.SUCCESS
        cprm = self._cprms_by_capus[capu]
        n_sample_expected = cprm.calc_capture_samples()
        n_sample_captured = self._capctrl.num_captured_samples(capu)
        if n_sample_captured == n_sample_expected:
            LOGGER.debug(
                f"the capture unit {self._ipaddr}:{capu} captured {n_sample_captured} samples"
            )
        else:
            # TODO: investigate the reason this happens
            LOGGER.warning(
                "the capture unit {self._ipaddr}:{capu} captured {n_sample_captured} samples, "
                "should be {n_sample_expected} samples"
            )
            status = CaptureReturnCode.BROKEN_DATA

        if DspUnit.CLASSIFICATION in cprm.dsp_units_enabled:
            d = np.array(
                list(
                    self._capctrl.get_classification_results(
                        capu,
                        n_sample_captured,
                    )
                ),
                dtype=np.int16,
            )
        else:
            c = np.array(
                self._capctrl.get_capture_data(
                    capu,
                    n_sample_captured,
                ),
                dtype=np.float32,
            )
        d = c[:, 0] + c[:, 1] * 1j

        if DspUnit.INTEGRATION in cprm.dsp_units_enabled:
            d = d.reshape(1, -1)
        else:
            d = d.reshape(cprm.num_integ_sections, -1)

        if DspUnit.SUM in cprm.dsp_units_enabled:
            e = np.hsplit(d, list(range(len(cprm.sum_section_list))[1:]))
        else:
            f = np.hsplit(
                d,
                np.cumsum(
                    np.array(
                        [
                            w
                            if DspUnit.DECIMATION not in cprm.dsp_units_enabled
                            else int(w / 4)
                            for w, _ in cprm.sum_section_list[:-1]
                        ]
                    )
                )
                * cprm.NUM_SAMPLES_IN_ADC_WORD,
            )
            e = [o.transpose() for o in f]
        return Result(
            status=status,
            data=e,
        )

    # TODO: direct access to the register will be implemented in the future
    def _get_dsp_units_enabled(self, capu: CaptureUnit) -> list[DspUnit]:
        """Locked capctrl is assumed."""
        base_addr = CaptureParamRegs.Addr.capture(capu)
        result = self._capctrl._CaptureCtrl__reg_access.read(
            base_addr, CaptureParamRegs.Offset.DSP_MODULE_ENABLE
        )
        return result

    def clear_before_starting_emission(self) -> None:
        """Locked awgctrl is assumed."""
        self._awgctrl.clear_awg_stop_flags(*self._awgs)


class WaveSequenceMultiplexer:
    def __init__(self, *wave_sequences: list[WaveSequence]) -> None:
        """多重化する WaveSequence を複製して保存する"""
        self._wave_sequences = [copy.deepcopy(wseq) for wseq in wave_sequences]


class WaveSequenceModifier:
    SAMPLEs = SAMPLING_PERIOD * 1e-9  # s
    WORDs = 4 * SAMPLEs  # s
    BLOCKs = 16 * WORDs  # s

    def __init__(self, wave_sequence: WaveSequence) -> None:
        """WaveSequence 複製を保存する"""
        self._wave_sequence = copy.deepcopy(wave_sequence)

    def finish(self) -> WaveSequence:
        return self._wave_sequence

    def modulated(self, frequency_in_MHz: float) -> WaveSequenceModifier:
        timings = self._create_timing()
        phase = [
            np.exp(2j * np.pi * frequency_in_MHz * 1e6 * timing) for timing in timings
        ]
        chunk_list = self._wave_sequence.chunk_list
        for index, (c, p) in enumerate(zip(chunk_list, phase)):
            w = np.array(c.wave_data.samples)
            iq = (w[:, 0] + 1j * w[:, 1]) * p
            s = IqWave.convert_to_iq_format(
                np.real(iq).astype(int),
                np.imag(iq).astype(int),
                WaveSequence.NUM_SAMPLES_IN_AWG_WORD,
            )
            self._replace_chunk(
                index,
                s,
                chunk_list[index].num_blank_words,
                chunk_list[index].num_repeats,
            )
        return self

    # def reset(
    # #     self,
    # #     timings: list[float],
    # #     base_frequency_in_MHz: list[npt.NDArray[np.float]],
    # # ) -> WaveSequenceModifier:
    # #     timings = self._create_timing()
    # #     return self
    # def _reset(self, timing_in_ns, frequency_in_GHz):
    #     phase = np.exp(-1j * 2 * np.pi * frequency_in_GHz * 1e9 * timing_in_ns)

    # def _shift(self, timing_in_ns: float, phase: float):
    #     """
    #     WaveChunk を変調する
    #     """
    #     phase = np.exp(2j * np.pi * frequency_in_MHz * 1e6 * timing_in_us)
    #     chunk_list = self._wave_sequence.chunk_list
    #     for index, chunk in enumerate(chunk_list):
    #         w = np.array(chunk.wave_data.samples)
    #         iq = (w[:, 0] + 1j * w[:, 1]) * phase
    #         s = IqWave.convert_to_iq_format(
    #             np.real(iq).astype(int),
    #             np.imag(iq).astype(int),
    #             WaveSequence.NUM_SAMPLES_IN_AWG_WORD,
    #         )
    #         self._replace_chunk(
    #             index,
    #             s,
    #             chunk.num_blank_words,
    #             chunk.num_repeats,
    #         )

    def _replace_chunk(
        self, index: int, iq_samples: int, num_blank_words: int, num_repeats: int
    ) -> None:
        """
        WaveChunk を置き換える
        """
        # 置き換え対象以降の WaveChunk のインデックス
        after_index = range(index, self._wave_sequence.num_chunks)
        # WaveChunk を退避
        saved_chunks = [
            {
                "iq_samples": chunk.wave_data.samples,
                "num_blank_words": chunk.num_blank_words,
                "num_repeats": chunk.num_repeats,
            }
            for chunk in self._wave_sequence.chunk_list
        ]
        # 置き換え対象以降の WaveChunk を削除
        for _ in after_index:
            self._wave_sequence.del_chunk(index)
        # 置き換え対象以降の WaveChunk を追加・復元
        for i in after_index:
            if i == index:
                self._wave_sequence.add_chunk(
                    iq_samples=iq_samples,
                    num_blank_words=num_blank_words,
                    num_repeats=num_repeats,
                )
            else:
                self._wave_sequence.add_chunk(**saved_chunks[i])

    def _create_timing(self) -> list[np.ndarray]:
        """
        WaveChunk を変調するためのタイミングを生成する
        """
        num_chunks = self._wave_sequence.num_chunks
        chunk_list = self._wave_sequence.chunk_list  # TODO ここで返ってくるのはコピーなので、元のWaveSequenceには影響しないことを確認する
        num_wave_words = [chunk.num_wave_words for chunk in chunk_list]
        num_blank_words = [chunk.num_blank_words for chunk in chunk_list]
        total_in_sec = [
            (wave + blank) * self.WORDs
            for wave, blank in zip(num_wave_words, num_blank_words)
        ]
        beginning_in_sec = [sum(total_in_sec[:index]) for index in range(num_chunks)]
        duration_in_samples = [wave * 4 for wave in num_wave_words]
        timings = [
            beginning + self.SAMPLEs * np.arange(duration)
            for beginning, duration in zip(beginning_in_sec, duration_in_samples)
        ]
        return timings


class CaptureParamModifier:
    def __init__(self, capture_param: CaptureParam) -> None:
        self._capture_param = copy.deepcopy(capture_param)
        self._saved_sum_start_word_no = capture_param.sum_start_word_no
        self._saved_num_words_to_sum = capture_param.num_words_to_sum
        self._saved_num_integ_sections = capture_param.num_integ_sections
        self._saved_complex_fir_coefs = capture_param.complex_fir_coefs
        self._saved_complex_window_coefs = capture_param.complex_window_coefs

    def finish(self) -> CaptureParam:
        return self._capture_param

    def modified_capture_delay(self, delay: int) -> CaptureParamModifier:
        self._capture_param.capture_delay = delay
        return self

    def enabled_sum_module(
        self, sum_start_word_no: int, num_words_to_sum: int
    ) -> CaptureParamModifier:
        self._enable_sum_module(sum_start_word_no, num_words_to_sum)
        return self

    def _enable_sum_module(self, sum_start_word_no: int, num_words_to_sum: int) -> None:
        self._enable_dspunits(DspUnit.SUM)
        capprm = self._capture_param
        self._saved_sum_start_word_no = capprm.sum_start_word_no
        self._saved_num_words_to_sum = capprm.num_words_to_sum
        capprm.sum_start_word_no = sum_start_word_no
        capprm.num_words_to_sum = num_words_to_sum

    def disabled_sum_module(self) -> CaptureParamModifier:
        self._disable_sum_module()
        return self

    def _disable_sum_module(self) -> None:
        self._disable_dspunits(DspUnit.SUM)
        self._capture_param.sum_start_word_no = self._saved_sum_start_word_no
        self._capture_param.num_words_to_sum = self._saved_num_words_to_sum

    def enabled_integration(self, num_integ_sections: int) -> CaptureParamModifier:
        self._enable_integration(num_integ_sections)
        return self

    def _enable_integration(self, num_integ_sections: int) -> None:
        self._enable_dspunits(DspUnit.INTEGRATION)
        self._saved_num_integ_sections = self._capture_param.num_integ_sections
        self._capture_param.num_integ_sections = num_integ_sections

    def disabled_integration(self) -> CaptureParamModifier:
        self._disable_integration()
        return self

    def _disable_integration(self) -> None:
        self._disable_dspunits(DspUnit.INTEGRATION)
        self._capture_param.num_integ_sections = self._saved_num_integ_sections

    def enabled_demodulation(self, frequency_in_MHz: float) -> CaptureParamModifier:
        self._enable_demodulation(frequency_in_MHz)
        return self

    def _enable_demodulation(self, frequency_in_MHz: float) -> None:
        self._enable_dspunits(DspUnit.COMPLEX_FIR, DspUnit.COMPLEX_WINDOW)
        self._enable_decimation()
        capprm = self._capture_param
        self._saved_complex_fir_coefs = capprm.complex_fir_coefs
        self._saved_complex_window_coefs = capprm.complex_window_coefs
        capprm.complex_fir_coefs = self._fir_coefficient(frequency_in_MHz)
        capprm.complex_window_coefs = self._window_coefficient(frequency_in_MHz)

    @classmethod
    def _fir_coefficient(cls, frequency_in_MHz: float) -> list[complex]:
        """
        Calculate FIR coefficients for a bandpass filter.

        Parameters
        ----------
        frequency_in_MHz : float
            Center frequency of the bandpass filter in MHz.

        Returns
        -------
        list[complex]
            FIR coefficients for the bandpass filter.
            Each part of a complex FIR coefficient must be an integer
            and in the range of [-2**15, 2**15 - 1].
        """
        N_COEFS = CaptureParam.NUM_COMPLEX_FIR_COEFS  # 16
        MAX_VAL = CaptureParam.MAX_FIR_COEF_VAL  # 32767
        t_ns = SAMPLING_PERIOD * np.arange(-N_COEFS + 1, 1)  # [-30, -28, ..., 0]

        # rect window
        # window_function = MAX_VAL * np.ones(N_COEFS).astype(complex)

        # gaussian window
        mu = (t_ns[-1] + t_ns[0]) / 2
        sigma = (t_ns[-1] - t_ns[0]) / 6
        window_function = MAX_VAL * np.exp(-0.5 * (t_ns - mu) ** 2 / (sigma**2))

        coefs = window_function * np.exp(
            1j * 2 * np.pi * frequency_in_MHz * 1e-3 * t_ns
        )
        result = coefs.round().tolist()
        return result

    @classmethod
    def _window_coefficient(cls, frequency_in_MHz: float) -> list[complex]:
        """
        Calculate window coefficients for a bandpass filter.

        Parameters
        ----------
        freuquency_in_MHz : float
            Center frequency of the bandpass filter in MHz.

        Returns
        -------
        list[complex]
            Window coefficients for the bandpass filter.
            Each part of a complex window coefficient must be an integer
            and in the range of [-2**31, 2**31 - 1].
        """
        N_DECIMATION = 4
        N_COEFS = CaptureParam.NUM_COMPLEXW_WINDOW_COEFS  # 2048
        MAX_VAL = CaptureParam.MAX_WINDOW_COEF_VAL  # 2147483647
        t_ns = N_DECIMATION * SAMPLING_PERIOD * np.arange(N_COEFS)  # [0, 8, ..., 16376]
        coefs = MAX_VAL * np.exp(-1j * 2 * np.pi * frequency_in_MHz * 1e-3 * t_ns)
        result = coefs.round().tolist()
        return result

    def disabled_demodulation(self) -> CaptureParamModifier:
        self._disable_dspunits(
            DspUnit.COMPLEX_FIR, DspUnit.DECIMATION, DspUnit.COMPLEX_WINDOW
        )
        self._capture_param.sum_start_word_no = self._saved_sum_start_word_no
        self._capture_param.num_words_to_sum = self._saved_num_words_to_sum
        return self

    def _disable_demodulation(self) -> None:
        self._disable_dspunits(DspUnit.COMPLEX_FIR, DspUnit.COMPLEX_WINDOW)
        self._disable_decimation()
        capprm = self._capture_param
        capprm.complex_fir_coefs = self._saved_complex_fir_coefs
        capprm.complex_window_coefs = self._saved_complex_window_coefs

    def enabled_decimation(self) -> CaptureParamModifier:
        self._enable_decimation()
        return self

    def _enable_decimation(self) -> None:
        self._enable_dspunits(DspUnit.DECIMATION)
        capprm = self._capture_param
        self._saved_sum_start_word_no = capprm.sum_start_word_no
        self._saved_num_words_to_sum = capprm.num_words_to_sum
        capprm.sum_start_word_no = self._saved_sum_start_word_no // 4
        capprm.num_words_to_sum = self._saved_num_words_to_sum // 4

    def disabled_decimation(self) -> CaptureParamModifier:
        self._disable_decimation()
        return self

    def _disable_decimation(self) -> None:
        self._disable_dspunits(DspUnit.DECIMATION)
        capprm = self._capture_param
        capprm.sum_start_word_no = self._saved_sum_start_word_no
        capprm.num_words_to_sum = self._saved_num_words_to_sum

    def _enable_dspunits(self, *dsp_units: DspUnit) -> CaptureParamModifier:
        enabled = self._capture_param.dsp_units_enabled
        for dsp_unit in dsp_units:
            if dsp_unit not in enabled:
                enabled.append(dsp_unit)
            else:
                raise ValueError(f"{dsp_unit} is already enabled")
        self._capture_param.sel_dsp_units_to_enable(*enabled)
        return self

    def _disable_dspunits(self, *dsp_units: DspUnit) -> CaptureParamModifier:
        enabled = self._capture_param.dsp_units_enabled
        for dsp_unit in dsp_units:
            if dsp_unit in enabled:
                enabled.remove(dsp_unit)
            else:
                raise ValueError(f"{dsp_unit} is not enabled")
        self._capture_param.sel_dsp_units_to_enable(*enabled)
        return self
