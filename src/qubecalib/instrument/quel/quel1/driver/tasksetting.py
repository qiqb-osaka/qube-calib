from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Final, Generator, NamedTuple, cast

import numpy as np
import numpy.typing as npt
from e7awgsw import CaptureParam, DecisionFunc, DspUnit, IqWave, WaveSequence
from quel_ic_config import Quel1PortType

FULL_SCALE = 2**15 - 1  # Full scale for complex64 in Quel1 AWG


def copy_capture_param(param: CaptureParam) -> CaptureParam:
    """CaptureParam の内容をすべて複製した新しいインスタンスを返す"""
    new_param = CaptureParam()
    new_param.num_integ_sections = param.num_integ_sections
    new_param.capture_delay = param.capture_delay
    new_param.sel_dsp_units_to_enable(*param.dsp_units_enabled)
    for length, blank in param.sum_section_list:
        new_param.add_sum_section(length, blank)
    new_param.sum_start_word_no = param.sum_start_word_no
    new_param.num_words_to_sum = param.num_words_to_sum
    new_param.complex_fir_coefs = param.complex_fir_coefs
    new_param.real_fir_i_coefs = param.real_fir_i_coefs
    new_param.real_fir_q_coefs = param.real_fir_q_coefs
    new_param.complex_window_coefs = param.complex_window_coefs
    for func in (0, 1):
        a, b, c = param.get_decision_func_params(func)
        new_param.set_decision_func_params(func, a, b, c)
    return new_param


class AwgId(NamedTuple):
    """Identifier for AWG channel on a specific port."""

    port: Quel1PortType
    channel: int


class AwgSetting(NamedTuple):
    """AWG configuration with associated WaveSequence."""

    awg: AwgId
    wseq: WaveSequence


class RunitId(NamedTuple):
    """Identifier for capture unit (CAPU) on a specific port."""

    port: int
    runit: int


class RunitSetting(NamedTuple):
    """CAPU configuration with associated CaptureParam."""

    runit: RunitId
    cprm: CaptureParam


class TriggerSetting(NamedTuple):
    """Trigger relationship between an AWG and a CAPU port (triggered port)."""

    trigger_awg: AwgId  # port, channel
    triggered_port: int


@dataclass(frozen=True)
class RawWaveformSet:
    port: int
    channel: int
    indexed_waveforms: list[list[tuple[int, npt.NDArray[np.complex64]]]]


@dataclass(frozen=True)
class TaskSetting:
    """Grouped configuration of waveform generators (AWG), capture units (CAPU), and their trigger relationships."""

    wseqs: dict[AwgId, WaveSequence]
    cprms: dict[RunitId, CaptureParam]
    triggers: dict[int, AwgId]
    #
    raw_waveforms: dict[AwgId, RawWaveformSet] = field(default_factory=dict)
    repetition_count: int | None = None
    coherent_integration_period: int | None = None
    runit_sample_indices: dict[RunitId, list[npt.NDArray[np.int32]]] = field(
        default_factory=dict
    )
    device_index_at_user_zero: dict[AwgId, int] = field(default_factory=dict)

    def get_wave_sequence(self, *, port: int, channel: int) -> WaveSequence:
        """
        Retrieve the WaveSequence for a specific AWG channel.

        Args:
            port (int): Port number of the AWG device.
            channel (int): Channel number within the AWG port.

        Returns:
            WaveSequence: The waveform sequence associated with the specified AWG channel.
        """
        awg_id = AwgId(port, channel)
        if awg_id not in self.wseqs:
            raise KeyError(f"WaveSequence for {awg_id} not found")
        return self.wseqs[awg_id]

    def get_capture_param(self, *, port: int, runit: int) -> CaptureParam:
        """
        Retrieve the CaptureParam for a specific CAPU unit.

        Args:
            port (int): CAPU device port number.
            runit (int): CAPU unit index on the specified port.

        Returns:
            CaptureParam: The capture parameters associated with the specified CAPU unit.
        """
        runit_id = RunitId(port, runit)
        if runit_id not in self.cprms:
            raise KeyError(f"CaptureParam for {runit_id} not found")
        return self.cprms[runit_id]

    def update_repetition_count(self, count: int) -> None:
        """
        Update the repetition count for all WaveSequences in the task setting.

        Args:
            count (int): New repetition count to set for all WaveSequences.
        """
        for wseq in self.wseqs.values():
            wseq.num_repeats = count
        for cprm in self.cprms.values():
            cprm.num_integ_sections = count

    def dsp_config(self, *, port: int, runit: int) -> Any:
        """
        Context manager interface:
        with DspConfigHelper.modify(capture_param) as dsp:
            dsp.enable_integration(...)
            ...
        """
        cprm = self.get_capture_param(port=port, runit=runit)
        return DspConfigHelper.modify(cprm)

    def ensure_wave_sequences_if_deferred(self) -> None:
        """
        Conditionally build WaveSequences from raw_waveforms only if wseqs are empty and raw_waveforms are present.
        This allows deferred construction of WaveSequences for modulation flexibility.
        """
        if self.raw_waveforms:
            self.ensure_wave_sequences(
                repetition_count=self.repetition_count or 1,
                integration_period=self.coherent_integration_period or 10240,
            )

    def ensure_wave_sequences(
        self, *, repetition_count: int, integration_period: int
    ) -> None:
        self.wseqs.clear() if self.wseqs else None
        from .tasksetting import TaskSettingBuilder

        builder = TaskSettingBuilder(
            coherent_integration_period=integration_period,
            repetition_count=repetition_count,
        )
        for awg, raw in self.raw_waveforms.items():
            builder.add_waveforms(
                indexed_waveforms=raw.indexed_waveforms,
                port=raw.port,
                channel=raw.channel,
            )
        # Since wseqs is frozen, but we want to update it, use object.__setattr__.
        object.__setattr__(self, "wseqs", builder.build().wseqs)


class RawBuilder(ABC):
    @abstractmethod
    def add_awg_setting(
        self,
        *,
        wseq: WaveSequence,
        port: int,
        channel: int,
    ) -> None: ...
    @abstractmethod
    def add_runit_setting(
        self,
        *,
        cprm: CaptureParam,
        port: int,
        runit: int,
        trigger_port: int | None = None,
        trigger_channel: int | None = None,
    ) -> None: ...


class RawTaskSettingBuilder(RawBuilder):
    """
    Helper class to incrementally construct a DeviceTask by specifying AWG, CAPU, and trigger settings.
    Once configured, it can produce a DeviceTask via `build()`.
    This builder produces raw task settings for a single box.
    """

    def __init__(self) -> None:
        self._settings: list[RunitSetting | AwgSetting | TriggerSetting] = []
        self._runit_sample_indices: dict[RunitId, list[npt.NDArray[np.int32]]] = {}
        self._device_index_at_user_zero: dict[AwgId, int] = {}

    def build(
        self,
        *,
        raw_waveforms: dict[AwgId, RawWaveformSet] = {},
        repetition_count: int | None = None,
        coherent_integration_period: int | None = None,
    ) -> TaskSetting:
        wseqs, cprms, triggers = self._parse_settings(self._settings)
        if wseqs:
            return TaskSetting(
                wseqs=wseqs,
                cprms=cprms,
                triggers=triggers,
                runit_sample_indices=self._runit_sample_indices,
                device_index_at_user_zero=self._device_index_at_user_zero,
            )
        else:
            return TaskSetting(
                wseqs=wseqs,
                cprms=cprms,
                triggers=triggers,
                raw_waveforms=raw_waveforms,
                repetition_count=repetition_count,
                coherent_integration_period=coherent_integration_period,
                runit_sample_indices=self._runit_sample_indices,
                device_index_at_user_zero=self._device_index_at_user_zero,
            )

    def add_awg_setting(
        self,
        *,
        wseq: WaveSequence,
        port: int,
        channel: int,
    ) -> None:
        """
        Add a waveform generation (AWG) configuration.

        Args:
            port (int): Port number of the AWG device.
            channel (int): Channel number within the AWG port.
            wseq (WaveSequence): The waveform sequence to configure.

        Returns:
            None
        """
        self._settings.append(AwgSetting(AwgId(port, channel), wseq))

    def add_runit_setting(
        self,
        *,
        cprm: CaptureParam,
        port: int,
        runit: int,
        trigger_port: int | None = None,
        trigger_channel: int | None = None,
    ) -> None:
        """
        Add a capture unit (CAPU (runit)) configuration with optional trigger source.

        Args:
            port (int): CAPU (runit) device port number.
            runit (int): CAPU (runit) unit index on the specified port.
            cprm (CaptureParam): Capture configuration parameters.
            trigger_port (int, optional): Port number of the triggering AWG.
            trigger_channel (int, optional): Channel of the triggering AWG.

        Raises:
            ValueError: If only one of `trigger_port` or `trigger_channel` is specified.

        Returns:
            None
        """
        # Enforce atomicity: both trigger port and channel must be provided together.
        if (trigger_port is None) != (trigger_channel is None):
            raise ValueError(
                "Both trigger port and channel must be provided or neither."
            )
        self._settings.append(RunitSetting(RunitId(port, runit), cprm))
        if (trigger_port is not None) and (trigger_channel is not None):
            self._settings.append(
                TriggerSetting(
                    trigger_awg=AwgId(trigger_port, trigger_channel),
                    triggered_port=port,
                )
            )

    @staticmethod
    def _parse_settings(
        settings: list[RunitSetting | AwgSetting | TriggerSetting],
    ) -> tuple[
        dict[AwgId, WaveSequence],
        dict[RunitId, CaptureParam],
        dict[int, AwgId],
    ]:
        """
        Parse a list of settings into grouped AWG, CAPU, and trigger dictionaries.

        Args:
            settings (list): List of RunitSetting, AwgSetting, or TriggerSetting objects.

        Raises:
            ValueError: If settings are empty or inconsistent trigger mappings.

        Returns:
            TaskSetting: Grouped settings suitable for DeviceTask configuration.
        """
        # Validity matrix:
        # (wseqs, cprms, triggers) => Valid or Error
        # (False, False, False) -> Error: no settings
        # (True,  False, False) -> OK: AWG only
        # (False, True,  False) -> OK: Capture only
        # (True,  True,  False) -> Error: AWG and CAPU without trigger
        # (False, False, True)  -> Error: Trigger without AWG or CAPU
        # (True,  False, True)  -> Error: Trigger without CAPU
        # (False, True,  True)  -> Error: Trigger without AWG
        # (True,  True,  True)  -> OK: Triggered capture
        if not settings:
            raise ValueError("no settings provided")
        wseqs, cprms, triggers = {}, {}, {}
        for setting in settings:
            if isinstance(setting, AwgSetting):
                wseqs[setting.awg] = setting.wseq
            elif isinstance(setting, RunitSetting):
                cprms[setting.runit] = setting.cprm
            elif isinstance(setting, TriggerSetting):
                triggers[setting.triggered_port] = setting.trigger_awg
            else:
                raise ValueError(f"unsupported setting: {setting}")
        if all([bool(wseqs), bool(cprms), not bool(triggers)]):
            raise ValueError("both wseqs and cprms are provided without triggers")
        if triggers:
            cap_ports = {runit.port for runit in cprms}
            for port in triggers:
                if port not in cap_ports:
                    raise ValueError(
                        f"triggered port {port} is not provided in runit settings"
                    )
            awgs = set(wseqs.keys())
            for port, awg in triggers.items():
                if awg not in awgs:
                    raise ValueError(
                        f"trigger {awg} for triggered port {port} is not provided"
                    )
        return wseqs, cprms, triggers


@dataclass(frozen=True)
class Waveform:
    start: int
    waveform: npt.NDArray[np.complex64]


@dataclass(frozen=True)
class CaptureWindow:
    start: int
    duration: int


class WaveSequenceBuilder:
    """
    Helper class to construct WaveSequence objects from raw waveform definitions.
    """

    @staticmethod
    def construct_from_indexed_waveforms(
        *,
        indexed_waveforms: list[list[tuple[int, npt.NDArray[np.complex64]]]],
        wait_samples: int,
        repetition_count: int,
        coherent_integration_period: int,
        modulation_frequency: float = 0.0,
    ) -> tuple[WaveSequence, int]:
        wait_words = wait_samples // 4  # Convert samples to words
        wait_sample_offset = wait_samples % 4
        if not indexed_waveforms:
            raise ValueError("indexed_waveforms must not be empty")
        if wait_words > 2**32 - 1:
            raise ValueError(
                f"wait_words must be less than 4294967296, but got {wait_words}"
            )

        for chunk_index, chunk in enumerate(indexed_waveforms):
            WaveSequenceBuilder.validate_waveform_chunk(chunk_index, chunk)

        first_wait_words = wait_words
        last_tail_words = 0
        buffer: list[int | npt.NDArray[np.complex64]] = []

        offset = wait_sample_offset
        indexed_waveforms_with_skew = [
            [(begin_index + offset, waveform) for begin_index, waveform in chunk]
            for chunk in indexed_waveforms
        ]

        for chunk in indexed_waveforms_with_skew:
            head_index = chunk[0][0]
            head_words = head_index // 4  # Convert to words
            tail_index = chunk[-1][0]
            tail_waveform = chunk[-1][1]
            wait_words = head_words - last_tail_words
            if wait_words < 0:
                raise ValueError(
                    f"Overlap detected between chunks: the chunk starting at sample index {head_index - offset} "
                    f"must be placed at or after {last_tail_words * 4 - offset} to avoid overlap."
                )
            wave = np.zeros(
                ((tail_index + len(tail_waveform) - (last_tail_words * 4) + 63) // 64)
                * 64,
                dtype=np.complex64,
            )
            buffer.append(wait_words)
            buffer.append(wave)
            last_tail_words = wait_words + len(wave) // 4
            for begin_index, waveform in chunk:
                if not isinstance(begin_index, int):
                    raise TypeError(
                        f"index must be an int, but got {type(begin_index).__name__}"
                    )
                # Validation: waveform must be np.ndarray of dtype np.complex64
                if not isinstance(waveform, np.ndarray):
                    raise TypeError(
                        f"waveform must be a numpy array, but got {type(waveform).__name__}"
                    )
                if waveform.dtype != np.complex64:
                    raise ValueError(
                        f"waveform must be of type np.complex64, but got {waveform.dtype}"
                    )
                sample_indices = np.arange(
                    begin_index, begin_index + len(waveform), dtype=np.float32
                )
                index = begin_index - head_words * 4
                wave[index : index + len(waveform)] = (
                    FULL_SCALE
                    * waveform
                    * np.exp(2j * np.pi * modulation_frequency * sample_indices)
                )

        wseq = WaveSequence(
            num_wait_words=first_wait_words + buffer[0],
            num_repeats=repetition_count,
            enable_lib_log=False,
        )
        device_index_at_user_zero = wait_sample_offset - cast(int, buffer[0]) * 4

        for iq, num_blank_words in zip(buffer[1:-1:2], buffer[2::2]):
            iq = cast(npt.NDArray[np.complex64], iq)
            wseq.add_chunk(
                iq_samples=IqWave.convert_to_iq_format(
                    np.real(iq).astype(int).tolist(),
                    np.imag(iq).astype(int).tolist(),
                    WaveSequence.NUM_SAMPLES_IN_WAVE_BLOCK,
                ),
                num_blank_words=num_blank_words,
                num_repeats=1,
            )
        total_length_words = sum(
            [
                wait_words + len(cast(npt.NDArray, iq)) // 4
                for wait_words, iq in zip(buffer[0::2], buffer[1::2])
            ]
        )
        iq = cast(npt.NDArray[np.complex64], buffer[-1])
        wseq.add_chunk(
            iq_samples=IqWave.convert_to_iq_format(
                np.real(iq).astype(int).tolist(),
                np.imag(iq).astype(int).tolist(),
                WaveSequence.NUM_SAMPLES_IN_WAVE_BLOCK,
            ),
            num_blank_words=coherent_integration_period // 4
            - total_length_words
            + buffer[0],
            num_repeats=1,
        )
        return wseq, device_index_at_user_zero

    @staticmethod
    def validate_waveform_array(array: npt.NDArray[np.complex64]) -> None:
        if not isinstance(array, np.ndarray):
            raise TypeError(
                f"waveform must be a numpy array, but got {type(array).__name__}"
            )
        if array.dtype != np.complex64:
            raise ValueError(
                f"waveform must be of type np.complex64, but got {array.dtype}"
            )

    @classmethod
    def validate_waveform_chunk(
        cls, chunk_index: int, chunk: list[tuple[int, npt.NDArray[np.complex64]]]
    ) -> None:
        if not chunk:
            raise ValueError(f"indexed_waveforms[{chunk_index}] is empty")
        head = chunk[0]
        if not (isinstance(head, tuple) and len(head) == 2):
            raise TypeError(
                f"indexed_waveforms[{chunk_index}] must be a list of tuples, "
                f"but got {type(head).__name__}"
            )
        head_index = head[0]
        if not isinstance(head_index, int):
            raise TypeError(
                f"index must be an int, but got {type(head_index).__name__}"
            )
        tail = chunk[-1]
        if not (isinstance(tail, tuple) and len(tail) == 2):
            raise TypeError(
                f"indexed_waveforms[{chunk_index}] must be a list of tuples, "
                f"but got {type(tail).__name__}"
            )
        tail_index = tail[0]
        if not isinstance(tail_index, int):
            raise TypeError(
                f"index must be an int, but got {type(tail_index).__name__}"
            )
        cls.validate_waveform_array(tail[1])


class CaptureParamBuilder:
    """
    Helper class to construct CaptureParam objects from abstract user inputs.
    This class provides methods to set up capture parameters for a CAPU (runit).
    """

    @classmethod
    def construct_from_capture_windows(
        cls,
        *,
        capture_windows: list[CaptureWindow],
        repetition_count: int,
        coherent_integration_period: int,
        capture_delay_blocks: int = 0,
    ) -> tuple[CaptureParam, list[npt.NDArray[np.int32]]]:
        """
        Construct a CaptureParam from a list of CaptureWindow objects.

        Args:
            capture_windows (list[CaptureWindow]): List of capture windows as sample index pairs.
            repetition_count (int): Number of integration sections.
            coherent_integration_period (int): Coherent integration period in words.
            capture_delay_blocks (int, optional): Delay before capture in units of 16 ADC words (default 0).

        Returns:
            CaptureParam: Configured CaptureParam object.
        """
        if not capture_windows:
            raise ValueError("capture_windows must not be empty")

        capture_windows = [
            CaptureWindow(start=win.start, duration=win.duration)
            for win in capture_windows
        ]
        capture_windows = sorted(capture_windows, key=lambda w: w.start)

        # Use the first window as the base
        cprm = CaptureParam()
        first = capture_windows[0]
        unit = CaptureParam.NUM_SAMPLES_IN_ADC_WORD
        aligned_start_first = cls._align_capture_first_window_start(first.start)
        capture_delay_words = 16 * capture_delay_blocks + aligned_start_first // unit
        cprm.capture_delay = capture_delay_words
        cprm.num_integ_sections = repetition_count

        # Add remaining windows if any
        last = capture_windows[-1]
        aligned_end_last = cls._align_capture_window_end(last.start + last.duration)
        if aligned_end_last > coherent_integration_period:
            raise ValueError(
                "Capture windows exceed coherent integration period: "
                f"{aligned_end_last} > {coherent_integration_period}"
            )

        sample_indices = []
        next_starts = [win.start for win in capture_windows[1:]] + [
            coherent_integration_period + aligned_start_first
        ]
        for win, next_start in zip(capture_windows, next_starts):
            aligned_start = (
                cls._align_capture_window_start(win.start)
                if first.start != win.start
                else cls._align_capture_first_window_start(win.start)
            )
            print(aligned_start, win.start, next_start)
            aligned_end = cls._align_capture_window_end(win.start + win.duration)
            aligned_next_start = cls._align_capture_window_start(next_start)
            aligned_duration = aligned_end - aligned_start

            blank = aligned_next_start - aligned_end
            if blank < 0:
                raise ValueError(
                    f"Each capture window must be separated by at least 1 word (4 samples) of blank space. "
                    f"Overlap detected between windows: from {aligned_end} to {aligned_next_start} "
                    f"(original next start index: {next_start})."
                )
            elif blank < 4:
                raise ValueError(
                    f"Each capture window must be separated by at least 1 word (4 samples) of blank space. "
                    f"Only {blank // 4} words found between windows: from {aligned_end} to {aligned_next_start} "
                    f"(corresponding to original start index {next_start})."
                )
            cprm.add_sum_section(
                num_words=aligned_duration // 4,
                num_post_blank_words=blank // 4,
            )
            sample_indices.append(np.arange(aligned_duration) + aligned_start)

        return cprm, sample_indices

    @staticmethod
    def _align_capture_first_window_start(start: int) -> int:
        """
        Align the start of the first capture window to the hardware-specific requirement.
        For Quel1 CAPU, must be 16-word (64-sample) aligned.
        """
        if start % 64 != 0:
            return (start // 64) * 64
        else:
            return start

    @staticmethod
    def _align_capture_window_start(start: int) -> int:
        """
        Align the start of subsequent capture windows to 1-word (4-sample) boundary.
        """
        return (start // 4) * 4

    @staticmethod
    def _align_capture_window_end(end: int) -> int:
        """
        Align the end of a capture window upward to the nearest 1-word (4-sample) boundary.
        """
        return ((end + 3) // 4) * 4


class TaskSettingBuilder:
    """
    High-level builder for TaskSetting from abstract user inputs.
    This class converts user-friendly readout/capture configurations
    into AWG/CAPU settings and trigger mappings.
    """

    def __init__(
        self,
        # context: DeviceContext,
        coherent_integration_period: int = 10240,  # word count
        repetition_count: int = 1,
    ) -> None:
        # Validation: coherent_integration_period must be a multiple of 16 words
        if coherent_integration_period % 16 != 0:
            raise ValueError(
                f"coherent_integration_period must be a multiple of 16 words, "
                f"but got {coherent_integration_period}"
            )
        self._builder = RawTaskSettingBuilder()
        self._coherent_integration_period: Final[int] = coherent_integration_period
        self._repetition_count: int = repetition_count
        self._awg_settings: list[AwgSetting] = []
        self._capu_settings: list[RunitSetting] = []
        self._trigger_settings: list[TriggerSetting] = []
        self._raw_waveforms: dict[AwgId, RawWaveformSet] = {}
        self._awg_wait_samples: dict[AwgId, int] = {}

    @property
    def coherent_integration_period(self) -> int:
        """
        Get the coherent integration period in words.

        Returns:
            int: Coherent integration period in words.
        """
        return self._coherent_integration_period

    def build(self) -> TaskSetting:
        return self._builder.build()

    def add_waveforms(
        self,
        *,
        indexed_waveforms: list[list[tuple[int, npt.NDArray[np.complex64]]]],
        port: int,
        channel: int,
        wait_samples: int = 0,  # in samples, default 0
        modulation_frequency: float = 0.0,
    ) -> None:
        """
        Add an AWG (arbitrary waveform generator) setting using abstract parameters.

        Args:
            port (int): AWG device port number.
            channel (int): Channel number within the AWG port.
            indexed_waveforms (list[list[tuple[int, npt.NDArray[np.complex64]]]]): A list of up to 16 waveform chunks,
                each chunk being a list of tuples containing start index in samples and complex waveform data.
            wait_words (int, optional): Number of words to wait before starting the waveform (default 0).
        """
        self._awg_wait_samples[AwgId(port, channel)] = wait_samples
        wseq, device_index_at_user_zero = (
            WaveSequenceBuilder.construct_from_indexed_waveforms(
                indexed_waveforms=indexed_waveforms,
                wait_samples=wait_samples,
                repetition_count=self._repetition_count,
                coherent_integration_period=self._coherent_integration_period,
                modulation_frequency=modulation_frequency,
            )
        )
        self._builder._device_index_at_user_zero[AwgId(port, channel)] = (
            device_index_at_user_zero
        )
        self._builder.add_awg_setting(
            port=port,
            channel=channel,
            wseq=wseq,
        )

    def add_capture_windows(
        self,
        *,
        capture_windows: list[CaptureWindow],
        port: int,
        runit: int,
        trigger_port: int | None = None,
        trigger_channel: int | None = None,
        capture_delay_blocks: int = 0,
    ) -> None:
        """
        Add a CAPU (runit) setting using abstract parameters.

        Args:
            port (int): CAPU (runit) device port number.
            runit (int): CAPU (runit) unit index on the specified port.
            capture_windows (list[CaptureWindow]): List of capture windows as sample index pairs.
            sample_rate (float, optional): Override sampling rate [Hz].
            trigger_port (int, optional): Port number of the triggering AWG.
            trigger_channel (int, optional): Channel of the triggering AWG.
            capture_delay_blocks (int, optional): Delay before capture in units of 16 ADC words (default 0).
        """

        cprm, sample_indices = CaptureParamBuilder.construct_from_capture_windows(
            capture_windows=capture_windows,
            repetition_count=self._repetition_count,
            coherent_integration_period=self._coherent_integration_period,
            capture_delay_blocks=capture_delay_blocks,
        )
        self._builder._runit_sample_indices[RunitId(port, runit)] = sample_indices

        self._builder.add_runit_setting(
            port=port,
            runit=runit,
            cprm=cprm,
            trigger_port=trigger_port,
            trigger_channel=trigger_channel,
        )


# --- DspConfigHelper helper ---
class DspConfigHelper:
    """
    A helper class to configure DSP-related settings on a CaptureParam instance.
    Intended for use with `with` block to modify the target instance in place.
    """

    def __init__(self, target: CaptureParam) -> None:
        self._target = target

    @staticmethod
    @contextmanager
    def modify(cprm: CaptureParam) -> Generator["DspConfigHelper", Any, None]:
        """
        Context manager interface:
        with DspConfigHelper.modify(capture_param) as dsp:
            dsp.enable_integration(...)
            ...
        """
        yield DspConfigHelper(cprm)

    def set_capture_delay(self, delay: int) -> None:
        self._target.capture_delay = delay

    def set_num_integ_sections(self, count: int) -> None:
        self._target.num_integ_sections = count

    def enable_integration(self) -> None:
        self._enable_dsp_option(DspUnit.INTEGRATION)

    def disable_integration(self) -> None:
        self._disable_dsp_option(DspUnit.INTEGRATION)

    def enable_sum(self) -> None:
        self._enable_dsp_option(DspUnit.SUM)

    def disable_sum(self) -> None:
        self._disable_dsp_option(DspUnit.SUM)

    def enable_demodulation(self, frequency: float) -> None:
        self.enable_band_pass_fir(center_frequency=frequency)
        factor = self.enable_decimation()
        self.enable_software_downconverter(
            frequency=frequency,
            decimation_factor=factor,
        )

    def disable_demodulation(self) -> None:
        """
        Disable demodulation DSP units.
        This will disable band-pass FIR, decimation, and software downconverter.
        """
        self.disable_band_pass_fir()
        self.disable_decimation()
        self.disable_software_downconverter()

    def enable_decimation(self) -> int:
        """
        Enable decimation DSP unit.
        """
        self._enable_dsp_option(DspUnit.DECIMATION)
        return 4

    def disable_decimation(self) -> None:
        """
        Disable decimation DSP unit.
        """
        self._disable_dsp_option(DspUnit.DECIMATION)

    def enable_classification(
        self,
        *,
        func_sel: DecisionFunc,
        coef_a: np.float32,
        coef_b: np.float32,
        const_c: np.float32,
    ) -> None:
        self._enable_dsp_option(DspUnit.CLASSIFICATION)
        self._target.set_decision_func_params(
            func_sel=func_sel,
            coef_a=coef_a,
            coef_b=coef_b,
            const_c=const_c,
        )

    def disable_classification(self) -> None:
        self._disable_dsp_option(DspUnit.CLASSIFICATION)
        self._target.set_decision_func_params(
            func_sel=DecisionFunc.U0,
            coef_a=0.0,
            coef_b=0.0,
            const_c=0.0,
        )

    def enable_band_pass_fir(self, center_frequency: float) -> None:
        """
        Enable band-pass FIR filter with given center frequency.

        Args:
            center_frequency (float): Center frequency of the band-pass filter.
        """
        coefs = FirCoefficients.band_pass(
            center_frequency=center_frequency,
            num_fir_taps=CaptureParam.NUM_COMPLEX_FIR_COEFS,
        )
        self.enable_complex_fir(coefs)

    def disable_band_pass_fir(self) -> None:
        """
        Disable band-pass FIR filter.
        """
        self.disable_complex_fir()

    def enable_complex_fir(self, coefs: npt.NDArray[np.complex64]) -> None:
        """
        Enable complex FIR filter with given coefficients.

        Args:
            coefs (npt.NDArray[np.complex64]): Coefficients for the complex FIR filter.
        """
        self._enable_dsp_option(DspUnit.COMPLEX_FIR)
        self._target.complex_fir_coefs = (FULL_SCALE * coefs).round().tolist()

    def disable_complex_fir(self) -> None:
        """
        Disable complex FIR filter.
        """
        self._disable_dsp_option(DspUnit.COMPLEX_FIR)
        zeros = np.zeros(
            CaptureParam.NUM_COMPLEX_FIR_COEFS,
            dtype=np.complex64,
        )
        self._target.complex_fir_coefs = zeros.round().tolist()

    def enable_complex_window(self, coefs: npt.NDArray[np.complex64]) -> None:
        self._enable_dsp_option(DspUnit.COMPLEX_WINDOW)
        self._target.complex_window_coefs = (FULL_SCALE * coefs).round().tolist()

    def disable_complex_window(self) -> None:
        self._disable_dsp_option(DspUnit.COMPLEX_WINDOW)
        zeros = np.zeros(
            CaptureParam.NUM_COMPLEXW_WINDOW_COEFS,
            dtype=np.complex64,
        )
        self._target.complex_window_coefs = zeros.round().tolist()

    def enable_software_downconverter(
        self, *, frequency: float, decimation_factor: int
    ) -> None:
        self._enable_dsp_option(DspUnit.COMPLEX_WINDOW)
        N_COEFS = CaptureParam.NUM_COMPLEXW_WINDOW_COEFS
        MAX_VAL = CaptureParam.MAX_WINDOW_COEF_VAL
        indices = decimation_factor * np.arange(N_COEFS)
        coefs = MAX_VAL * np.exp(-2j * np.pi * frequency * indices)
        self._target.complex_window_coefs = coefs.round().tolist()

    def disable_software_downconverter(self) -> None:
        """
        Disable software downconverter.
        This will disable the complex window DSP unit.
        """
        self._disable_dsp_option(DspUnit.COMPLEX_WINDOW)
        zeros = np.zeros(
            CaptureParam.NUM_COMPLEXW_WINDOW_COEFS,
            dtype=np.complex64,
        )
        self._target.complex_window_coefs = zeros.round().tolist()

    def _enable_dsp_option(self, option: DspUnit) -> None:
        dspunits = self._target.dsp_units_enabled
        if option not in dspunits:
            dspunits.append(option)
            self._target.sel_dsp_units_to_enable(*dspunits)

    def _disable_dsp_option(self, option: DspUnit) -> None:
        dspunits = self._target.dsp_units_enabled
        if option in dspunits:
            dspunits.remove(option)
            self._target.sel_dsp_units_to_enable(*dspunits)


class FirCoefficients:
    """
    Helper class to generate FIR coefficients for various filter types.
    """

    @staticmethod
    def band_pass(
        *,
        center_frequency: float,
        # bandwidth: float,
        num_fir_taps: int,
    ) -> npt.NDArray[np.complex64]:
        """
        Generate band-pass filter coefficients.

        Args:
            center_frequency (float) : Center frequency, defined as the inverse of the period (period is measured in sample_indices units).
            period_in_samples (int): Period of the filter in samples.
            center_frequency (float): Center frequency of the band-pass filter.
            bandwidth (float): Bandwidth of the band-pass filter.

        Returns:
            list[complex]: List of complex coefficients for the band-pass filter.
        """

        indices = np.arange(num_fir_taps)
        mu = (indices[-1] + indices[0]) / 2
        sigma = (indices[-1] - indices[0]) / 6
        gaussian_window = np.exp(-0.5 * (indices - mu) ** 2 / sigma**2)
        coefs = gaussian_window * np.exp(2j * np.pi * center_frequency * indices)
        return coefs


# class FirCoefficients:
#     """
#     Helper class to generate FIR coefficients for various filter types.
#     """

#     @staticmethod
#     def gaussian(period_in_samples: int) -> list[complex]:
#         """
#         Generate band-pass filter coefficients.

#         Args:
#             period_in_samples (int): Period of the filter in samples.

#         Returns:
#             list[complex]: List of complex coefficients for the band-pass filter.
#         """

#         return [
#             complex(np.sin(2 * np.pi * i / period_in_samples), 0)
#             for i in range(period_in_samples)
#         ]


class WindowCoefficients:
    """
    Helper class to generate window coefficients for various window types.
    """

    @staticmethod
    def gaussian(length: int, mu: float, sigma: float) -> list[complex]:
        """
        Generate Gaussian window coefficients.

        Args:
            length (int): Length of the window in samples.
            mu (float): Mean of the Gaussian distribution.
            sigma (float): Standard deviation of the Gaussian distribution.

        Returns:
            list[complex]: List of complex coefficients for the Gaussian window.
        """
        return [
            complex(np.exp(-((i - mu) ** 2) / (2 * sigma**2)), 0) for i in range(length)
        ]


#     @staticmethod
#     def hamming(length: int) -> list[complex]:
#         """
#         Generate Hamming window coefficients.

#         Args:
#             length (int): Length of the window in samples.

#         Returns:
#             list[complex]: List of complex coefficients for the Hamming window.
#         """
#         return [
#             complex(0.54 - 0.46 * np.cos(2 * np.pi * i / (length - 1)), 0)
#             for i in range(length)
#         ]
