from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Final, NamedTuple, cast

import numpy as np
import numpy.typing as npt
from e7awgsw import CaptureParam, IqWave, WaveSequence
from quel_ic_config import Quel1PortType

FULL_SCALE = 2**15 - 1  # Full scale for complex64 in Quel1 AWG


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
class TaskSetting:
    """Grouped configuration of waveform generators (AWG), capture units (CAPU), and their trigger relationships."""

    wseqs: dict[AwgId, WaveSequence]
    cprms: dict[RunitId, CaptureParam]
    triggers: dict[int, AwgId]

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

    def build(self) -> TaskSetting:
        return self._parse_settings(self._settings)

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
        Add a capture unit (CAPU) configuration with optional trigger source.

        Args:
            port (int): CAPU device port number.
            runit (int): CAPU unit index on the specified port.
            cprm (CaptureParam): Capture configuration parameters.
            triggr_port (int, optional): Port number of the triggering AWG.
            triggr_channel (int, optional): Channel of the triggering AWG.

        Raises:
            ValueError: If only one of `triggr_port` or `triggr_channel` is specified.

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
    ) -> TaskSetting:
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
        return TaskSetting(wseqs, cprms, triggers)


@dataclass(frozen=True)
class Waveform:
    start: int
    waveform: npt.NDArray[np.complex64]


@dataclass(frozen=True)
class CaptureWindow:
    start: int
    duration: int


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
        self._builder = RawTaskSettingBuilder()
        # self._context = context
        self._coherent_integration_period: Final[int] = coherent_integration_period
        self._repetition_count: Final[int] = repetition_count
        self._awg_settings: list[AwgSetting] = []
        self._capu_settings: list[RunitSetting] = []
        self._trigger_settings: list[TriggerSetting] = []
        # Validation: coherent_integration_period must be a multiple of 16 words
        if self._coherent_integration_period % 16 != 0:
            raise ValueError(
                f"coherent_integration_period must be a multiple of 16 words, "
                f"but got {self._coherent_integration_period}"
            )

    def build(self) -> TaskSetting:
        return self._builder.build()

    def add_waveforms(
        self,
        *,
        indexed_waveforms: list[list[tuple[int, npt.NDArray[np.complex64]]]],
        port: int,
        channel: int,
        wait_words: int = 0,  # in words, default 0
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
        if not indexed_waveforms:
            raise ValueError("indexed_waveforms must not be empty")

        if wait_words > 2**32 - 1:
            raise ValueError(
                f"wait_words must be less than 4294967296, but got {wait_words}"
            )

        first_wait_words = wait_words
        last_tail_words = 0
        buffer: list[int | npt.NDArray[np.complex64]] = []
        for chunk_index, chunk in enumerate(indexed_waveforms):
            if not chunk:
                raise ValueError(f"indexed_waveforms[{chunk_index}] is empty")
            head = chunk[0]
            if not isinstance(head, tuple) and len(head) != 2:
                raise TypeError(
                    f"indexed_waveforms[{chunk_index}] must be a list of tuples, "
                    f"but got {type(head).__name__}"
                )
            head_index = head[0]
            if not isinstance(head_index, int):
                raise TypeError(
                    f"index must be an int, but got {type(head_index).__name__}"
                )
            head_words = head_index // 4  # Convert to words
            tail = chunk[-1]
            if not isinstance(tail, tuple) and len(tail) != 2:
                raise TypeError(
                    f"indexed_waveforms[{chunk_index}] must be a list of tuples, "
                    f"but got {type(tail).__name__}"
                )
            tail_index = tail[0]
            if not isinstance(tail_index, int):
                raise TypeError(
                    f"index must be an int, but got {type(tail_index).__name__}"
                )
            tail_waveform = tail[1]
            if not isinstance(tail_waveform, np.ndarray):
                raise TypeError(
                    f"waveform must be a numpy array, but got {type(tail_waveform).__name__}"
                )
            if tail_waveform.dtype != np.complex64:
                raise ValueError(
                    f"waveform must be of type np.complex64, but got {tail_waveform.dtype}"
                )
            wait_words = head_words - last_tail_words
            if wait_words < 0:
                raise ValueError(
                    f"Too close chunk: the chunk starting at sample index {head_index} "
                    f"must be placed at or after {last_tail_words * 4} to avoid overlap."
                )
            wave = np.zeros(
                ((tail_index + len(tail_waveform) - (head_words * 4) + 63) // 64) * 64,
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
                if not isinstance(waveform, np.ndarray):
                    raise TypeError(
                        f"waveform must be a numpy array, but got {type(waveform).__name__}"
                    )
                if waveform.dtype != np.complex64:
                    raise ValueError(
                        "waveform must be of type np.complex64, "
                        f"but got {waveform.dtype}"
                    )
                index = begin_index - head_words * 4
                wave[index : index + len(waveform)] = FULL_SCALE * waveform

            # w = indexed_waveforms
            # head_index = (w[0][0][0] // 64) * 64  # Align to 64-samples boundary
            # head_words = head_index // 4  # Convert to words
            # wseq = WaveSequence(
            #     num_wait_words=wait_words + head_words,
            #     num_repeats=self._repetition_count,
            #     enable_lib_log=False,
            # )
            # tail_index = (
            #     (w[-1][-1][0] + len(w[-1][-1][1]) + 63) // 64
            # ) * 64  # Align to 64-samples boundary
            # iq = np.zeros(tail_index - head_index, dtype=np.complex64)
            # tail_words = tail_index // 4  # Convert to words
            # for chunk_idx, chunk in enumerate(indexed_waveforms):
            #     # Write the waveform data into the iq array
            #     for index, waveform in chunk:
            #         begin_index = index - head_index
            #         end_index = begin_index + len(waveform)
            #         iq[begin_index:end_index] = waveform
            #     chunk_tail_idx = (
            #         (chunk[-1][0] + len(chunk[-1][1]) + 63) // 64 * 64
            #     )  # Align to 64-samples boundary
            #     if chunk_idx < len(indexed_waveforms) - 1:
            #         next_head_idx = (
            #             indexed_waveforms[chunk_idx + 1][0][0] // 64
            #         ) * 64  # Align to 64-samples boundary
            #         num_blank_words = (
            #             next_head_idx - chunk_tail_idx
            #         ) // 4  # Convert to words
            #     else:
            #         num_blank_words = self._coherent_integration_period - tail_words

        wseq = WaveSequence(
            num_wait_words=first_wait_words + buffer[0],
            num_repeats=self._repetition_count,
            enable_lib_log=False,
        )

        # assert buffer[0] == 0
        # wave = cast(npt.NDArray[np.complex64], buffer[1])
        # assert len(wave) == 64
        # assert (
        #     np.real(wave).tolist()
        #     == [FULL_SCALE] * 10 + [0] * 10 + [FULL_SCALE] * 10 + [0] * 34
        # )
        # assert np.imag(wave).tolist() == [0] * 64
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
            num_blank_words=self._coherent_integration_period - total_length_words,
            num_repeats=1,
        )
        self._builder.add_awg_setting(port=port, channel=channel, wseq=wseq)

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
        Add a CAPU (capture unit) setting using abstract parameters.

        Args:
            port (int): CAPU device port number.
            runit (int): CAPU unit index on the specified port.
            capture_windows (list[CaptureWindow]): List of capture windows as sample index pairs.
            sample_rate (float, optional): Override sampling rate [Hz].
            triggr_port (int, optional): Port number of the triggering AWG.
            triggr_channel (int, optional): Channel of the triggering AWG.
            capture_delay_blocks (int, optional): Delay before capture in units of 16 ADC words (default 0).
        """
        if not capture_windows:
            raise ValueError("capture_windows must not be empty")

        # Align start and duration to word alignment requirements
        capture_windows = sorted(capture_windows, key=lambda w: w.start)
        aligned_windows: list[CaptureWindow] = []
        for idx, win in enumerate(capture_windows):
            start = win.start
            end = win.start + win.duration
            if idx == 0:
                # First capture window: align start to 16-word boundary if needed
                if start % 16 != 0:
                    start_aligned = (start // 16) * 16 * 4
                else:
                    start_aligned = start
            else:
                # Subsequent windows: align start to 4-word boundary
                start_aligned = (start // 4) * 4

            end_aligned = (
                (end + 3) // 4
            ) * 4  # always round up to next 4-word boundary

            aligned_windows.append(
                CaptureWindow(
                    start=start_aligned,
                    duration=end_aligned - start_aligned,
                )
            )

        # Use the first window as the base
        cprm = CaptureParam()
        first = aligned_windows[0]
        unit = CaptureParam.NUM_SAMPLES_IN_ADC_WORD
        capture_delay_words = (
            (16 * capture_delay_blocks + (first.start // unit * unit)) // 16 * 16
        )
        cprm.capture_delay = capture_delay_words
        cprm.num_integ_sections = self._repetition_count

        # Add remaining windows if any
        durations = [win.duration for win in aligned_windows]
        last = aligned_windows[-1]
        if last.start + last.duration > self._coherent_integration_period:
            raise ValueError(
                "Capture windows exceed coherent integration period: "
                f"{last.start + last.duration} > {self._coherent_integration_period}"
            )

        next_starts = [win.start for win in aligned_windows[1:]] + [
            self._coherent_integration_period // 4
        ]
        loc = 0
        for duration, next_start in zip(durations, next_starts):
            loc += duration
            blank = next_start - loc
            cprm.add_sum_section(num_words=duration // 4, num_post_blank_words=blank)

        self._builder.add_runit_setting(
            port=port,
            runit=runit,
            cprm=cprm,
            trigger_port=trigger_port,
            trigger_channel=trigger_channel,
        )

        # if (trigger_port is not None) and (trigger_channel is not None):
        #     self._trigger_settings.append(
        #         TriggerSetting(
        #             trigger_awg=AwgId(trigger_port, trigger_channel),
        #             triggered_port=port,
        #         )
        #     )
        # elif (trigger_port is not None) or (trigger_channel is not None):
        #     raise ValueError(
        #         "Both trigger_port and trigger_channel must be provided or neither."
        #     )
