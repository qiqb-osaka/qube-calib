"""
DeviceTask and related classes for managing AWG/CAPU configurations and coordinated execution
in a Quel1 quantum control box system.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import Future
from logging import getLogger
from typing import Final, Optional, Union, cast

import numpy as np
import numpy.typing as npt
from e7awgsw import CaptureParam, DspUnit, WaveSequence
from quel_ic_config import CaptureReturnCode, Quel1BoxWithRawWss, Quel1WaveSubsystem

from .boxtask import BoxTask
from .tasksetting import RawBuilder, RawTaskSettingBuilder, RunitId, TaskSetting

Quel1PortType = Union[int, tuple[int, int]]

FULL_SCALE = 2**15 - 1  # Full scale for complex64 in Quel1 AWG

logger = getLogger(__name__)


class SingleBoxTask(BoxTask):
    """
    Encapsulates the coordination of waveform emission and data capture using
    a configured quantum control box (Quel1). Supports standalone or triggered execution
    of waveform and capture tasks.

    Call `load()` before `execute()` to apply the configuration to the hardware.

    Attributes:
        is_loaded (bool): Indicates whether settings have been loaded into the box.
    """

    def __init__(
        self,
        setting: TaskSetting,
    ) -> None:
        self._box: Quel1BoxWithRawWss | None = None
        self._setting: Final[TaskSetting] = setting
        self._is_loaded: bool = False

    def __repr__(self) -> str:
        return (
            f"<SingleBoxTask("
            f"wseqs={len(self._setting.wseqs)}, "
            f"cprms={len(self._setting.cprms)}, "
            f"triggers={len(self._setting.triggers)}, "
            f"is_loaded={self._is_loaded})>"
        )

    @property
    def box(self) -> Quel1BoxWithRawWss:
        """
        Access the underlying hardware control box.

        Returns:
            Quel1BoxWithRawWss: The hardware control box instance.
        """
        if self._box is None:
            raise RuntimeError("DeviceTask is not loaded with a box")
        return self._box

    @property
    def setting(self) -> TaskSetting:
        """
        Access the task settings.

        Returns:
            TaskSetting: The current task settings.
        """
        return self._setting

    # @classmethod
    # def build(
    #     cls,
    #     *,
    #     settings: list[RunitSetting | AwgSetting | TriggerSetting],
    # ) -> SingleBoxTask:
    #     """
    #     Create a DeviceTask instance from a list of settings without loading it.

    #     Args:
    #         box (Quel1BoxWithRawWss): The hardware control box instance.
    #         settings (list): List of RunitSetting, AwgSetting, or TriggerSetting objects.

    #     Raises:
    #         ValueError: If no settings are provided.

    #     Returns:
    #         DeviceTask: New instance with parsed settings.
    #     """
    #     if not settings:
    #         raise ValueError("settings must be provided")
    #     setting = cls.parse_settings(settings)
    #     return cls(setting)

    def load(self, box: Quel1BoxWithRawWss) -> None:
        """
        Load the configured settings into the hardware device and prepare for emission.

        This must be called before `execute()` to apply the configuration.

        Returns:
            None
        """
        self._box = box
        self._load_to_device()
        awgs = set([(s.port, s.channel) for s in self._setting.wseqs])
        self._box.prepare_for_emission(awgs)
        self._is_loaded = True

    @property
    def is_loaded(self) -> bool:
        """
        Check whether the task settings have been loaded into the device.

        Returns:
            bool: True if loaded, False otherwise.
        """
        return self._is_loaded

    def _load_to_device(self) -> None:
        # Ensure the box is assigned before attempting to configure the device.
        if self._box is None:
            raise RuntimeError("DeviceTask must be provided with a box before use.")
        # self._setting.ensure_wave_sequences_if_defferred()
        for awg, wseq in self._setting.wseqs.items():
            self.box.config_channel(
                port=awg.port,
                channel=awg.channel,
                wave_param=wseq,
            )
        for runit, cprm in self._setting.cprms.items():
            self.box.config_runit(
                port=runit.port,
                runit=runit.runit,
                capture_param=cprm,
            )

    def capture_start(
        self,
        *,
        timeout: Optional[float] = None,
    ) -> dict[
        int, Future[tuple[CaptureReturnCode, dict[int, npt.NDArray[np.complex64]]]]
    ]:
        """
        Initiate capture operations on configured CAPUs, optionally using triggers.

        Args:
            timeout (float, optional): Maximum time to wait for capture completion.

        Raises:
            ValueError: If trigger or runit port configurations are inconsistent.

        Returns:
            dict: Mapping from port number to Future objects representing ongoing captures.
        """
        # Prevent CAPM from stalling by ensuring that the trigger source AWG is part of the configured AWG channels.
        channels = {awg for awg in self._setting.wseqs}
        runits_by_ports = defaultdict(list)
        for runit in self._setting.cprms:
            runits_by_ports[runit.port].append(runit.runit)
        for port, trigger in self._setting.triggers.items():
            if trigger not in channels:
                raise ValueError(
                    f"trigger {trigger} for triggered port {port} is not provided"
                )
            if port not in runits_by_ports:
                raise ValueError(
                    f"triggered port {port} is not provided in runit settings"
                )
        if timeout is None:
            timeout = Quel1WaveSubsystem.DEFAULT_CAPTURE_TIMEOUT
        if runits_by_ports:
            return {
                port: cast(Quel1BoxWithRawWss, self._box).capture_start(
                    port,
                    runits,
                    triggering_channel=(
                        self._setting.triggers[port].port,
                        self._setting.triggers[port].channel,
                    )
                    if port in self._setting.triggers
                    else None,
                    timeout=timeout,
                )
                for port, runits in runits_by_ports.items()
            }
        else:
            return {}

    def start_emission(self) -> None:
        """
        Start waveform emission on all configured AWG channels.

        If no AWG channels are configured, this method does nothing.

        Returns:
            None
        """
        awg_specs = set([(s.port, s.channel) for s in self._setting.wseqs])
        if awg_specs:  # _channels が空の場合は AWG は起動しない
            cast(Quel1BoxWithRawWss, self._box).start_emission(awg_specs)

    def capture_stop(
        self, futures: dict[int, Future]
    ) -> tuple[
        dict[int, CaptureReturnCode], dict[tuple[int, int], npt.NDArray[np.complex64]]
    ]:
        """
        Wait for capture futures to complete and collect results.

        Args:
            futures (dict): Mapping from port number to Future objects.

        Returns:
            tuple:
                - dict mapping port to CaptureReturnCode status.
                - dict mapping (port, runit) to captured complex data arrays.
        """
        status, data = {}, {}
        for port, future in futures.items():
            capt_return_code, runit_data = future.result()
            status[port] = capt_return_code
            for runit, d in runit_data.items():
                data[(port, runit)] = d
        return status, data

    def execute(
        self,
    ) -> tuple[
        dict[int, CaptureReturnCode],
        dict[tuple[int, int], list[npt.NDArray[np.complex64]]],
        dict[tuple[int, int], list[npt.NDArray[np.int32]]],
    ]:
        """
        Execute the configured task: start capture and/or emission depending on configuration.

        Raises:
            RuntimeError: If called before `load()`.
            ValueError: If configuration combination is unsupported.

        Returns:
            tuple:
                - dict mapping port to CaptureReturnCode status.
                - dict mapping (port, runit) to captured complex data arrays.
        """
        if not self._is_loaded:
            raise RuntimeError("DeviceTask must be loaded before action() is called")
        # Execution mode based on available settings:
        # - AWG only:               wseqs=True,  cprms=False, triggers=False
        # - Capture only:           wseqs=False, cprms=True,  triggers=False
        # - Triggered capture:      wseqs=True,  cprms=True,  triggers=True
        # - All other combinations: unsupported
        if all(
            [
                bool(self._setting.wseqs),
                bool(self._setting.cprms),
                bool(self._setting.triggers),
            ]
        ):
            futures = self.capture_start()
            self.start_emission()
            status, raw_results = self.capture_stop(futures)
            results = {
                (port, runit): self.parse_capture_result(
                    result, self._setting.cprms[RunitId(port, runit)]
                )
                for (port, runit), result in raw_results.items()
            }
            indices = {
                (runit_id.port, runit_id.runit): indices
                for runit_id, indices in self._setting.runit_sample_indices.items()
            }
            return (
                status,
                results,
                indices,
            )
        # Awg only
        elif all(
            [
                bool(self._setting.wseqs),
                not bool(self._setting.cprms),
                not bool(self._setting.triggers),
            ]
        ):
            self.start_emission()
            return {}, {}, {}
        # Capture only
        elif all(
            [
                not bool(self._setting.wseqs),
                bool(self._setting.cprms),
                not bool(self._setting.triggers),
            ]
        ):
            futures = self.capture_start()
            status, raw_results = self.capture_stop(futures)
            results = {
                (port, runit): self.parse_capture_result(
                    result, self._setting.cprms[RunitId(port, runit)]
                )
                for (port, runit), result in raw_results.items()
            }
            indices = {
                (runit_id.port, runit_id.runit): indices
                for runit_id, indices in self._setting.runit_sample_indices.items()
            }
            return (
                status,
                results,
                indices,
            )
        else:
            # Any other combination is considered invalid.
            raise ValueError(
                "unsupported combination of AWG, CAPU, and trigger settings"
            )

    def parse_capture_result(
        self,
        result: npt.NDArray[np.complex64],
        cprm: CaptureParam,
    ) -> list[npt.NDArray[np.complex64]]:
        if DspUnit.INTEGRATION in cprm.dsp_units_enabled:
            result = result.reshape(1, -1)
        else:
            result = result.reshape(cprm.num_integ_sections, -1)
        if DspUnit.SUM in cprm.dsp_units_enabled:
            width = len(cprm.sum_section_list)
            data = np.hsplit(result, width)
        else:
            b = DspUnit.DECIMATION not in cprm.dsp_units_enabled
            ssl = cprm.sum_section_list
            ws = [w if b else int(w // 4) for w, _ in ssl[:-1]]
            word = cprm.NUM_SAMPLES_IN_ADC_WORD
            width = np.cumsum(np.array(ws))
            data = np.hsplit(result, width * word)
        return data


class RawSingleBoxTaskBuilder(RawBuilder):
    """
    High-level builder for SingleBoxTask from abstract user inputs.
    This class converts user-friendly readout/capture configurations
    into AWG/CAPU settings and trigger mappings.
    """

    def __init__(self) -> None:
        self._builder = RawTaskSettingBuilder()

    def add_awg_setting(
        self,
        *,
        wseq: WaveSequence,
        port: int,
        channel: int,
    ) -> None:
        self._builder.add_awg_setting(
            port=port,
            channel=channel,
            wseq=wseq,
        )

    def add_runit_setting(
        self,
        *,
        cprm: CaptureParam,
        port: int,
        runit: int,
        trigger_port: int | None = None,
        trigger_channel: int | None = None,
    ) -> None:
        self._builder.add_runit_setting(
            cprm=cprm,
            port=port,
            runit=runit,
            trigger_port=trigger_port,
            trigger_channel=trigger_channel,
        )

    def build(self) -> SingleBoxTask:
        return SingleBoxTask(self._builder.build())
