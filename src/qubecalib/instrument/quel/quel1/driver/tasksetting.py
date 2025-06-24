from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from e7awgsw import CaptureParam, WaveSequence
from quel_ic_config import Quel1PortType


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


class RawTaskSettingBuilder:
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
        port: int,
        channel: int,
        wseq: WaveSequence,
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
        port: int,
        runit: int,
        cprm: CaptureParam,
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
