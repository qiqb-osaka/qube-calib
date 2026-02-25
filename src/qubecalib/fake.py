from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, MutableMapping, cast

import numpy as np

from .qubecalib import _Quel1SystemBoxProtocol

Quel1PortType = int | tuple[int, int]


@dataclass
class Quel1SystemModel:
    config_cache: dict[str, dict[str, Any]]
    config_fetched_at: datetime.datetime | None


class FakeQuel1System:
    def __init__(self, model: Quel1SystemModel):
        self.boxes: MutableMapping[str, _Quel1SystemBoxProtocol] = {}
        self.box: MutableMapping[str, _Quel1SystemBoxProtocol] = {}
        self.trigger: MutableMapping[
            tuple[str, Quel1PortType], tuple[str, Quel1PortType, int]
        ] = {}
        self.timing_shift: MutableMapping[str, int] = {}
        self.displacement: Any = None

        self.model = model

    @classmethod
    def create_from_config(cls, config: dict[str, Any]) -> FakeQuel1System:
        model = Quel1SystemModel(
            config_cache=config,
            config_fetched_at=datetime.datetime.now(),
        )
        return FakeQuel1System(model)

    def dump_box(self, box_name: str) -> dict[str, Any]:
        if self.model.config_fetched_at is None:
            raise ValueError("config cache is empty")
        if box_name not in self.model.config_cache:
            raise ValueError(f"box {box_name} not found in system")
        return self.model.config_cache[box_name]

    def dump_port(self, box_name: str, port: Quel1PortType) -> dict[str, Any]:
        if self.model.config_fetched_at is None:
            raise ValueError("config cache is empty")
        if box_name not in self.model.config_cache:
            raise ValueError(f"box {box_name} not found in system")
        return self.model.config_cache[box_name]["ports"][port]

    def is_output_port(self, box_name: str, port: Quel1PortType) -> bool:
        if self.model.config_fetched_at is None:
            raise ValueError("config cache is empty")
        if box_name not in self.model.config_cache:
            raise ValueError(f"box {box_name} not found in system")
        return self.model.config_cache[box_name]["ports"][port]["direction"] == "out"

    def is_input_port(self, box_name: str, port: Quel1PortType) -> bool:
        if self.model.config_fetched_at is None:
            raise ValueError("config cache is empty")
        if box_name not in self.model.config_cache:
            raise ValueError(f"box {box_name} not found in system")
        return self.model.config_cache[box_name]["ports"][port]["direction"] == "in"

    def get_monitor_input_ports(self, box_name: str) -> set[int | tuple[int, int]]:
        if self.model.config_fetched_at is None:
            raise ValueError("config cache is empty")
        if box_name not in self.model.config_cache:
            raise ValueError(f"box {box_name} not found in system")
        return self.model.config_cache[box_name]["monitor_input_ports"]

    def get_lo_freq(self, box_name: str, port: Quel1PortType) -> float | None:
        port_cfg = self.dump_port(box_name, port)
        return cast(float, port_cfg["lo_freq"]) if "lo_freq" in port_cfg else None

    def get_cnco_freq(self, box_name: str, port: Quel1PortType) -> float:
        port_cfg = self.dump_port(box_name, port)
        return cast(float, port_cfg["cnco_freq"])

    def get_fnco_freq(self, box_name: str, port: Quel1PortType, channel: int) -> float:
        port_cfg = self.dump_port(box_name, port)
        if "channels" in port_cfg:
            key = "channels"
        elif "runits" in port_cfg:
            key = "runits"
        else:
            raise ValueError(
                f"no channel information found in port-{port} of {box_name}"
            )
        ch_cfgs = cast(dict[int, dict[str, float]], port_cfg[key])
        return ch_cfgs[channel]["fnco_freq"]

    def get_sideband(self, box_name: str, port: Quel1PortType) -> str | None:
        port_cfg = self.dump_port(box_name, port)
        return cast(str, port_cfg["sideband"]) if "sideband" in port_cfg else None


class _FakeBoxHandle:
    """Opaque box handle for APIs that only need box identity."""


class _FakeSequencerClientHandle:
    """Opaque sequencer client handle for BoxPool compatibility."""


class FakeBoxPool:
    """Minimal BoxPool-compatible fake for Sequencer.generate_e7_settings().

    Assumes ``Sequencer.driver`` is set (the ``driver=None`` path is obsolete).
    """

    def __init__(self) -> None:
        self._boxes: dict[str, tuple[Any, Any]] = {}
        self._port_direction: dict[tuple[str, Quel1PortType], str] = {}
        self._box_config_cache: dict[str, dict[str, Any]] = {}

    @classmethod
    def create_from_quel1system(cls, system: FakeQuel1System) -> FakeBoxPool:
        pool = cls()
        if system.model.config_fetched_at is None:
            raise ValueError("config cache is empty")

        for box_name, box_cfg in system.model.config_cache.items():
            pool._box_config_cache[box_name] = box_cfg
            pool._boxes[box_name] = (_FakeBoxHandle(), _FakeSequencerClientHandle())
            for port, port_cfg in box_cfg.get("ports", {}).items():
                direction = port_cfg.get("direction")
                if isinstance(direction, str):
                    pool._port_direction[(box_name, cast(Quel1PortType, port))] = (
                        direction
                    )

        return pool

    def get_box(self, name: str) -> tuple[Any, Any]:
        if name not in self._boxes:
            raise ValueError(f"invalid name of box: '{name}'")
        return self._boxes[name]

    def get_port_direction(self, box_name: str, port: Quel1PortType) -> str:
        key = (box_name, port)
        if key in self._port_direction:
            return self._port_direction[key]
        if box_name not in self._box_config_cache:
            raise ValueError(f"invalid name of box: '{box_name}'")
        direction = self._box_config_cache[box_name]["ports"][port]["direction"]
        if not isinstance(direction, str):
            raise ValueError(f"direction is not defined for {box_name}:{port}")
        self._port_direction[key] = direction
        return direction


class FakeDspUnit:
    INTEGRATION = "INTEGRATION"
    SUM = "SUM"
    DECIMATION = "DECIMATION"
    COMPLEX_FIR = "COMPLEX_FIR"
    COMPLEX_WINDOW = "COMPLEX_WINDOW"
    CLASSIFICATION = "CLASSIFICATION"


class FakeIqWave:
    @staticmethod
    def convert_to_iq_format(
        i: np.ndarray, q: np.ndarray, block_size: int
    ) -> np.ndarray:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if i.shape != q.shape:
            raise ValueError("i and q must have the same shape")
        if i.ndim != 1 or q.ndim != 1:
            raise ValueError("i and q must be 1D arrays")
        pad = (-len(i)) % block_size
        if pad:
            i = np.pad(i, (0, pad))
            q = np.pad(q, (0, pad))
        return i.astype(np.int16) + 1j * q.astype(np.int16)


class FakeWaveSequence:
    NUM_SAMPLES_IN_AWG_WORD = 4
    NUM_SAMPLES_IN_WAVE_BLOCK = 16

    def __init__(self, num_wait_words: int, num_repeats: int):
        self.num_wait_words = num_wait_words
        self.num_repeats = num_repeats
        self.chunks: list[dict[str, Any]] = []

    def add_chunk(
        self, iq_samples: Any, num_blank_words: int, num_repeats: int
    ) -> None:
        self.chunks.append(
            {
                "iq_samples": iq_samples,
                "num_blank_words": num_blank_words,
                "num_repeats": num_repeats,
            }
        )


class FakeCaptureParam:
    NUM_SAMPLES_IN_ADC_WORD = 4
    NUM_COMPLEX_FIR_COEFS = 16
    MAX_FIR_COEF_VAL = 32767
    NUM_COMPLEXW_WINDOW_COEFS = 2048
    MAX_WINDOW_COEF_VAL = 2147483647

    def __init__(self) -> None:
        self.capture_delay: int = 0
        self.num_integ_sections: int = 0
        self.sum_sections: list[dict[str, int]] = []
        self.dsp_units_enabled: list[Any] = []
        self.complex_fir_coefs: list[complex] = []
        self.complex_window_coefs: list[complex] = []
        self.decision_func_params: dict[int, dict[str, np.float32]] = {}

    def add_sum_section(self, num_words: int, num_post_blank_words: int) -> None:
        self.sum_sections.append(
            {
                "num_words": num_words,
                "num_post_blank_words": num_post_blank_words,
            }
        )

    def sel_dsp_units_to_enable(self, *dsp_units: Any) -> None:
        self.dsp_units_enabled = list(dsp_units)

    def set_decision_func_params(
        self,
        *,
        func_sel: int,
        coef_a: np.float32,
        coef_b: np.float32,
        const_c: np.float32,
    ) -> None:
        self.decision_func_params[func_sel] = {
            "coef_a": coef_a,
            "coef_b": coef_b,
            "const_c": const_c,
        }


def install_fake_e7awgsw() -> None:
    from . import e7utils

    e7utils.set_e7awgsw_overrides(
        dsp_unit=FakeDspUnit,
        iq_wave=FakeIqWave,
        wave_sequence_cls=FakeWaveSequence,
        capture_param_cls=FakeCaptureParam,
    )


def reset_fake_e7awgsw() -> None:
    from . import e7utils

    e7utils.reset_e7awgsw_overrides()
