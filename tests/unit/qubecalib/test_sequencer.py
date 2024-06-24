from __future__ import annotations

from typing import Any

from e7awgsw import AWG, CaptureUnit, WaveSequence
from pytest_mock import MockerFixture
from qubecalib.general_looptest_common_mod import BoxPool
from qubecalib.neopulse import DEFAULT_SAMPLING_PERIOD, Capture, Rectangle, Sequence
from qubecalib.qubecalib import BoxSetting, PortSetting, Sequencer
from quel_ic_config import Quel1BoxType


class ResourceMap:
    def get_capture_module_of_rline(self, group: int, rline: str) -> CaptureUnit:
        return CaptureUnit.U0

    def get_awg_of_channel(self, group: int, line: int, channel: int) -> AWG:
        return AWG.U0


class Box:
    def __init__(self) -> None:
        self.rmap = ResourceMap()
        self.wss = None

    def dump_box(self) -> dict[str, Any]:
        return {}

    def dump_port(
        self, port: int
    ) -> dict[str, str | float | int | dict[int, dict[str, float]]]:
        if port in [1, 4, 9, 12]:
            return {
                "direction": "in",
                "lo_freq": 12000_000_000,
                "cnco_freq": 1500000000.0,
                "runits": {
                    0: {"fnco_freq": 0.0},
                    1: {"fnco_freq": 0.0},
                    2: {"fnco_freq": 0.0},
                    3: {"fnco_freq": 0.0},
                },
            }
        elif port in [5, 6, 7, 8]:
            return {
                "direction": "out",
                "channels": {
                    0: {"fnco_freq": 0.0},
                    1: {"fnco_freq": 0.0},
                    2: {"fnco_freq": 0.0},
                    3: {"fnco_freq": 0.0},
                },
                "lo_freq": 12000_000_000,
                "cnco_freq": 1500000000.0,
                "sideband": "L",
            }
        elif port in [0, 13]:
            return {
                "direction": "out",
                "channels": {
                    0: {"fnco_freq": 0.0},
                },
                "lo_freq": 12000_000_000,
                "cnco_freq": 1500000000.0,
                "sideband": "L",
            }
        else:
            return {}

    def link_status(self) -> dict[int, bool]:
        return {
            0: True,
            1: True,
        }

    def decode_port(self, port: int) -> tuple[int, int]:
        return (port, 0)

    def _convert_any_port(self, port: int) -> tuple[int, int | str]:
        if port == 0:
            return (0, 0)
        elif port == 1:
            return (0, "r")
        else:
            raise ValueError(f"Invalid port number: {port}")


class Quel1WaveSubsystemMod:
    def set_wave(
        self,
        wss: None,
        awg: AWG,
        waveseq: WaveSequence,
    ) -> None:
        pass


def test_execute(mocker: MockerFixture) -> None:
    mocker.patch(
        "qubecalib.general_looptest_common_mod.BoxPool.get_box",
        return_value=(Box(), None),
    )
    mocker.patch(
        "qubecalib.quel1_wave_subsystem_mod.Quel1WaveSubsystemMod.set_wave",
        return_value=None,
    )
    # mocker.patch(
    #     "qubecalib.quel1_wave_subsystem_mod.Quel1WaveSubsystemMod.simple_capture_start",
    # )
    mocker.patch(
        "qubecalib.general_looptest_common_mod.PulseCap.capture_at_trigger_of",
        return_value=None,
    )
    mocker.patch(
        "qubecalib.general_looptest_common_mod.PulseGen.emit_now",
        return_value=None,
    )
    mocker.patch(
        "qubecalib.general_looptest_common_mod.PulseCap.wait_until_capture_finishes",
        return_value=(None, None),
    )
    mocker.patch(
        "qubecalib.Sequencer.convert_key_from_bmu_to_target",
        return_value=(None, None),
    )

    TARGET = "RQ00"
    DT = DEFAULT_SAMPLING_PERIOD
    N_BLANK = 5
    N_CAPTURE = 10
    with Sequence() as seq:
        Rectangle(duration=N_BLANK * DT).target(TARGET)
        Capture(duration=N_CAPTURE * DT).target(TARGET)

    gen_sampled_sequence, cap_sampled_sequence = seq.convert_to_sampled_sequence()
    resource_map = {
        TARGET: [
            {
                "box": BoxSetting("QUBE", "0.0.0.0", Quel1BoxType.QuBE_RIKEN_TypeA),
                "port": PortSetting("QUBE.PORT0", "QUBE", 0, ndelay_or_nwait=(0,)),
                "channel_number": 0,
                "target": {"frequency": 1000},
            },
            {
                "box": BoxSetting("QUBE", "0.0.0.0", Quel1BoxType.QuBE_RIKEN_TypeA),
                "port": PortSetting(
                    "QUBE.PORT1", "QUBE", 1, ndelay_or_nwait=(0, 0, 0, 0)
                ),
                "channel_number": 0,
                "target": {"frequency": 1000},
            },
        ]
    }
    sequencer = Sequencer(
        gen_sampled_sequence,
        cap_sampled_sequence,
        resource_map,
    )
    sequencer.set_measurement_option(
        repeats=1000,
        interval=10024,
        integral_mode="",
        dsp_demodulation=True,
        software_demodulation=False,
    )

    boxpool = BoxPool()

    sequencer.execute(boxpool)
