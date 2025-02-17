from __future__ import annotations

from typing import Any, Type

import error

error

from e7awgsw import AWG, CaptureUnit, WaveSequence
from pytest_mock import MockerFixture
from qubecalib.neopulse import (
    DEFAULT_SAMPLING_PERIOD,
    CapSampledSequence,
    Capture,
    Flushleft,
    Rectangle,
    Sequence,
    Slot,
    padding,
)
from qubecalib.qubecalib import (
    BoxPool,
    BoxSetting,
    PortSetting,
    Sequencer,
    SystemConfigDatabase,
)
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

    def dump_box(self) -> dict[str, dict[int, Any]]:
        dump: dict[str, dict[int, Any]] = {"ports": {}}
        for i in range(14):
            dump["ports"][i] = self.dump_port(i)
        return dump

    def dump_port(
        self, port: int
    ) -> dict[str, str | float | int | dict[int, dict[str, float]]]:
        if port in [1, 4, 9, 12]:
            return {
                "direction": "in",
                "lo_freq": 8_000_000_000,
                "cnco_freq": 1_500_000_000,
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
                "lo_freq": 12_000_000_000,
                "cnco_freq": 1_500_000_000,
                "sideband": "L",
            }
        elif port in [0, 2, 11, 13]:
            return {
                "direction": "out",
                "channels": {
                    0: {"fnco_freq": 0.0},
                },
                "lo_freq": 8_000_000_000,
                "cnco_freq": 1_500_000_000,
                "sideband": "U",
            }
        else:
            return {}

    def link_status(self) -> dict[int, bool]:
        return {
            0: True,
            1: True,
        }

    def _decode_port(self, port: int) -> tuple[int, int]:
        return (port, 0)

    def _convert_any_port(self, port: int) -> tuple[int, int | str]:
        # QUBE_OU_TypeA の場合
        mapper: dict[int, tuple[int, int | str]] = {
            0: (0, 0),
            1: (0, "r"),
            2: (0, 1),
            5: (0, 2),
            6: (0, 3),
            7: (1, 3),
            8: (1, 2),
            11: (1, 1),
            12: (1, "r"),
            13: (1, 0),
        }
        if port not in mapper:
            raise ValueError(f"Invalid port: {port}")
        return mapper[port]

    def get_output_ports(self) -> set[int]:
        return {0, 2, 5, 6, 7, 8, 11, 13}

    def get_input_ports(self) -> set[int]:
        return {1, 12}

    def get_read_input_ports(self) -> set[int]:
        return {1, 12}

    def get_loopbacks_of_port(self, port: int) -> set[int]:
        if port in self.get_read_input_ports():
            if port == 1:
                return {0}
            elif port == 12:
                return {13}
            else:
                return set()
        else:
            if port == 4:
                return {0, 2, 5, 6}
            elif port == 9:
                return {7, 8, 11, 13}
            else:
                return set()


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
        "qubecalib.qubecalib.BoxPool.get_box",
        return_value=(Box(), None),
    )
    # mocker.patch(
    #     "qubecalib.quel1_wave_subsystem_mod.Quel1WaveSubsystemMod.set_wave",
    #     return_value=None,
    # )
    # mocker.patch(
    #     "qubecalib.general_looptest_common_mod.PulseCap.capture_at_trigger_of",
    #     return_value=None,
    # )
    # mocker.patch(
    #     "qubecalib.general_looptest_common_mod.PulseGen.emit_now",
    #     return_value=None,
    # )
    # mocker.patch(
    #     "qubecalib.general_looptest_common_mod.PulseCap.wait_until_capture_finishes",
    #     return_value=(None, None),
    # )
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
                "target": {"frequency": 9.6},
            },
            {
                "box": BoxSetting("QUBE", "0.0.0.0", Quel1BoxType.QuBE_RIKEN_TypeA),
                "port": PortSetting(
                    "QUBE.PORT1", "QUBE", 1, ndelay_or_nwait=(0, 0, 0, 0)
                ),
                "channel_number": 0,
                "target": {"frequency": 9.6},
            },
        ]
    }
    sequencer = Sequencer(
        gen_sampled_sequence=gen_sampled_sequence,
        cap_sampled_sequence=cap_sampled_sequence,
        group_items_by_target=seq._get_group_items_by_target(),
        resource_map=resource_map,
        sysdb=SystemConfigDatabase(),
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


def test_execute_backward_compat(mocker: MockerFixture) -> None:
    mocker.patch(
        "qubecalib.qubecalib.BoxPool.get_box",
        return_value=(Box(), None),
    )
    # mocker.patch(
    #     "qubecalib.quel1_wave_subsystem_mod.Quel1WaveSubsystemMod.set_wave",
    #     return_value=None,
    # )
    # mocker.patch(
    #     "qubecalib.general_looptest_common_mod.PulseCap.capture_at_trigger_of",
    #     return_value=None,
    # )
    # mocker.patch(
    #     "qubecalib.general_looptest_common_mod.PulseGen.emit_now",
    #     return_value=None,
    # )
    # mocker.patch(
    #     "qubecalib.general_looptest_common_mod.PulseCap.wait_until_capture_finishes",
    #     return_value=(None, None),
    # )
    # mocker.patch(
    #     "qubecalib.Sequencer.convert_key_from_bmu_to_target",
    #     return_value=(None, None),
    # )

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
                "target": {"frequency": 9.6},
            },
            {
                "box": BoxSetting("QUBE", "0.0.0.0", Quel1BoxType.QuBE_RIKEN_TypeA),
                "port": PortSetting(
                    "QUBE.PORT1", "QUBE", 1, ndelay_or_nwait=(0, 0, 0, 0)
                ),
                "channel_number": 0,
                "target": {"frequency": 9.6},
            },
        ]
    }
    sequencer = Sequencer(
        gen_sampled_sequence=gen_sampled_sequence,
        cap_sampled_sequence=cap_sampled_sequence,
        resource_map=resource_map,
        sysdb=SystemConfigDatabase(),
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


def property_of_capture_slot(
    csseq: CapSampledSequence,
    target_name: str,
    subseq_index: int,
    slot_index: int,
) -> tuple[int, int]:
    slot = csseq[target_name].sub_sequences[subseq_index].capture_slots[slot_index]
    return slot.duration, slot.post_blank


def property_of_slot(slot: Slot) -> tuple[Type[Slot], int, int]:
    return slot.__class__, slot.duration, slot.begin


def test_make_e7_settings(mocker: MockerFixture) -> None:
    mocker.patch(
        "qubecalib.qubecalib.BoxPool.get_box",
        return_value=(Box(), None),
    )
    # mocker.patch(
    #     "qubecalib.quel1_wave_subsystem_mod.Quel1WaveSubsystemMod.set_wave",
    #     return_value=None,
    # )
    # mocker.patch(
    #     "qubecalib.general_looptest_common_mod.PulseCap.capture_at_trigger_of",
    #     return_value=None,
    # )
    # mocker.patch(
    #     "qubecalib.general_looptest_common_mod.PulseGen.emit_now",
    #     return_value=None,
    # )
    # mocker.patch(
    #     "qubecalib.general_looptest_common_mod.PulseCap.wait_until_capture_finishes",
    #     return_value=(None, None),
    # )
    # mocker.patch(
    #     "qubecalib.Sequencer.convert_key_from_bmu_to_target",
    #     return_value=(None, None),
    # )

    with Sequence() as seq:
        Rectangle(duration=21).target("Q00")
        with Flushleft():
            Rectangle(duration=9).target("RQ00")
            Capture(duration=9).target("RQ00")
            Capture(duration=9).target("RQ02")
        padding(10.5)
        with Flushleft():
            Rectangle(duration=121).target("RQ00")
            Capture(duration=121).target("RQ00")
        with Flushleft():
            Rectangle(duration=31).target("RQ01")
            Capture(duration=31).target("RQ01")
        with Flushleft():
            Rectangle(duration=51).target("RQ01")
            Capture(duration=51).target("RQ01")

    gen_sampled_sequence, cap_sampled_sequence = seq.convert_to_sampled_sequence()
    group_items_by_target = seq._get_group_items_by_target()

    box = BoxSetting("QUBE", "0.0.0.0", Quel1BoxType.QuBE_RIKEN_TypeA)
    port0 = PortSetting("QUBE.PORT0", "QUBE", 0, ndelay_or_nwait=(0,))
    port1 = PortSetting("QUBE.PORT1", "QUBE", 1, ndelay_or_nwait=(1, 1, 1, 1))
    port5 = PortSetting("QUBE.PORT5", "QUBE", 5, ndelay_or_nwait=(0, 0, 0))
    port13 = PortSetting("QUBE.PORT13", "QUBE", 13, ndelay_or_nwait=(0,))
    resource_map = {
        "RQ00": [
            {
                "box": box,
                "port": port1,
                "channel_number": 0,
                "target": {"frequency": 9.6},
            },
            {
                "box": box,
                "port": port0,
                "channel_number": 0,
                "target": {"frequency": 9.6},
            },
        ],
        "RQ01": [
            {
                "box": box,
                "port": port1,
                "channel_number": 1,
                "target": {"frequency": 9.6},
            },
            {
                "box": box,
                "port": port0,
                "channel_number": 0,
                "target": {"frequency": 9.6},
            },
        ],
        "RQ02": [
            {
                "box": box,
                "port": port1,
                "channel_number": 2,
                "target": {"frequency": 9.7},
            },
            {
                "box": box,
                "port": port13,
                "channel_number": 0,
                "target": {"frequency": 9.7},
            },
        ],
        "Q00": [
            {
                "box": box,
                "port": port5,
                "channel_number": 0,
                "target": {"frequency": 10.4},
            },
        ],
    }

    sequencer = Sequencer(
        gen_sampled_sequence,
        cap_sampled_sequence,
        resource_map,
        group_items_by_target=group_items_by_target,
        sysdb=SystemConfigDatabase(),
    )

    sequencer.set_measurement_option(
        repeats=1000,
        interval=10024,
        integral_mode="",
        dsp_demodulation=True,
        software_demodulation=False,
        phase_compensation=True,
    )

    boxpool = BoxPool()

    status, iqs, config = sequencer.execute(boxpool)

    print()
