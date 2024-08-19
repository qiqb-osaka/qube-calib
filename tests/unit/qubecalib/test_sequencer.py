from __future__ import annotations

from typing import Any, Type

from e7awgsw import AWG, CaptureUnit, WaveSequence
from pytest_mock import MockerFixture
from quel_ic_config import Quel1BoxType

from qubecalib.general_looptest_common_mod import BoxPool
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
    BoxSetting,
    PortSetting,
    Sequencer,
)


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
        gen_sampled_sequence=gen_sampled_sequence,
        cap_sampled_sequence=cap_sampled_sequence,
        group_items_by_target=seq._get_group_items_by_target(),
        resource_map=resource_map,
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
        "qubecalib.general_looptest_common_mod.BoxPool.get_box",
        return_value=(Box(), None),
    )
    mocker.patch(
        "qubecalib.quel1_wave_subsystem_mod.Quel1WaveSubsystemMod.set_wave",
        return_value=None,
    )
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
        gen_sampled_sequence=gen_sampled_sequence,
        cap_sampled_sequence=cap_sampled_sequence,
        resource_map=resource_map,
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


# def property_of_generator_slot(
#     gsseq: GenSampledSequence,
#     target_name: str,
#     subseqid: int,
#     slotid: int,
# ) -> tuple[int, int]:
#     slot = gsseq[target_name]


def property_of_slot(slot: Slot) -> tuple[Type[Slot], int, int]:
    return slot.__class__, slot.duration, slot.begin


# def test_make_cap_gen_e7_settings(mocker: MockerFixture) -> None:
#     with Sequence() as seq:
#         Rectangle(duration=21).target("Q00")
#         with Flushleft():
#             Rectangle(duration=9).target("RQ00")
#             Capture(duration=9).target("RQ00")
#             Capture(duration=9).target("RQ02")
#         padding(10.5)
#         with Flushleft():
#             Rectangle(duration=121).target("RQ00")
#             Capture(duration=121).target("RQ00")
#         with Flushleft():
#             Rectangle(duration=31).target("RQ01")
#             Capture(duration=31).target("RQ01")
#         with Flushleft():
#             Rectangle(duration=51).target("RQ01")
#             Capture(duration=51).target("RQ01")
#     seq._tree.place_slots()
#     group_items = (
#         seq._get_group_items_by_target()
#     )  # target, nodeid of SubSequence 毎に属する item をまとめる
#     assert property_of_slot(group_items["Q00"][1][0]) == (Rectangle, 21, 0.0)
#     assert property_of_slot(group_items["RQ00"][1][0]) == (Rectangle, 9, 21.0)
#     assert property_of_slot(group_items["RQ00"][1][1]) == (Capture, 9, 21.0)
#     assert property_of_slot(group_items["RQ00"][1][2]) == (Rectangle, 121, 40.5)
#     assert property_of_slot(group_items["RQ00"][1][3]) == (Capture, 121, 40.5)
#     assert property_of_slot(group_items["RQ01"][1][0]) == (Rectangle, 31, 161.5)
#     assert property_of_slot(group_items["RQ01"][1][1]) == (Capture, 31, 161.5)
#     assert property_of_slot(group_items["RQ01"][1][2]) == (Rectangle, 51, 192.5)
#     assert property_of_slot(group_items["RQ01"][1][3]) == (Capture, 51, 192.5)
#     genitems_by_target_with_empty_subseqids = {
#         target_name: {
#             subseqid: [item for item in items if isinstance(item, (Waveform, Modifier))]
#             for subseqid, items in items_by_subseqids.items()
#         }
#         for target_name, items_by_subseqids in group_items.items()
#     }  # group_items の中から gen に関係する item のみを取り出す
#     genitems_by_target_with_empty_targets: dict[
#         str, dict[int, MutableSequence[Waveform | Modifier]]
#     ] = {
#         target_name: {
#             subseqid: items for subseqid, items in items_by_subseqids.items() if items
#         }
#         for target_name, items_by_subseqids in genitems_by_target_with_empty_subseqids.items()
#     }  # gen に関係しない item を削除したので空の subseqid を削除
#     genitems_by_target: dict[str, dict[int, MutableSequence[Waveform | Modifier]]] = {
#         target_name: {
#             subseqid: items for subseqid, items in items_by_subseqids.items() if items
#         }
#         for target_name, items_by_subseqids in genitems_by_target_with_empty_targets.items()
#         if items_by_subseqids
#     }  # gen に関係しない subseq を削除したので空の target を削除
#     gsseq = {
#         target: seq._create_gen_sampled_sequence(target, genitems_by_target)
#         for target in genitems_by_target
#     }

#     capitems_by_target_with_empty_subseqids = {
#         target_name: {
#             subseqid: [item for item in items if isinstance(item, Capture)]
#             for subseqid, items in items_by_subseqids.items()
#         }
#         for target_name, items_by_subseqids in group_items.items()
#     }  # group_items の中から gen に関係する item のみを取り出す
#     capitems_by_target_with_empty_targets: dict[
#         str, dict[int, MutableSequence[Waveform | Modifier]]
#     ] = {
#         target_name: {
#             subseqid: items for subseqid, items in items_by_subseqids.items() if items
#         }
#         for target_name, items_by_subseqids in capitems_by_target_with_empty_subseqids.items()
#     }  # gen に関係しない item を削除したので空の subseqid を削除
#     capitems_by_target: dict[str, dict[int, MutableSequence[Waveform | Modifier]]] = {
#         target_name: {
#             subseqid: items for subseqid, items in items_by_subseqids.items() if items
#         }
#         for target_name, items_by_subseqids in capitems_by_target_with_empty_targets.items()
#         if items_by_subseqids
#     }  # gen に関係しない subseq を削除したので空の target を削除
#     csseq = {
#         target: seq._create_cap_sampled_sequence(target, capitems_by_target)
#         for target in capitems_by_target
#     }

#     # targets = set([target for target in gsseq] + [target for target in csseq])

#     # channels_by_targets: MutableSequence[tuple[str, set[str]]] = []

#     # bpc_targets = {target_name: [] for target_name, channels in channels_by_targets}

#     # resource_map = {target_name: [{"box": BoxSetting}] for target_name in targets}
#     # gsseq, csseq = seq.convert_to_sampled_sequence()

#     assert csseq["RQ00"].sub_sequences[0].prev_blank == 10
#     assert property_of_capture_slot(csseq, "RQ00", 0, 0) == (5, 5)  # Sa
#     assert property_of_capture_slot(csseq, "RQ00", 0, 1) == (60, 41)  # Sa
#     assert csseq["RQ01"].sub_sequences[0].prev_blank == 80  # Sa
#     assert property_of_capture_slot(csseq, "RQ01", 0, 0) == (16, 0)  # Sa
#     assert property_of_capture_slot(csseq, "RQ01", 0, 1) == (25, 0)  # Sa
#     assert csseq["RQ02"].sub_sequences[0].prev_blank == 10  # Sa
#     assert property_of_capture_slot(csseq, "RQ02", 0, 0) == (5, 106)  # Sa

#     # Converter.convert_to_cap_device_specific_sequence(csseq)

#     # bpc_targets = {target_name: [] for target_name, channels in targets_channels}

#     # _create_target_resource_map
#     box = BoxSetting("QUBE", "0.0.0.0", Quel1BoxType.QuBE_RIKEN_TypeA)
#     port1 = PortSetting("QUBE.PORT1", "QUBE", 1, ndelay_or_nwait=(1, 1, 1, 1))
#     caprmap = {
#         "RQ00": {
#             "box": box,
#             "port": port1,
#             "channel_number": 0,
#             "target": {"frequency": 9.8},
#         },
#         "RQ01": {
#             "box": box,
#             "port": port1,
#             "channel_number": 1,
#             "target": {"frequency": 9.9},
#         },
#         "RQ02": {
#             "box": box,
#             "port": port1,
#             "channel_number": 2,
#             "target": {"frequency": 10.1},
#         },
#     }

#     cap_target_bpc: dict[str, TargetBPC] = {
#         target_name: TargetBPC(
#             box=Box(),
#             port=m["port"].port if isinstance(m["port"], PortSetting) else m["port"],
#             channel=m["channel_number"],
#         )
#         for target_name, m in caprmap.items()
#     }

#     cap_target_portconf = {
#         target_name: PortConfigAcquirer(
#             box=m["box"],
#             port=m["port"],
#             channel=m["channel"],
#         )
#         for target_name, m in cap_target_bpc.items()
#     }

#     first_blank = min(
#         [seq.prev_blank for sseq in csseq.values() for seq in sseq.sub_sequences]
#     )
#     # WaveSequence の単位に合わせた先頭 padding
#     # 暗黙に CaptureParam でも同じ処理が入る
#     first_padding = (first_blank // 64 + 1) * 64 - first_blank  # Sa

#     cap_e7_settings: dict[tuple[str, int, int], CaptureParam] = (
#         Converter.convert_to_cap_device_specific_sequence(
#             gen_sampled_sequence=gsseq,
#             cap_sampled_sequence=csseq,
#             resource_map=caprmap,
#             port_config=cap_target_portconf,
#             repeats=1,
#             interval=10240,
#             integral_mode="integral",
#             dsp_demodulation=True,
#             software_demodulation=False,
#         )
#     )  # convert_to_device_specific_sequence

#     E7_RQ00 = cap_e7_settings[("QUBE", 1, 0)]
#     assert E7_RQ00.capture_delay == 32
#     E7_RQ01 = cap_e7_settings[("QUBE", 1, 1)]
#     assert E7_RQ01.capture_delay == 48
#     E7_RQ02 = cap_e7_settings[("QUBE", 1, 2)]
#     assert E7_RQ02.capture_delay == 32

#     port0 = PortSetting("QUBE.PORT0", "QUBE", 0, ndelay_or_nwait=(0,))
#     port5 = PortSetting("QUBE.PORT5", "QUBE", 5, ndelay_or_nwait=(0, 0, 0))
#     port13 = PortSetting("QUBE.PORT13", "QUBE", 13, ndelay_or_nwait=(0,))
#     genrmap = {
#         "RQ00": {
#             "box": box,
#             "port": port0,
#             "channel_number": 0,
#             "target": {"frequency": 9.8},
#         },
#         "RQ01": {
#             "box": box,
#             "port": port0,
#             "channel_number": 0,
#             "target": {"frequency": 9.9},
#         },
#         "RQ02": {
#             "box": box,
#             "port": port13,
#             "channel_number": 0,
#             "target": {"frequency": 10.1},
#         },
#         "Q00": {
#             "box": box,
#             "port": port5,
#             "channel_number": 0,
#             "target": {"frequency": 10.2},
#         },
#     }
#     gen_target_bpc: dict[str, TargetBPC] = {
#         target_name: TargetBPC(
#             box=Box(),
#             port=m["port"].port
#             if isinstance(m["port"], PortSetting)
#             else m["port"].port,
#             channel=m["channel_number"],
#         )
#         for target_name, m in genrmap.items()
#     }

#     gen_target_portconf = {
#         target_name: PortConfigAcquirer(
#             box=m["box"], port=m["port"], channel=m["channel"]
#         )
#         for target_name, m in gen_target_bpc.items()
#     }

#     # gen_e7_settings
#     gen_e7_settings: dict[tuple[str, int, int], WaveSequence] = (
#         Converter.convert_to_gen_device_specific_sequence(
#             gen_sampled_sequence=gsseq,
#             cap_sampled_sequence=csseq,
#             resource_map=genrmap,
#             port_config=gen_target_portconf,
#             repeats=1,
#             interval=10240,
#             padding=first_padding,
#         )
#     )

#     assert len(gen_e7_settings[("QUBE", 5, 0)].chunk_list) == 1
#     assert gen_e7_settings[("QUBE", 5, 0)].chunk_list[0].num_words == 1280
#     assert gen_e7_settings[("QUBE", 5, 0)].chunk_list[0].num_wave_words == 48
#     assert gen_e7_settings[("QUBE", 5, 0)].chunk_list[0].num_blank_words == 1232


def test_make_e7_settings(mocker: MockerFixture) -> None:
    mocker.patch(
        "qubecalib.general_looptest_common_mod.BoxPool.get_box",
        return_value=(Box(), None),
    )
    mocker.patch(
        "qubecalib.quel1_wave_subsystem_mod.Quel1WaveSubsystemMod.set_wave",
        return_value=None,
    )
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
    # readout_targets = {
    #     target
    #     for target, subseq in items_by_target.items()
    #     for items in subseq.values()
    #     for item in items
    #     if isinstance(item, Capture)
    # }
    # readout_timings = {
    #     target: {
    #         nodeid: [
    #             (item.begin, item.begin + item.duration)
    #             for item in items
    #             if isinstance(item, Waveform)
    #         ]
    #         for nodeid, items in items_by_target[target].items()
    #     }
    #     for target in readout_targets
    # }
    # readout_timings = {
    #     target: {nodeid: items for nodeid, items in subseqs.items() if items}
    #     for target, subseqs in readout_timings.items()
    # }
    # readout_timings = {
    #     target: subseqs for target, subseqs in readout_timings.items() if subseqs
    # }

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
                "target": {"frequency": 9.8},
            },
            {
                "box": box,
                "port": port0,
                "channel_number": 0,
                "target": {"frequency": 9.8},
            },
        ],
        "RQ01": [
            {
                "box": box,
                "port": port1,
                "channel_number": 1,
                "target": {"frequency": 9.9},
            },
            {
                "box": box,
                "port": port0,
                "channel_number": 0,
                "target": {"frequency": 9.9},
            },
        ],
        "RQ02": [
            {
                "box": box,
                "port": port1,
                "channel_number": 2,
                "target": {"frequency": 10.1},
            },
            {
                "box": box,
                "port": port13,
                "channel_number": 0,
                "target": {"frequency": 10.1},
            },
        ],
        "Q00": [
            {
                "box": box,
                "port": port5,
                "channel_number": 0,
                "target": {"frequency": 10.2},
            },
        ],
    }

    sequencer = Sequencer(
        gen_sampled_sequence,
        cap_sampled_sequence,
        resource_map,
        group_items_by_target=group_items_by_target,
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
    # group_items = (
    #     seq._get_group_items_by_target()
    # )  # target, nodeid of SubSequence 毎に属する item をまとめる
    # assert property_of_slot(group_items["Q00"][1][0]) == (Rectangle, 21, 0.0)
    # assert property_of_slot(group_items["RQ00"][1][0]) == (Rectangle, 9, 21.0)
    # assert property_of_slot(group_items["RQ00"][1][1]) == (Capture, 9, 21.0)
    # assert property_of_slot(group_items["RQ00"][1][2]) == (Rectangle, 121, 40.5)
    # assert property_of_slot(group_items["RQ00"][1][3]) == (Capture, 121, 40.5)
    # assert property_of_slot(group_items["RQ01"][1][0]) == (Rectangle, 31, 161.5)
    # assert property_of_slot(group_items["RQ01"][1][1]) == (Capture, 31, 161.5)
    # assert property_of_slot(group_items["RQ01"][1][2]) == (Rectangle, 51, 192.5)
    # assert property_of_slot(group_items["RQ01"][1][3]) == (Capture, 51, 192.5)

    # assert csseq["RQ00"].sub_sequences[0].prev_blank == 10
    # assert property_of_capture_slot(csseq, "RQ00", 0, 0) == (5, 5)  # Sa
    # assert property_of_capture_slot(csseq, "RQ00", 0, 1) == (60, 41)  # Sa
    # assert csseq["RQ01"].sub_sequences[0].prev_blank == 80  # Sa
    # assert property_of_capture_slot(csseq, "RQ01", 0, 0) == (16, 0)  # Sa
    # assert property_of_capture_slot(csseq, "RQ01", 0, 1) == (25, 0)  # Sa
    # assert csseq["RQ02"].sub_sequences[0].prev_blank == 10  # Sa
    # assert property_of_capture_slot(csseq, "RQ02", 0, 0) == (5, 106)  # Sa

    # E7_RQ00 = cap_e7_settings[("QUBE", 1, 0)]
    # assert E7_RQ00.capture_delay == 32
    # E7_RQ01 = cap_e7_settings[("QUBE", 1, 1)]
    # assert E7_RQ01.capture_delay == 48
    # E7_RQ02 = cap_e7_settings[("QUBE", 1, 2)]
    # assert E7_RQ02.capture_delay == 32

    # assert len(gen_e7_settings[("QUBE", 5, 0)].chunk_list) == 1
    # assert gen_e7_settings[("QUBE", 5, 0)].chunk_list[0].num_words == 1280
    # assert gen_e7_settings[("QUBE", 5, 0)].chunk_list[0].num_wave_words == 48
    # assert gen_e7_settings[("QUBE", 5, 0)].chunk_list[0].num_blank_words == 1232
