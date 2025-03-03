from __future__ import annotations

from typing import Any, Iterable, MutableSequence, Optional, TypedDict

import numpy as np
from e7awgsw import CaptureParam, DspUnit, WaveSequence
from numpy import typing as npt
from quel_ic_config import CaptureModule, CaptureReturnCode, Quel1BoxWithRawWss

from ....command import Command
from ....neopulse import CapSampledSequence, Capture, GenSampledSequence, Slot, Waveform
from ....sysconfdb import BoxSetting, PortSetting, SystemConfigDatabase
from . import driver as direct
from .converter import Converter
from .portconfacq import PortConfigAcquirer
from .system import BoxPool


class TargetBPC(TypedDict):
    box: Quel1BoxWithRawWss
    port: int | tuple[int, int]
    channel: int
    box_name: str


class Sequencer(Command):
    def __init__(
        self,
        gen_sampled_sequence: dict[str, GenSampledSequence],
        cap_sampled_sequence: dict[str, CapSampledSequence],
        resource_map: dict[
            str, Iterable[dict[str, BoxSetting | PortSetting | int | dict[str, Any]]]
        ],
        *,
        sysdb: SystemConfigDatabase,
        time_offset: dict[str, int] = {},
        time_to_start: dict[str, int] = {},
        group_items_by_target: dict[str, dict[int, MutableSequence[Slot]]] = {},
        interval: Optional[float] = None,
    ):
        self.gen_sampled_sequence = gen_sampled_sequence
        self.cap_sampled_sequence = cap_sampled_sequence
        self.group_items_by_terget = group_items_by_target  # TODO ここは begin, end の境界だけわかれば良いので過剰
        # むしろオブジェクトは不要（シリアライズして送るのに面倒）
        self.resource_map = resource_map
        self.syncoffset_by_boxname = time_offset  # taps
        self.timetostart_by_boxname = time_to_start  # sysref
        self.interval = interval

        settings = sysdb._target_settings
        for target_name, gss in gen_sampled_sequence.items():
            if target_name not in settings:
                raise ValueError(f"target({target_name}) is not defined")
            box_names = sysdb.get_boxes_by_target(target_name)
            if not box_names:
                raise ValueError(f"target({target_name}) is not assigned to any box")
            if len(box_names) > 1:
                raise ValueError(f"target({target_name}) is assigned to multiple boxes")
            # tgtset = settings[target_name]
            # skew = tgtset["skew"] if "skew" in tgtset else 0
            box_name = list(box_names)[0]
            skew = sysdb.skew[box_name] if box_name in sysdb.skew else 0
            gss.padding += skew

        # resource_map は以下の形式
        # {
        #   "box": db._box_settings[box_name],
        #   "port": db._port_settings[port_name],
        #   "channel_number": channel_number,
        #   "target": db._target_settings[target_name],
        # }
        self.sysdb = sysdb
        self._sideload_settings: list[
            direct.AwgSetting | direct.RunitSetting | direct.TriggerSetting
        ] = []  # サイドロード用の設定

        # readout の target set を作る
        readout_targets = {
            target
            for target, subseq in group_items_by_target.items()
            for items in subseq.values()
            for item in items
            if isinstance(item, Waveform)
        }
        # readout のタイミング (begin, end) 辞書を作る
        readout_timings: dict[str, MutableSequence[list[tuple[float, float]]]] = {
            target: [
                [
                    (begin, begin + duration)
                    for item in items
                    if isinstance(item, Waveform)
                    if (begin := item.begin) is not None
                    and (duration := item.duration) is not None
                ]
                for items in group_items_by_target[target].values()
            ]
            for target in readout_targets
        }
        # remove empty items
        readout_timings = {
            target: [item for item in items if item]
            for target, items in readout_timings.items()
        }
        # remove empty subseqs
        readout_timings = {
            target: items for target, items in readout_timings.items() if items
        }
        # readout_timings が 空なら後方互換
        if readout_timings:
            for target_name, gseq in gen_sampled_sequence.items():
                # readout_timings に target_name は含まれているはず
                gseq.readout_timings = readout_timings[target_name]

        readin_targets = {
            target
            for target, subseq in group_items_by_target.items()
            for items in subseq.values()
            for item in items
            if isinstance(item, Capture)
        }
        readin_offsets: dict[str, MutableSequence[list[tuple[float, float]]]] = {
            target: [
                [
                    (begin, begin + duration)
                    for item in items
                    if isinstance(item, Capture)
                    if (begin := item.begin) is not None
                    and (duration := item.duration) is not None
                ]
                for nodeid, items in group_items_by_target[target].items()
            ]
            for target in readin_targets
        }
        # remove empty items
        readin_offsets = {
            target: [item for item in items if item]
            for target, items in readin_offsets.items()
        }
        # remove empty subseqs
        readin_offsets = {
            target: items for target, items in readin_offsets.items() if items
        }
        if readin_offsets:
            for target_name, cseq in cap_sampled_sequence.items():
                cseq.readin_offsets = readin_offsets[target_name]

    def set_measurement_option(
        self,
        repeats: int,
        interval: float,
        integral_mode: str,
        dsp_demodulation: bool = True,
        software_demodulation: bool = False,
        phase_compensation: bool = True,  # TODO not work
    ) -> None:
        self.repeats = repeats
        self.interval = interval
        self.integral_mode = integral_mode
        self.dsp_demodulation = dsp_demodulation
        self.software_demodulation = software_demodulation
        self.phase_compensation = phase_compensation

    def generate_cap_resource_map(self, boxpool: BoxPool) -> dict[str, Any]:
        _cap_resource_map: dict[str, MutableSequence[dict[str, Any]]] = {}
        for target_name, ms in self.resource_map.items():
            for m in ms:
                if isinstance(m["box"], BoxSetting):
                    box_name = m["box"].box_name
                else:
                    raise ValueError("box_name is not defined")
                if isinstance(m["port"], PortSetting):
                    port = m["port"].port
                else:
                    raise ValueError("port is not defined")
                if (
                    boxpool.get_port_direction(box_name, port) == "in"
                    and target_name in self.cap_sampled_sequence
                ):
                    if target_name in _cap_resource_map:
                        _cap_resource_map[target_name].append(m)
                    else:
                        _cap_resource_map[target_name] = [m]
        return {
            target_name: next(iter(maps))
            for target_name, maps in _cap_resource_map.items()
            if maps
        }

    def calc_first_padding(self) -> int:
        csseq = self.cap_sampled_sequence
        first_blank = min(
            [seq.prev_blank for sseq in csseq.values() for seq in sseq.sub_sequences]
        )
        return ((first_blank - 1) // 64 + 1) * 64 - first_blank  # Sa

    def generate_e7_settings(
        self,
        boxpool: BoxPool,
    ) -> tuple[
        dict[tuple[str, int, int], CaptureParam],
        dict[tuple[str, int, int], WaveSequence],
        dict[str, Any],
    ]:
        # cap 用の cap_e7_setting と gen 用の gen_e7setting を作る
        cap_resource_map = self.generate_cap_resource_map(boxpool)
        _gen_resource_map: dict[str, MutableSequence[dict[str, Any]]] = {}
        for target_name, ms in self.resource_map.items():
            for m in ms:
                if isinstance(m["box"], BoxSetting):
                    box_name = m["box"].box_name
                else:
                    raise ValueError("box_name is not defined")
                if isinstance(m["port"], PortSetting):
                    port = m["port"].port
                else:
                    raise ValueError("port is not defined")
                if (
                    boxpool.get_port_direction(box_name, port) == "out"
                    and target_name in self.gen_sampled_sequence
                ):
                    if target_name in _gen_resource_map:
                        _gen_resource_map[target_name].append(m)
                    else:
                        _gen_resource_map[target_name] = [m]
        gen_resource_map: dict[str, Any] = {
            target_name: next(iter(maps))
            for target_name, maps in _gen_resource_map.items()
            if maps
        }

        # TODO ここで caps や gens が二つ以上だとエラーを出すこと
        # e7 の生成に必要な lo_hz などをまとめた辞書を作る
        cap_target_bpc: dict[str, TargetBPC] = {
            target_name: TargetBPC(
                box=boxpool.get_box(m["box"].box_name)[0],
                port=m["port"].port if isinstance(m["port"], PortSetting) else 0,
                channel=m["channel_number"],
                box_name=m["box"].box_name,
            )
            for target_name, m in cap_resource_map.items()
        }
        gen_target_bpc: dict[str, TargetBPC] = {
            target_name: TargetBPC(
                box=boxpool.get_box(m["box"].box_name)[0],
                port=m["port"].port if isinstance(m["port"], PortSetting) else 0,
                channel=m["channel_number"],
                box_name=m["box"].box_name,
            )
            for target_name, m in gen_resource_map.items()
        }
        cap_target_portconf = {
            target_name: PortConfigAcquirer(
                boxpool=boxpool,
                box_name=m["box_name"],
                box=m["box"],
                port=m["port"],
                channel=m["channel"],
            )
            for target_name, m in cap_target_bpc.items()
        }

        # first_blank = min(
        #     [seq.prev_blank for sseq in csseq.values() for seq in sseq.sub_sequences]
        # )
        # first_padding = ((first_blank - 1) // 64 + 1) * 64 - first_blank  # Sa
        # ref_sequence = next(iter(csseq.values()))
        first_padding = self.calc_first_padding()

        for target_name, cseq in self.cap_sampled_sequence.items():
            cseq.padding += first_padding
        for target_name, gseq in self.gen_sampled_sequence.items():
            gseq.padding += first_padding

        interval = self.interval if self.interval is not None else 10240
        cap_e7_settings: dict[tuple[str, int, int], CaptureParam] = (
            Converter.convert_to_cap_device_specific_sequence(
                gen_sampled_sequence=self.gen_sampled_sequence,
                cap_sampled_sequence=self.cap_sampled_sequence,
                resource_map=cap_resource_map,
                # target_freq=target_freq,
                port_config=cap_target_portconf,
                repeats=self.repeats,
                interval=interval,
                integral_mode=self.integral_mode,
                dsp_demodulation=self.dsp_demodulation,
                software_demodulation=self.software_demodulation,
            )
        )
        # phase_offset_list_by_target = {
        #     target: [-2 * np.pi * cap_fmod[target] * t for t in reference_time_list]
        #     for target, reference_time_list in reference_time_list_by_target.items()
        # }

        gen_target_portconf = {
            target_name: PortConfigAcquirer(
                boxpool=boxpool,
                box_name=m["box_name"],
                box=m["box"],
                port=m["port"],
                channel=m["channel"],
            )
            for target_name, m in gen_target_bpc.items()
        }
        gen_e7_settings: dict[tuple[str, int, int], WaveSequence] = (
            Converter.convert_to_gen_device_specific_sequence(
                gen_sampled_sequence=self.gen_sampled_sequence,
                cap_sampled_sequence=self.cap_sampled_sequence,
                resource_map=gen_resource_map,
                port_config=gen_target_portconf,
                repeats=self.repeats,
                interval=interval,
            )
        )
        return cap_e7_settings, gen_e7_settings, cap_resource_map

    def execute(
        self,
        boxpool: BoxPool,
    ) -> tuple[dict[str, CaptureReturnCode], dict[str, list], dict]:
        quel1system = self.create_quel1system(boxpool)
        c, g, m = self.generate_e7_settings(boxpool)

        settings: list[
            direct.RunitSetting | direct.AwgSetting | direct.TriggerSetting
        ] = []
        for (name, port, runit), cprm in c.items():
            settings.append(
                direct.RunitSetting(
                    runit=direct.RunitId(
                        box=name,
                        port=port,
                        runit=runit,
                    ),
                    cprm=cprm,
                )
            )
        for (name, port, channel), wseq in g.items():
            settings.append(
                direct.AwgSetting(
                    awg=direct.AwgId(
                        box=name,
                        port=port,
                        channel=channel,
                    ),
                    wseq=wseq,
                )
            )
        settings += self.select_trigger(quel1system, settings)
        if len(settings) == 0:
            raise ValueError("no settings")

        if self._sideload_settings:
            action = direct.Action.build(
                system=quel1system, settings=self._sideload_settings
            )
        else:
            action = direct.Action.build(system=quel1system, settings=settings)
        status, results = action.action()
        return self.parse_capture_results(status, results, action, m)

    def parse_capture_results(
        self,
        status: dict[tuple[str, int], CaptureReturnCode],
        results: dict[tuple[str, int, int], npt.NDArray[np.complex64]],
        action: direct.Action,
        crmap: dict[str, Any],
    ) -> tuple[dict[str, CaptureReturnCode], dict[str, list], dict]:
        bpc2target = {}
        for target, m in crmap.items():
            box, port, channel = m["box"].box_name, m["port"].port, m["channel_number"]
            bpc2target[(box, port, channel)] = target
        # status = {}
        # for (box, port), code in status.items():
        #     status[(box, port)] = code
        data = {}
        for (box, port, runit), datum in results.items():
            data[(box, port, runit)] = datum
        cprms = {}
        if isinstance(action._action, direct.multi.Action):
            for box, act in action._action._actions.items():
                for runit_id, cprm in act._cprms.items():
                    cprms[(box, runit_id.port, runit_id.runit)] = cprm
        elif isinstance(action._action, tuple):
            box, act = action._action
            for runit_id, cprm in act._cprms.items():
                cprms[(box, runit_id.port, runit_id.runit)] = cprm
        rstatus, rresults = {}, {}
        for (box, port, runit), target in bpc2target.items():
            s, r = self.parse_capture_result(
                status[(box, port)],
                data[(box, port, runit)],
                cprms[(box, port, runit)],
            )
            target = bpc2target[(box, port, runit)]
            rstatus[target] = s
            rresults[target] = r
        return rstatus, rresults, {}

    def parse_capture_result(
        self,
        status: CaptureReturnCode,
        data: npt.NDArray[np.complex64],
        cprm: CaptureParam,
    ) -> tuple[CaptureReturnCode, list[npt.NDArray[np.complex64]]]:
        # num_expected_words = cprm.calc_capture_samples()
        if DspUnit.INTEGRATION in cprm.dsp_units_enabled:
            data = data.reshape(1, -1)
        else:
            data = data.reshape(cprm.num_integ_sections, -1)
        if DspUnit.SUM in cprm.dsp_units_enabled:
            width = list(range(len(cprm.sum_section_list))[1:])
            result = np.hsplit(data, width)
        else:
            b = DspUnit.SUM not in cprm.dsp_units_enabled
            ssl = cprm.sum_section_list
            ws = [w if b else int(w / 4) for w in ssl[:-1]]
            word = cprm.NUM_SAMPLES_IN_ADC_WORD
            width = np.cumsum(np.array(ws))
            c = np.hsplit(data, width * word)
            result = [di.transpose() for di in c]
        return status, result

    def create_quel1system(self, boxpool: BoxPool) -> direct.Quel1System:
        quel1system = direct.Quel1System.create(
            clockmaster=boxpool._clock_master,
            boxes=[
                direct.NamedBox(
                    name,
                    box,
                )
                for name, (box, _) in boxpool._boxes.items()
            ],
        )
        quel1system.trigger = self.sysdb.trigger
        for box_name, timing_shift in self.sysdb.timing_shift.items():
            quel1system.timing_shift[box_name] = timing_shift
        quel1system.displacement = self.sysdb.time_to_start
        return quel1system

    def convert(
        self,
        cap_e7_settings: dict[tuple[str, int, int], CaptureParam],
        gen_e7_settings: dict[tuple[str, int, int], WaveSequence],
    ) -> list[direct.AwgSetting | direct.RunitSetting | direct.TriggerSetting]:
        settings: list[
            direct.AwgSetting | direct.RunitSetting | direct.TriggerSetting
        ] = []
        for (box_name, port, runit), e7 in cap_e7_settings.items():
            settings.append(
                direct.RunitSetting(
                    runit=direct.RunitId(box=box_name, port=port, runit=runit),
                    cprm=e7,
                )
            )
        for (box_name, port, channel), e7 in gen_e7_settings.items():
            settings.append(
                direct.AwgSetting(
                    awg=direct.AwgId(box=box_name, port=port, channel=channel),
                    wseq=e7,
                )
            )
        return settings

    @staticmethod
    def is_empty_trigger(
        settings: list[direct.AwgSetting | direct.RunitSetting | direct.TriggerSetting],
    ) -> bool:
        for s in settings:
            if isinstance(s, direct.TriggerSetting):
                return False
        return True

    def select_trigger(
        self,
        quel1system: direct.Quel1System,
        settings: list[direct.AwgSetting | direct.RunitSetting | direct.TriggerSetting],
    ) -> list[direct.TriggerSetting]:
        if not self.is_empty_trigger(settings):
            raise ValueError("trigger is already set")

        # トリガを自動で設定する
        result: list[direct.TriggerSetting] = []
        caps: list[tuple[int, direct.RunitId]] = []
        gens: list[tuple[int, direct.AwgId]] = []
        for setting in settings:
            if isinstance(setting, direct.RunitSetting):
                # 右肺か左肺かの情報を付加して runit の設定を抽出する
                box = quel1system.box[setting.runit.box]
                port, subport = box._decode_port(setting.runit.port)
                group, rline = box._convert_any_port(port)
                # capmod = box.rmap.get_capture_module_of_rline(group, rline)
                caps.append((group, setting.runit))
            elif isinstance(setting, direct.AwgSetting):
                # 右肺か左肺かの情報を付加して awg の設定を抽出する
                box = quel1system.box[setting.awg.box]
                port, subport = box._decode_port(setting.awg.port)
                group, rline = box._convert_any_port(port)
                gens.append((group, setting.awg))
        # もし quel1system に明示的に trigger が設定されているならそれを使う
        defined_awgs = [s.awg for s in settings if isinstance(s, direct.AwgSetting)]
        for runit_group, runit_id in caps:
            if (runit_id.box, runit_id.port) in quel1system.trigger:
                trig_name, trig_nport, trig_nchannel = quel1system.trigger[
                    (runit_id.box, runit_id.port)
                ]
                awg = direct.AwgId(
                    box=trig_name, port=trig_nport, channel=trig_nchannel
                )
                if runit_id.box != trig_name:
                    raise ValueError(
                        f"invalid trigger {runit_id.box, runit_id.port} for {trig_name, trig_nport, trig_nchannel}"
                    )
                if awg not in defined_awgs:
                    raise ValueError(
                        f"trigger {trig_name, trig_nport, trig_nchannel} not found in settings"
                    )
                result.append(
                    direct.TriggerSetting(
                        triggerd_port=runit_id.port,
                        trigger_awg=awg,
                    )
                )
        # もし capture のみあるいは awgs のみなら tigger は設定しない
        if all([bool(caps), not bool(gens)]) or all([not bool(caps), bool(gens)]):
            return result
        pre_defined_triggers = {(s.trigger_awg.box, s.triggerd_port) for s in result}
        for runit_group, runit_id in caps:
            if (runit_id.box, runit_id.port) in pre_defined_triggers:
                continue  # もしすでに trigger が設定されているならスキップ
            for awg_group, awg_id in gens:
                if runit_id.box == awg_id.box and runit_group == awg_group:
                    result.append(
                        direct.TriggerSetting(
                            triggerd_port=runit_id.port,
                            trigger_awg=awg_id,
                        )
                    )
                    break
            # 別の group の trigger を割り当てるようにチャレンジ
            for awg_group, awg_id in gens:
                if runit_id.box == awg_id.box:
                    result.append(
                        direct.TriggerSetting(
                            triggerd_port=runit_id.port,
                            trigger_awg=awg_id,
                        )
                    )
                    break
            else:
                raise ValueError("invalid trigger")
        return result

    @classmethod
    def convert_key_from_bmu_to_target(
        cls,
        bmc_target: dict[tuple[Optional[str], CaptureModule, Optional[int]], str],
        status: dict[tuple[str, CaptureModule], CaptureReturnCode],
        iqs: dict[tuple[str, CaptureModule], dict[int, list]],
    ) -> tuple[dict[str, CaptureReturnCode], dict[str, list]]:
        _iqs = {
            bmc_target[(box_name, capm, capu)]: __iqs
            for (box_name, capm), _ in iqs.items()
            for capu, __iqs in _.items()
        }
        _status = {
            bmc_target[(box_name, capm, capu)]: status[(box_name, capm)]
            for (box_name, capm), _ in iqs.items()
            for capu in _
        }

        # sort keys of iqs by target name
        sorted_iqs = {key: _iqs[key] for key in sorted(_iqs)}

        return _status, sorted_iqs
