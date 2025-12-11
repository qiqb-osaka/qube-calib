from __future__ import annotations

import json
import uuid
from os.path import expanduser
from typing import Mapping

import numpy as np
from e7awgsw import CaptureParam, WaveSequence
from numpy.typing import NDArray
from quel_ic_config import AwgParam, CapParam, CapSection, WaveChunk
from quel_ic_config_utils import deskew_tools

from .....core.context import ExecutionContext, TaskResult
from .....drivers.quel1.bandref import BandRef
from .....drivers.quel1.deskew_env import Quel1DeskewEnv
from .....drivers.quel1.quel1task import (
    Quel1ClearWaveformTask,
    Quel1PulseCaptureTask,
    Quel1WaveformLoadTask,
    Quel1Waveforms,
)

# Quel1Waveform ?
from .common import AwgId, AwgSetting, RunitId, RunitSetting, TriggerSetting
from .multi import Quel1System


def convert_settings_to_params(
    settings: list[AwgSetting | RunitSetting | TriggerSetting],
) -> tuple[Mapping[BandRef, AwgParam | CapParam], list[Quel1Waveforms]]:
    # TODO Support for TriggerSetting
    commands: dict[BandRef, AwgParam | CapParam] = {}
    waveforms: list[Quel1Waveforms] = []
    for setting in settings:
        if isinstance(setting, AwgSetting):
            awg_param, awg_waveforms = convert_wave_sequence_to_awg_param(setting.wseq)
            band_ref = convert_awg_id_to_band_ref(setting.awg)
            commands[band_ref] = awg_param
            for wave_key, chunk_waveform in awg_waveforms.items():
                waveforms.append(
                    Quel1Waveforms(
                        band=band_ref,
                        wave_key=wave_key,
                        iq=chunk_waveform,
                    )
                )
        elif isinstance(setting, RunitSetting):
            cap_param = convert_capture_param_to_cap_param(setting.cprm)
            band_ref = convert_runit_id_to_band_ref(setting.runit)
            commands[band_ref] = cap_param
        elif isinstance(setting, TriggerSetting):
            Warning("TriggerSetting is not converted to any param.")
            continue
        else:
            raise TypeError(f"unsupported setting type: {type(setting)}")
    return commands, waveforms


def convert_wave_sequence_to_awg_param(
    wave_sequence: WaveSequence,
) -> tuple[AwgParam, dict[str, NDArray[np.complex64]]]:
    prefix = "wave_" + uuid.uuid4().hex
    wseq = wave_sequence
    awg_param = AwgParam(
        num_wait_word=wseq.num_wait_words,
        num_repeat=wseq.num_repeats,
    )
    wave_data_dict: dict[str, NDArray[np.complex64]] = {}
    for i, chunk in enumerate(wseq.chunk_list):
        name_of_wavedata = prefix + f"_chunk{i}"
        awg_param.chunks.append(
            WaveChunk(
                name_of_wavedata=name_of_wavedata,
                num_blank_word=chunk.num_blank_words,
                num_repeat=chunk.num_repeats,
            )
        )
        i_data = np.array([i for i, _ in chunk.wave_data.samples])
        q_data = np.array([q for _, q in chunk.wave_data.samples])
        iq_data = (i_data + 1j * q_data).astype(np.complex64)
        wave_data_dict[name_of_wavedata] = iq_data
    return awg_param, wave_data_dict


def convert_awg_id_to_band_ref(awg_id: AwgId) -> BandRef:
    return BandRef(box_key=awg_id.box, port=awg_id.port, band_key=awg_id.channel)


def convert_capture_param_to_cap_param(capture_param: CaptureParam) -> CapParam:
    prefix = "cap_" + uuid.uuid4().hex
    cprm = capture_param
    cap_param = CapParam(
        num_wait_word=cprm.capture_delay,
        num_repeat=cprm.num_integ_sections,
    )
    for i, sum_section in enumerate(cprm.sum_section_list):
        cap_param.sections.append(
            CapSection(
                name=prefix + f"_section{i}",
                num_capture_word=sum_section[0],
                num_blank_word=sum_section[1],
            )
        )
    return cap_param


def convert_runit_id_to_band_ref(runit_id: RunitId) -> BandRef:
    return BandRef(box_key=runit_id.box, port=runit_id.port, band_key=runit_id.runit)


def run(
    system: Quel1System,
    *,
    band_refs: list[BandRef],
    waveforms: list[Quel1Waveforms],
    commands: Mapping[BandRef, AwgParam | CapParam],
) -> dict[str, TaskResult]:
    for band_ref in band_refs:
        print(f"Running band: {band_ref}")

    clear_task = Quel1ClearWaveformTask(
        task_id="clear_waveforms",
        band_refs=band_refs,
    )
    load_task = Quel1WaveformLoadTask(
        task_id="load_waveforms",
        waveforms=waveforms,
    )
    pulse_capture_task = Quel1PulseCaptureTask(
        task_id="pulse_capture",
        commands=commands,
    )

    ctx = create_execution_context(system)
    clear_result = clear_task.run(context=ctx)
    load_result = load_task.run(context=ctx)
    pulse_capture_result = pulse_capture_task.run(context=ctx)

    return {
        "clear_task": clear_result,
        "load_task": load_result,
        "pulse_capture_task": pulse_capture_result,
    }


def create_execution_context(system: Quel1System) -> ExecutionContext:
    ctx = ExecutionContext()

    ctx.drivers["quel1"] = system._quel1driver

    with open(expanduser("~/.config/quelware/deskew.json")) as file:
        obj = json.load(file)
        deskew_config = deskew_tools.DeskewConfiguration.model_validate(obj)

    ctx.backend_env["quel1.deskew"] = Quel1DeskewEnv(deskew_config)

    return ctx
