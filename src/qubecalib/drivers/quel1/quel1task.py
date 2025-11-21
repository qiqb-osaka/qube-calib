from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from quel_ic_config import AwgParam, CapParam, Quel1PortType
from quel_ic_config_utils import deskew_tools

from ...core.task import ConcreteTaskBase, TemplateTaskBase
from .bandref import BandRef
from .deskew_env import Quel1DeskewEnv
from .quel1driver import Quel1Driver


@dataclass
class BoxAdjustParam:
    pass


@dataclass
class BandParam:
    pass


@dataclass
class Quel1PulseCaptureTemplate(TemplateTaskBase):
    pass


@dataclass
class Quel1ClearWaveformTask(ConcreteTaskBase):
    """Quel1Box 上の波形データを全てクリアする Task"""

    device_key: str = "quel1"
    band_refs: list[BandRef] = field(default_factory=list)

    def execute(self) -> Any:
        drv = self.driver()
        if not isinstance(drv, Quel1Driver):
            raise TypeError("Driver must be an instance of Quel1Driver.")

        for band in self.band_refs:
            box = drv.box(band.box_key)
            port = band.port
            band_key = band.band_key
            for name in box.get_names_of_wavedata(port, band_key):
                if name == "null":
                    continue
                box.delete_wavedata(port, band_key, name)
        return {"cleared_boxes": {band.box_key for band in self.band_refs}}


@dataclass
class Quel1Waveforms:
    band: BandRef
    wave_key: str
    iq: NDArray[np.complex64]


@dataclass
class Quel1WaveformLoadTask(ConcreteTaskBase):
    """regiseter_wavedata によって波形を Box にロードする Task"""

    device_key: str = "quel1"
    waveforms: list[Quel1Waveforms] = field(default_factory=list)

    def execute(self) -> Any:
        drv = self.driver()
        if not isinstance(drv, Quel1Driver):
            raise TypeError("Driver must be an instance of Quel1Driver.")

        for wf in self.waveforms:
            band = wf.band
            box = drv.box(band.box_key)
            box.register_wavedata(
                port=band.port,
                channel=band.band_key,
                name=wf.wave_key,
                iq=wf.iq.astype(np.complex64, copy=False),
            )
        return {"loaded": [(wf.band, wf.wave_key) for wf in self.waveforms]}


@dataclass
class Quel1PulseCaptureTask(ConcreteTaskBase):
    device_key: str = "quel1"
    box_adjust: dict[str, BoxAdjustParam] = field(default_factory=dict)
    commands: Mapping[BandRef, CapParam | AwgParam] = field(default_factory=dict)

    delay_sec: float = 0.15

    def execute(self) -> Any:
        """
        複数 Quel1Box 上の Awg/Cap に対し：
        1) DeskewConfig に基づいて AwgParam / CapParam を補正
        2) config_channel / config_runit で設定
        3) count_proposer による共通トリガで同期実行
        4) CAP の結果を BandRef -> wave_dict の dict として返す
        """
        drv = self.driver()
        if not isinstance(drv, Quel1Driver):
            raise TypeError("Driver must be an instance of Quel1Driver.")

        # --- 0) Context から deskew env を取得
        ctx = self._current_context
        if ctx is None:
            raise RuntimeError("ExecutionContext is not set for Quel1PulseCaptureTask.")

        deskew_env = ctx.get_backend_env("quel1.deskew")
        if not isinstance(deskew_env, Quel1DeskewEnv):
            raise TypeError("backend_env['quel1.deskew'] must be Quel1DeskewEnv.")

        wait_amount_resolver = deskew_env.wait_amount_resolver
        delay_compensator = deskew_env.delay_compensator
        count_proposer = deskew_env.count_proposer

        # --- 1) Box ごとに AwgParam / CapParam を仕分け ---
        box_to_awg: dict[str, list[tuple[BandRef, AwgParam]]] = {}
        box_to_cap: dict[str, list[tuple[BandRef, CapParam]]] = {}

        for band, command in self.commands.items():
            if isinstance(command, AwgParam):
                box_to_awg.setdefault(band.box_key, []).append((band, command))
            elif isinstance(command, CapParam):
                box_to_cap.setdefault(band.box_key, []).append((band, command))
            else:
                raise TypeError(
                    f"Unsupported command type for band {band}: {type(command)}"
                )

        if not box_to_awg and not box_to_cap:
            raise ValueError("No commands are specified for Quel1PulseCaptureTask.")

        target_boxes = sorted(set(box_to_awg.keys()) | set(box_to_cap.keys()))
        if not target_boxes:
            raise ValueError("No target boxes found from commands.")

        # --- 2) Deskew 補正をした上で config_* に流し込む ---
        for box_name in target_boxes:
            box = drv.box(box_name)

            # AwgParam 補正と設定
            for band, awg_param in box_to_awg.get(box_name, []):
                port = band.port
                deskew_tools.register_blank_wavedata(box, port, band.band_key)
                word_to_wait = wait_amount_resolver.get_word_to_wait(box_name, port)
                adjusted_awg = delay_compensator.adjust_awg_param(
                    awg_param, word_to_wait
                )
                box.config_channel(
                    port=band.port,
                    channel=band.band_key,
                    awg_param=adjusted_awg,
                )

            # CapParam 補正と設定
            for band, cap_param in box_to_cap.get(box_name, []):
                port = band.port
                word_to_wait = wait_amount_resolver.get_word_to_wait(box_name, port)
                adjusted_cap = delay_compensator.adjust_cap_param(
                    cap_param, word_to_wait
                )
                box.config_runit(
                    port=band.port,
                    runit=band.band_key,
                    capture_param=adjusted_cap,
                )

        # --- 4-1) gentask と captask を整理
        box_to_channels: dict[str, set[tuple[Quel1PortType, int]]] = {}
        box_to_runits: dict[str, set[tuple[Quel1PortType, int]]] = {}

        for box_name in target_boxes:
            awg_bands = box_to_awg.get(box_name, [])
            cap_bands = box_to_cap.get(box_name, [])
            box_to_channels[box_name] = {
                (band.port, band.band_key) for band, _ in awg_bands
            }
            box_to_runits[box_name] = {
                (band.port, band.band_key) for band, _ in cap_bands
            }

        # --- 3) count_proposer で各 Box のトリガカウンタを決定 ---
        first_box = target_boxes[0]
        t0 = drv.box(first_box).get_current_timecounter()
        trigger_counts = count_proposer.propose_trigger_counts(
            t0,
            target_boxes,
            delay_sec=self.delay_sec,
        )

        # --- 4-2) 各 Box で wavegen / capture を起動 ---
        name_to_gentask: dict[str, Any] = {}
        name_to_captask: dict[str, Any] = {}

        for box_name in target_boxes:
            box = drv.box(box_name)
            tc = trigger_counts[box_name]
            channels = box_to_channels.get(box_name)
            runits = box_to_runits.get(box_name)

            if channels and runits:
                captask, gentask = box.start_capture_by_awg_trigger(
                    runits=box_to_runits[box_name],
                    channels=box_to_channels[box_name],
                    timecounter=tc,
                )
                name_to_captask[box_name] = captask
                name_to_gentask[box_name] = gentask
            elif channels and not runits:
                gentask = box.start_wavegen(
                    channels=box_to_channels[box_name],
                    timecounter=tc,
                )
                name_to_gentask[box_name] = gentask
            elif runits and not channels:
                capnowtask = box.start_capture_now(
                    runits=runits,
                )
                name_to_captask[box_name] = capnowtask

        # --- 5) Wavegen 側の完了を待つ ---
        for box_name, gentask in name_to_gentask.items():
            gentask.result()

        # --- 6) Capture 側の結果を集約し， BandRef -> wave_dict にして返す ---
        wave_dict: dict[BandRef, dict[str, NDArray[np.complex64]]] = {}

        for box_name, captask in name_to_captask.items():
            iqs_ri_readers = captask.result()  # {(port, runit): readers}

            for (port, band_key), readers in iqs_ri_readers.items():
                band = BandRef(box_key=box_name, port=port, band_key=band_key)
                wave_dict[band] = readers.as_wave_dict()

        return wave_dict

        # for band, command in self.commands.items():
        #     if isinstance(command, AwgParam):
        #         drv.box(band.box_key).config_channel(
        #             port=band.port,
        #             channel=band.band_key,
        #             awg_param=cast(AwgParam, command),
        #         )
        #     elif isinstance(command, CapParam):
        #         drv.box(band.box_key).config_runit(
        #             port=band.port,
        #             runit=band.band_key,
        #             capture_param=cast(CapParam, command),
        #         )
        #     else:
        #         raise TypeError(
        #             f"Unsupported command type for band {band}: {type(command)}"
        #         )

        # boxes = {band.box_key for band in self.commands.keys()}

        # if len(boxes) == 1:
        #     box = drv.box(next(iter(boxes)))
        #     runits = {
        #         (band.port, band.band_key)
        #         for band, command in self.commands.items()
        #         if band.box_key == box.name
        #         if isinstance(command, CapParam)
        #     }
        #     channels = [
        #         (band.port, band.band_key)
        #         for band, command in self.commands.items()
        #         if band.box_key == box.name
        #         if isinstance(command, AwgParam)
        #     ]
        #     cur = box.get_current_timecounter()
        #     thunk_ri, thunk_ro = box.start_capture_by_awg_trigger(
        #         runits=runits,
        #         channels=channels,
        #         timecounter=cur + 125_000_000 // 10,
        #     )
        #     thunk_ro.result()
        #     iqs_ri_readers = thunk_ri.result()
        #     wave_dict = {
        #         BandRef(
        #             box_key=box.name, port=port, band_key=band_key
        #         ): readers.as_wave_dict()
        #         for (port, band_key), readers in iqs_ri_readers.items()
        #     }
        #     return wave_dict
        # else:
        #     pass
