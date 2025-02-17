from __future__ import annotations

import math
import warnings
from collections import Counter
from enum import Enum
from typing import MutableMapping

import numpy as np
from e7awgsw import CaptureParam, WaveSequence

from ....instrument.etrees.e7awgsw.e7utils import (
    CaptureParamTools,
    WaveSequenceTools,
    _convert_gen_sampled_sequence_to_blanks_and_waves_chain,
)
from ....neopulse import CapSampledSequence, GenSampledSequence, GenSampledSubSequence
from ....sysconfdb import BoxSetting, PortSetting
from .portconfacq import PortConfigAcquirer


class Sideband(Enum):
    UpperSideBand = "U"
    LowerSideBand = "L"


class Converter:
    @classmethod
    def convert_to_device_specific_sequence(
        cls,
        gen_sampled_sequence: dict[str, GenSampledSequence],
        cap_sampled_sequence: dict[str, CapSampledSequence],
        resource_map: dict[str, dict[str, BoxSetting | PortSetting | int]],
        port_config: dict[str, PortConfigAcquirer],
        repeats: int,
        interval: float,
        integral_mode: str,
        dsp_demodulation: bool,
        software_demodulation: bool,
    ) -> dict[tuple[str, int, int], WaveSequence | CaptureParam]:
        # sampled_sequence と resource_map から e7 データを生成する
        # gen と cap を分離する
        capseq = cls.convert_to_cap_device_specific_sequence(
            gen_sampled_sequence=gen_sampled_sequence,
            cap_sampled_sequence=cap_sampled_sequence,
            resource_map={
                target_name: _
                for target_name, _ in resource_map.items()
                if target_name in cap_sampled_sequence
            },
            port_config={
                target_name: _
                for target_name, _ in port_config.items()
                if target_name in cap_sampled_sequence
            },
            repeats=repeats,
            interval=interval,
            integral_mode=integral_mode,
            dsp_demodulation=dsp_demodulation,
            software_demodulation=software_demodulation,
        )
        genseq = cls.convert_to_gen_device_specific_sequence(
            gen_sampled_sequence=gen_sampled_sequence,
            cap_sampled_sequence=cap_sampled_sequence,
            resource_map={
                target_name: _
                for target_name, _ in resource_map.items()
                if target_name in gen_sampled_sequence
            },
            port_config={
                target_name: _
                for target_name, _ in port_config.items()
                if target_name in gen_sampled_sequence
            },
            repeats=repeats,
            interval=interval,
        )
        return genseq | capseq

    @classmethod
    def convert_to_cap_device_specific_sequence(
        cls,
        gen_sampled_sequence: dict[str, GenSampledSequence],
        cap_sampled_sequence: dict[str, CapSampledSequence],
        resource_map: dict[str, dict[str, BoxSetting | PortSetting | int]],
        port_config: dict[str, PortConfigAcquirer],
        repeats: int,
        interval: float,
        integral_mode: str,
        dsp_demodulation: bool,
        software_demodulation: bool,
    ) -> dict[tuple[str, int, int], CaptureParam]:
        # 線路に起因する遅延
        ndelay_or_nwait_by_target = {
            target_name: _["port"].ndelay_or_nwait[_["channel_number"]]
            if _["port"].ndelay_or_nwait is not None
            else 0
            for target_name, _ in resource_map.items()
        }
        # CaptureParam の生成
        # target 毎の変調周波数の計算
        targets_freqs: MutableMapping[str, float] = {
            target_name: cls.calc_modulation_frequency(
                f_target=_["target"]["frequency"],
                port_config=port_config[target_name],
            )
            for target_name, _ in resource_map.items()
        }
        for target_name, freq in targets_freqs.items():
            cap_sampled_sequence[target_name].modulation_frequency = freq
        # target_name と (box_name, port_number, channel_number) のマップを作成する
        # 1:1 の対応を仮定
        targets_ids = {
            target_name: (_["box"].box_name, _["port"].port, _["channel_number"])
            for target_name, _ in resource_map.items()
        }
        ids_targets = {id: target_name for target_name, id in targets_ids.items()}
        if len(targets_ids) != len(ids_targets):
            raise ValueError("multiple targets are assigned.")

        # ハードウェア復調の場合 channel (unit) に対して単一の target を仮定する
        # ソフトウェア復調の場合は channel 毎にデータを束ねる必要がある TODO 後日実装
        # cap channel と target が 1:1 で対応しているか確認
        if not all(
            [
                _ == 1
                for _ in Counter(
                    [targets_ids[_] for _ in cap_sampled_sequence]
                ).values()
            ]
        ):
            raise ValueError(
                "multiple access for single runit will be supported, not now"
            )
        # 戻り値は {(box_name, port_number, channel_number): CaptureParam} の dict
        sseqs = [seq for seq in cap_sampled_sequence.values()]
        # fps = [padding] + len(sseqs[1:]) * [0]
        # lbs = len(sseqs[:-1]) * [0] + [padding]
        # padding は WaveSequence の長さと合わせるために設けた
        # 原則 WaveSequence は wait = 0 とする
        # TODO Skew の調整はこれから実装する。　wait の単位を確認すること。1Saで調整できるように。
        ids_e7 = {
            targets_ids[sseq.target_name]: CaptureParamTools.create(
                sequence=sseq,
                capture_delay_words=ndelay_or_nwait_by_target[sseq.target_name]
                * 16,  # ndelay は 16 words = 1 block の単位
                repeats=repeats,
                interval_samples=int(interval / sseq.sampling_period),  # samples
            )
            for sseq in sseqs
        }
        if integral_mode == "integral":
            ids_e7 = {
                id: CaptureParamTools.enable_integration(capprm=e7)
                for id, e7 in ids_e7.items()
            }
        if dsp_demodulation:
            ids_e7 = {
                id: CaptureParamTools.enable_demodulation(
                    capprm=e7,
                    f_GHz=targets_freqs[ids_targets[id]],
                )
                for id, e7 in ids_e7.items()
            }
        return ids_e7

    @classmethod
    def convert_to_gen_device_specific_sequence(
        cls,
        gen_sampled_sequence: dict[str, GenSampledSequence],
        cap_sampled_sequence: dict[str, CapSampledSequence],
        resource_map: dict[
            str, dict[str, BoxSetting | PortSetting | int | dict[str, float]]
        ],
        port_config: dict[str, PortConfigAcquirer],
        repeats: int,
        interval: float,
    ) -> dict[tuple[str, int, int], WaveSequence]:
        # for target_name, gss in gen_sampled_sequence.items():
        #     print(target_name, gss.padding, end=" ")
        #     # for sub in gss.sub_sequences:
        #     #     print(sub.padding, end=" ")
        # print()
        # WaveSequence の生成
        SAMPLING_PERIOD = 2

        # target 毎の変調周波数の計算と channel 毎にデータを束ねる
        targets_freqs: MutableMapping[str, float] = {}
        targets_ids: MutableMapping[str, tuple[str, int, int]] = {}
        for target_name, rmap in resource_map.items():
            # target 毎の変調周波数の計算
            rmap_target = rmap["target"]
            if not isinstance(rmap_target, dict):
                raise ValueError("target is not defined")
            targets_freqs[target_name] = cls.calc_modulation_frequency(
                f_target=rmap_target["frequency"],
                port_config=port_config[target_name],
            )

            # targets_freqs = {
            #     target_name: cls.calc_modulation_frequency(
            #         f_target=rmap["target"]["frequency"],
            #         port_config=port_config[target_name],
            #     )
            #     for target_name, rmap in resource_map.items()
            # }

            # channel 毎 (awg 毎) にデータを束ねる
            # target_name と (box_name, port_number, channel_number) のマップを作成する
            rmap_box = rmap["box"]
            if not isinstance(rmap_box, BoxSetting):
                raise ValueError("box is not defined")
            rmap_port = rmap["port"]
            if not isinstance(rmap_port, PortSetting):
                raise ValueError("port is not defined")
            rmap_channel_number = rmap["channel_number"]
            if not isinstance(rmap_channel_number, int):
                raise ValueError("channel_number is not defined")
            targets_ids[target_name] = (
                rmap_box.box_name,
                rmap_port.port,
                rmap_channel_number,
            )

        # readout のタイミングを考慮して位相補償を行う
        for target_name, freq in targets_freqs.items():
            gen_sampled_sequence[target_name].modulation_frequency = freq
        for target, seq in gen_sampled_sequence.items():
            # readout target でない場合はスキップ
            if target not in cap_sampled_sequence:
                continue
            modulation_angular_frequency = 2 * np.pi * targets_freqs[target]
            timing_list = seq.readout_timings
            offset_list = cap_sampled_sequence[target].readin_offsets
            # もし readout_timings と readin_offsets が None なら後方互換でスキップ
            if timing_list is None and offset_list is None:
                continue
            if timing_list is None:
                raise ValueError("readout_timings is not defined")
            if offset_list is None:
                raise ValueError("readin_offsets is not defined")
            for subseq, timings, offsets in zip(
                seq.sub_sequences, timing_list, offset_list
            ):
                wave = subseq.real + 1j * subseq.imag
                for (begin, end), (offsetb, _) in zip(timings, offsets):
                    offset_phase = modulation_angular_frequency * offsetb
                    b = math.floor(begin / SAMPLING_PERIOD)
                    e = math.floor(end / SAMPLING_PERIOD)
                    wave[b:e] = wave[b:e] * np.exp(-1j * offset_phase)
                subseq.real = np.real(wave)
                subseq.imag = np.imag(wave)

        ndelay_or_nwait_by_id = {
            id: next(
                iter(
                    {
                        _["port"].ndelay_or_nwait[_["channel_number"]]
                        for _ in resource_map.values()
                        if _["box"].box_name == id[0]
                        and _["port"].port == id[1]
                        and _["channel_number"] == id[2]
                    }
                )
            )
            for id in targets_ids.values()
        }
        # (box_name, port_number, channel_number) と {target_name: sampled_sequence} とのマップを作成する
        ids_sampled_sequences: dict[
            tuple[str, int, int], dict[str, GenSampledSequence]
        ] = {}
        for target, box_port_channel in targets_ids.items():
            if target in gen_sampled_sequence:
                if box_port_channel not in ids_sampled_sequences:
                    ids_sampled_sequences[box_port_channel] = {}
                ids_sampled_sequences[box_port_channel][target] = gen_sampled_sequence[
                    target
                ]
        # ids_sampled_sequences = {
        #     id: {
        #         _: sampled_sequence[_]
        #         for _, _id in targets_ids.items()
        #         if _id == id and _ in sampled_sequence
        #     }
        #     for id in targets_ids.values()
        # }
        # (box_name, port_number, channel_number) と {target_name: modfreq} とのマップを作成する
        ids_modfreqs = {
            id: {_: targets_freqs[_] for _, _id in targets_ids.items() if _id == id}
            for id in targets_ids.values()
        }
        # 最初の subsequence の先頭に padding 分の 0 を追加する
        # TODO sampling_period を見るようにする
        for target, seq in gen_sampled_sequence.items():
            padding = seq.padding
            subseq = seq.sub_sequences[0]
            subseq.real = np.concatenate([np.zeros(padding), subseq.real])
            subseq.imag = np.concatenate([np.zeros(padding), subseq.imag])
        # ここから全部の subseq で位相回転 (omega(t-t0))しないといけない
        # channel 毎に WaveSequence を生成する
        # 戻り値は {(box_name, port_number, channel_number): WaveSequence} の dict
        # 周波数多重した sampled_sequence を作成する
        ids_muxed_sequences = {
            id: cls.multiplex(
                sequences=ids_sampled_sequences[id],
                modfreqs=ids_modfreqs[id],
            )
            for id in ids_sampled_sequences
        }
        return {
            id: WaveSequenceTools.create(
                sequence=ids_muxed_sequences[id],
                # wait_words=wait_words,
                wait_words=ndelay_or_nwait_by_id[id],
                repeats=repeats,
                interval_samples=int(
                    interval / ids_muxed_sequences[id].sampling_period
                ),
            )
            for id in ids_sampled_sequences
        }

    @classmethod
    def calc_modulation_frequency(
        cls,
        f_target: float,
        port_config: PortConfigAcquirer,
    ) -> float:
        """
        Calculate modulation frequency from target frequency and port configuration.

        Parameters
        ----------
        f_target : float
            Target frequency in GHz.
        port_config : PortConfigAcquirer
            Port configuration.

        Returns
        -------
        float
            Modulation frequency in GHz.
        """
        # Note that port_config has frequencies in Hz.
        f_lo = port_config.lo_freq * 1e-9  # Hz -> GHz
        f_cnco = port_config.cnco_freq * 1e-9  # Hz -> GHz
        f_fnco = port_config.fnco_freq * 1e-9  # Hz -> GHz
        sideband = port_config.sideband
        if sideband == Sideband.UpperSideBand.value:
            f_diff = f_target - f_lo - (f_cnco + f_fnco)
        elif sideband == Sideband.LowerSideBand.value:
            f_diff = -(f_target - f_lo) - (f_cnco + f_fnco)
        else:
            raise ValueError("invalid ssb mode")

        if 0.25 < abs(f_diff):
            p = port_config
            warnings.warn(
                f"Modulation frequency of {p._box_name}:{p._port}:{p._channel} is too high. f_target={f_target} GHz, f_lo={f_lo} GHz, f_cnco={f_cnco} GHz, f_fnco={f_fnco} GHz, sideband={sideband}"
            )

        return f_diff  # GHz

    @classmethod
    def multiplex(
        cls,
        sequences: MutableMapping[str, GenSampledSequence],
        modfreqs: MutableMapping[str, float],
    ) -> GenSampledSequence:
        # sequences は {target_name: GenSampledSequence} の dict だが
        # 基本データを取得するために代表する先頭の sequence を取得する
        # 基本データは全て同じであることを前提とする
        cls.validate_geometry_identity(sequences)
        if not cls.validate_geometry_identity(sequences):
            raise ValueError(
                "All geometry of sub sequences belonging to the same awg must be equal"
            )
        sequence = sequences[next(iter(sequences))]
        padding = sequence.padding

        chain = {
            target_name: _convert_gen_sampled_sequence_to_blanks_and_waves_chain(subseq)
            for target_name, subseq in sequences.items()
        }
        begins = {
            target_name: [
                sum(chain[target_name][: i + 1])
                for i, _ in enumerate(chain[target_name][1:])
            ]
            for target_name in sequences
        }
        # 変調する時間軸を生成する
        # padding 分基準時間をずらす t - t0 処理が加わっている
        SAMPLING_PERIOD = sequence.sampling_period
        times = {
            target_name: [
                (begin + np.arange(subseq.real.shape[0]) - padding) * SAMPLING_PERIOD
                for begin, subseq in zip(
                    begins[target_name][::2],
                    sequence.sub_sequences,
                )
            ]
            for target_name, sequence in sequences.items()
        }
        # 変調および多重化した複素信号
        waves = [
            np.array(
                [
                    (
                        sequences[target].sub_sequences[i].real
                        + 1j * sequences[target].sub_sequences[i].imag
                    )
                    * np.exp(1j * 2 * np.pi * (modfreqs[target] * times[target][i]))
                    for target in sequences
                ]
            ).sum(axis=0)
            for i, _ in enumerate(sequence.sub_sequences)
        ]

        return GenSampledSequence(
            target_name="",
            prev_blank=sequence.prev_blank,
            post_blank=sequence.post_blank,
            repeats=sequence.repeats,
            sampling_period=sequence.sampling_period,
            sub_sequences=[
                GenSampledSubSequence(
                    real=np.real(waves[i]),
                    imag=np.imag(waves[i]),
                    post_blank=subseq.post_blank,
                    repeats=subseq.repeats,
                )
                for i, subseq in enumerate(sequence.sub_sequences)
            ],
        )

    @classmethod
    def validate_geometry_identity(
        cls,
        sequences: MutableMapping[str, GenSampledSequence],
    ) -> bool:
        # 同一 awg から出すすべての信号の GenSampledSequence の波形構造が同一であることを確認する
        _ = {
            target_name: [
                (_.real.shape, _.imag.shape, _.post_blank, _.repeats)
                for _ in sequence.sub_sequences
            ]
            for target_name, sequence in sequences.items()
        }
        if not all(
            sum(
                [[geometry[0] == __ for __ in geometry] for _, geometry in _.items()],
                [],
            )
        ):
            return False
        else:
            return True
