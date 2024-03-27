from __future__ import annotations

import math
from typing import Any, MutableMapping, MutableSequence

import numpy as np

# from dataclasses import dataclass
from e7awgsw import CaptureParam, IqWave, WaveSequence

from .neopulse import CapSampledSequence, GenSampledSequence

# import json


class WaveSequenceTools:
    @classmethod
    def quantize_duration(
        cls,
        duration: float,
        constrain: float = 10_240e-9,
    ) -> float:
        # nco 達をデータフレームの長さと同期させると位相が揃う
        # 15.625MHz の整数倍/割にすると良い
        # 15.625MHz(64ns) の1/160 97.65625kHz(10_240ns) を繰り返し周期にすると位相が揃う
        return duration // constrain * constrain

    @classmethod
    def validate_e7_compatibility(
        cls,
        sequences: MutableMapping[str, GenSampledSequence],
    ) -> bool:
        # wait word: 0 - 4,294,967,295 awg words 32 bits 34.359_758_360 sec
        # wave chunk 1 ~ 16 chunks max 4,294,967,295 repeats 32 bits
        # post blank 0 - 4,294,967,295 awg words 32 bits
        # duration of wave part -> N x 16 awg words
        # total length of wave part -> 1,677,216 awg words 24 bits 13.417728 ms
        return False

    @classmethod
    def create_chunk(cls) -> MutableMapping[str, Any]:
        return {}

    @classmethod
    def create(
        cls,
        sequence: GenSampledSequence,
    ) -> WaveSequence:
        # 同一 awg に所属する target を束ねた WaveSequence を生成する
        # TODO もし sequence の geometry が e7 compatible だった場合 wave_chunk を活用して変換する
        # そうでなかったらすべての subseq を 単一の wave_chunk に入れる
        return cls.create_single_chunked_wave_sequence(sequence)
        # # アライメント境界を生成
        # unit = WaveSequence.NUM_SAMPLES_IN_AWG_WORD * 16
        # grid = [math.floor(_ / unit) * unit for _ in bounds]
        # lower = [
        #     align - unit if bound < align else align
        #     for bound, align in zip(bounds[1::2], grid[1::2])
        # ]
        # upper = [
        #     align + unit if bound > align else align
        #     for bound, align in zip(bounds[2::2], grid[2::2])
        # ]
        # # print(bounds[1::2], aligns[1::2], lower)
        # # print(bounds[2::2], aligns[2::2], upper)
        # aligned: MutableSequence = sum(
        #     [[bounds[0]]] + [[low, up] for low, up in zip(lower, upper)],
        #     [],
        # )
        # # print(aligned)
        # new_chain = [
        #     int((up - low) / unit) if low is not None and up is not None else None
        #     for low, up in zip(aligned[:-1], aligned[1:])
        # ] + [chain[-1]]
        # print(first_target_name, new_chain)
        # # TODO new_chain の blank 要素が潰れることがある（capt は拡張なので潰れない）この処理を追加しないといけない
        # # TODO 最終の post_blank の処理をまだ入れていない
        # # TODO post_blank は 1 以上の制約がある

        # # WaveSequence の制約を満たすようにする

        # wseq = WaveSequence(
        #     num_wait_words=0,
        #     num_repeats=1,
        #     # num_wait_words=seq.prev_blank,
        #     # num_repeats=seq.repeats if seq.repeats is not None else 1,
        # )
        # # wseq.add_chunk(
        # #     iq_samples=,
        # #     num_blank_words=,
        # #     num_repeats=,
        # # )

        # return wseq

    @classmethod
    def create_single_chunked_wave_sequence(
        cls,
        sequence: GenSampledSequence,
    ) -> WaveSequence:
        """単一の WaveChunk にすべての iq をまとめた WaveSequence を作る"""
        # 市松模様チェーンを生成
        chain = _convert_gen_sampled_sequence_to_blanks_and_waves_chain(sequence)
        # print(chain)
        # 境界を抽出
        bounds = [sum(chain[: i + 1]) for i, _ in enumerate(chain)]
        # print(bounds)
        i, q = np.zeros(bounds[-1]).astype(int), np.zeros(bounds[-1]).astype(int)
        for begin, subseq in zip(bounds[::2], sequence.sub_sequences):
            if max(np.abs(subseq.real)) > 1 or max(np.abs(subseq.imag)) > 1:
                raise ValueError("magnitude of iq signal must not exceed 1")
            i[begin : begin + subseq.real.shape[0]] = (32767 * subseq.real).astype(int)
            q[begin : begin + subseq.imag.shape[0]] = (32767 * subseq.imag).astype(int)

        wseq = WaveSequence(
            num_wait_words=0,
            num_repeats=sequence.repeats if sequence.repeats is not None else 1,
        )

        s = IqWave.convert_to_iq_format(i, q, WaveSequence.NUM_SAMPLES_IN_WAVE_BLOCK)
        wseq.add_chunk(
            iq_samples=s,
            num_blank_words=0,
            num_repeats=1,
        )

        # r = e7awgsw.AwgCtrl.SAMPLING_RATE
        # blank = self.num_blank_words * WORDs
        # b = int(r * blank * 1e-9)
        # n = WaveSequence.NUM_SAMPLES_IN_AWG_WORD

        # return {
        #     "iq_samples": s,
        #     "num_blank_words": b // n,
        #     "num_repeats": 1,
        # }

        return wseq


class CaptureParamTools:
    @classmethod
    def create(cls, sequence: CapSampledSequence) -> CaptureParam:
        unit = CaptureParam.NUM_SAMPLES_IN_ADC_WORD
        # print(sequence)
        # 市松模様チェーンを生成
        chain = _convert_cap_sampled_sequence_to_blanks_and_durations_chain(sequence)
        # print(chain)
        # 境界を抽出
        bounds = [sum(chain[:i]) for i, _ in enumerate(chain)]
        # print(bounds)
        # アライメント境界を生成
        grid = [math.floor(_ / unit) * unit for _ in bounds]
        lower = [
            align - unit if bound < align else align
            for bound, align in zip(bounds[1::2], grid[1::2])
        ]
        upper = [
            align + unit if bound > align else align
            for bound, align in zip(bounds[2::2], grid[2::2])
        ]
        # print(bounds[1::2], aligns[1::2], lower)
        # print(bounds[2::2], aligns[2::2], upper)
        aligned: MutableSequence = sum(
            [[bounds[0]]] + [[low, up] for low, up in zip(lower, upper)],
            [],
        )
        # print(aligned)
        new_chain = [
            int((up - low) / unit) if low is not None and up is not None else None
            for low, up in zip(aligned[:-1], aligned[1:])
        ] + [chain[-1]]
        # TODO new_chain の blank 要素が潰れることがある（capt は拡張なので潰れない）この処理を追加しないといけない
        # TODO 最終の post_blank の処理をまだ入れていない
        # TODO post_blank は 1 以上の制約がある
        capprm = CaptureParam()
        capprm.capture_delay = int(new_chain[0] / unit)
        for duration, blank in zip(new_chain[1::2], new_chain[2::2]):
            capprm.add_sum_section(
                num_words=duration,
                num_post_blank_words=blank if blank is not None else 1,
            )
        return capprm


def _convert_gen_sampled_sequence_to_blanks_and_waves_chain(
    sequence: GenSampledSequence,
) -> MutableSequence:
    # return sum(
    #     # 先頭につく blank
    #     [[sequence.prev_blank]]
    #     # 最終以外の sub sequence の wave, blank
    #     + sum(
    #         # 最終以外の subseq の 最終以外の slot の wave, blank chain
    #         [
    #             [
    #                 subseq.real.shape[0],
    #                 subseq.post_blank if subseq.post_blank is not None else 0,
    #             ]
    #             for subseq in sequence.sub_sequences[:-1]
    #         ],
    #         [],
    #     ),
    #     # お尻につく wave, blank
    #     [],
    # )
    return sum(
        [[sequence.prev_blank]]
        + [[_.real.shape[0], _.post_blank] for _ in sequence.sub_sequences[:-1]]
        + [
            [
                sequence.sub_sequences[-1].real.shape[0],
                sequence.sub_sequences[-1].post_blank + sequence.post_blank
                if sequence.post_blank is not None
                and sequence.sub_sequences[-1].post_blank is not None
                else 0,
            ]
        ],
        [],
    )


def _convert_cap_sampled_sequence_to_blanks_and_durations_chain(
    sequence: CapSampledSequence,
) -> MutableSequence:
    """cap sampled sequence を blank - duration チェーンに変換する"""
    seq = sequence

    # sub sequence 間を橋渡しするブランクのリスト MutableSequence[Optional[int|float]]
    # 最終スロットの blank, 前 subseq の post_blank, 後 subseq の prev_blank
    blank_bridges = [
        lo.capture_slots[-1].post_blank + lo.post_blank + hi.prev_blank
        if lo.capture_slots[-1].post_blank is not None
        and lo.post_blank is not None
        and hi.prev_blank is not None
        else None
        for lo, hi in zip(seq.sub_sequences[:-1], seq.sub_sequences[1:])
    ]

    # お尻につく blank
    last_blank = (
        seq.sub_sequences[-1].capture_slots[-1].post_blank
        + seq.sub_sequences[-1].post_blank
        + seq.post_blank
        if seq.sub_sequences[-1].capture_slots[-1].post_blank is not None
        and seq.sub_sequences[-1].post_blank is not None
        and seq.post_blank is not None
        else None
    )

    return sum(
        # 先頭の blank
        [[seq.prev_blank + seq.sub_sequences[0].prev_blank]]
        + [
            sum(
                # 最終以外の subseq の 最終以外の slot の wave, blank chain
                [[slot.duration, slot.post_blank] for slot in subseq.capture_slots[:-1]]
                # 最終以外の subseq の 最終 slot の wave, blank
                # blank は 橋渡し blank の値を使う
                + [[subseq.capture_slots[-1].duration, blank]],
                [],
            )
            for subseq, blank in zip(seq.sub_sequences[:-1], blank_bridges)
        ]
        # 最終 subseq の 最終以外の slot の wave, blank chain
        + [
            [slot.duration, slot.post_blank]
            for slot in seq.sub_sequences[-1].capture_slots[:-1]
        ]
        # お尻につく wave, blank
        + [[seq.sub_sequences[-1].capture_slots[-1].duration, last_blank]],
        [],
    )
