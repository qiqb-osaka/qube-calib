from __future__ import annotations

import math
from typing import MutableSequence

# from dataclasses import dataclass
from e7awgsw import CaptureParam, WaveSequence

from .neopulse import CapSampledSequence, GenSampledSequence

# import json


class WaveSequenceTools:
    @classmethod
    def create(cls, sequence: GenSampledSequence) -> WaveSequence:
        unit = WaveSequence.NUM_SAMPLES_IN_AWG_WORD
        seq = sequence

        wseq = WaveSequence(
            num_wait_words=seq.prev_blank,
            num_repeats=seq.repeats if seq.repeats is not None else 1,
        )

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


def _convert_cap_sampled_sequence_to_blanks_and_durations_chain(
    sequence: CapSampledSequence,
) -> MutableSequence:
    """cap sampled sequence を blank - duration チェーンに変換する"""
    seq = sequence

    blank_bridges = [
        lo.capture_slots[-1].post_blank + lo.post_blank + hi.prev_blank
        if lo.capture_slots[-1].post_blank is not None
        and lo.post_blank is not None
        and hi.prev_blank is not None
        else None
        for lo, hi in zip(seq.sub_sequences[:-1], seq.sub_sequences[1:])
    ]

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
        [[seq.prev_blank + seq.sub_sequences[0].prev_blank]]
        + [
            sum(
                [[slot.duration, slot.post_blank] for slot in subseq.capture_slots[:-1]]
                + [[subseq.capture_slots[-1].duration, blank]],
                [],
            )
            for subseq, blank in zip(seq.sub_sequences[:-1], blank_bridges)
        ]
        + [
            [slot.duration, slot.post_blank]
            for slot in seq.sub_sequences[-1].capture_slots[:-1]
        ]
        + [[seq.sub_sequences[-1].capture_slots[-1].duration, last_blank]],
        [],
    )
