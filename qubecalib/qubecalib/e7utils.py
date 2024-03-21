from __future__ import annotations

from typing import MutableSequence

# from dataclasses import dataclass
from e7awgsw import CaptureParam, WaveSequence

from .neopulseexp import CapSampledSequence

# import json


def create_wseq() -> WaveSequence:
    return WaveSequence()


def create_cprm() -> CaptureParam:
    return CaptureParam()


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
