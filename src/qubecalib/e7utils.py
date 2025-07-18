from __future__ import annotations

import math
import sys
from typing import Any, MutableMapping, MutableSequence

import numpy as np

# from dataclasses import dataclass
from e7awgsw import CaptureParam, DspUnit, IqWave, WaveSequence

from .neopulse import CapSampledSequence, GenSampledSequence

# import json

SAMPLING_PERIOD = 2  # [ns]


class WaveSequenceTools:
    @classmethod
    def quantize_duration(
        cls,
        duration: float,
        constrain: int = 10_240,
    ) -> int:
        # nco 達をデータフレームの長さと同期させると位相が揃う
        # 15.625MHz の整数倍/割にすると良い
        # 15.625MHz(64ns) の1/160 97.65625kHz(10_240ns) を繰り返し周期にすると位相が揃う
        return int(duration // constrain) * constrain

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
        wait_words: int,
        repeats: int,
        interval_samples: int,
    ) -> WaveSequence:
        unit = WaveSequence.NUM_SAMPLES_IN_AWG_WORD
        # 同一 awg に所属する target を束ねた WaveSequence を生成する
        # TODO もし sequence の geometry が e7 compatible だった場合 wave_chunk を活用して変換する
        # そうでなかったらすべての subseq を 単一の wave_chunk に入れる
        return cls.create_single_chunked_wave_sequence(
            sequence=sequence,
            wait_words=wait_words,
            repeats=repeats,
            interval_words=int(interval_samples / unit),
        )
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
        wait_words: int,
        repeats: int,
        interval_words: int,
    ) -> WaveSequence:
        """単一の WaveChunk にすべての iq をまとめた WaveSequence を作る"""
        # 市松模様チェーンを生成
        chain = _convert_gen_sampled_sequence_to_blanks_and_waves_chain(sequence)
        # print(chain)
        # 境界を抽出
        bounds = [sum(chain[: i + 1]) for i, _ in enumerate(chain)]
        # print(bounds)
        i, q = np.zeros(bounds[-1]).astype(int), np.zeros(bounds[-1]).astype(int)
        EPS = sys.float_info.epsilon
        for begin, subseq in zip(bounds[::2], sequence.sub_sequences):
            if max(np.abs(subseq.real)) - 1 > EPS or max(np.abs(subseq.imag)) - 1 > EPS:
                raise ValueError("magnitude of iq signal must not exceed 1")
            i[begin : begin + subseq.real.shape[0]] = (32767 * subseq.real).astype(int)
            q[begin : begin + subseq.imag.shape[0]] = (32767 * subseq.imag).astype(int)
        # 繰り返し周期を設定する
        wseq = WaveSequence(
            num_wait_words=wait_words,
            num_repeats=sequence.repeats if sequence.repeats is not None else repeats,
        )

        s = IqWave.convert_to_iq_format(i, q, WaveSequence.NUM_SAMPLES_IN_WAVE_BLOCK)
        total_duration_in_words = int(len(s) // WaveSequence.NUM_SAMPLES_IN_AWG_WORD)
        wseq.add_chunk(
            iq_samples=s,
            num_blank_words=interval_words - total_duration_in_words,
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
    def create(
        cls,
        sequence: CapSampledSequence,
        capture_delay_words: int,
        repeats: int,
        interval_samples: int,
    ) -> CaptureParam:
        unit = CaptureParam.NUM_SAMPLES_IN_ADC_WORD
        # 市松模様チェーンを生成
        # WaveChunk に相当するものがないので subsequence はマージする
        chain = _convert_cap_sampled_sequence_to_blanks_and_durations_chain(sequence)
        # 最初のブランクは Block = 16 words 単位でしか反応しないので 16 の整数倍数に合うよう拡大する
        # WaveSequence の単位に合わせた先頭 padding
        chain[0] += sequence.padding
        # 境界を抽出
        bounds = [sum(chain[:i]) for i, _ in enumerate(chain)]
        # アライメント境界を生成
        # delay と sum_section の長さは ADC_WORD が最小単位
        # ただし pre_blank は 1 block = 16 words が最小単位
        # ticks = [0, math.floor(bounds[1] / (16 * unit)) * 16 * unit] + [
        #     math.floor(bound / unit) * unit for bound in bounds[2:]
        # ]
        # lower = [
        #     tick - unit if bound < tick else tick
        #     for bound, tick in zip(bounds[1::2], ticks[1::2])
        # ]
        # upper = [
        #     tick - unit if bound < tick else tick
        #     for bound, tick in zip(bounds[2::2], ticks[2::2])
        # ]
        # aligned: MutableSequence = sum(
        #     [[bounds[0]]] + [[low, up] for low, up in zip(lower, upper)],
        #     [],
        # )
        aligned: MutableSequence[int] = [
            0,
            math.floor(bounds[1] / (16 * unit)) * 16 * unit,
        ] + [math.floor(bound / unit) * unit for bound in bounds[2:]]
        # アライメントされた境界を使って新しい市松模様チェーンを生成
        # 最後の要素が None だと以降の処理でエラーにならないよう 0 へ変換
        new_chain = [
            int((up - low) / unit) if low is not None and up is not None else None
            for low, up in zip(aligned[:-1], aligned[1:])
        ] + [chain[-1] if chain[-1] is not None else 0]
        total_duration_words = sum(new_chain)
        interval_words = int(interval_samples // unit)
        # 先頭に blank を入れた分末尾にも同じ長さの blank を入れないと wave と同期しなくなる TOOO ?
        # wave の先頭を wait で合わせるという選択もあり（そっちの方が良さげ） <- これはダメ　 X wait は先頭のみ
        new_chain[-1] = interval_words - total_duration_words + new_chain[0]
        for i in range(2, len(new_chain), 2):
            if new_chain[i] == 0:
                if new_chain[i - 1] == 1:
                    raise ValueError("Capture is too short")
                new_chain[i - 1] -= 1
                new_chain[i] = 1
        # if new_chain[-1] == 0:
        #     new_chain[-2] -= 1
        #     new_chain[-1] = 1
        # TODO new_chain の blank 要素が潰れることがある（capt は拡張なので潰れない）この処理を追加しないといけない
        # TODO post_blank は 1 以上の制約がある
        capprm = CaptureParam()
        capprm.capture_delay = capture_delay_words + new_chain[0]
        capprm.num_integ_sections = repeats
        for duration, blank in zip(new_chain[1::2], new_chain[2::2]):
            capprm.add_sum_section(
                num_words=duration,
                num_post_blank_words=blank,
            )

        return capprm

    @classmethod
    def enable_integration(
        cls,
        capprm: CaptureParam,
    ) -> CaptureParam:
        dsp = capprm.dsp_units_enabled
        dsp.append(DspUnit.INTEGRATION)
        capprm.sel_dsp_units_to_enable(*dsp)
        return capprm

    @classmethod
    def enable_sum(
        cls,
        capprm: CaptureParam,
    ) -> CaptureParam:
        dsp = capprm.dsp_units_enabled
        dsp.append(DspUnit.SUM)
        capprm.sel_dsp_units_to_enable(*dsp)
        return capprm

    @classmethod
    def enable_demodulation(
        cls,
        capprm: CaptureParam,
        f_GHz: float,
    ) -> CaptureParam:
        """
        Enable demodulation of the captured signal.

        Parameters
        ----------
        capprm : CaptureParam
            Capture parameters.
        f_GHz : float
            Frequency in GHz to demodulate the captured signal.

        Returns
        -------
        CaptureParam
            CaptureParam object with demodulation enabled.
        """
        capprm.complex_fir_coefs = cls.fir_coefficient(f_GHz)
        capprm.complex_window_coefs = cls.window_coefficient(f_GHz)

        dspunits = capprm.dsp_units_enabled
        dspunits.append(DspUnit.COMPLEX_FIR)
        dspunits.append(DspUnit.DECIMATION)
        dspunits.append(DspUnit.COMPLEX_WINDOW)
        capprm.sel_dsp_units_to_enable(*dspunits)
        return capprm

    @classmethod
    def fir_coefficient(
        cls,
        f_GHz: float,
    ) -> list[complex]:
        """
        Calculate FIR coefficients for a bandpass filter.

        Parameters
        ----------
        f_GHz : float
            Center frequency of the bandpass filter in GHz.

        Returns
        -------
        list[complex]
            FIR coefficients for the bandpass filter.
            Each part of a complex FIR coefficient must be an integer
            and in the range of [-2**15, 2**15 - 1].
        """
        N_COEFS = CaptureParam.NUM_COMPLEX_FIR_COEFS  # 16
        MAX_VAL = CaptureParam.MAX_FIR_COEF_VAL  # 32767
        t_ns = SAMPLING_PERIOD * np.arange(-N_COEFS + 1, 1)  # [-30, -28, ..., 0]

        # rect window
        # window_function = MAX_VAL * np.ones(N_COEFS).astype(complex)

        # gaussian window
        mu = (t_ns[-1] + t_ns[0]) / 2
        sigma = (t_ns[-1] - t_ns[0]) / 6
        window_function = MAX_VAL * np.exp(-0.5 * (t_ns - mu) ** 2 / (sigma**2))

        coefs = window_function * np.exp(1j * 2 * np.pi * f_GHz * t_ns)
        result = coefs.round().tolist()
        return result

    @classmethod
    def window_coefficient(
        cls,
        f_GHz: float,
    ) -> list[complex]:
        """
        Calculate window coefficients for a bandpass filter.

        Parameters
        ----------
        f_GHz : float
            Center frequency of the bandpass filter in GHz.

        Returns
        -------
        list[complex]
            Window coefficients for the bandpass filter.
            Each part of a complex window coefficient must be an integer
            and in the range of [-2**31, 2**31 - 1].
        """
        N_DECIMATION = 4
        N_COEFS = CaptureParam.NUM_COMPLEXW_WINDOW_COEFS  # 2048
        MAX_VAL = CaptureParam.MAX_WINDOW_COEF_VAL  # 2147483647
        t_ns = N_DECIMATION * SAMPLING_PERIOD * np.arange(N_COEFS)  # [0, 8, ..., 16376]
        coefs = MAX_VAL * np.exp(-1j * 2 * np.pi * f_GHz * t_ns)
        result = coefs.round().tolist()
        return result

    @classmethod
    def enable_classification(
        cls,
        capprm: CaptureParam,
        *,
        func_sel: Any,
        coef_a: Any,
        coef_b: Any,
        const_c: Any,
    ) -> CaptureParam:
        """
        Enable classification in the capture parameters.

        Parameters
        ----------
        capprm : CaptureParam
            Capture parameters.

        Returns
        -------
        CaptureParam
            CaptureParam object with classification enabled.
        """
        dspunits = capprm.dsp_units_enabled
        dspunits.append(DspUnit.CLASSIFICATION)
        capprm.sel_dsp_units_to_enable(*dspunits)
        capprm.set_decision_func_params(
            func_sel=func_sel,
            coef_a=coef_a,
            coef_b=coef_b,
            const_c=const_c,
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
                (
                    sequence.sub_sequences[-1].post_blank + sequence.post_blank
                    if sequence.post_blank is not None
                    and sequence.sub_sequences[-1].post_blank is not None
                    else 0
                ),
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
        (
            lo.capture_slots[-1].post_blank + lo.post_blank + hi.prev_blank
            if lo.capture_slots[-1].post_blank is not None
            and lo.post_blank is not None
            and hi.prev_blank is not None
            else None
        )
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


def _convert_cap_sampled_sequence_to_blanks_and_durations_chain_use_original_values(
    sequence: CapSampledSequence,
) -> MutableSequence:
    """cap sampled sequence を blank - duration チェーンに変換する（ただし、オリジナルの値を使う）"""
    seq = sequence

    # sub sequence 間を橋渡しするブランクのリスト MutableSequence[Optional[int|float]]
    # 最終スロットの blank, 前 subseq の post_blank, 後 subseq の prev_blank
    blank_bridges = [
        (
            lo.capture_slots[-1].original_post_blank
            + lo.original_post_blank
            + hi.original_prev_blank
            if lo.capture_slots[-1].original_post_blank is not None
            and lo.original_post_blank is not None
            and hi.original_prev_blank is not None
            else None
        )
        for lo, hi in zip(seq.sub_sequences[:-1], seq.sub_sequences[1:])
    ]

    # お尻につく blank
    last_blank = (
        seq.sub_sequences[-1].capture_slots[-1].original_post_blank
        + seq.sub_sequences[-1].original_post_blank
        + seq.original_post_blank
        if seq.sub_sequences[-1].capture_slots[-1].original_post_blank is not None
        and seq.sub_sequences[-1].original_post_blank is not None
        and seq.original_post_blank is not None
        else None
    )

    prev_blank = seq.original_prev_blank
    if prev_blank is None:
        raise ValueError("original_prev_blank must be set")
    subseq_prev_blank = seq.sub_sequences[0].original_prev_blank
    if subseq_prev_blank is None:
        raise ValueError("original_prev_blank of subseq must be set")

    return sum(
        # 先頭の blank
        [[prev_blank + subseq_prev_blank]]
        + [
            sum(
                # 最終以外の subseq の 最終以外の slot の wave, blank chain
                [
                    [slot.original_duration, slot.original_post_blank]
                    for slot in subseq.capture_slots[:-1]
                ]
                # 最終以外の subseq の 最終 slot の wave, blank
                # blank は 橋渡し blank の値を使う
                + [[subseq.capture_slots[-1].original_duration, blank]],
                [],
            )
            for subseq, blank in zip(seq.sub_sequences[:-1], blank_bridges)
        ]
        # 最終 subseq の 最終以外の slot の wave, blank chain
        + [
            [slot.original_duration, slot.original_post_blank]
            for slot in seq.sub_sequences[-1].capture_slots[:-1]
        ]
        # お尻につく wave, blank
        + [[seq.sub_sequences[-1].capture_slots[-1].original_duration, last_blank]],
        [],
    )
