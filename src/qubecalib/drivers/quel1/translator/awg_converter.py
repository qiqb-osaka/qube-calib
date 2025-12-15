from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class QcAwgChunk:
    wave_id: int
    prev_blank_words: int
    pad_front: int = 0
    pad_back: int = 0
    loops: int = 1


@dataclass
class QcAwgSegment:
    begin: int
    aligned_length: int


@dataclass
class AwgProgramTemplate:
    SAMPLES_PER_WORD: int = 4  # 1 word = 4 samples = 8 ns

    wait_words: int = 0  # 1 word = 4  samples = 8 ns, min=0
    waves: list[NDArray] = field(default_factory=list)  # waveform samples (500MHz)
    chunks: list[QcAwgChunk] = field(default_factory=list)

    # def __init__(self, awg_param: AwgParam, waves: list[NDArray]) -> None:
    #     self.wait_words = awg_param.num_wait_word
    #     self.waves = waves
    #     self.chunks = [
    #         QcAwgChunk(
    #             wave_id=chunk,
    #             prev_blank_words=chunk["blank_words"],
    #             pad_front=chunk.get("pad_front", 0),
    #             pad_back=chunk.get("pad_back", 0),
    #             loops=chunk.get("loops", 1),
    #         )
    #         for chunk in awg_param.chunks
    #     ]


@dataclass
class QcAwgProgram:
    SAMPLES_PER_WORD: int = 4  # 1 word = 4 samples = 8 ns

    wait_words: int = 0  # 1 word = 4  samples = 8 ns, min=0
    waves: list[NDArray] = field(default_factory=list)  # waveform samples (500MHz)
    chunks: list[QcAwgChunk] = field(default_factory=list)
    # chunk = {"wave_id": int, "blank_words": int}
    # pad_fronts: list[int] = field(default_factory=list)
    # pad_backs: list[int] = field(default_factory=list)

    def get_timeline(self) -> list[QcAwgSegment]:
        cur = self.wait_words * self.SAMPLES_PER_WORD
        segments: list[QcAwgSegment] = []

        for chunk in self.chunks:
            # blank before this chunk
            blank_samples = chunk.prev_blank_words * self.SAMPLES_PER_WORD
            cur += blank_samples

            wave = self.waves[chunk.wave_id]

            begin = cur
            duration = len(wave)
            segments.append(QcAwgSegment(begin, duration))

            cur += duration

        return segments

    def get_timeline_detailed(self) -> list[dict]:
        cur = self.wait_words * self.SAMPLES_PER_WORD
        segments: list[dict] = []

        for chunk in self.chunks:
            # blank before this chunk
            blank_samples = chunk.prev_blank_words * self.SAMPLES_PER_WORD
            cur += blank_samples

            wave = self.waves[chunk.wave_id]

            begin = cur
            duration = len(wave)
            user_length = duration - chunk.pad_front - chunk.pad_back

            segments.append(
                {
                    "wave_id": chunk.wave_id,
                    "begin": begin,
                    "aligned_length": duration,
                    "user_length": user_length,
                    "pad_front": chunk.pad_front,
                    "pad_back": chunk.pad_back,
                    "loops": chunk.loops,
                    "prev_blank_words": chunk.prev_blank_words,
                }
            )

            cur += duration

        return segments


class BeginArrayAwgConverter:
    SAMPLES_PER_WORD: int = 4  # 1 word = 4 samples = 8 ns
    ALIGN_SAMPLES: int = 64

    def convert(
        self,
        data: list[tuple[int, np.ndarray]],
        *,
        repeats: int = 1,
    ) -> QcAwgProgram:
        # sort by begin index
        data = sorted(data, key=lambda t: t[0])

        # calculate auto_wait_words
        first_begin = data[0][0]
        wait_words = first_begin // self.SAMPLES_PER_WORD

        ap = QcAwgProgram(wait_words=wait_words)

        # cursor in machine coordinates
        prev_end = wait_words * self.SAMPLES_PER_WORD

        for begin, arr in data:
            arr_front_aligned, pad_front, prev_blank_words = self._align_front_waveform(
                prev_end, begin, arr
            )
            arr_aligned, pad_back = self._align_back_waveform(arr_front_aligned)
            aligned_len = len(arr_aligned)

            # Step 5: register wave and chunk
            wave_id = len(ap.waves)
            ap.waves.append(arr_aligned)

            ap.chunks.append(
                QcAwgChunk(
                    wave_id=wave_id,
                    prev_blank_words=prev_blank_words,
                    pad_back=pad_back,
                    pad_front=pad_front,
                )
            )

            # update machine/user positions
            prev_end += prev_blank_words * self.SAMPLES_PER_WORD + aligned_len

        return ap

    def _align_front_waveform(
        self,
        prev_end: int,
        begin: int,
        arr: np.ndarray,
    ) -> tuple[NDArray, int, int]:
        prev_blank = (begin - prev_end) // self.SAMPLES_PER_WORD
        if prev_blank < 0:
            raise ValueError("Waveform overlap in machine coordinates.")
        offset = begin - (prev_end + prev_blank * self.SAMPLES_PER_WORD)
        if offset < 0:
            raise ValueError("Waveform overlap in machine coordinates.")
        if offset == 0:
            pad_front = 0
        else:
            pad_front = offset
            arr = np.concatenate([np.zeros(pad_front, dtype=arr.dtype), arr])
        return (
            arr,
            pad_front,
            prev_blank,
        )

    def _align_back_waveform(self, arr: np.ndarray) -> tuple[NDArray, int]:
        n = len(arr)
        a = self.ALIGN_SAMPLES  # e.g., 64
        r = n % a

        if r == 0:
            return arr, 0
        else:
            pad_back = a - r
            return np.concatenate([arr, np.zeros(pad_back, dtype=arr.dtype)]), pad_back
