import numpy as np
import pytest
from qubecalib import neopulse
from qubecalib.neopulse import (
    Blank,
    CapSampledSequence,
    Capture,
    Flushleft,
    Flushright,
    GenSampledSequence,
    Rectangle,
    Sequence,
    Series,
)


def test_convert_to_sampled_sequence():
    """Sequence should convert to sampled sequence."""
    target = "RQ00"

    with Sequence() as seq:
        Rectangle(duration=10).target(target)
        Capture(duration=10).target(target)

    sampled_sequence = seq.convert_to_sampled_sequence()
    gen_sampled_sequence = sampled_sequence[0][target]
    cap_sampled_sequence = sampled_sequence[1][target]

    assert isinstance(gen_sampled_sequence, GenSampledSequence)
    assert isinstance(cap_sampled_sequence, CapSampledSequence)


def test_reuse_slot_instance():
    """Slot instances should be reusable."""
    target = "RQ00"
    dt = neopulse.DEFAULT_SAMPLING_PERIOD
    n = 2

    rect = Rectangle(duration=n * dt)

    with Sequence() as seq:
        Blank(duration=n * dt).target(target)
        Rectangle(duration=n * dt).target(target)
        rect.scaled(2).shifted(np.pi).target(target)
        rect.target(target)

    gen_sampled_sequence, _ = seq.convert_to_sampled_sequence()
    gen_sub_sequence = gen_sampled_sequence[target].sub_sequences[0]
    assert gen_sub_sequence.real == pytest.approx(
        np.array([0.0, 0.0, 1.0, 1.0, -2.0, -2.0, 1.0, 1.0])
    )


def test_gen_sampled_sequence():
    """GenSampledSequence should return the correct values."""
    target = "RQ00"
    dt = neopulse.DEFAULT_SAMPLING_PERIOD
    n = 5

    with Sequence() as seq:
        Blank(duration=n * dt).target(target)
        Rectangle(duration=n * dt).target(target)

    sampled_sequence = seq.convert_to_sampled_sequence()
    gen_sampled_sequence = sampled_sequence[0][target]
    assert gen_sampled_sequence.asdict() == {
        "target_name": target,
        "prev_blank": 0,
        "post_blank": None,
        "padding": 0,
        "sampling_period": dt,
        "repeats": None,
        "original_prev_blank": None,
        "original_post_blank": None,
        "modulation_frequency": None,
        "sub_sequences": [
            {
                "real": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "imag": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "repeats": 1,
                "post_blank": None,
                "original_post_blank": None,
            }
        ],
        "readout_timings": None,
        "class": "GenSampledSequence",
    }


def test_cap_sampled_sequence():
    """CapSampledSequence should return the correct values."""
    target = "RQ00"
    dt = neopulse.DEFAULT_SAMPLING_PERIOD
    n_blank = 5
    n_capture = 10

    with Sequence() as seq:
        Blank(duration=n_blank * dt).target(target)
        Capture(duration=n_capture * dt).target(target)

    sampled_sequence = seq.convert_to_sampled_sequence()
    cap_sampled_sequence = sampled_sequence[1][target]
    assert cap_sampled_sequence.asdict() == {
        "target_name": target,
        "prev_blank": 0,
        "padding": 0,
        "sampling_period": dt,
        "post_blank": None,
        "repeats": None,
        "original_prev_blank": 0,
        "original_post_blank": None,
        "modulation_frequency": None,
        "sub_sequences": [
            {
                "capture_slots": [
                    {
                        "duration": n_capture,
                        "original_duration": 20.0,
                        "original_post_blank": 0.0,
                        "post_blank": 0,
                    }
                ],
                "prev_blank": n_blank,
                "post_blank": 0,
                "original_prev_blank": 10.0,
                "original_post_blank": 0.0,
                "repeats": 1,
            }
        ],
        "readin_offsets": None,
        "class": "CapSampledSequence",
    }


def test_series():
    """Series should add slots in series."""
    target0 = "RQ00"
    target1 = "RQ01"
    dt = neopulse.DEFAULT_SAMPLING_PERIOD
    n = 5

    with Sequence() as seq:
        with Series():
            Rectangle(duration=n * dt).target(target0)
            Rectangle(duration=n * dt).target(target1)

    target0_sequence = seq.convert_to_sampled_sequence()[0][target0].asdict()
    target1_sequence = seq.convert_to_sampled_sequence()[0][target1].asdict()

    assert target0_sequence == {
        "target_name": target0,
        "prev_blank": 0,
        "post_blank": None,
        "repeats": None,
        "sampling_period": dt,
        "sub_sequences": [
            {
                "real": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "imag": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "original_post_blank": None,
                "post_blank": None,
                "repeats": 1,
            }
        ],
        "class": "GenSampledSequence",
        "modulation_frequency": None,
        "original_post_blank": None,
        "original_prev_blank": None,
        "padding": 0,
        "readout_timings": None,
    }

    assert target1_sequence == {
        "target_name": target1,
        "prev_blank": 0,
        "post_blank": None,
        "repeats": None,
        "sampling_period": dt,
        "sub_sequences": [
            {
                "real": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "imag": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "original_post_blank": None,
                "post_blank": None,
                "repeats": 1,
            }
        ],
        "class": "GenSampledSequence",
        "modulation_frequency": None,
        "original_post_blank": None,
        "original_prev_blank": None,
        "padding": 0,
        "readout_timings": None,
    }


def test_series_as_default():
    """Series should be the default slot."""
    target0 = "RQ00"
    target1 = "RQ01"
    duration = 10

    with Sequence() as seq1:
        Rectangle(duration=duration).target(target0)
        Rectangle(duration=duration).target(target1)

    with Sequence() as seq2:
        with Series():
            Rectangle(duration=duration).target(target0)
            Rectangle(duration=duration).target(target1)

    target0_sequence1 = seq1.convert_to_sampled_sequence()[0][target0].asdict()
    target0_sequence2 = seq2.convert_to_sampled_sequence()[0][target0].asdict()
    target1_sequence1 = seq1.convert_to_sampled_sequence()[0][target1].asdict()
    target1_sequence2 = seq2.convert_to_sampled_sequence()[0][target1].asdict()
    assert target0_sequence1 == target0_sequence2
    assert target1_sequence1 == target1_sequence2


def test_flushleft():
    """Flushleft should add slots in parallel to the left."""
    target0 = "RQ00"
    target1 = "RQ01"
    dt = neopulse.DEFAULT_SAMPLING_PERIOD
    n0 = 5
    n1 = 10

    with Sequence() as seq:
        with Flushleft():
            Rectangle(duration=n0 * dt).target(target0)
            Rectangle(duration=n1 * dt).target(target1)

    target0_sequence = seq.convert_to_sampled_sequence()[0][target0].asdict()
    target1_sequence = seq.convert_to_sampled_sequence()[0][target1].asdict()

    assert target0_sequence == {
        "target_name": target0,
        "prev_blank": 0,
        "post_blank": None,
        "repeats": None,
        "sampling_period": dt,
        "sub_sequences": [
            {
                "real": [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "imag": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "original_post_blank": None,
                "post_blank": None,
                "repeats": 1,
            }
        ],
        "class": "GenSampledSequence",
        "modulation_frequency": None,
        "original_post_blank": None,
        "original_prev_blank": None,
        "padding": 0,
        "readout_timings": None,
    }

    assert target1_sequence == {
        "target_name": target1,
        "prev_blank": 0,
        "post_blank": None,
        "repeats": None,
        "sampling_period": dt,
        "sub_sequences": [
            {
                "real": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "imag": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "original_post_blank": None,
                "post_blank": None,
                "repeats": 1,
            }
        ],
        "class": "GenSampledSequence",
        "modulation_frequency": None,
        "original_post_blank": None,
        "original_prev_blank": None,
        "padding": 0,
        "readout_timings": None,
    }


def test_flushright():
    """Flushright should add slots in parallel to the right."""
    target0 = "RQ00"
    target1 = "RQ01"
    dt = neopulse.DEFAULT_SAMPLING_PERIOD
    n0 = 5
    n1 = 10

    with Sequence() as seq:
        with Flushright():
            Rectangle(duration=n0 * dt).target(target0)
            Rectangle(duration=n1 * dt).target(target1)

    target0_sequence = seq.convert_to_sampled_sequence()[0][target0].asdict()
    target1_sequence = seq.convert_to_sampled_sequence()[0][target1].asdict()

    assert target0_sequence == {
        "target_name": target0,
        "prev_blank": 0,
        "post_blank": None,
        "repeats": None,
        "sampling_period": dt,
        "sub_sequences": [
            {
                "real": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "imag": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "original_post_blank": None,
                "post_blank": None,
                "repeats": 1,
            }
        ],
        "class": "GenSampledSequence",
        "modulation_frequency": None,
        "original_post_blank": None,
        "original_prev_blank": None,
        "padding": 0,
        "readout_timings": None,
    }

    assert target1_sequence == {
        "target_name": target1,
        "prev_blank": 0,
        "post_blank": None,
        "repeats": None,
        "sampling_period": dt,
        "sub_sequences": [
            {
                "real": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                "imag": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "original_post_blank": None,
                "post_blank": None,
                "repeats": 1,
            }
        ],
        "class": "GenSampledSequence",
        "modulation_frequency": None,
        "original_post_blank": None,
        "original_prev_blank": None,
        "padding": 0,
        "readout_timings": None,
    }
