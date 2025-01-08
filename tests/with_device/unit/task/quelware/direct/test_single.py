from __future__ import annotations

from concurrent.futures import Future

import pytest
from e7awgsw import CaptureParam, DspUnit, WaveSequence
from qubecalib.task.quelware.direct.single import (
    Action,
    AwgId,
    AwgSetting,
    RunitId,
    RunitSetting,
    TriggerSetting,
)
from quel_ic_config import CaptureReturnCode, Quel1BoxWithRawWss

PERIOD = 1280 * 128

WAVE_SEQUENCE = WaveSequence(num_wait_words=0, num_repeats=1)
WAVE_SEQUENCE.add_chunk(
    iq_samples := 4 * 1 * 1024 * [(32767, 0)],
    num_blank_words=PERIOD - len(iq_samples) // 4,
    num_repeats=100,
)

CAPTURE_PARAM = CaptureParam()
CAPTURE_PARAM.num_integ_sections = 100
CAPTURE_PARAM.add_sum_section(
    num_words := int(8 * 4096 / 16),
    num_post_blank_words=PERIOD - num_words,
)
CAPTURE_PARAM.sel_dsp_units_to_enable(DspUnit.INTEGRATION)


def test_create_box(box: Quel1BoxWithRawWss) -> None:
    assert isinstance(box, Quel1BoxWithRawWss)


def create_capture_params(
    num_words: int = int(8 * 4096 / 16),
    period: int = 1280 * 128,
) -> CaptureParam:
    c = CaptureParam()
    c.num_integ_sections = 100
    c.add_sum_section(
        num_words := num_words,
        num_post_blank_words=period - num_words,
    )
    c.sel_dsp_units_to_enable(DspUnit.INTEGRATION)


def test_action_runitonly(box: Quel1BoxWithRawWss) -> None:
    a = Action.build(
        box=box,
        settings=[
            RunitSetting(runit=RunitId(port=1, runit=0), cprm=CAPTURE_PARAM),
            RunitSetting(runit=RunitId(port=12, runit=0), cprm=CAPTURE_PARAM),
            RunitSetting(runit=RunitId(port=12, runit=1), cprm=CAPTURE_PARAM),
            RunitSetting(runit=RunitId(port=12, runit=2), cprm=CAPTURE_PARAM),
            RunitSetting(runit=RunitId(port=12, runit=3), cprm=CAPTURE_PARAM),
        ],
    )
    assert isinstance(a, Action)
    assert isinstance(a.box, Quel1BoxWithRawWss)
    assert a._wseqs == {}
    assert a._cprms == {
        RunitId(port=1, runit=0): CAPTURE_PARAM,
        RunitId(port=12, runit=0): CAPTURE_PARAM,
        RunitId(port=12, runit=1): CAPTURE_PARAM,
        RunitId(port=12, runit=2): CAPTURE_PARAM,
        RunitId(port=12, runit=3): CAPTURE_PARAM,
    }
    assert a._triggers == {}

    futures = a.capture_start()
    a.start_emission()
    assert futures.keys() == {1, 12}

    futures_1, future_12 = futures[1], futures[12]
    assert isinstance(futures_1, Future) and isinstance(future_12, Future)

    status, data = a.capture_stop(futures)
    assert status.keys() == {1, 12}
    assert status[1] == CaptureReturnCode.SUCCESS
    assert status[12] == CaptureReturnCode.SUCCESS
    assert data.keys() == {(1, 0), (12, 0), (12, 1), (12, 2), (12, 3)}
    assert data[(1, 0)].shape == (4 * CAPTURE_PARAM.num_samples_to_sum(0),)
    assert data[(12, 0)].shape == (4 * CAPTURE_PARAM.num_samples_to_sum(0),)
    assert data[(12, 1)].shape == (4 * CAPTURE_PARAM.num_samples_to_sum(0),)
    assert data[(12, 2)].shape == (4 * CAPTURE_PARAM.num_samples_to_sum(0),)
    assert data[(12, 3)].shape == (4 * CAPTURE_PARAM.num_samples_to_sum(0),)

    a = Action.build(
        box=box,
        settings=[
            RunitSetting(runit=RunitId(port=1, runit=0), cprm=CAPTURE_PARAM),
            RunitSetting(runit=RunitId(port=12, runit=0), cprm=CAPTURE_PARAM),
            RunitSetting(runit=RunitId(port=12, runit=1), cprm=CAPTURE_PARAM),
            RunitSetting(runit=RunitId(port=12, runit=2), cprm=CAPTURE_PARAM),
            RunitSetting(runit=RunitId(port=12, runit=3), cprm=CAPTURE_PARAM),
        ],
    )
    status, data = a.capture_stop(futures)
    assert status.keys() == {1, 12}
    assert status[1] == CaptureReturnCode.SUCCESS
    assert status[12] == CaptureReturnCode.SUCCESS
    assert data.keys() == {(1, 0), (12, 0), (12, 1), (12, 2), (12, 3)}
    assert data[(1, 0)].shape == (4 * CAPTURE_PARAM.num_samples_to_sum(0),)
    assert data[(12, 0)].shape == (4 * CAPTURE_PARAM.num_samples_to_sum(0),)
    assert data[(12, 1)].shape == (4 * CAPTURE_PARAM.num_samples_to_sum(0),)
    assert data[(12, 2)].shape == (4 * CAPTURE_PARAM.num_samples_to_sum(0),)
    assert data[(12, 3)].shape == (4 * CAPTURE_PARAM.num_samples_to_sum(0),)


def test_action_awgonly(box: Quel1BoxWithRawWss) -> None:
    a = Action.build(
        box=box,
        settings=[
            AwgSetting(
                awg=AwgId(port=0, channel=0),
                wseq=WAVE_SEQUENCE,
            ),
            AwgSetting(
                awg=AwgId(port=13, channel=0),
                wseq=WAVE_SEQUENCE,
            ),
        ],
    )
    assert a._wseqs == {
        AwgId(port=0, channel=0): WAVE_SEQUENCE,
        AwgId(port=13, channel=0): WAVE_SEQUENCE,
    }
    assert a._cprms == {}
    assert a._triggers == {}
    futures = a.capture_start()
    a.start_emission()
    assert futures == {}
    assert list(futures.keys()) == []
    status, data = a.capture_stop(futures)
    assert status == {}
    assert data == {}
    assert list(data.keys()) == []

    a = Action.build(
        box=box,
        settings=[
            AwgSetting(
                awg=AwgId(port=0, channel=0),
                wseq=WAVE_SEQUENCE,
            ),
            AwgSetting(
                awg=AwgId(port=13, channel=0),
                wseq=WAVE_SEQUENCE,
            ),
        ],
    )
    status, data = a.action()
    assert list(data.keys()) == []


def test_action_runit_awg_notrig(box: Quel1BoxWithRawWss) -> None:
    with pytest.raises(ValueError) as e:
        Action.build(
            box=box,
            settings=[
                AwgSetting(
                    awg=AwgId(port=0, channel=0),
                    wseq=WAVE_SEQUENCE,
                ),
                RunitSetting(
                    runit=RunitId(port=1, runit=0),
                    cprm=CAPTURE_PARAM,
                ),
            ],
        )
    assert str(e.value) == "both wseqs and cprms are provided without triggers"
    # a = Action.build(
    #     box=box,
    #     settings=[
    #         AwgSetting(
    #             awg=AwgId(port=0, channel=0),
    #             wseq=WAVE_SEQUENCE,
    #         ),
    #         RunitSetting(
    #             runit=RunitId(port=1, runit=0),
    #             cprm=CAPTURE_PARAM,
    #         ),
    #     ],
    # )
    # assert a._wseqs == {AwgId(port=0, channel=0): WAVE_SEQUENCE}
    # assert a._cprms == {RunitId(port=1, runit=0): CAPTURE_PARAM}
    # assert a._triggers == {}
    # futures = a.capture_start()
    # a.start_emission()
    # assert futures.keys() == {1}
    # assert isinstance(futures[1], Future)
    # results = a.capture_stop(futures)
    # assert results.keys() == {(1, 0)}
    # assert results[(1, 0)].shape == (4 * CAPTURE_PARAM.num_samples_to_sum(0),)

    # a = Action.build(
    #     box=box,
    #     settings=[
    #         AwgSetting(
    #             awg=AwgId(port=0, channel=0),
    #             wseq=WAVE_SEQUENCE,
    #         ),
    #         RunitSetting(
    #             runit=RunitId(port=1, runit=0),
    #             cprm=CAPTURE_PARAM,
    #         ),
    #     ],
    # )
    # results = a.action()
    # assert results.keys() == {(1, 0)}
    # assert results[(1, 0)].shape == (4 * CAPTURE_PARAM.num_samples_to_sum(0),)


def test_action_runit_awg_trig(box: Quel1BoxWithRawWss) -> None:
    a = Action.build(
        box=box,
        settings=[
            AwgSetting(
                awg=AwgId(port=0, channel=0),
                wseq=WAVE_SEQUENCE,
            ),
            RunitSetting(
                runit=RunitId(port=1, runit=0),
                cprm=CAPTURE_PARAM,
            ),
            TriggerSetting(
                triggerd_port=1,
                trigger_awg=AwgId(port=0, channel=0),
            ),
        ],
    )
    assert a._wseqs == {AwgId(port=0, channel=0): WAVE_SEQUENCE}
    assert a._cprms == {RunitId(port=1, runit=0): CAPTURE_PARAM}
    assert a._triggers == {1: AwgId(port=0, channel=0)}
    futures = a.capture_start()
    a.start_emission()
    assert isinstance(futures[1], Future)
    status, data = a.capture_stop(futures)
    assert status.keys() == {1}
    assert status[1] == CaptureReturnCode.SUCCESS
    assert data.keys() == {(1, 0)}
    assert data[(1, 0)].shape == (4 * CAPTURE_PARAM.num_samples_to_sum(0),)

    a = Action.build(
        box=box,
        settings=[
            AwgSetting(
                awg=AwgId(port=0, channel=0),
                wseq=WAVE_SEQUENCE,
            ),
            RunitSetting(
                runit=RunitId(port=1, runit=0),
                cprm=CAPTURE_PARAM,
            ),
            TriggerSetting(
                triggerd_port=1,
                trigger_awg=AwgId(port=0, channel=0),
            ),
        ],
    )
    status, data = a.action()
    assert data.keys() == {(1, 0)}
    assert data[(1, 0)].shape == (4 * CAPTURE_PARAM.num_samples_to_sum(0),)


def test_action_raise_no_settings(box: Quel1BoxWithRawWss) -> None:
    with pytest.raises(ValueError) as e:
        Action.build(box=box, settings=[])
    assert str(e.value) == "no settings provided"


# def test_action_raise_already_loaded(box: Quel1BoxWithRawWss) -> None:
#     a = Action.build(
#         box=box,
#         settings=[
#             RunitSetting(runit=RunitId(port=1, runit=0), cprm=CAPTURE_PARAM),
#         ],
#     )
#     with pytest.raises(ValueError) as e:
#         a._load([])
#     assert str(e.value) == "already loaded"


def test_action_invalid_trigger(box: Quel1BoxWithRawWss) -> None:
    # noawg for triggerd port 1
    NUM_WORDS = int(8 * 4096 / 16)
    PERIOD = 1280 * 128
    c = create_capture_params(NUM_WORDS, PERIOD)
    with pytest.raises(ValueError) as e:
        Action.build(
            box=box,
            settings=[
                RunitSetting(runit=RunitId(port=1, runit=0), cprm=c),
                RunitSetting(runit=RunitId(port=1, runit=0), cprm=c),
                TriggerSetting(triggerd_port=1, trigger_awg=AwgId(port=0, channel=0)),
            ],
        )
    assert (
        str(e.value)
        == "trigger AwgId(port=0, channel=0) for triggerd port 1 is not provided"
    )
    # a = Action.build(
    #     box=box,
    #     settings=[
    #         RunitSetting(runit=RunitId(port=1, runit=0), cprm=c),
    #         RunitSetting(runit=RunitId(port=1, runit=0), cprm=c),
    #         TriggerSetting(triggerd_port=1, trigger_awg=AwgId(port=0, channel=0)),
    #     ],
    # )
    # with pytest.raises(ValueError) as e:
    #     a.action()
    # assert (
    #     str(e.value)
    #     == "trigger AwgId(port=0, channel=0) for triggerd port 1 is not provided"
    # )
    # norunit for triggerd port 1
    with pytest.raises(ValueError) as e:
        Action.build(
            box=box,
            settings=[
                AwgSetting(awg=AwgId(port=0, channel=0), wseq=WAVE_SEQUENCE),
                TriggerSetting(triggerd_port=1, trigger_awg=AwgId(port=0, channel=0)),
            ],
        )
    assert str(e.value) == "triggerd port 1 is not provided in runit settings"
    # a = Action.build(
    #     box=box,
    #     settings=[
    #         AwgSetting(awg=AwgId(port=0, channel=0), wseq=WAVE_SEQUENCE),
    #         TriggerSetting(triggerd_port=1, trigger_awg=AwgId(port=0, channel=0)),
    #     ],
    # )
    # with pytest.raises(ValueError) as e:
    #     a.action()
    # assert str(e.value) == "triggerd port 1 is not provided in runit settings"
