from __future__ import annotations

import pytest
import qubecalib.task.quelware.direct.single as single
import quel_ic_config
from e7awgsw import CaptureParam, DspUnit, WaveSequence
from qubecalib.task.quelware.direct.single import Awg, Runit
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


def test_quel_ic_config_version() -> None:
    assert quel_ic_config.__version__ == "0.8.10"


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


def test_run_runitonly(box: Quel1BoxWithRawWss) -> None:
    task = single.Task(
        box=box,
        settings=[
            single.RunitSetting(runit=Runit(port=1, runit=0), cprm=CAPTURE_PARAM),
            single.RunitSetting(runit=Runit(port=12, runit=0), cprm=CAPTURE_PARAM),
            single.RunitSetting(runit=Runit(port=12, runit=1), cprm=CAPTURE_PARAM),
            single.RunitSetting(runit=Runit(port=12, runit=2), cprm=CAPTURE_PARAM),
            single.RunitSetting(runit=Runit(port=12, runit=3), cprm=CAPTURE_PARAM),
        ],
    )
    assert task._runits_by_ports == {1: [0], 12: [0, 1, 2, 3]}
    assert task._channels == []
    assert task._triggers == {}
    futures = task.run()
    assert futures.keys() == {1, 12}
    # for port 1
    future_1 = futures[1]
    assert isinstance(future_1, single.Future)
    result_1 = future_1.result()
    assert isinstance(result_1[0], CaptureReturnCode)
    assert result_1[1].keys() == {0}
    for value in result_1[1].values():
        # num_samples_to_sum なのに words_to_sum が返ってくるのは仕様か？
        assert value.shape == (4 * CAPTURE_PARAM.num_samples_to_sum(0),)
    # for port 12
    future_12 = futures[12]
    assert isinstance(future_12, single.Future)
    result_12 = future_12.result()
    assert isinstance(result_12[0], CaptureReturnCode)
    assert result_12[1].keys() == {0, 1, 2, 3}
    for value in result_12[1].values():
        assert value.shape == (4 * CAPTURE_PARAM.num_samples_to_sum(0),)


def test_run_awgonly(box: Quel1BoxWithRawWss) -> None:
    task = single.Task(
        box=box,
        settings=[
            single.AwgSetting(
                awg=Awg(port=0, channel=0),
                wseq=WAVE_SEQUENCE,
            ),
            single.AwgSetting(
                awg=Awg(port=13, channel=0),
                wseq=WAVE_SEQUENCE,
            ),
        ],
    )
    assert task._runits_by_ports == {}
    assert task._channels == [Awg(port=0, channel=0), Awg(port=13, channel=0)]
    assert task._triggers == {}
    futures = task.run()
    assert futures is None


def test_run_runit_awg_notrig(box: Quel1BoxWithRawWss) -> None:
    task = single.Task(
        box=box,
        settings=[
            single.AwgSetting(
                awg=Awg(port=0, channel=0),
                wseq=WAVE_SEQUENCE,
            ),
            single.RunitSetting(
                runit=Runit(port=1, runit=0),
                cprm=CAPTURE_PARAM,
            ),
        ],
    )
    assert task._runits_by_ports == {1: [0]}
    assert task._channels == [Awg(port=0, channel=0)]
    assert task._triggers == {}
    futures = task.run()
    # for port 1
    future_1 = futures[1]
    assert isinstance(future_1, single.Future)
    result_1 = future_1.result()
    assert isinstance(result_1[0], CaptureReturnCode)
    assert result_1[1].keys() == {0}
    for value in result_1[1].values():
        assert value.shape == (4 * CAPTURE_PARAM.num_samples_to_sum(0),)


def test_run_runit_awg_trig(box: Quel1BoxWithRawWss) -> None:
    task = single.Task(
        box=box,
        settings=[
            single.AwgSetting(
                awg=Awg(port=0, channel=0),
                wseq=WAVE_SEQUENCE,
            ),
            single.RunitSetting(
                runit=Runit(port=1, runit=0),
                cprm=CAPTURE_PARAM,
            ),
            single.TriggerSetting(
                triggerd_port=1,
                trigger_awg=Awg(port=0, channel=0),
            ),
        ],
    )
    assert task._runits_by_ports == {1: [0]}
    assert task._channels == [Awg(port=0, channel=0)]
    assert task._triggers == {1: Awg(port=0, channel=0)}
    futures = task.run()
    # for port 1
    future_1 = futures[1]
    assert isinstance(future_1, single.Future)
    result_1 = future_1.result()
    assert isinstance(result_1[0], CaptureReturnCode)
    assert result_1[1].keys() == {0}
    for value in result_1[1].values():
        assert value.shape == (4 * CAPTURE_PARAM.num_samples_to_sum(0),)


def test_run_raise_no_settings(box: Quel1BoxWithRawWss) -> None:
    with pytest.raises(ValueError) as e:
        single.Task(box=box, settings=[])
    assert str(e.value) == "no settings provided"


def test_run_raise_already_loaded(box: Quel1BoxWithRawWss) -> None:
    task = single.Task(
        box=box,
        settings=[
            single.RunitSetting(runit=Runit(port=1, runit=0), cprm=CAPTURE_PARAM),
        ],
    )
    with pytest.raises(ValueError) as e:
        task._load([])
    assert str(e.value) == "already loaded"


def test_run_invalid_trigger(box: Quel1BoxWithRawWss) -> None:
    # noawg for triggerd port 1
    NUM_WORDS = int(8 * 4096 / 16)
    PERIOD = 1280 * 128
    c = create_capture_params(NUM_WORDS, PERIOD)
    task = single.Task(
        box=box,
        settings=[
            single.RunitSetting(runit=Runit(port=1, runit=0), cprm=c),
            single.RunitSetting(runit=Runit(port=1, runit=0), cprm=c),
            single.TriggerSetting(triggerd_port=1, trigger_awg=Awg(port=0, channel=0)),
        ],
    )
    with pytest.raises(ValueError) as e:
        task.run()
    assert str(e.value) == "trigger awg (0, 0) for triggerd port 1 is not provided"
    # norunit for triggerd port 1
    task = single.Task(
        box=box,
        settings=[
            single.AwgSetting(awg=Awg(port=0, channel=0), wseq=WAVE_SEQUENCE),
            single.TriggerSetting(triggerd_port=1, trigger_awg=Awg(port=0, channel=0)),
        ],
    )
    with pytest.raises(ValueError) as e:
        task.run()
    assert str(e.value) == "triggerd port 1 is not provided in runit settings"
