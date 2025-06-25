import pytest


def test_add_waveforms_multi_chunk_edge() -> None:
    import numpy as np
    from qubecalib.instrument.quel.quel1.driver.tasksetting import TaskSettingBuilder

    b = TaskSettingBuilder()
    with pytest.raises(ValueError) as e:
        b.add_waveforms(
            indexed_waveforms=[
                [
                    (0, np.ones(65, dtype=np.complex64)),
                ],
                [
                    (127, np.ones(32, dtype=np.complex64)),
                ],
            ],
            port=0,
            channel=0,
            wait_words=0,
        )
    assert (
        str(e.value)
        == "Too close chunk: the chunk starting at sample index 127 must be placed at or after 128 to avoid overlap."
    )


def test_add_waveforms_multi_chunk_close_edge() -> None:
    import numpy as np
    from qubecalib.instrument.quel.quel1.driver.tasksetting import TaskSettingBuilder

    b = TaskSettingBuilder()
    b.add_waveforms(
        indexed_waveforms=[
            [
                (0, np.ones(64, dtype=np.complex64)),
            ],
            [
                (64, np.ones(32, dtype=np.complex64)),
            ],
        ],
        port=0,
        channel=0,
        wait_words=0,
    )
