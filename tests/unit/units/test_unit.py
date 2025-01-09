import numpy as np
from qubecalib.units import (
    BLOCK,
    DEG,
    RAD,
    WORD,
    BLOCKs,
    GHz,
    Hz,
    MHz,
    Sec,
    WORDs,
    kHz,
    mS,
    nS,
    uS,
)


def test_units() -> None:
    """Units should have the correct values."""
    assert 0.2 * Sec == 0.2 * 1e9
    assert 0.2 * mS == 0.2 * 1e6
    assert 0.2 * uS == 0.2 * 1e3
    assert 0.2 * nS == 0.2 * 1.0
    assert 0.2 * Hz == 0.2 * 1e-9
    assert 0.2 * kHz == 0.2 * 1e-6
    assert 0.2 * MHz == 0.2 * 1e-3
    assert 0.2 * GHz == 0.2 * 1.0
    assert 2 * BLOCK == 2 * 128.0
    assert 2 * BLOCKs == 2 * 128.0
    assert 2 * WORD == 2 * 8.0
    assert 2 * WORDs == 2 * 8.0
    assert np.pi * RAD == np.pi * 1.0
    assert 90 * DEG == 90 * np.pi / 180.0
