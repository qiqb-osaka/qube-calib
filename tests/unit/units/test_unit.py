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


def test_units():
    """Units should have the correct values."""
    assert 0.2 * Sec == 0.2
    assert 0.2 * mS == 0.2e-3
    assert 0.2 * uS == 0.2e-6
    assert 0.2 * nS == 0.1e-9
    assert 0.2 * Hz == 0.2
    assert 0.2 * kHz == 0.2e3
    assert 0.2 * MHz == 0.2e6
    assert 0.2 * GHz == 0.2e9
    assert 0.2 * BLOCK == 0.2 * 128.0
    assert 0.2 * BLOCKs == 0.2 * 128.0
    assert 0.2 * WORD == 0.2 * 8.0
    assert 0.2 * WORDs == 0.2 * 8.0
    assert 0.2 * RAD == 0.2 * 1.0
    assert 0.2 * DEG == 0.2 * 3.141592653589793 / 180.0
