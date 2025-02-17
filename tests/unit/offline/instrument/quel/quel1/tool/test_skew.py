from __future__ import annotations

from qubecalib.instrument.quel.quel1.tool.skew import SkewAdjust
from qubecalib.qubecalib import SystemConfigDatabase


def test_create_database() -> None:
    """create_database should create a database."""
    db = SkewAdjust(SystemConfigDatabase())
    assert isinstance(db, SkewAdjust)
